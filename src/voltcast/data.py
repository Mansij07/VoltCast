from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from voltcast.config import file_sha256, save_manifest

REQUIRED_RAW_COLS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

NUMERIC_COLS = REQUIRED_RAW_COLS[2:]


def fetch_uci_dataset(cfg: dict[str, Any]) -> Path:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_txt = raw_dir / cfg["paths"]["raw_filename"]
    if target_txt.exists():
        return target_txt

    zip_path = raw_dir / "uci_household_power.zip"
    if not zip_path.exists():
        url = cfg["paths"]["raw_zip_url"]
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    if not target_txt.exists():
        raise FileNotFoundError(
            f"Expected {target_txt} after extracting {zip_path}. "
            "If download is blocked, place household_power_consumption.txt in data/raw/."
        )
    return target_txt


def generate_synthetic_raw_fixture(out_path: Path, days: int = 35, seed: int = 42) -> Path:
    """Create a realistic minute-level semicolon-delimited fixture for smoke tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 00:00:00", periods=days * 24 * 60, freq="min")
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    # Two recurring daily patterns + weekend boost + diurnal wave
    morning_peak = np.exp(-0.5 * ((hour - 8) / 2.0) ** 2) * 1.4
    evening_peak = np.exp(-0.5 * ((hour - 19) / 2.2) ** 2) * 2.2
    weekend_bump = np.where(dow >= 5, 0.45, 0.0)
    day_regime = np.where((idx.day % 3) == 0, 0.8, 0.0) * np.exp(-0.5 * ((hour - 13) / 2.5) ** 2)

    active = np.clip(
        0.45 + morning_peak + evening_peak + weekend_bump + day_regime + rng.normal(0, 0.25, len(idx)),
        0.1,
        8.5,
    )
    reactive = np.clip(0.08 + 0.05 * active + rng.normal(0, 0.02, len(idx)), 0.0, 1.5)
    voltage = 240.0 - 0.8 * active + rng.normal(0, 1.0, len(idx))
    intensity = np.clip((active * 1000.0) / np.maximum(voltage, 200.0), 0.1, 40.0)

    total_wh_min = active * (1000.0 / 60.0)
    sub1 = np.where((hour >= 7) & (hour <= 9) | (hour >= 18) & (hour <= 20), total_wh_min * 0.22, 0.0)
    sub2 = np.where((dow >= 5) & (hour >= 10) & (hour <= 14), total_wh_min * 0.18, 0.0)
    sub3 = np.clip(total_wh_min * 0.35 + rng.normal(0, 1.0, len(idx)), 0.0, total_wh_min * 0.6)

    df = pd.DataFrame({
        "Date": idx.strftime("%d/%m/%Y"),
        "Time": idx.strftime("%H:%M:%S"),
        "Global_active_power": np.round(active, 3),
        "Global_reactive_power": np.round(reactive, 3),
        "Voltage": np.round(voltage, 2),
        "Global_intensity": np.round(intensity, 2),
        "Sub_metering_1": np.round(sub1, 2),
        "Sub_metering_2": np.round(sub2, 2),
        "Sub_metering_3": np.round(sub3, 2),
    })

    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(object)
    df.loc[500:525, NUMERIC_COLS] = "?"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=";", index=False)
    return out_path


def parse_raw_minute_data(raw_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values=["?", ""],
        low_memory=False,
    )
    missing_cols = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")

    raw_rows = len(df)
    ts = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    parse_errors = int(ts.isna().sum())
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).copy()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag physically invalid negative values
    invalid_neg = 0
    for col in ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        mask = df[col] < 0
        invalid_neg += int(mask.sum())
        df.loc[mask, col] = np.nan

    dup_count = int(df["timestamp"].duplicated(keep="first").sum())
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)

    quality = {
        "raw_rows": raw_rows,
        "timestamp_parse_errors": parse_errors,
        "duplicate_timestamps_dropped": dup_count,
        "invalid_negative_values_nulled": invalid_neg,
        "missing_active_power_minutes": int(df["Global_active_power"].isna().sum()),
        "start_timestamp": str(df["timestamp"].min()),
        "end_timestamp": str(df["timestamp"].max()),
    }
    return df, quality


def aggregate_to_hourly(
    minute_df: pd.DataFrame,
    min_valid_minutes_target: int = 45,
    min_valid_minutes_circuit: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = minute_df.copy()
    df["hour_start"] = df["timestamp"].dt.floor("h")

    agg = df.groupby("hour_start", as_index=True).agg(
        active_kw_mean=("Global_active_power", "mean"),
        reactive_kw_mean=("Global_reactive_power", "mean"),
        voltage_v_mean=("Voltage", "mean"),
        intensity_a_mean=("Global_intensity", "mean"),
        sub1_wh_sum=("Sub_metering_1", "sum"),
        sub2_wh_sum=("Sub_metering_2", "sum"),
        sub3_wh_sum=("Sub_metering_3", "sum"),
        valid_active_minutes=("Global_active_power", "count"),
        valid_sub1_minutes=("Sub_metering_1", "count"),
        valid_sub2_minutes=("Sub_metering_2", "count"),
        valid_sub3_minutes=("Sub_metering_3", "count"),
    )

    # Regularize hourly timeline so row shifts always equal 1 hour
    full_grid = pd.date_range(agg.index.min(), agg.index.max(), freq="h", name="hour_start")
    hourly = agg.reindex(full_grid)
    count_cols = ["valid_active_minutes", "valid_sub1_minutes", "valid_sub2_minutes", "valid_sub3_minutes"]
    hourly[count_cols] = hourly[count_cols].fillna(0).astype(int)

    # Null out sums where 0 valid minutes exist
    for s_col, c_col in [("sub1_wh_sum", "valid_sub1_minutes"),
                         ("sub2_wh_sum", "valid_sub2_minutes"),
                         ("sub3_wh_sum", "valid_sub3_minutes")]:
        hourly.loc[hourly[c_col] == 0, s_col] = np.nan

    hourly["is_eligible_target"] = hourly["valid_active_minutes"] >= min_valid_minutes_target
    hourly["is_complete_circuit_hour"] = (
        (hourly["valid_sub1_minutes"] >= min_valid_minutes_circuit)
        & (hourly["valid_sub2_minutes"] >= min_valid_minutes_circuit)
        & (hourly["valid_sub3_minutes"] >= min_valid_minutes_circuit)
    )
    # Treat ineligible target hours as missing target
    hourly.loc[~hourly["is_eligible_target"], "active_kw_mean"] = np.nan
    hourly = hourly.reset_index()

    stats = {
        "total_regularized_hours": int(len(hourly)),
        "missing_gap_hours": int((hourly["valid_active_minutes"] == 0).sum()),
        "partial_hours_below_threshold": int(
            ((hourly["valid_active_minutes"] > 0) & (~hourly["is_eligible_target"])).sum()
        ),
        "eligible_target_hours": int(hourly["is_eligible_target"].sum()),
        "complete_circuit_hours": int(hourly["is_complete_circuit_hour"].sum()),
        "excluded_circuit_hours": int((~hourly["is_complete_circuit_hour"]).sum()),
    }
    return hourly, stats


def generate_eda_plots(hourly: pd.DataFrame, plots_dir: Path) -> list[str]:
    saved: list[str] = []
    df = hourly.copy()
    df["hour"] = df["hour_start"].dt.hour
    df["is_weekend"] = df["hour_start"].dt.dayofweek >= 5
    df["month"] = df["hour_start"].dt.month

    # 1. Hourly & Weekday/Weekend profile
    fig, ax = plt.subplots(figsize=(8, 4))
    for flag, label in [(False, "Weekday"), (True, "Weekend")]:
        sub = df[df["is_weekend"] == flag].groupby("hour")["active_kw_mean"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=label)
    ax.set_title(f"Mean Hourly Active Power (n={int(df['active_kw_mean'].notna().sum())} eligible hours)")
    ax.set_xlabel("Hour of Day (local clock)")
    ax.set_ylabel("Active Power (kW)")
    ax.legend()
    ax.grid(alpha=0.3)
    p1 = plots_dir / "eda_hourly_weekday_weekend.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Distribution & Missingness
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["active_kw_mean"].dropna().hist(bins=40, ax=axes[0], color="steelblue")
    axes[0].set_title("Hourly Active Power Distribution")
    axes[0].set_xlabel("Active Power (kW)")
    axes[0].set_ylabel("Hours")
    axes[1].plot(df["hour_start"], df["valid_active_minutes"], lw=0.7, color="darkorange")
    axes[1].set_title("Observed Minutes per Hour")
    axes[1].set_ylabel("Valid Minutes (0-60)")
    p2 = plots_dir / "eda_distribution_missingness.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    saved.append(str(p2))

    # 3. Correlation heatmap
    corr_cols = ["active_kw_mean", "reactive_kw_mean", "voltage_v_mean", "intensity_a_mean",
                 "sub1_wh_sum", "sub2_wh_sum", "sub3_wh_sum"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Hourly Feature Correlations")
    p3 = plots_dir / "eda_correlations.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    saved.append(str(p3))

    # 4. Complete-hour sub-metering totals by hour
    comp = df[df["is_complete_circuit_hour"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    if not comp.empty:
        by_h = comp.groupby("hour")[["sub1_wh_sum", "sub2_wh_sum", "sub3_wh_sum"]].mean()
        by_h.plot(ax=ax)
    ax.set_title(
        f"Mean Sub-Metered Energy by Hour (Complete 60-min hours n={len(comp)}, excluded={len(df)-len(comp)})"
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Energy (Wh)")
    ax.grid(alpha=0.3)
    p4 = plots_dir / "eda_submetering_hourly.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=120)
    plt.close(fig)
    saved.append(str(p4))

    # 5. Seasonal patterns: monthly mean and month x hour heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
    by_m = df.groupby("month")["active_kw_mean"].mean().reindex(range(1, 13))
    axes[0].bar(by_m.index, by_m.values, color="steelblue")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_title("Mean Hourly Active Power by Month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Active Power (kW)")
    mh = df.pivot_table(index="month", columns="hour", values="active_kw_mean", aggfunc="mean").reindex(range(1, 13))
    sns.heatmap(mh, cmap="viridis", ax=axes[1], cbar_kws={"label": "kW"})
    axes[1].set_title("Mean Active Power: Month x Hour of Day")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Month")
    p5 = plots_dir / "eda_seasonal.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=120)
    plt.close(fig)
    saved.append(str(p5))
    return saved


def run_data_stage(cfg: dict[str, Any], artifacts_dir: Path, smoke: bool = False) -> Path:
    if smoke:
        raw_path = artifacts_dir / "synthetic_raw.txt"
        generate_synthetic_raw_fixture(raw_path, days=35, seed=cfg["project"]["random_seed"])
    else:
        raw_path = Path(cfg["paths"]["raw_dir"]) / cfg["paths"]["raw_filename"]
        if not raw_path.exists():
            raw_path = fetch_uci_dataset(cfg)

    minute_df, raw_q = parse_raw_minute_data(raw_path)
    hourly_df, hour_q = aggregate_to_hourly(
        minute_df,
        min_valid_minutes_target=cfg["data"]["min_valid_minutes_target"],
        min_valid_minutes_circuit=cfg["data"]["min_valid_minutes_circuit_full_hour"],
    )

    hourly_path = artifacts_dir / "hourly_observations.csv"
    hourly_df.to_csv(hourly_path, index=False)

    # Save minute-level matched subset stats or path reference for M6 circuit analysis
    quality_summary = {**raw_q, **hour_q, "timezone_note": cfg["data"]["timezone_note"]}
    q_path = artifacts_dir / "data_quality_summary.json"
    q_path.write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")

    eda_plots = generate_eda_plots(hourly_df, artifacts_dir / "plots")

    save_manifest(
        artifacts_dir,
        {
            "config_version": cfg["project"]["config_version"],
            "run_mode": "smoke" if smoke else "full",
            "raw_data_path": str(raw_path),
            "dataset_fingerprint": file_sha256(raw_path),
            "hourly_observations_path": str(hourly_path),
            "data_quality_summary_path": str(q_path),
            "eda_plots": eda_plots,
        },
    )
    return hourly_path