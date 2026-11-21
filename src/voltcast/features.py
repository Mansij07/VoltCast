from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from voltcast.config import save_manifest

FEATURE_COLUMNS = [
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "is_weekend",
    "target_month",
    "active_kw_origin",
    "active_kw_lag_2h_from_target",
    "active_kw_lag_3h_from_target",
    "active_kw_lag_24h_from_target",
    "active_kw_lag_48h_from_target",
    "active_kw_lag_168h_from_target",
    "active_kw_roll_mean_3h",
    "active_kw_roll_std_3h",
    "active_kw_roll_mean_6h",
    "active_kw_roll_std_6h",
    "active_kw_roll_mean_24h",
    "active_kw_roll_std_24h",
    "reactive_kw_origin",
    "voltage_v_origin",
    "intensity_a_origin",
    "sub1_wh_origin",
    "sub2_wh_origin",
    "sub3_wh_origin",
    "origin_valid_minutes",
]


def compute_chronological_day_splits(
    target_hours: pd.Series, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.Series, dict[str, str]]:
    days = np.sort(target_hours.dt.normalize().unique())
    n_days = len(days)
    if n_days < 5:
        raise ValueError(f"Need at least 5 calendar days to split, got {n_days}")

    train_end_idx = max(1, int(np.floor(n_days * train_frac)) - 1)
    val_end_idx = max(train_end_idx + 1, int(np.floor(n_days * (train_frac + val_frac))) - 1)
    val_end_idx = min(val_end_idx, n_days - 2)

    train_end_day = pd.Timestamp(days[train_end_idx])
    val_end_day = pd.Timestamp(days[val_end_idx])

    norm_days = target_hours.dt.normalize()
    split = np.where(
        norm_days <= train_end_day,
        "train",
        np.where(norm_days <= val_end_day, "val", "test"),
    )
    boundaries = {
        "train_start": str(pd.Timestamp(days[0]).date()),
        "train_end": str(train_end_day.date()),
        "val_start": str(pd.Timestamp(days[train_end_idx + 1]).date()),
        "val_end": str(val_end_day.date()),
        "test_start": str(pd.Timestamp(days[val_end_idx + 1]).date()),
        "test_end": str(pd.Timestamp(days[-1]).date()),
    }
    return pd.Series(split, index=target_hours.index, name="split"), boundaries


def build_feature_table(hourly: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    df = hourly.sort_values("hour_start").reset_index(drop=True).copy()
    df["hour_start"] = pd.to_datetime(df["hour_start"])

    # Verify regular 1-hour grid
    diffs = df["hour_start"].diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError("Hourly table is not on a strictly regular 1-hour grid.")

    # Row i represents target_hour = hour_start[i], origin_hour = hour_start[i] - 1h
    out = pd.DataFrame()
    out["target_hour"] = df["hour_start"]
    out["origin_hour"] = df["hour_start"] - pd.Timedelta(hours=1)
    out["actual_kw"] = df["active_kw_mean"]
    out["is_eligible_target"] = df["is_eligible_target"].astype(bool)

    # Calendar features known in advance for target_hour
    h = out["target_hour"].dt.hour
    dow = out["target_hour"].dt.dayofweek
    out["target_hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    out["target_hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    out["target_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["target_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(int)
    out["target_month"] = out["target_hour"].dt.month

    # Causal lags: shift(1) is origin_hour (t = target_hour - 1h)
    active = df["active_kw_mean"]
    out["active_kw_origin"] = active.shift(1)
    for lag in [2, 3, 24, 48, 168]:
        out[f"active_kw_lag_{lag}h_from_target"] = active.shift(lag)

    # Causal rolling windows ending at origin_hour: shift(1).rolling(w)
    shifted = active.shift(1)
    for w in cfg["features"]["rolling_windows_hours"]:
        out[f"active_kw_roll_mean_{w}h"] = shifted.rolling(w, min_periods=1).mean()
        out[f"active_kw_roll_std_{w}h"] = shifted.rolling(w, min_periods=2).std()

    # Causal electrical & circuit readings at origin_hour
    out["reactive_kw_origin"] = df["reactive_kw_mean"].shift(1)
    out["voltage_v_origin"] = df["voltage_v_mean"].shift(1)
    out["intensity_a_origin"] = df["intensity_a_mean"].shift(1)
    out["sub1_wh_origin"] = df["sub1_wh_sum"].shift(1)
    out["sub2_wh_origin"] = df["sub2_wh_sum"].shift(1)
    out["sub3_wh_origin"] = df["sub3_wh_sum"].shift(1)
    out["origin_valid_minutes"] = df["valid_active_minutes"].shift(1)

    # Keep only eligible target rows with valid origin_hour
    out = out[out["is_eligible_target"] & out["actual_kw"].notna()].copy()
    out = out.iloc[1:].reset_index(drop=True)  # drop very first hour (no origin)

    splits, boundaries = compute_chronological_day_splits(
        out["target_hour"],
        train_frac=cfg["splits"]["train_frac"],
        val_frac=cfg["splits"]["val_frac"],
    )
    out.insert(2, "split", splits)
    return out, boundaries


def run_features_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    hourly = pd.read_csv(artifacts_dir / "hourly_observations.csv", parse_dates=["hour_start"])
    feat_df, boundaries = build_feature_table(hourly, cfg)

    feat_path = artifacts_dir / "features.csv"
    feat_df.to_csv(feat_path, index=False)

    split_counts = feat_df["split"].value_counts().to_dict()
    split_manifest = {"boundaries": boundaries, "counts": split_counts}
    (artifacts_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    save_manifest(
        artifacts_dir,
        {
            "features_path": str(feat_path),
            "feature_columns": FEATURE_COLUMNS,
            "split_boundaries": boundaries,
            "split_counts": split_counts,
        },
    )
    return feat_path