from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from voltcast.cli import run_all
from voltcast.config import load_config
from voltcast.data import aggregate_to_hourly, parse_raw_minute_data
from voltcast.features import build_feature_table
from voltcast.profile_forecast import assign_causal_profile_routes


def test_parsing_missing_duplicates_and_hourly_coverage(tmp_path: Path):
    raw = tmp_path / "mini.txt"
    lines = ["Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3"]
    # Hour 00: 50 valid minutes (eligible target, incomplete circuit)
    for m in range(50):
        lines.append(f"01/01/2024;00:{m:02d}:00;1.2;0.1;240.0;5.0;1.0;0.0;2.0")
    # Duplicate timestamp
    lines.append("01/01/2024;00:00:00;9.9;0.1;240.0;5.0;1.0;0.0;2.0")
    # Hour 01: missing '?' for all 60 minutes (gap)
    lines.append("01/01/2024;01:00:00;?;?;?;?;?;?;?")
    # Hour 02: 60 valid minutes (eligible target + complete circuit)
    for m in range(60):
        lines.append(f"01/01/2024;02:{m:02d}:00;2.4;0.2;240.0;10.0;1.0;1.0;1.0")
    raw.write_text("\n".join(lines), encoding="utf-8")

    mdf, q = parse_raw_minute_data(raw)
    assert q["duplicate_timestamps_dropped"] == 1
    assert q["missing_active_power_minutes"] == 1

    hdf, hq = aggregate_to_hourly(mdf, min_valid_minutes_target=45, min_valid_minutes_circuit=60)
    assert len(hdf) == 3  # regularized 00, 01, 02
    assert bool(hdf.loc[0, "is_eligible_target"]) is True
    assert bool(hdf.loc[0, "is_complete_circuit_hour"]) is False
    assert bool(hdf.loc[1, "is_eligible_target"]) is False
    assert bool(hdf.loc[2, "is_complete_circuit_hour"]) is True
    assert np.isclose(hdf.loc[2, "sub1_wh_sum"], 60.0)


def test_no_future_leakage_in_features():
    cfg = load_config("config/default.yaml")
    idx = pd.date_range("2024-01-01 00:00:00", periods=240, freq="h")
    hourly = pd.DataFrame({
        "hour_start": idx,
        "active_kw_mean": np.linspace(1.0, 3.0, len(idx)),
        "reactive_kw_mean": 0.1,
        "voltage_v_mean": 240.0,
        "intensity_a_mean": 5.0,
        "sub1_wh_sum": 10.0,
        "sub2_wh_sum": 5.0,
        "sub3_wh_sum": 15.0,
        "valid_active_minutes": 60,
        "is_eligible_target": True,
    })
    f1, _ = build_feature_table(hourly, cfg)

    # Mutate future rows after index 150
    hourly_mut = hourly.copy()
    hourly_mut.loc[150:, "active_kw_mean"] = 999.0
    f2, _ = build_feature_table(hourly_mut, cfg)

    # All feature rows with target_hour <= idx[149] must be identical
    cutoff = idx[149]
    left = f1[f1["target_hour"] <= cutoff].drop(columns=["split"]).reset_index(drop=True)
    right = f2[f2["target_hour"] <= cutoff].drop(columns=["split"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_causal_profile_routing_never_uses_incomplete_current_day():
    daily_labels = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "cluster": [0, 1],
    })
    # At origin_hour 2024-01-02 15:00, 2024-01-02 is NOT complete; must route to 2024-01-01 (cluster 0)
    # At origin_hour 2024-01-02 23:00 (interval ends 2024-01-03 00:00), 2024-01-02 IS complete (cluster 1)
    origins = pd.Series([pd.Timestamp("2024-01-02 15:00:00"), pd.Timestamp("2024-01-02 23:00:00")])
    routed = assign_causal_profile_routes(origins, daily_labels, max_lookback_days=7)
    assert routed.loc[0, "route_cluster"] == 0
    assert routed.loc[0, "profile_source_date"] == "2024-01-01"
    assert routed.loc[1, "route_cluster"] == 1
    assert routed.loc[1, "profile_source_date"] == "2024-01-02"


def test_end_to_end_smoke_pipeline(tmp_path: Path):
    cfg = load_config("config/default.yaml")
    cfg["paths"]["smoke_artifacts_dir"] = str(tmp_path / "smoke_out")
    art = run_all(cfg, smoke=True)
    assert (art / "run_manifest.json").exists()
    assert (art / "regression_metrics.csv").exists()
    assert (art / "classification_metrics.csv").exists()
    assert (art / "profile_forecast_comparison.json").exists()
    assert (art / "circuit_analysis.json").exists()