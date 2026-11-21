from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from voltcast.config import load_manifest, save_manifest
from voltcast.data import parse_raw_minute_data


def run_circuit_analysis_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    manifest = load_manifest(artifacts_dir)
    raw_path = Path(manifest["raw_data_path"])
    peak_kw_cutoff = float(manifest["peak_kw_cutoff"])

    minute_df, _ = parse_raw_minute_data(raw_path)
    cols = ["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    matched = minute_df.dropna(subset=cols).copy()

    # Exact unit conversion: 1 kW for 1 minute = 1000 / 60 Wh
    matched["total_active_wh"] = matched["Global_active_power"] * (1000.0 / 60.0)
    matched["unmetered_wh"] = (
        matched["total_active_wh"]
        - matched["Sub_metering_1"]
        - matched["Sub_metering_2"]
        - matched["Sub_metering_3"]
    ).clip(lower=0.0)
    matched["is_peak_minute"] = matched["Global_active_power"] >= peak_kw_cutoff

    total_wh_sum = float(matched["total_active_wh"].sum())
    summary = {
        "matched_complete_minutes": int(len(matched)),
        "excluded_incomplete_minutes": int(len(minute_df) - len(matched)),
        "unit_conversion": "1 kW over 1 minute = 1000/60 Wh",
        "note": "Sub-metering channels represent groups of loads (kitchen, laundry, climate control); associations are observational, not causal.",
        "energy_shares_pct": {
            "sub1_kitchen_pct": round(100.0 * float(matched["Sub_metering_1"].sum()) / max(total_wh_sum, 1e-9), 2),
            "sub2_laundry_pct": round(100.0 * float(matched["Sub_metering_2"].sum()) / max(total_wh_sum, 1e-9), 2),
            "sub3_climate_pct": round(100.0 * float(matched["Sub_metering_3"].sum()) / max(total_wh_sum, 1e-9), 2),
            "unmetered_other_pct": round(100.0 * float(matched["unmetered_wh"].sum()) / max(total_wh_sum, 1e-9), 2),
        },
        "mean_wh_per_min_by_peak_state": {
            "peak_minutes": matched[matched["is_peak_minute"]][["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "unmetered_wh"]].mean().round(3).to_dict(),
            "ordinary_minutes": matched[~matched["is_peak_minute"]][["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "unmetered_wh"]].mean().round(3).to_dict(),
        },
    }

    out_path = artifacts_dir / "circuit_analysis.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_manifest(artifacts_dir, {"circuit_analysis_path": str(out_path)})
    return out_path