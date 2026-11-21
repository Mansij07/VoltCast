from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from voltcast.baselines import evaluate_regression_metrics
from voltcast.config import load_manifest, save_manifest
from voltcast.features import FEATURE_COLUMNS


def assign_causal_profile_routes(
    origin_hours: pd.Series,
    daily_labels: pd.DataFrame,
    max_lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Strictly causal routing: at origin_hour t (interval [t, t+1h) just closed),
    a day d is fully observed iff d + 24h <= t + 1h.
    Select the most recent complete day within max_lookback_days.
    """
    dl = daily_labels.copy()
    dl["date_ts"] = pd.to_datetime(dl["date"])
    dl["day_complete_at"] = dl["date_ts"] + pd.Timedelta(hours=24)
    dl = dl.sort_values("day_complete_at").reset_index(drop=True)

    complete_times = dl["day_complete_at"].to_numpy()
    clusters = dl["cluster"].to_numpy()
    dates_str = dl["date"].to_numpy()

    routes = []
    source_dates = []
    obs_end = (pd.to_datetime(origin_hours) + pd.Timedelta(hours=1)).to_numpy()

    for t_end in obs_end:
        idx = int(np.searchsorted(complete_times, t_end, side="right")) - 1
        if idx >= 0:
            age_days = (pd.Timestamp(t_end) - pd.Timestamp(complete_times[idx])).total_seconds() / 86400.0
            if age_days <= max_lookback_days:
                routes.append(int(clusters[idx]))
                source_dates.append(str(dates_str[idx]))
                continue
        routes.append(-1)
        source_dates.append("")

    return pd.DataFrame({"route_cluster": routes, "profile_source_date": source_dates}, index=origin_hours.index)


def run_profile_forecast_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    manifest = load_manifest(artifacts_dir)
    global_model_name = manifest["selected_global_regressor"]
    global_est = joblib.load(artifacts_dir / "models" / f"reg_{global_model_name}.joblib")

    df = pd.read_csv(artifacts_dir / "features.csv", parse_dates=["origin_hour", "target_hour"])
    daily_labels = pd.read_csv(artifacts_dir / "daily_profile_labels.csv")

    routing = assign_causal_profile_routes(
        df["origin_hour"], daily_labels, max_lookback_days=cfg["profile_forecast"]["max_lookback_days"]
    )
    df = pd.concat([df, routing], axis=1)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    min_rows = cfg["profile_forecast"]["min_segment_train_rows"]
    seg_models = {}
    for cid, grp in train_df[train_df["route_cluster"] >= 0].groupby("route_cluster"):
        if len(grp) >= min_rows:
            est = clone(global_est)
            est.fit(grp[FEATURE_COLUMNS], grp["actual_kw"].to_numpy())
            seg_models[int(cid)] = est

    global_preds = global_est.predict(test_df[FEATURE_COLUMNS])
    prof_preds = np.empty(len(test_df), dtype=float)
    used_fallback = np.zeros(len(test_df), dtype=bool)

    for i, (_, row) in enumerate(test_df.iterrows()):
        cid = int(row["route_cluster"])
        if cid in seg_models:
            prof_preds[i] = seg_models[cid].predict(test_df.iloc[[i]][FEATURE_COLUMNS])[0]
        else:
            prof_preds[i] = global_preds[i]
            used_fallback[i] = True

    out = test_df[["origin_hour", "target_hour", "split", "actual_kw", "route_cluster", "profile_source_date"]].copy()
    out["global_pred_kw"] = global_preds
    out["profile_aware_pred_kw"] = prof_preds
    out["used_fallback"] = used_fallback

    preds_path = artifacts_dir / "profile_forecast_predictions.csv"
    out.to_csv(preds_path, index=False)

    # Fair comparison on identical test timestamps
    comp = {
        "global_model_family": global_model_name,
        "test_n_samples": int(len(out)),
        "fallback_rate": float(used_fallback.mean()),
        "trained_segments": sorted(seg_models.keys()),
        "global_test_metrics": evaluate_regression_metrics(out["actual_kw"].to_numpy(), out["global_pred_kw"].to_numpy()),
        "profile_aware_test_metrics": evaluate_regression_metrics(out["actual_kw"].to_numpy(), out["profile_aware_pred_kw"].to_numpy()),
    }
    comp_path = artifacts_dir / "profile_forecast_comparison.json"
    comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")

    save_manifest(artifacts_dir, {"profile_forecast_predictions_path": str(preds_path), "profile_forecast_comparison_path": str(comp_path)})
    return preds_path