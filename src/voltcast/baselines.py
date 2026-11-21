from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_kw": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kw": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def run_baselines_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    df = pd.read_csv(artifacts_dir / "features.csv", parse_dates=["origin_hour", "target_hour"])
    preds = []
    metrics_rows = []

    for model_name, col in [
        ("persistence_1h", "active_kw_origin"),
        ("seasonal_naive_24h", "active_kw_lag_24h_from_target"),
    ]:
        sub = df[["origin_hour", "target_hour", "split", "actual_kw", col]].copy()
        sub = sub.rename(columns={col: "predicted_kw"})
        sub["model"] = model_name
        preds.append(sub)

        for split_name in ["train", "val", "test"]:
            s = sub[sub["split"] == split_name]
            avail = s.dropna(subset=["predicted_kw"])
            m = evaluate_regression_metrics(avail["actual_kw"].to_numpy(), avail["predicted_kw"].to_numpy())
            m.update({
                "model": model_name,
                "split": split_name,
                "availability_rate": float(len(avail) / max(len(s), 1)),
            })
            metrics_rows.append(m)

    pred_df = pd.concat(preds, ignore_index=True)
    pred_path = artifacts_dir / "baseline_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    pd.DataFrame(metrics_rows).to_csv(artifacts_dir / "baseline_metrics.csv", index=False)
    return pred_path