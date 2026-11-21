from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from voltcast.baselines import evaluate_regression_metrics
from voltcast.config import save_manifest
from voltcast.features import FEATURE_COLUMNS


def _build_regression_specs(cfg: dict[str, Any]) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    seed = cfg["project"]["random_seed"]
    alphas = cfg["regression"]["alpha_grid"]
    depths = cfg["regression"]["max_depth_grid"]

    return {
        "linear_regression": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("reg", LinearRegression())]),
            {},
        ),
        "ridge": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("reg", Ridge(random_state=seed))]),
            {"reg__alpha": alphas},
        ),
        "lasso": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("reg", Lasso(random_state=seed, max_iter=3000))]),
            {"reg__alpha": alphas},
        ),
        "decision_tree": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", DecisionTreeRegressor(random_state=seed))]),
            {"reg__max_depth": depths, "reg__min_samples_leaf": [5, 15]},
        ),
        "random_forest": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("reg", RandomForestRegressor(n_estimators=cfg["regression"]["rf_n_estimators"], random_state=seed, n_jobs=-1)),
            ]),
            {"reg__max_depth": depths[:2], "reg__min_samples_leaf": [5, 10]},
        ),
        "gradient_boosting": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("reg", GradientBoostingRegressor(n_estimators=cfg["regression"]["gb_n_estimators"], random_state=seed)),
            ]),
            {"reg__max_depth": [3, 5], "reg__learning_rate": [0.05, 0.1]},
        ),
    }


def run_regression_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    df = pd.read_csv(artifacts_dir / "features.csv", parse_dates=["origin_hour", "target_hour"])
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["actual_kw"].to_numpy()
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["actual_kw"].to_numpy()
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["actual_kw"].to_numpy()

    peak_kw_cutoff = float(np.quantile(y_train, cfg["classification"]["peak_quantile"]))
    tscv = TimeSeriesSplit(n_splits=cfg["regression"]["cv_splits"])

    specs = _build_regression_specs(cfg)
    metrics_rows = []
    pred_frames = []
    fitted_models = {}

    for name, (pipe, grid) in specs.items():
        t0 = time.perf_counter()
        if grid:
            gs = GridSearchCV(pipe, grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_est = gs.best_estimator_
            best_params = gs.best_params_
        else:
            pipe.fit(X_train, y_train)
            best_est = pipe
            best_params = {}
        fit_sec = time.perf_counter() - t0

        fitted_models[name] = best_est
        joblib.dump(best_est, artifacts_dir / "models" / f"reg_{name}.joblib")

        for split_name, part_df, X_p, y_p in [
            ("val", val_df, X_val, y_val),
            ("test", test_df, X_test, y_test),
        ]:
            y_hat = best_est.predict(X_p)
            m = evaluate_regression_metrics(y_p, y_hat)
            # Peak vs ordinary slice metrics
            is_pk = y_p >= peak_kw_cutoff
            m_pk = evaluate_regression_metrics(y_p[is_pk], y_hat[is_pk]) if is_pk.any() else {"mae_kw": np.nan, "n_samples": 0}
            m_ord = evaluate_regression_metrics(y_p[~is_pk], y_hat[~is_pk]) if (~is_pk).any() else {"mae_kw": np.nan, "n_samples": 0}

            m.update({
                "model": name,
                "split": split_name,
                "fit_time_sec": round(fit_sec, 3),
                "best_params": json.dumps(best_params),
                "peak_mae_kw": m_pk["mae_kw"],
                "peak_n": m_pk["n_samples"],
                "ordinary_mae_kw": m_ord["mae_kw"],
                "ordinary_n": m_ord["n_samples"],
            })
            metrics_rows.append(m)

            pf = part_df[["origin_hour", "target_hour", "split", "actual_kw"]].copy()
            pf["model"] = name
            pf["predicted_kw"] = y_hat
            pred_frames.append(pf)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = artifacts_dir / "regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    preds_df = pd.concat(pred_frames, ignore_index=True)
    preds_path = artifacts_dir / "regression_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Select best model strictly on validation MAE
    val_metrics = metrics_df[metrics_df["split"] == "val"].sort_values("mae_kw")
    best_model_name = str(val_metrics.iloc[0]["model"])

    # Interpretation: standardized linear coefficients + best tree/permutation importance
    interp = {}
    if "ridge" in fitted_models:
        coef = fitted_models["ridge"].named_steps["reg"].coef_
        interp["ridge_standardized_coef"] = dict(zip(FEATURE_COLUMNS, [float(c) for c in coef]))
    best_est = fitted_models[best_model_name]
    if hasattr(best_est.named_steps["reg"], "feature_importances_"):
        fi = best_est.named_steps["reg"].feature_importances_
        interp["best_model_tree_importance"] = dict(zip(FEATURE_COLUMNS, [float(v) for v in fi]))
    perm = permutation_importance(best_est, X_val, y_val, n_repeats=5, random_state=cfg["project"]["random_seed"], scoring="neg_mean_absolute_error")
    interp["best_model_val_permutation_importance"] = dict(zip(FEATURE_COLUMNS, [float(v) for v in perm.importances_mean]))
    (artifacts_dir / "regression_interpretation.json").write_text(json.dumps(interp, indent=2), encoding="utf-8")

    # Diagnostic plots on test slice
    best_test = preds_df[(preds_df["model"] == best_model_name) & (preds_df["split"] == "test")].copy()
    best_test["residual"] = best_test["actual_kw"] - best_test["predicted_kw"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    win = best_test.head(168)
    axes[0].plot(win["target_hour"], win["actual_kw"], label="Observed (kW)")
    axes[0].plot(win["target_hour"], win["predicted_kw"], label=f"Predicted ({best_model_name})", alpha=0.85)
    axes[0].set_title(f"Test Window Forecast vs Observed ({best_model_name}, first {len(win)} test hours)")
    axes[0].set_ylabel("Active Power (kW)")
    axes[0].legend()
    axes[1].hist(best_test["residual"], bins=40, color="slategray")
    axes[1].set_title(f"Test Residual Distribution (n={len(best_test)})")
    axes[1].set_xlabel("Observed - Predicted (kW)")
    fig.tight_layout()
    p_diag = artifacts_dir / "plots" / "regression_diagnostics.png"
    fig.savefig(p_diag, dpi=120)
    plt.close(fig)

    save_manifest(
        artifacts_dir,
        {
            "regression_metrics_path": str(metrics_path),
            "regression_predictions_path": str(preds_path),
            "selected_global_regressor": best_model_name,
            "peak_kw_cutoff": peak_kw_cutoff,
        },
    )
    return preds_path