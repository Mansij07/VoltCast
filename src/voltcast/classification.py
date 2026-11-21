from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from voltcast.config import save_manifest
from voltcast.features import FEATURE_COLUMNS


def select_validation_threshold(
    y_val: np.ndarray, probs_val: np.ndarray, min_precision: float = 0.50
) -> tuple[float, str]:
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.50, "default_empty_thresholds"

    p = precisions[:-1]
    r = recalls[:-1]
    qual = np.where(p >= min_precision)[0]
    if len(qual) > 0:
        best_idx = qual[np.argmax(r[qual])]
        return float(thresholds[best_idx]), f"max_recall_precision_ge_{min_precision:.2f}"

    f1s = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), "fallback_max_f1"


def _eval_cls_slice(y_true: np.ndarray, probs: np.ndarray, cutoff: float) -> dict[str, Any]:
    y_pred = (probs >= cutoff).astype(int)
    one_class = len(np.unique(y_true)) < 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(y_true)),
        "peak_rate": float(y_true.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if not one_class else np.nan,
        "average_precision": float(average_precision_score(y_true, probs)) if not one_class else np.nan,
    }


def run_classification_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    df = pd.read_csv(artifacts_dir / "features.csv", parse_dates=["origin_hour", "target_hour"])
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Train-only kW peak cutoff (frozen)
    peak_kw_cutoff = float(np.quantile(train_df["actual_kw"].to_numpy(), cfg["classification"]["peak_quantile"]))
    y_train = (train_df["actual_kw"].to_numpy() >= peak_kw_cutoff).astype(int)
    y_val = (val_df["actual_kw"].to_numpy() >= peak_kw_cutoff).astype(int)
    y_test = (test_df["actual_kw"].to_numpy() >= peak_kw_cutoff).astype(int)

    X_train, X_val, X_test = train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS], test_df[FEATURE_COLUMNS]
    seed = cfg["project"]["random_seed"]
    tscv = TimeSeriesSplit(n_splits=cfg["classification"]["cv_splits"])

    specs = {
        "logistic_regression": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))]),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "gaussian_nb": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", GaussianNB())]),
            {},
        ),
        "knn": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": cfg["classification"]["knn_neighbors_grid"]},
        ),
        "decision_tree": (
            Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", DecisionTreeClassifier(random_state=seed))]),
            {"clf__max_depth": [3, 6, 10], "clf__min_samples_leaf": [5, 15]},
        ),
    }

    metrics_rows = []
    pred_frames = []
    fig, ax = plt.subplots(figsize=(7, 5))

    # Simple baseline: peak persistence (origin_hour >= peak_kw_cutoff)
    for split_name, part_df, y_true in [("val", val_df, y_val), ("test", test_df, y_test)]:
        avail_mask = part_df["active_kw_origin"].notna().to_numpy()
        p_base = (part_df.loc[avail_mask, "active_kw_origin"].to_numpy() >= peak_kw_cutoff).astype(float)
        m = _eval_cls_slice(y_true[avail_mask], p_base, 0.50)
        m.update({"model": "peak_persistence_baseline", "split": split_name, "threshold_type": "0.50", "prob_cutoff": 0.50, "rule": "baseline", "availability_rate": float(avail_mask.mean())})
        metrics_rows.append(m)

    for name, (pipe, grid) in specs.items():
        if grid:
            gs = GridSearchCV(pipe, grid, cv=tscv, scoring="average_precision", n_jobs=-1)
            gs.fit(X_train, y_train)
            best_est = gs.best_estimator_
        else:
            pipe.fit(X_train, y_train)
            best_est = pipe

        joblib.dump(best_est, artifacts_dir / "models" / f"clf_{name}.joblib")
        probs_val = best_est.predict_proba(X_val)[:, 1]
        probs_test = best_est.predict_proba(X_test)[:, 1]

        # Threshold tuned strictly on validation
        prob_cutoff, rule_used = select_validation_threshold(
            y_val, probs_val, min_precision=cfg["classification"]["min_precision_for_recall_opt"]
        )

        if len(np.unique(y_test)) == 2:
            prec, rec, _ = precision_recall_curve(y_test, probs_test)
            ax.plot(rec, prec, label=f"{name} (cut={prob_cutoff:.2f})")

        for split_name, part_df, y_true, probs in [("val", val_df, y_val, probs_val), ("test", test_df, y_test, probs_test)]:
            for t_label, cut in [("default_050", 0.50), ("tuned_val", prob_cutoff)]:
                m = _eval_cls_slice(y_true, probs, cut)
                m.update({"model": name, "split": split_name, "threshold_type": t_label, "prob_cutoff": cut, "rule": rule_used, "availability_rate": 1.0})
                metrics_rows.append(m)

            pf = part_df[["origin_hour", "target_hour", "split", "actual_kw"]].copy()
            pf["model"] = name
            pf["y_true_peak"] = y_true
            pf["peak_prob"] = probs
            pf["y_pred_default_050"] = (probs >= 0.50).astype(int)
            pf["y_pred_tuned"] = (probs >= prob_cutoff).astype(int)
            pf["prob_cutoff"] = prob_cutoff
            pf["peak_kw_cutoff"] = peak_kw_cutoff
            pred_frames.append(pf)

    ax.set_title("Test Precision-Recall Curves (Thresholds Frozen from Validation)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    p_pr = artifacts_dir / "plots" / "classification_pr_curves.png"
    fig.tight_layout()
    fig.savefig(p_pr, dpi=120)
    plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = artifacts_dir / "classification_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    preds_df = pd.concat(pred_frames, ignore_index=True)
    preds_path = artifacts_dir / "classification_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    save_manifest(
        artifacts_dir,
        {
            "peak_kw_cutoff": peak_kw_cutoff,
            "classification_metrics_path": str(metrics_path),
            "classification_predictions_path": str(preds_path),
        },
    )
    return preds_path