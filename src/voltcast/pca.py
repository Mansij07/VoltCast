from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from voltcast.config import save_manifest
from voltcast.features import FEATURE_COLUMNS


def run_pca_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    df = pd.read_csv(artifacts_dir / "features.csv", parse_dates=["target_hour"])
    train_df = df[df["split"] == "train"]
    X_train = train_df[FEATURE_COLUMNS]

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=cfg["project"]["random_seed"])),
    ])
    pipe.fit(X_train)
    joblib.dump(pipe, artifacts_dir / "models" / "pca_pipeline.joblib")

    pca_step: PCA = pipe.named_steps["pca"]
    evr = pca_step.explained_variance_ratio_
    cum = np.cumsum(evr)

    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": evr,
        "cumulative_variance_ratio": cum,
    })
    var_path = artifacts_dir / "pca_variance.csv"
    var_df.to_csv(var_path, index=False)

    loadings = pd.DataFrame(
        pca_step.components_.T,
        index=FEATURE_COLUMNS,
        columns=[f"PC{i+1}" for i in range(len(evr))],
    )
    loadings_path = artifacts_dir / "pca_loadings.csv"
    loadings.to_csv(loadings_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr, alpha=0.6, label="Individual")
    ax.plot(x, cum, marker="o", color="crimson", label="Cumulative")
    ax.axhline(cfg["pca"]["variance_threshold"], linestyle="--", color="gray", label="90% threshold")
    ax.set_title("PCA Scree Plot (Fitted on Training Data Only)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifacts_dir / "plots" / "pca_scree.png", dpi=120)
    plt.close(fig)

    save_manifest(artifacts_dir, {"pca_variance_path": str(var_path), "pca_loadings_path": str(loadings_path)})
    return var_path