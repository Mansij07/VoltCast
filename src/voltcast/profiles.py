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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from voltcast.config import save_manifest


def build_daily_vectors(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = hourly.copy()
    df["date"] = df["hour_start"].dt.normalize()
    df["hour"] = df["hour_start"].dt.hour

    piv = df.pivot(index="date", columns="hour", values="active_kw_mean")
    piv = piv.reindex(columns=list(range(24)))
    complete_mask = piv.notna().all(axis=1)
    complete_days = piv[complete_mask].copy()
    complete_days.columns = [f"h{h:02d}" for h in range(24)]

    summary = {
        "total_calendar_days": int(len(piv)),
        "complete_24h_days": int(complete_mask.sum()),
        "excluded_incomplete_days": int((~complete_mask).sum()),
    }
    return complete_days, summary


def normalize_daily_profiles(daily_df: pd.DataFrame, eps: float = 1e-6) -> np.ndarray:
    arr = daily_df.to_numpy(dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > eps, row_sums, 1.0)
    norm = np.where(row_sums > eps, arr / safe_sums, 1.0 / 24.0)
    return norm


def run_profiles_stage(cfg: dict[str, Any], artifacts_dir: Path) -> Path:
    hourly = pd.read_csv(artifacts_dir / "hourly_observations.csv", parse_dates=["hour_start"])
    split_manifest = json.loads((artifacts_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_end = pd.Timestamp(split_manifest["boundaries"]["train_end"])

    daily_all, day_stats = build_daily_vectors(hourly)
    train_days = daily_all[daily_all.index <= train_end].copy()
    if len(train_days) < 4:
        raise ValueError(f"Not enough complete training days for clustering: {len(train_days)}")

    eps = float(cfg["clustering"]["zero_day_eps"])
    X_train_norm = normalize_daily_profiles(train_days, eps=eps)
    seed = cfg["project"]["random_seed"]

    comparison_rows = []
    Z = linkage(X_train_norm, method="ward")

    best_k = None
    best_sil = -2.0
    best_km = None

    for k in cfg["clustering"]["k_candidates"]:
        if k >= len(train_days):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km_labels = km.fit_predict(X_train_norm)
        km_sil = float(silhouette_score(X_train_norm, km_labels)) if len(np.unique(km_labels)) > 1 else np.nan

        hc_labels = fcluster(Z, t=k, criterion="maxclust") - 1
        hc_sil = float(silhouette_score(X_train_norm, hc_labels)) if len(np.unique(hc_labels)) > 1 else np.nan

        comparison_rows.append({
            "k": k,
            "kmeans_silhouette": km_sil,
            "kmeans_min_cluster_size": int(pd.Series(km_labels).value_counts().min()),
            "hierarchical_silhouette": hc_sil,
            "hierarchical_min_cluster_size": int(pd.Series(hc_labels).value_counts().min()),
        })
        if not np.isnan(km_sil) and km_sil > best_sil:
            best_sil = km_sil
            best_k = k
            best_km = km

    if best_km is None:
        best_k = 2
        best_km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X_train_norm)

    # Save frozen K-Means artifact for M6
    km_path = artifacts_dir / "models" / "kmeans_profiles.joblib"
    joblib.dump({"kmeans": best_km, "eps": eps, "k": best_k}, km_path)

    # Assign labels to all complete days (causal usage enforced in M6)
    X_all_norm = normalize_daily_profiles(daily_all, eps=eps)
    daily_labels = pd.DataFrame({
        "date": daily_all.index.strftime("%Y-%m-%d"),
        "cluster": best_km.predict(X_all_norm),
        "is_train_day": daily_all.index <= train_end,
        "daily_mean_kw": daily_all.mean(axis=1).to_numpy(),
    })
    labels_path = artifacts_dir / "daily_profile_labels.csv"
    daily_labels.to_csv(labels_path, index=False)

    # Original-scale average curves on training days
    train_days_labeled = train_days.copy()
    train_days_labeled["cluster"] = best_km.predict(X_train_norm)
    curves = train_days_labeled.groupby("cluster").mean()
    curves.to_csv(artifacts_dir / "profile_average_curves_kw.csv")

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(artifacts_dir / "clustering_comparison.csv", index=False)
    (artifacts_dir / "daily_profile_summary.json").write_text(json.dumps(day_stats, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    for c_id, row in curves.iterrows():
        n_c = int((train_days_labeled["cluster"] == c_id).sum())
        ax.plot(range(24), row.values, marker="o", label=f"Cluster {c_id} (n={n_c} train days)")
    ax.set_title(f"Training Daily Load Profiles (K-Means, k={best_k})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Active Power (kW)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "plots" / "daily_load_profiles.png", dpi=120)
    plt.close(fig)

    hc_labels_best = fcluster(Z, t=best_k, criterion="maxclust") - 1
    hc_curves = train_days.assign(cluster=hc_labels_best).groupby("cluster").mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=axes[0], color_threshold=Z[-(best_k - 1), 2] if best_k > 1 else None)
    axes[0].set_title(f"Ward Dendrogram of Training Days (truncated, cut at k={best_k})")
    axes[0].set_xlabel("Day clusters (size in brackets)")
    axes[0].set_ylabel("Ward distance")
    for c_id, row in hc_curves.iterrows():
        n_c = int((hc_labels_best == c_id).sum())
        axes[1].plot(range(24), row.values, marker="o", label=f"HC cluster {c_id} (n={n_c})")
    axes[1].set_title(f"Hierarchical (Ward) Daily Profiles, k={best_k}")
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Mean Active Power (kW)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "plots" / "hierarchical_profiles.png", dpi=120)
    plt.close(fig)

    km_train_labels = train_days_labeled["cluster"].to_numpy()
    agreement = pd.crosstab(pd.Series(km_train_labels, name="kmeans"), pd.Series(hc_labels_best, name="hierarchical"))
    agreement.to_csv(artifacts_dir / "kmeans_vs_hierarchical_crosstab.csv")

    save_manifest(
        artifacts_dir,
        {
            "selected_k": best_k,
            "kmeans_artifact_path": str(km_path),
            "daily_profile_labels_path": str(labels_path),
        },
    )
    return labels_path