from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="VoltCast Dashboard", layout="wide")
st.title("VoltCast: Household Electricity Demand Forecasting & Load Profiling")

run_mode = st.sidebar.radio("Artifact Run", ["full", "smoke"], index=1)
art_dir = Path("artifacts") / run_mode
manifest_path = art_dir / "run_manifest.json"

if not manifest_path.exists():
    cmd = "python -m voltcast.cli smoke" if run_mode == "smoke" else "python -m voltcast.cli run --config config/default.yaml"
    st.error(f"No artifacts found in `{art_dir}`. Generate them first with:\n\n`{cmd}`")
    st.stop()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
st.sidebar.caption(f"Config v{manifest.get('config_version')} | Fingerprint: {manifest.get('dataset_fingerprint')}")

tabs = st.tabs(["1. Data & EDA", "2. Regression", "3. Peak Classification", "4. PCA & Profiles", "5. Profile-Aware & Circuits", "6. Historical Replay"])

with tabs[0]:
    q = json.loads((art_dir / "data_quality_summary.json").read_text(encoding="utf-8"))
    st.json(q)
    for img in ["eda_hourly_weekday_weekend.png", "eda_distribution_missingness.png", "eda_correlations.png", "eda_submetering_hourly.png"]:
        p = art_dir / "plots" / img
        if p.exists():
            st.image(str(p), use_column_width=True)

with tabs[1]:
    st.subheader("Regression Metrics (Validation & Test)")
    st.dataframe(pd.read_csv(art_dir / "regression_metrics.csv"), use_container_width=True)
    st.subheader("Naive Baselines")
    st.dataframe(pd.read_csv(art_dir / "baseline_metrics.csv"), use_container_width=True)
    p = art_dir / "plots" / "regression_diagnostics.png"
    if p.exists():
        st.image(str(p), use_column_width=True)

with tabs[2]:
    st.subheader(f"Peak Classification (Train 90th-pct cutoff = {manifest['peak_kw_cutoff']:.3f} kW)")
    st.dataframe(pd.read_csv(art_dir / "classification_metrics.csv"), use_container_width=True)
    p = art_dir / "plots" / "classification_pr_curves.png"
    if p.exists():
        st.image(str(p), use_column_width=True)

with tabs[3]:
    st.subheader("PCA Variance Explained")
    st.dataframe(pd.read_csv(art_dir / "pca_variance.csv").head(10), use_container_width=True)
    if (art_dir / "plots" / "pca_scree.png").exists():
        st.image(str(art_dir / "plots" / "pca_scree.png"), use_column_width=True)
    st.subheader("Daily Load Profiles")
    st.dataframe(pd.read_csv(art_dir / "clustering_comparison.csv"), use_container_width=True)
    if (art_dir / "plots" / "daily_load_profiles.png").exists():
        st.image(str(art_dir / "plots" / "daily_load_profiles.png"), use_column_width=True)

with tabs[4]:
    st.subheader("Global vs Profile-Aware Comparison (Identical Test Timestamps)")
    st.json(json.loads((art_dir / "profile_forecast_comparison.json").read_text(encoding="utf-8")))
    st.subheader("Sub-Metered Circuit Analysis (Observational)")
    st.json(json.loads((art_dir / "circuit_analysis.json").read_text(encoding="utf-8")))

with tabs[5]:
    st.info("Historical Replay: displays stored backtest predictions on the held-out test period.")
    prof_preds = pd.read_csv(art_dir / "profile_forecast_predictions.csv")
    cls_preds = pd.read_csv(art_dir / "classification_predictions.csv")
    cls_test = cls_preds[(cls_preds["split"] == "test") & (cls_preds["model"] == "logistic_regression")]

    choice = st.selectbox("Select held-out target_hour", prof_preds["target_hour"].tolist())
    row = prof_preds[prof_preds["target_hour"] == choice].iloc[0]
    crow = cls_test[cls_test["target_hour"] == choice].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observed (kW)", f"{row['actual_kw']:.3f}")
    c2.metric("Global Forecast (kW)", f"{row['global_pred_kw']:.3f}")
    c3.metric("Profile-Aware (kW)", f"{row['profile_aware_pred_kw']:.3f}")
    c4.metric("Peak Prob (LogReg)", f"{crow['peak_prob']:.2f}", f"Pred Peak: {bool(crow['y_pred_tuned'])}")
    st.write(f"**Origin Hour:** `{row['origin_hour']}` | **Route Cluster:** `{row['route_cluster']}` | **Profile Source Date:** `{row['profile_source_date']}` | **Fallback Used:** `{row['used_fallback']}`")