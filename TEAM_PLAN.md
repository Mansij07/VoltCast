# VoltCast Seven-Member Team Plan

| Member | Branch | Reviewer | Core Modules | Verification Command |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | `work/data-eda` | M2 | `src/voltcast/data.py` | `python -m voltcast.cli data --smoke` |
| **M2** | `work/features-baselines` | M3 | `src/voltcast/features.py`, `baselines.py` | `python -m voltcast.cli features --smoke` |
| **M3** | `work/regression` | M4 | `src/voltcast/regression.py` | `python -m voltcast.cli regression --smoke` |
| **M4** | `work/peak-classification` | M6 | `src/voltcast/classification.py` | `python -m voltcast.cli classification --smoke` |
| **M5** | `work/pca-clustering` | M7 | `src/voltcast/pca.py`, `profiles.py` | `python -m voltcast.cli profiles --smoke` |
| **M6** | `work/profile-aware` | M3 | `src/voltcast/profile_forecast.py`, `circuit_analysis.py` | `python -m voltcast.cli profile-forecast --smoke` |
| **M7** | `work/integration` | M1 | `cli.py`, `dashboard/app.py`, CI, docs | `python -m voltcast.cli smoke && pytest -q` |

## Viva Talking Points by Member
- **M1**: Why 45 valid minutes for target eligibility vs 60 for circuit Wh totals; how `?`, duplicates, and gaps are regularized on a 1-hour grid.
- **M2**: Exact `origin_hour` $\rightarrow$ `target_hour` alignment; why future-mutation tests prove zero leakage; baseline availability on shared timestamps.
- **M3**: Expanding-window CV inside train; linear vs tree performance on peaks vs ordinary hours; why collinearity affects linear coefficients and tree importances.
- **M4**: Separation of train-only kW peak cutoff (90th pct) from validation-only probability cutoff; precision-recall trade-off and cost of missed peaks.
- **M5**: Train-only standardized PCA loadings; 24h profile normalization; K-Means vs Hierarchical clustering and why hierarchical is descriptive.
- **M6**: Causal routing using the most recent fully observed eligible day; fallback rules; matched-minute Wh unit conversion ($1\text{ kW} = 1000/60\text{ Wh}$) and observational caveats.
- **M7**: End-to-end reproducibility via `run_manifest.json`, smoke vs full separation, CI checks, and historical replay in Streamlit.

## PR Checklist (All Workstreams)
- [ ] Inputs and outputs match `ARCHITECTURE.md` schemas.
- [ ] Stage command executed and outputs inspected.
- [ ] `python -m pytest -q` passes locally.
- [ ] Any shared schema change documented in `ARCHITECTURE.md` and approved by reviewer.

## Contribution Log Template
| Date | Member | PR / Commit | Summary of Work | Verified By |
| :--- | :--- | :--- | :--- | :--- |
| YYYY-MM-DD | M1–M7 | `#0` | (Record actual student contributions here) | Reviewer |