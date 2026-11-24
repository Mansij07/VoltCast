# VoltCast — Final Report

**Electricity Demand Forecasting and Load Profiling on the UCI Individual Household Electric Power Consumption dataset**

All numbers below come from `python -m voltcast.cli run` (config `1.0.0`, seed 42, dataset fingerprint `4259c9d7ece5dbee`).
Plots referenced are written to `artifacts/full/plots/` by that command and are also shown in the Streamlit dashboard.

---

## 1. Problem

Household demand changes with time of day, weekday, season and appliance use. The project asks two linked questions:

1. **Forecasting:** how accurately can next-hour household demand be predicted from information available now, and which model family does best?
2. **Profiling:** what recurring daily consumption shapes exist, when are peaks likely, and does knowing the current profile improve forecasts?

The central difficulty is generalising to *future* periods: observations are autocorrelated, electrical variables are collinear, and peaks are rare. Every design choice below is made to keep the evaluation honest under those conditions.

## 2. Data and preprocessing

| Item | Value |
|---|---|
| Raw minute rows | 2,075,259 (2006-12-16 17:24 → 2010-11-26 21:02) |
| Minutes with missing active power (`?`) | 25,979 (1.25%) |
| Duplicate timestamps / negative values / parse errors | 0 / 0 / 0 |
| Regular hourly grid | 34,589 hours |
| Fully missing hours (gaps) | 421 |
| Partial hours below 45 valid minutes | 17 |
| Eligible target hours (≥ 45 valid minutes) | 34,151 |
| Complete 60-minute circuit hours (for Wh totals) | 34,085 |
| Complete 24-hour days (for profiling) | 1,411 of 1,442 |

- Minute data is aggregated to a strict 1-hour grid. Gaps stay as explicit missing rows, so every row shift is exactly one hour and lags never silently skip a gap.
- An hour is a valid **target** if ≥ 45 of its 60 minutes were observed (a mean over 45 minutes is a reliable estimate of the hourly mean). Sub-metering **energy totals** require all 60 minutes, because a sum over a partial hour would understate energy.

### Decisions fixed after exploration (Proposal §8)

| Decision | Choice |
|---|---|
| Aggregation | Hourly mean active power (kW) |
| Horizon | 1 hour ahead: forecast issued when hour *t* closes, for hour *t+1* |
| Split | Chronological by whole calendar days of the target hour: train 2006-12-16 → 2009-09-15 (23,959 rows), validation 2009-09-16 → 2010-04-19 (5,098), test 2010-04-20 → 2010-11-26 (5,093) |
| Peak definition | Target hour ≥ **2.395 kW**, the 90th percentile of *training* demand only |
| Leakage control | Imputation, scaling, PCA, clustering and peak cutoff are all fitted on training data only; classification thresholds are tuned on validation only; the test set is touched once |

### Features (26)
- **Calendar for the target hour:** sin/cos of hour of day, day of week and month, plus a weekend flag. Cyclic encoding keeps 23:00 next to 00:00 and December next to January.
- **Lags of active power:** 1, 2, 3, 24, 48 and 168 hours before the target.
- **Rolling mean and std** over 3, 6 and 24 hours ending at the origin hour.
- **Electrical state at the origin hour:** reactive power, voltage, current intensity, the three sub-metering channels, and the number of valid minutes.

A unit test mutates all data after a cutoff and verifies that no feature row before the cutoff changes (`tests/test_voltcast.py::test_no_future_leakage_in_features`).

## 3. Exploratory analysis

Plots: `eda_hourly_weekday_weekend.png`, `eda_distribution_missingness.png`, `eda_correlations.png`, `eda_submetering_hourly.png`, `eda_seasonal.png`.

- **Daily shape:** weekdays show a sharp morning peak (~1.7 kW at 07:00), a midday trough (~0.8 kW at 16:00) and an evening peak (~1.85 kW at 20:00–21:00). Weekends have no early spike, a sustained midday plateau (~1.4–1.5 kW) and the highest evening peak overall (~2.05 kW at 19:00–20:00).
- **Seasonality:** mean hourly demand ranges from ~1.49 kW in December to ~0.57 kW in August. The month × hour heatmap shows the evening peak is strongest in winter, and August is uniformly low (holiday absence).
- **Distribution:** right-skewed, with most hours under 1.5 kW and a long tail of high-load hours. This motivates both MAE (robust) and RMSE (penalises big misses), and the separate peak classifier.
- **Correlation:** active power and current intensity are almost perfectly correlated (intensity ≈ power / voltage), a collinearity that matters for interpretation (§5.3).

## 4. Regression: next-hour demand

Hyperparameters were tuned with 3-fold `TimeSeriesSplit` (expanding window) inside the training period, using MAE. The model was selected on **validation MAE only**.

### 4.1 Results

| Model | Best params | Val MAE | **Test MAE** | Test RMSE | Test R² | Test peak-hour MAE | Test ordinary-hour MAE |
|---|---|---|---|---|---|---|---|
| Persistence (ŷ = last hour) | – | 0.467 | 0.386 | 0.584 | 0.322 | – | – |
| Seasonal naive (ŷ = same hour yesterday) | – | 0.664 | 0.512 | 0.761 | −0.145 | – | – |
| Linear regression | – | 0.412 | 0.352 | 0.493 | 0.517 | 1.167 | 0.314 |
| Ridge | α = 10 | 0.412 | 0.352 | 0.493 | 0.517 | 1.167 | 0.314 |
| Lasso | α = 0.01 | 0.411 | 0.351 | 0.493 | 0.517 | 1.167 | 0.313 |
| Regression tree | depth 8, leaf 15 | 0.387 | 0.344 | 0.497 | 0.509 | 1.046 | 0.311 |
| **Random forest** (selected) | depth 8, leaf 10, 60 trees | **0.364** | **0.322** | **0.464** | **0.573** | 1.001 | 0.290 |
| Gradient boosting | depth 3, lr 0.1, 60 trees | 0.376 | 0.329 | 0.465 | 0.571 | 1.011 | 0.297 |

(Peak/ordinary split uses the 2.395 kW training cutoff: 228 peak hours in test.)

### 4.2 Findings
- **The random forest beats persistence by 16.5% in test MAE** (0.386 → 0.322) and 20.6% in RMSE. On validation the gain is 22%. Seasonal naive is worse than persistence: yesterday's same hour is a poor guide at this horizon.
- **Tree ensembles beat linear models by ~8% MAE.** The gain is largest on peak hours (1.17 → 1.00 kW), where the response to recent load is non-linear: a high reading predicts a high next hour, but only up to a point.
- **Regularisation barely matters.** Ridge and Lasso match OLS because 24k rows against 26 features is far from overfitting territory. The selected α values are at the grid edges, which shows the penalty is weak.
- **Peaks remain hard:** peak-hour MAE is ~3.5× the ordinary-hour MAE for every model. One-hour-ahead models react to a peak only after it starts, which is why a dedicated peak classifier is useful.
- Test errors are lower than validation errors because the test period (late April to November) contains summer, when absolute demand is lower. R² is lower on test for the same reason: there is less variance to explain.

Plots: `regression_diagnostics.png` (first test week, residual histogram).

### 4.3 Interpretation (`regression_importance.png`)
- **Current intensity at the origin hour dominates** every method: random forest impurity importance 0.68, permutation importance 0.37 MAE, and the largest Ridge coefficient (+0.45). Active power at the origin hour comes second.
- This is **collinearity, not a separate physical driver**: intensity and active power carry nearly the same information. The forest arbitrarily favours one of them, and Ridge splits weight between them. Removing either would barely change accuracy.
- After that come **calendar position** (hour of day sin/cos) and the **168-hour (same hour last week) lag**, which shows the weekly routine matters more than yesterday's.
- In Ridge, the sub-metering channels get **negative** coefficients once total power is held fixed. One plausible reading: when the current load comes from a metered appliance cycle (water heater, washer, oven), that cycle tends to end, so demand falls back next hour. Collinearity with total power also distorts these signs, so this is an association, not a causal effect.

## 5. Peak-hour classification

Label: is the target hour ≥ 2.395 kW (train 90th percentile)? Peak rates: train 10%, validation 11.7%, **test 4.5%** (summer test period).
Models were tuned on average precision with time-series CV. The probability threshold was chosen on validation as **the highest-recall threshold that keeps precision ≥ 0.50**, then frozen for test.

| Model | Tuned cutoff | Val recall | Val F1 | **Test precision** | **Test recall** | **Test F1** | Test ROC-AUC | Test AP |
|---|---|---|---|---|---|---|---|---|
| Peak persistence (last hour ≥ cutoff) | – | 0.462 | 0.462 | 0.329 | 0.329 | 0.329 | 0.649 | 0.138 |
| Logistic regression | 0.30 | 0.513 | 0.507 | 0.383 | 0.250 | 0.302 | **0.876** | 0.301 |
| Gaussian NB | 1.00* | 0.389 | 0.438 | 0.391 | 0.259 | 0.311 | 0.849 | 0.254 |
| KNN (k tuned) | 0.38 | 0.409 | 0.457 | 0.422 | 0.250 | 0.314 | 0.859 | 0.288 |
| **Decision tree** | 0.42 | 0.438 | 0.472 | 0.391 | **0.298** | **0.338** | 0.874 | **0.313** |

\*Naive Bayes outputs near-saturated probabilities (independence assumption on highly correlated lags), so the only threshold reaching 0.50 precision is ~1.0. Its default 0.5 threshold gives the highest test recall of any model (0.42), at precision 0.25.

### Findings
- **Ranking ability is solid:** ROC-AUC 0.85–0.88 for all learned models vs 0.65 for persistence. The models know which hours are riskier.
- **The tuned threshold matters.** For logistic regression, moving from 0.50 to the validation-tuned 0.30 raises test recall from 0.09 to 0.25 (F1 0.16 → 0.30). Using the default 0.5 would miss 91% of peaks.
- **Test F1 is only slightly above persistence (0.34 vs 0.33)** and much lower than on validation. The cause is **prior shift**: the 2.395 kW cutoff fixed on mostly winter-heavy training data makes peaks 2.6× rarer in the summer test period. With a lower base rate, the same ranking quality gives lower precision and recall at a fixed threshold. This is the main limitation of a fixed absolute peak definition. A seasonal or rolling-percentile definition would be the natural next step.
- **Cost of a missed peak:** with the frozen thresholds, the decision tree still misses 160 of 228 test peaks. If missed peaks cost far more than false alarms, Naive Bayes at 0.5 (96 caught, 292 false alarms) is the operationally better choice, and the precision/recall curves (`classification_pr_curves.png`) show the full trade-off.

## 6. PCA of the feature space

Standardised PCA was fitted on the 26 training features (`pca_scree.png`, `pca_loadings.png`).

- **PC1 explains 29.9%** of variance: almost every load variable (current/lagged power, rolling means and stds, intensity, sub-metering) loads positively, ~0.2–0.34. It is a general **"how much power is the house drawing"** axis, and confirms the heavy redundancy among load features.
- **PC2 (9.0%)** pairs `month_cos` (+0.42), 24-hour rolling mean/std (+0.38/+0.35) and voltage (+0.38) against reactive power (−0.35): a **seasonal/winter-level** axis.
- **PC3/PC4 (7.3%, 6.9%)** are driven by `is_weekend` and day-of-week vs the 24/48/168-hour lags: a **weekly-routine** axis.
- Beyond PC1 the spectrum is flat: **15 components are needed for 90% of the variance.** The features are redundant in one direction (load level) but otherwise carry mostly independent information. Dimensionality reduction would therefore not help prediction much, so PCA is used here for interpretation, as the proposal intended, not as a model input.

## 7. Daily load-profile clustering

Each complete day is a 24-dimensional vector of hourly kW, **normalised to sum to 1**, so clusters capture *shape* rather than level. Both k-means and Ward hierarchical clustering were fitted on the 990 training days; k was chosen by k-means silhouette over k ∈ {2,3,4,5}.

| k | k-means silhouette | smallest k-means cluster | Ward silhouette | smallest Ward cluster |
|---|---|---|---|---|
| **2** | **0.136** | 488 | 0.117 | 387 |
| 3 | 0.124 | 266 | 0.097 | 158 |
| 4 | 0.103 | 195 | 0.098 | 121 |
| 5 | 0.102 | 146 | 0.098 | 106 |

**Selected k = 2** (`daily_load_profiles.png`, `hierarchical_profiles.png`):

| Profile | Shape (training average) | Days (all) | Weekend share | Mean kW |
|---|---|---|---|---|
| **0 — "Out during the day"** | Sharp 07:00 peak (1.9 kW), low midday (~0.6 kW), large 20–21:00 peak (2.2 kW) | 674 | 9% | 1.03 |
| **1 — "Home all day"** | Gradual rise to a midday plateau (~1.5 kW), moderate evening peak (1.7 kW) | 737 | 45% | 1.15 |

- The shapes are easy to read: profile 0 is a working weekday; profile 1 covers weekends, holidays and days at home. August is 72% profile 1, consistent with the summer holiday.
- **Ward and k-means agree on 83.5% of training days** (827 of 990, from `kmeans_vs_hierarchical_crosstab.csv`). Ward's dendrogram shows the same two-way top split. The Ward clusters are less balanced, and its silhouette is lower at every k.
- **Silhouette values are low (≈0.1).** Days lie on a continuum between the two shapes rather than in sharply separated groups. We report the profiles as a useful *description*, not as discrete natural classes.

## 8. Profile-aware forecasting vs a single global model

Each forecast is routed to the cluster of the **most recent fully observed day** (never the current, incomplete day) within 7 days. One random-forest model, with the same hyperparameters as the global model, is trained per cluster on training rows, and the global model is the fallback. Both approaches are scored on the same 5,093 test hours.

| | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Global random forest | **0.3222** | **0.4640** | **0.5726** |
| Profile-aware (2 segment models) | 0.3261 | 0.4683 | 0.5646 |

Fallback rate 0% (both segments had enough data, and every test hour had a recent complete day).

**Finding: segmentation does not help (MAE +1.2%).** Three reasons:
1. Yesterday's profile is a weak predictor of today's (the weekday/weekend switch flips it), so routing often sends the hour to the "wrong" specialist.
2. The global model already sees hour of day, weekday and weekly lags, so it can represent both shapes itself.
3. Each segment model sees only ~half the training data.

This is an honest negative result: the proposal asked *whether* segmentation gives a meaningful benefit, and at a 1-hour horizon it does not.

## 9. Sub-metered circuit analysis

On the 2,049,280 minutes where all four channels are observed (1 kW for 1 minute = 1000/60 Wh):

| Circuit | Share of total active energy |
|---|---|
| Sub-3: water heater & air conditioner | 35.5% |
| Sub-2: laundry | 7.1% |
| Sub-1: kitchen | 6.2% |
| Not sub-metered (lighting, other) | 51.2% |

Comparing minutes whose instantaneous power is ≥ 2.395 kW with all other minutes (mean Wh per minute):

| Circuit | High-load minutes | Other minutes | Ratio |
|---|---|---|---|
| Kitchen | 9.63 | 0.10 | ~100× |
| Laundry | 7.95 | 0.50 | ~16× |
| Water heater / AC | 13.83 | 5.57 | 2.5× |
| Unmetered | 25.90 | 7.31 | 3.5× |

The water heater/AC circuit dominates **total energy**, but the **kitchen and laundry circuits are what separate high-load moments** from ordinary ones: they are almost off normally and switch on during peaks. These are observational associations within one household. The channels group several appliances each, and the minute-level threshold reuses the hourly cutoff, so the high-load minute set is a stricter, spikier selection than the hourly peak label.

## 10. Limitations

- **One household:** profiles and accuracy will not transfer directly to other homes, climates or tariffs.
- **One horizon:** only 1-hour-ahead was evaluated. Profile information may matter more for day-ahead forecasts, which is untested.
- **Fixed absolute peak cutoff:** it is sensitive to seasonal prior shift (§5). Test conclusions about peaks mostly describe a summer period.
- **Single chronological test window (~7 months):** there is no rolling-origin backtest or confidence intervals, so small differences such as RF vs GB (0.322 vs 0.329) and global vs profile-aware should not be over-interpreted.
- **Sub-metering covers ~49% of energy:** appliance-level attribution is impossible.

## 11. Compliance with the proposal

| Proposal item | Delivered |
|---|---|
| Data cleaning, missingness, quality checks | §2, `data_quality_summary.json` |
| Temporal, lag and rolling features from available information only | §2, leakage unit test |
| EDA: daily/weekly trends, distributions, missingness, correlations, **seasonal patterns**, sub-metering vs total | §3, five EDA plots |
| Linear, Ridge, Lasso, regression tree, random forest, gradient boosting with time-aware validation | §4 |
| MAE, RMSE, R² on held-out chronological test | §4.1 |
| Peak label from training percentile; logistic, NB, KNN, tree; precision/recall/F1/confusion/ROC-AUC; threshold from validation | §5, `classification_metrics.csv` (tp/fp/fn/tn per model) |
| PCA on standardised features fitted on train; temporal features encoded appropriately | §6 (cyclic sin/cos encoding) |
| 24-dim normalised daily vectors; k-means and hierarchical; silhouette, cluster sizes, average profiles | §7 |
| Global vs profile-specific models on the same test dates | §8 |
| Coefficients, tree importances, sub-metering associations (non-causal) | §4.3, §9 |
| Reproducible pipeline | CLI, `run_manifest.json`, CI smoke run, Streamlit dashboard |

## 12. Reproducing

```bash
python -m pip install -r requirements.txt && python -m pip install -e .
python -m voltcast.cli run          # ~a few minutes; writes artifacts/full/
python -m pytest -q
streamlit run dashboard/app.py
```
