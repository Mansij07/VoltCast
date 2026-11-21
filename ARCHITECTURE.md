# VoltCast Architecture & Shared Data Contracts

## 1. Time & Forecast Alignment Contract
- **Clock convention**: Timestamps use the source dataset's recorded local clock as naive `datetime64[ns]`.
- **Hourly row convention**: `hour_start` = `t` represents `[t, t + 1h)`.
- **Forecast alignment**: A prediction is issued immediately after `origin_hour = t` closes, using only data available up to `t + 1h`. The target is `target_hour = t + 1h`, representing `[t + 1h, t + 2h)`.
- **Partitioning**: Supervised rows are assigned to `train`, `val`, or `test` chronologically by `target_hour` on whole calendar-day boundaries (~70% / 15% / 15%). Historical rows before a split boundary may supply causal lags to rows after it.

## 2. Canonical Artifact Schemas
All tables are saved as CSV (UTF-8, ISO-8601 timestamps) and metadata as JSON for universal readability.

### `hourly_observations.csv` (Key: `hour_start`)
- `hour_start` (ISO timestamp, strictly monotonic 1-hour grid)
- `active_kw_mean`, `reactive_kw_mean`, `voltage_v_mean`, `intensity_a_mean`
- `sub1_wh_sum`, `sub2_wh_sum`, `sub3_wh_sum` (Wh)
- `valid_active_minutes`, `valid_sub1_minutes`, `valid_sub2_minutes`, `valid_sub3_minutes`
- `is_eligible_target` (bool: `valid_active_minutes >= min_valid_minutes_target`)
- `is_complete_circuit_hour` (bool: all sub-metering minutes == 60)

### `features.csv` (Key: `target_hour`)
- `origin_hour`, `target_hour`, `split` (`train` | `val` | `test`), `actual_kw`
- Calendar (known for `target_hour`): `target_hour_sin`, `target_hour_cos`, `target_dow_sin`, `target_dow_cos`, `is_weekend`, `target_month`
- Causal lags & rolling stats (strictly $\le$ `origin_hour`): `active_kw_origin`, `active_kw_lag_2h_from_target`, `active_kw_lag_3h_from_target`, `active_kw_lag_24h_from_target`, `active_kw_lag_48h_from_target`, `active_kw_lag_168h_from_target`, `active_kw_roll_mean_3h`, `active_kw_roll_std_3h`, `active_kw_roll_mean_6h`, `active_kw_roll_std_6h`, `active_kw_roll_mean_24h`, `active_kw_roll_std_24h`
- Causal electrical/circuit state at `origin_hour`: `reactive_kw_origin`, `voltage_v_origin`, `intensity_a_origin`, `sub1_wh_origin`, `sub2_wh_origin`, `sub3_wh_origin`, `origin_valid_minutes`

### `regression_predictions.csv` (Key: `(target_hour, model)`)
- `origin_hour`, `target_hour`, `split`, `model`, `actual_kw`, `predicted_kw`

### `classification_predictions.csv` (Key: `(target_hour, model)`)
- `origin_hour`, `target_hour`, `split`, `model`, `actual_kw`, `y_true_peak`, `peak_prob`, `y_pred_default_050`, `y_pred_tuned`, `prob_cutoff`, `peak_kw_cutoff`

### `profile_forecast_predictions.csv` (Key: `target_hour`)
- `origin_hour`, `target_hour`, `split`, `actual_kw`, `global_pred_kw`, `profile_aware_pred_kw`, `route_cluster`, `profile_source_date`, `used_fallback`

### `run_manifest.json`
Records `config_version`, `run_mode` (`full` | `smoke`), `dataset_fingerprint`, `split_boundaries`, `peak_kw_cutoff`, `selected_global_regressor`, `selected_k`, `sample_counts`, and relative paths to all generated artifacts.