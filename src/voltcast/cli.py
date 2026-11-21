from __future__ import annotations

import argparse
from pathlib import Path

from voltcast.baselines import run_baselines_stage
from voltcast.circuit_analysis import run_circuit_analysis_stage
from voltcast.classification import run_classification_stage
from voltcast.config import get_artifacts_dir, load_config
from voltcast.data import fetch_uci_dataset, run_data_stage
from voltcast.features import run_features_stage
from voltcast.pca import run_pca_stage
from voltcast.profile_forecast import run_profile_forecast_stage
from voltcast.profiles import run_profiles_stage
from voltcast.regression import run_regression_stage


def run_all(cfg: dict, smoke: bool = False) -> Path:
    art = get_artifacts_dir(cfg, smoke=smoke)
    run_data_stage(cfg, art, smoke=smoke)
    run_features_stage(cfg, art)
    run_baselines_stage(cfg, art)
    run_regression_stage(cfg, art)
    run_classification_stage(cfg, art)
    run_pca_stage(cfg, art)
    run_profiles_stage(cfg, art)
    run_profile_forecast_stage(cfg, art)
    run_circuit_analysis_stage(cfg, art)
    return art


def main() -> None:
    parser = argparse.ArgumentParser(prog="voltcast", description="VoltCast ML Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["fetch-data", "run", "smoke", "data", "features", "regression", "classification", "pca", "profiles", "profile-forecast", "circuits"]:
        p = sub.add_parser(name)
        p.add_argument("--config", default="config/default.yaml")
        if name not in ["fetch-data", "run", "smoke"]:
            p.add_argument("--smoke", action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "fetch-data":
        path = fetch_uci_dataset(cfg)
        print(f"Dataset ready at: {path}")
    elif args.cmd == "run":
        art = run_all(cfg, smoke=False)
        print(f"Full pipeline complete. Artifacts in: {art}")
    elif args.cmd == "smoke":
        art = run_all(cfg, smoke=True)
        print(f"Smoke pipeline complete. Artifacts in: {art}")
    else:
        art = get_artifacts_dir(cfg, smoke=args.smoke)
        stage_map = {
            "data": lambda: run_data_stage(cfg, art, smoke=args.smoke),
            "features": lambda: (run_features_stage(cfg, art), run_baselines_stage(cfg, art)),
            "regression": lambda: run_regression_stage(cfg, art),
            "classification": lambda: run_classification_stage(cfg, art),
            "pca": lambda: run_pca_stage(cfg, art),
            "profiles": lambda: run_profiles_stage(cfg, art),
            "profile-forecast": lambda: run_profile_forecast_stage(cfg, art),
            "circuits": lambda: run_circuit_analysis_stage(cfg, art),
        }
        stage_map[args.cmd]()
        print(f"Stage '{args.cmd}' complete in: {art}")


if __name__ == "__main__":
    main()