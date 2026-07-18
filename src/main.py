"""CLI orchestration: acquire Kaggle data -> canonical table -> 2030 forecast -> report."""
from __future__ import annotations

import argparse
import os

from src.agent.loaders.kaggle_csv import KaggleCsvLoader
from src.forecast.diet import forecast_shot_diet
from src.forecast.backtest import backtest_shares
from src.report.tables import projection_table, backtest_table
from src.report.plots import plot_fan_charts
from src.utils import helpers


def _acquire_zip(zip_path: str | None) -> str:
    if zip_path and os.path.exists(zip_path):
        return zip_path
    helpers.log_message("Downloading Kaggle NBA shots dataset via kagglehub...")
    import kagglehub  # imported lazily so tests never hit the network
    path = kagglehub.dataset_download("mexwell/nba-shots")
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".zip"):
                return os.path.join(root, f)
    raise FileNotFoundError("Could not locate a .zip in the downloaded dataset")


def run(zip_path: str, target_year: int = 2030, out_dir: str = "artifacts") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    helpers.log_message(f"Loading shots from {zip_path}")
    canonical = KaggleCsvLoader(zip_path).load()

    errors = helpers.validate_data(canonical)
    if errors:
        raise ValueError("Canonical data failed validation:\n- " + "\n- ".join(errors))
    helpers.write_canonical(canonical, os.path.join(out_dir, "shot_diet.parquet"))

    forecast = forecast_shot_diet(canonical, target_year)
    backtest = backtest_shares(canonical)
    plots = plot_fan_charts(canonical, forecast, out_dir, target_year)

    proj = projection_table(forecast, target_year)
    acc = backtest_table(backtest)
    helpers.log_message(f"\n=== Projected {target_year} shot diet ===\n{proj.to_string(index=False)}")
    helpers.log_message(f"\n=== Backtest accuracy ===\n{acc.to_string(index=False)}")
    return {"forecast": forecast, "backtest": backtest, "plots": plots}


def main() -> None:
    ap = argparse.ArgumentParser(description="NBA 2030 shot-diet forecast")
    ap.add_argument("--zip", dest="zip_path", default=None,
                    help="Path to archive.zip; downloaded via kagglehub if omitted")
    ap.add_argument("--year", dest="target_year", type=int, default=2030)
    ap.add_argument("--out", dest="out_dir", default="artifacts")
    args = ap.parse_args()
    run(_acquire_zip(args.zip_path), args.target_year, args.out_dir)


if __name__ == "__main__":
    main()
