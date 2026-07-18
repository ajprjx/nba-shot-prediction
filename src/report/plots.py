"""Fan charts: history plus the projected point and prediction band per zone."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless / test-safe
import matplotlib.pyplot as plt
import pandas as pd

from src.forecast import compositional as comp


def fan_chart_data(df: pd.DataFrame, forecast_df: pd.DataFrame, zone: str,
                   target_year: int) -> dict:
    years, mat = comp.shares_matrix(df, [zone], value_col="share")
    row = forecast_df[forecast_df["zone"] == zone].iloc[0]
    return {
        "hist_years": [int(y) for y in years],
        "hist_shares": [float(v) for v in mat[:, 0]],
        "target_year": int(target_year),
        "point": float(row["projected_share"]),
        "lo": float(row["share_lo"]),
        "hi": float(row["share_hi"]),
    }


def plot_fan_charts(df: pd.DataFrame, forecast_df: pd.DataFrame, out_dir: str,
                    target_year: int) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for zone in forecast_df["zone"]:
        d = fan_chart_data(df, forecast_df, zone, target_year)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(d["hist_years"], d["hist_shares"], marker="o", label="history")
        ax.plot([d["hist_years"][-1], d["target_year"]],
                [d["hist_shares"][-1], d["point"]], "--", color="orange", label="forecast")
        ax.fill_between([d["hist_years"][-1], d["target_year"]],
                        [d["hist_shares"][-1], d["lo"]],
                        [d["hist_shares"][-1], d["hi"]],
                        color="orange", alpha=0.2, label="95% PI")
        ax.set_title(f"{zone} share of shots -> {d['target_year']}")
        ax.set_xlabel("Season"); ax.set_ylabel("Share of shots")
        ax.legend(); ax.grid(True)
        path = os.path.join(out_dir, f"fan_{zone}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
