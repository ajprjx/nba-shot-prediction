"""Human-readable projection and backtest-accuracy tables."""
from __future__ import annotations

import pandas as pd


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def projection_table(forecast_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    df = forecast_df.sort_values("projected_share", ascending=False).reset_index(drop=True)
    return pd.DataFrame({
        "zone": df["zone"],
        "target_year": target_year,
        "projected_share_pct": df["projected_share"].map(_pct),
        "share_ci": [f"{_pct(lo)}-{_pct(hi)}" for lo, hi in zip(df["share_lo"], df["share_hi"])],
        "pts_per_shot": df["projected_pts_per_shot"].round(3),
    })


def _trust(mae: float) -> str:
    if mae < 0.02:
        return "high"
    if mae < 0.05:
        return "medium"
    return "low"


def backtest_table(backtest_df: pd.DataFrame) -> pd.DataFrame:
    df = backtest_df.copy()
    return pd.DataFrame({
        "zone": df["zone"],
        "mae": df["mae"].round(4),
        "coverage_rate": df["coverage_rate"].round(2),
        "trust": df["mae"].map(_trust),
    })
