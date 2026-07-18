import numpy as np
from src.forecast.backtest import backtest_shares


def test_backtest_reports_per_zone_metrics(synthetic_canonical):
    out = backtest_shares(synthetic_canonical, min_train=5, n_boot=100, seed=4)
    assert set(out["zone"]) == {"restricted_area", "above_break_3"}
    assert (out["n_folds"] > 0).all()
    # a clean linear trend should be forecastable to low error
    assert (out["mae"] < 0.05).all()
    assert (out["coverage_rate"] >= 0).all() and (out["coverage_rate"] <= 1).all()


def test_backtest_returns_empty_when_too_short(synthetic_canonical):
    short = synthetic_canonical[synthetic_canonical["season"] <= 2017]  # 3 seasons
    out = backtest_shares(short, min_train=5, n_boot=50, seed=4)
    assert len(out) == 0
