import pandas as pd
from src.report.tables import projection_table, backtest_table


def _fc():
    return pd.DataFrame([
        {"zone": "restricted_area", "projected_share": 0.52, "share_lo": 0.49, "share_hi": 0.55,
         "projected_pts_per_shot": 1.28, "pts_lo": 1.2, "pts_hi": 1.36},
        {"zone": "above_break_3", "projected_share": 0.48, "share_lo": 0.45, "share_hi": 0.51,
         "projected_pts_per_shot": 1.10, "pts_lo": 1.0, "pts_hi": 1.2},
    ])


def test_projection_table_formats_and_orders():
    t = projection_table(_fc(), target_year=2030)
    assert list(t["zone"]) == ["restricted_area", "above_break_3"]  # desc by share
    assert t.iloc[0]["projected_share_pct"] == "52.0%"
    assert t.iloc[0]["share_ci"] == "49.0%-55.0%"
    assert t.iloc[0]["target_year"] == 2030


def test_backtest_table_assigns_trust():
    bt = pd.DataFrame([
        {"zone": "restricted_area", "n_folds": 4, "mae": 0.01, "rmse": 0.01, "coverage_rate": 1.0},
        {"zone": "above_break_3", "n_folds": 4, "mae": 0.08, "rmse": 0.09, "coverage_rate": 0.5},
    ])
    t = backtest_table(bt)
    trust = dict(zip(t["zone"], t["trust"]))
    assert trust["restricted_area"] == "high"
    assert trust["above_break_3"] == "low"
