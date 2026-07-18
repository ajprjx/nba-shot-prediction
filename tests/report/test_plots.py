import os
import pandas as pd
from src.report.plots import fan_chart_data, plot_fan_charts
from src.forecast.diet import forecast_shot_diet


def test_fan_chart_data_extracts_history(synthetic_canonical):
    fc = forecast_shot_diet(synthetic_canonical, 2030, n_boot=50, seed=5)
    d = fan_chart_data(synthetic_canonical, fc, "above_break_3", 2030)
    assert d["hist_years"][0] == 2015 and d["hist_years"][-1] == 2024
    assert len(d["hist_shares"]) == 10
    assert d["lo"] <= d["point"] <= d["hi"]
    assert d["target_year"] == 2030


def test_plot_fan_charts_writes_files(synthetic_canonical, tmp_path):
    fc = forecast_shot_diet(synthetic_canonical, 2030, n_boot=50, seed=5)
    paths = plot_fan_charts(synthetic_canonical, fc, str(tmp_path), 2030)
    assert len(paths) == 2
    for p in paths:
        assert os.path.exists(p) and p.endswith(".png")
