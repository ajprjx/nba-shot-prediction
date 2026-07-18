import numpy as np
from src.forecast.diet import forecast_shot_diet


def test_projected_shares_sum_to_one(synthetic_canonical):
    out = forecast_shot_diet(synthetic_canonical, target_year=2030, n_boot=100, seed=3)
    assert np.isclose(out["projected_share"].sum(), 1.0, atol=1e-6)
    assert set(out["zone"]) == {"restricted_area", "above_break_3"}


def test_rising_three_share_continues_upward(synthetic_canonical):
    out = forecast_shot_diet(synthetic_canonical, target_year=2030, n_boot=100, seed=3)
    three_2030 = out.loc[out["zone"] == "above_break_3", "projected_share"].iloc[0]
    three_2024 = 0.30 + 0.02 * 9  # last observed value from the fixture (0.48)
    assert three_2030 > three_2024


def test_intervals_bracket_point(synthetic_canonical):
    out = forecast_shot_diet(synthetic_canonical, target_year=2030, n_boot=200, seed=3)
    assert (out["share_lo"] <= out["projected_share"]).all()
    assert (out["projected_share"] <= out["share_hi"]).all()
