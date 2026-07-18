import numpy as np
from src.forecast.model import DampedTrendForecaster, ForecastResult


def test_recovers_linear_trend_in_sample():
    years = np.arange(2015, 2025)
    values = 0.30 + 0.02 * (years - 2015)  # perfectly linear
    f = DampedTrendForecaster(damping=1.0, n_boot=200, seed=1).fit(years, values)
    res = f.forecast(2024)
    assert isinstance(res, ForecastResult)
    assert np.isclose(res.point, values[-1], atol=1e-6)


def test_damping_pulls_extrapolation_below_linear():
    years = np.arange(2015, 2025)
    values = 0.30 + 0.02 * (years - 2015)
    undamped = DampedTrendForecaster(damping=1.0, n_boot=50, seed=1).fit(years, values).forecast(2030)
    damped = DampedTrendForecaster(damping=0.6, n_boot=50, seed=1).fit(years, values).forecast(2030)
    # both rise above the last value, but damping rises less far
    assert damped.point < undamped.point
    assert damped.point > values[-1]


def test_interval_brackets_point_and_widens_with_noise():
    rng = np.random.default_rng(0)
    years = np.arange(2010, 2025)
    values = 0.30 + 0.015 * (years - 2010) + rng.normal(0, 0.02, size=years.size)
    res = DampedTrendForecaster(damping=0.9, n_boot=400, seed=2).fit(years, values).forecast(2030)
    assert res.lo < res.point < res.hi
