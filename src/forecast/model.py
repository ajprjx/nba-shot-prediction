"""Damped linear-trend forecaster with residual-bootstrap prediction intervals.

Designed for short annual series (~10-20 points). Damping prevents runaway
multi-year extrapolation; the bootstrap captures parameter + residual uncertainty.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

ForecastResult = namedtuple("ForecastResult", ["point", "lo", "hi"])


def _fit_line(years: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(years, values, 1)
    return float(slope), float(intercept)


class DampedTrendForecaster:
    def __init__(self, damping: float = 0.85, n_boot: int = 500, seed: int = 0):
        self.damping = damping
        self.n_boot = n_boot
        self.seed = seed

    def fit(self, years: np.ndarray, values: np.ndarray) -> "DampedTrendForecaster":
        self.years = np.asarray(years, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.slope_, self.intercept_ = _fit_line(self.years, self.values)
        self.last_year_ = float(self.years.max())
        self.fitted_ = self.intercept_ + self.slope_ * self.years
        self.resid_ = self.values - self.fitted_
        return self

    def _predict(self, slope: float, intercept: float, target_year: int) -> float:
        last = self.last_year_
        if target_year <= last:
            return intercept + slope * target_year
        base = intercept + slope * last
        h = int(round(target_year - last))
        # geometric-damped cumulative increments beyond the last observed year
        increment = slope * sum(self.damping ** k for k in range(1, h + 1))
        return base + increment

    def forecast(self, target_year: int, alpha: float = 0.05) -> ForecastResult:
        point = self._predict(self.slope_, self.intercept_, target_year)
        rng = np.random.default_rng(self.seed)
        draws = np.empty(self.n_boot)
        n = self.years.size
        for b in range(self.n_boot):
            resampled = rng.choice(self.resid_, size=n, replace=True)
            y_star = self.fitted_ + resampled
            slope_b, intercept_b = _fit_line(self.years, y_star)
            noise = rng.choice(self.resid_)  # predictive residual
            draws[b] = self._predict(slope_b, intercept_b, target_year) + noise
        lo = float(np.quantile(draws, alpha / 2))
        hi = float(np.quantile(draws, 1 - alpha / 2))
        return ForecastResult(point=float(point), lo=lo, hi=hi)
