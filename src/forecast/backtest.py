"""Rolling-origin backtest: for each origin >= min_train, train on seasons up to
the origin and predict `horizon` seasons ahead, scoring against actuals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast import compositional as comp
from src.forecast.model import DampedTrendForecaster
from src.forecast.diet import _present_zones


def backtest_shares(df: pd.DataFrame, damping: float = 0.85, min_train: int = 5,
                    horizon: int = 1, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    zones = _present_zones(df)
    years, share_mat = comp.shares_matrix(df, zones, value_col="share")
    n = years.size

    rows = []
    for j, zone in enumerate(zones):
        errs, hits, folds = [], [], 0
        for origin in range(min_train, n - horizon + 1):
            train_years = years[:origin]
            train_vals = share_mat[:origin, j]
            target_idx = origin + horizon - 1
            if target_idx >= n:
                break
            actual = share_mat[target_idx, j]
            f = DampedTrendForecaster(damping=damping, n_boot=n_boot, seed=seed + origin)
            r = f.fit(train_years, train_vals).forecast(int(years[target_idx]))
            errs.append(r.point - actual)
            hits.append(1.0 if (r.lo <= actual <= r.hi) else 0.0)
            folds += 1
        if folds == 0:
            continue
        errs = np.asarray(errs)
        rows.append({
            "zone": zone,
            "n_folds": folds,
            "mae": float(np.mean(np.abs(errs))),
            "rmse": float(np.sqrt(np.mean(errs ** 2))),
            "coverage_rate": float(np.mean(hits)),
        })
    return pd.DataFrame(rows, columns=["zone", "n_folds", "mae", "rmse", "coverage_rate"])
