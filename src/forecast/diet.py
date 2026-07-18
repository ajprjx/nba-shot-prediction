"""Assemble the 2030 shot-diet forecast: zone shares projected compositionally
(ALR space) and per-zone shot value projected independently."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import schema
from src.forecast import compositional as comp
from src.forecast.model import DampedTrendForecaster


def _present_zones(df: pd.DataFrame) -> list[str]:
    present = set(df[df["context"] == "any"]["zone"])
    return [z for z in schema.ZONES if z in present]


def forecast_shot_diet(df: pd.DataFrame, target_year: int, damping: float = 0.85,
                       n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    zones = _present_zones(df)
    years, mat = comp.shares_matrix(df, zones, value_col="share")

    # Forecast each ALR coordinate independently, then inverse-transform the point.
    coord_series = np.array([comp.alr(mat[i]) for i in range(mat.shape[0])])  # (T, k-1)
    point_coords = np.empty(coord_series.shape[1])
    for j in range(coord_series.shape[1]):
        f = DampedTrendForecaster(damping=damping, n_boot=n_boot, seed=seed + j)
        point_coords[j] = f.fit(years, coord_series[:, j]).forecast(target_year).point
    projected_shares = comp.inverse_alr(point_coords)

    # Per-zone share intervals via raw-share bootstrap, clamped to [0,1] and forced to
    # bracket the ALR point (the two paths can diverge slightly on short series).
    share_lo, share_hi = {}, {}
    _, share_mat = comp.shares_matrix(df, zones, value_col="share")
    for j, zone in enumerate(zones):
        f = DampedTrendForecaster(damping=damping, n_boot=n_boot, seed=seed + 100 + j)
        r = f.fit(years, share_mat[:, j]).forecast(target_year)
        share_lo[zone] = max(0.0, min(r.lo, projected_shares[j]))
        share_hi[zone] = min(1.0, max(r.hi, projected_shares[j]))

    # Per-zone shot value.
    _, val_mat = comp.shares_matrix(df, zones, value_col="pts_per_shot")
    rows = []
    for j, zone in enumerate(zones):
        vf = DampedTrendForecaster(damping=damping, n_boot=n_boot, seed=seed + 200 + j)
        vr = vf.fit(years, val_mat[:, j]).forecast(target_year)
        rows.append({
            "zone": zone,
            "projected_share": projected_shares[j],
            "share_lo": share_lo[zone],
            "share_hi": share_hi[zone],
            "projected_pts_per_shot": vr.point,
            "pts_lo": vr.lo,
            "pts_hi": vr.hi,
        })
    return pd.DataFrame(rows)
