"""Additive log-ratio (ALR) transforms so forecast outputs are valid shares
that sum to 1. The last component is the reference."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def alr(shares: np.ndarray) -> np.ndarray:
    """ALR transform. Requires all shares > 0 (zero shares are out of domain)."""
    shares = np.asarray(shares, dtype=float)
    shares = np.clip(shares, _EPS, None)
    ref = shares[-1]
    return np.log(shares[:-1] / ref)


def inverse_alr(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    exps = np.exp(coords)
    denom = 1.0 + exps.sum()
    head = exps / denom
    ref = 1.0 / denom
    return np.concatenate([head, [ref]])


def shares_matrix(df: pd.DataFrame, zones: list[str], value_col: str = "share"):
    any_rows = df[df["context"] == "any"]
    years = np.sort(any_rows["season"].unique())
    pivot = (any_rows.pivot_table(index="season", columns="zone", values=value_col)
             .reindex(index=years, columns=zones))
    return years, pivot.to_numpy(dtype=float)
