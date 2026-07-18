"""Canonical shot-diet table contract: the single interface every loader,
forecaster, and report agrees on."""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = ["restricted_area", "paint_non_ra", "midrange", "corner_3", "above_break_3"]
CONTEXTS = ["any", "catch_and_shoot", "pull_up"]
COVERAGES = ["full_history", "tracking_era"]

CANONICAL_COLUMNS = [
    "season", "zone", "context", "attempts", "share",
    "fg_pct", "efg_pct", "pts_per_shot",
    "xefg", "expected_pts_per_shot", "shot_making_delta", "ft_value",
    "coverage", "source_id",
]

# Populated only in tracking-era (Plan 2); NaN otherwise.
TRACKING_ONLY_COLUMNS = ["xefg", "expected_pts_per_shot", "shot_making_delta", "ft_value"]

_FLOAT_COLS = ["share", "fg_pct", "efg_pct", "pts_per_shot"] + TRACKING_ONLY_COLUMNS


def empty_canonical() -> pd.DataFrame:
    """Zero-row frame with the exact canonical columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS})


def validate_canonical(df: pd.DataFrame, share_tol: float = 0.02) -> list[str]:
    """Return a list of human-readable problems. Empty list == valid."""
    errors: list[str] = []

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    for c in missing:
        errors.append(f"missing required column: {c}")
    if missing:
        return errors  # further checks assume columns exist

    bad_zones = sorted(set(df["zone"]) - set(ZONES))
    if bad_zones:
        errors.append(f"invalid zone values: {bad_zones}")
    bad_ctx = sorted(set(df["context"]) - set(CONTEXTS))
    if bad_ctx:
        errors.append(f"invalid context values: {bad_ctx}")
    bad_cov = sorted(set(df["coverage"]) - set(COVERAGES))
    if bad_cov:
        errors.append(f"invalid coverage values: {bad_cov}")

    if df.duplicated(subset=["season", "zone", "context"]).any():
        errors.append("duplicate rows for a (season, zone, context) grain")

    if (df["attempts"] < 0).any():
        errors.append("attempts must be non-negative")

    # Shares must sum to ~1 within each season for context == 'any'.
    any_rows = df[df["context"] == "any"]
    for season, grp in any_rows.groupby("season"):
        total = grp["share"].sum()
        if not np.isclose(total, 1.0, atol=share_tol):
            errors.append(
                f"season {season}: context='any' shares sum to {total:.3f}, expected sum to 1"
            )
    return errors
