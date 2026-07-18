"""Loader for the Kaggle 'NBA shots' shot-level dataset (archive.zip).

Produces canonical rows at season x zone grain, context='any' only. Prefers the
dataset's BASIC_ZONE text column; falls back to coordinate geometry when absent.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from src.data import schema
from src.agent.loaders.base import Loader

_BASIC_ZONE_MAP = {
    "restricted area": "restricted_area",
    "in the paint (non-ra)": "paint_non_ra",
    "mid-range": "midrange",
    "left corner 3": "corner_3",
    "right corner 3": "corner_3",
    "above the break 3": "above_break_3",
}

# LOC_X/LOC_Y are in tenths of a foot; hoop at (0,0). Corner-3 baseline band.
_CORNER_Y_TENTHS = 92.5


def map_basic_zone(basic_zone: str) -> str | None:
    if not isinstance(basic_zone, str):
        return None
    return _BASIC_ZONE_MAP.get(basic_zone.strip().lower())


def derive_zone(is_three: bool, shot_distance: float, loc_x: float, loc_y: float) -> str:
    if is_three:
        return "corner_3" if abs(loc_y) <= _CORNER_Y_TENTHS else "above_break_3"
    if shot_distance <= 4:
        return "restricted_area"
    if abs(loc_x) <= 80 and loc_y <= 142.5:  # ~16ft-wide, ~14ft-deep paint box
        return "paint_non_ra"
    return "midrange"


def _season_ending_year(dates: pd.Series) -> pd.Series:
    """NBA season spans Oct-Jun; label by the ending calendar year."""
    dt = pd.to_datetime(dates)
    return np.where(dt.dt.month >= 10, dt.dt.year + 1, dt.dt.year)


class KaggleCsvLoader(Loader):
    def __init__(self, zip_path: str, source_id: str = "kaggle.nba_shots"):
        self.zip_path = zip_path
        self.source_id = source_id

    def fetch(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            csvs = [f for f in archive.namelist() if f.endswith(".csv")]
            if not csvs:
                raise FileNotFoundError(f"No CSV in {self.zip_path}")
            with archive.open(csvs[0]) as fh:
                return pd.read_csv(fh)

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = [c.upper() for c in df.columns]
        df["SHOT_MADE"] = df["SHOT_MADE"].astype(bool)
        is_three = df["SHOT_TYPE"].astype(str).str.contains("3")

        if "BASIC_ZONE" in df.columns:
            zone = df["BASIC_ZONE"].map(map_basic_zone)
            fallback = zone.isna()
            if fallback.any():
                zone.loc[fallback] = [
                    derive_zone(t, d, x, y)
                    for t, d, x, y in zip(
                        is_three[fallback], df.loc[fallback, "SHOT_DISTANCE"],
                        df.loc[fallback, "LOC_X"], df.loc[fallback, "LOC_Y"])
                ]
        else:
            zone = pd.Series([
                derive_zone(t, d, x, y)
                for t, d, x, y in zip(is_three, df["SHOT_DISTANCE"], df["LOC_X"], df["LOC_Y"])
            ], index=df.index)

        work = pd.DataFrame({
            "season": _season_ending_year(df["GAME_DATE"]),
            "zone": zone,
            "made": df["SHOT_MADE"].astype(int),
            "is_three": is_three.astype(int),
        })
        work = work[work["zone"].notna()]  # drop backcourt/unmapped

        grouped = work.groupby(["season", "zone"])
        agg = grouped.agg(
            attempts=("made", "size"),
            makes=("made", "sum"),
            threes_made=("is_three", lambda s: (s & work.loc[s.index, "made"].astype(bool)).sum()),
        ).reset_index()

        season_totals = agg.groupby("season")["attempts"].transform("sum")
        agg["share"] = agg["attempts"] / season_totals
        agg["fg_pct"] = agg["makes"] / agg["attempts"]
        points = agg["makes"] * 2 + agg["threes_made"]  # each 3 adds 1 extra pt
        agg["efg_pct"] = (agg["makes"] + 0.5 * agg["threes_made"]) / agg["attempts"]
        agg["pts_per_shot"] = points / agg["attempts"]

        out = schema.empty_canonical()
        out = pd.concat([out, pd.DataFrame({
            "season": agg["season"].astype(int),
            "zone": agg["zone"],
            "context": "any",
            "attempts": agg["attempts"].astype(int),
            "share": agg["share"],
            "fg_pct": agg["fg_pct"],
            "efg_pct": agg["efg_pct"],
            "pts_per_shot": agg["pts_per_shot"],
            "xefg": np.nan,
            "expected_pts_per_shot": np.nan,
            "shot_making_delta": np.nan,
            "ft_value": np.nan,
            "coverage": "full_history",
            "source_id": self.source_id,
        })], ignore_index=True)
        return out[schema.CANONICAL_COLUMNS]
