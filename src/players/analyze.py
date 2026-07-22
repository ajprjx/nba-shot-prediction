"""Player shot-diet analysis: load, aggregate, benchmark, and rank players by 2030 upside."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from src.agent.loaders.kaggle_csv import map_basic_zone

# ── Action category classification ───────────────────────────────────────────
# Extend these frozensets to support new categories without changing logic.
_ATTACK_TOKENS = frozenset(["Layup", "Dunk", "Finger Roll", "Floating"])
_PULL_UP_TOKENS = frozenset(["Pullup", "Pull-Up", "Step Back", "Fadeaway", "Turnaround"])


def _classify_action(action_type: str) -> str:
    for token in _ATTACK_TOKENS:
        if token in action_type:
            return "attack"
    for token in _PULL_UP_TOKENS:
        if token in action_type:
            return "pull_up"
    return "catch_shoot"


# ── Sub-area and slot-label mappings ─────────────────────────────────────────
# Add entries here to support new SHOT_ZONE_AREA values from future API changes.
_SUB_AREA_LABEL: dict[str, str] = {
    "Left Side(L)": "left",
    "Left Side Center(LC)": "left wing",
    "Center(C)": "center",
    "Right Side Center(RC)": "right wing",
    "Right Side(R)": "right",
    "Back Court(BC)": "backcourt",
}

# Zones already directionally specific — no sub_area suffix in the label.
_SELF_DESCRIBING_ZONES = frozenset(["Left Corner 3", "Right Corner 3", "Restricted Area"])

_ZONE_SHORT: dict[str, str] = {
    "Restricted Area": "Restricted Area",
    "Left Corner 3": "Left Corner 3",
    "Right Corner 3": "Right Corner 3",
    "Above the Break 3": "Above Break 3",
    "In The Paint (Non-RA)": "Paint",
    "Mid-Range": "Mid-Range",
}

_ACTION_LABEL: dict[str, str] = {
    "attack": "attack",
    "pull_up": "pull-up",
    "catch_shoot": "catch-and-shoot",
}


def _make_slot_label(shot_zone_basic: str, sub_area: str, action_category: str) -> str:
    base = _ZONE_SHORT.get(shot_zone_basic, shot_zone_basic)
    location = base if shot_zone_basic in _SELF_DESCRIBING_ZONES else f"{base} ({sub_area})"
    action = _ACTION_LABEL.get(action_category, action_category)
    return f"{location} [{action}]"


def _normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api shotchartdetail rows to the canonical player-shots format."""
    raw = raw.copy()
    raw.columns = [c.upper() for c in raw.columns]

    zone = raw["SHOT_ZONE_BASIC"].map(map_basic_zone)
    sub_area = raw["SHOT_ZONE_AREA"].map(_SUB_AREA_LABEL)
    action_cat = raw["ACTION_TYPE"].apply(_classify_action)
    is_three = raw["SHOT_TYPE"].str.contains("3", na=False)
    pts_value = np.where(is_three, 3, 2)

    out = pd.DataFrame({
        "player_name": raw["PLAYER_NAME"],
        "zone": zone,
        "shot_zone_basic": raw["SHOT_ZONE_BASIC"],
        "sub_area": sub_area,
        "action_category": action_cat,
        "made": raw["SHOT_MADE_FLAG"].astype(int),
        "pts_value": pts_value,
    })
    return out[out["zone"].notna()].reset_index(drop=True)


def load_player_shots(season: str = "2025-26", request_delay: float = 0.6) -> pd.DataFrame:
    """Pull shot-level data for all 30 teams via nba_api and return normalised rows.

    Lazy-imports nba_api so the module loads without the package installed (tests
    call _normalize_raw directly and never reach this function).
    """
    from nba_api.stats.endpoints import shotchartdetail  # noqa: PLC0415
    from nba_api.stats.static import teams as nba_teams  # noqa: PLC0415

    frames = []
    for team in nba_teams.get_teams():
        sc = shotchartdetail.ShotChartDetail(
            team_id=team["id"],
            player_id=0,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
        )
        frames.append(sc.get_data_frames()[0])
        time.sleep(request_delay)

    return _normalize_raw(pd.concat(frames, ignore_index=True))
