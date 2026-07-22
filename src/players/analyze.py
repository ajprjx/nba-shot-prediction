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


def player_shot_diet(shots_df: pd.DataFrame, min_attempts: int = 1000) -> pd.DataFrame:
    """Aggregate shots to player × zone grain; filter players below min_attempts."""
    work = shots_df.assign(pts=shots_df["pts_value"] * shots_df["made"])

    agg = work.groupby(["player_name", "zone"]).agg(
        attempts=("made", "size"),
        makes=("made", "sum"),
        pts=("pts", "sum"),
    ).reset_index()

    player_totals = agg.groupby("player_name")["attempts"].transform("sum")
    agg = agg[player_totals >= min_attempts].copy()

    season_totals = agg.groupby("player_name")["attempts"].transform("sum")
    agg["share"] = agg["attempts"] / season_totals
    agg["fg_pct"] = agg["makes"] / agg["attempts"]
    agg["pts_per_shot"] = agg["pts"] / agg["attempts"]

    return agg[["player_name", "zone", "attempts", "share", "fg_pct", "pts_per_shot"]].reset_index(drop=True)


def league_slot_stats(shots_df: pd.DataFrame) -> pd.DataFrame:
    """League-wide efficiency at zone × sub_area × action_category grain.

    Returns the benchmark used by specific_shot_rec. Can be pre-computed once
    and passed to multiple player lookups.
    """
    work = shots_df.assign(pts=shots_df["pts_value"] * shots_df["made"])

    agg = work.groupby(["zone", "shot_zone_basic", "sub_area", "action_category"]).agg(
        attempts=("made", "size"),
        makes=("made", "sum"),
        pts=("pts", "sum"),
    ).reset_index()

    total = agg["attempts"].sum()
    agg["share"] = agg["attempts"] / total
    agg["fg_pct"] = agg["makes"] / agg["attempts"]
    agg["pts_per_shot"] = agg["pts"] / agg["attempts"]
    agg["slot_label"] = agg.apply(
        lambda r: _make_slot_label(r["shot_zone_basic"], r["sub_area"], r["action_category"]),
        axis=1,
    )

    return agg[["zone", "shot_zone_basic", "sub_area", "action_category", "slot_label",
                "attempts", "share", "fg_pct", "pts_per_shot"]].reset_index(drop=True)


def specific_shot_rec(
    player_name: str,
    shots_df: pd.DataFrame,
    league_stats_df: pd.DataFrame,
    league_mix_pts: float,
) -> str:
    """Return the slot label of the highest-value, most under-used shot for a player.

    league_stats_df should be the output of league_slot_stats(shots_df).
    league_mix_pts is the weighted-average projected pts/shot under the 2030 mix.
    """
    player_shots = shots_df[shots_df["player_name"] == player_name]
    if player_shots.empty:
        return "insufficient data"

    total = len(player_shots)
    player_slot = (
        player_shots
        .groupby(["zone", "shot_zone_basic", "sub_area", "action_category"])
        .size()
        .reset_index(name="attempts")
    )
    player_slot["player_share"] = player_slot["attempts"] / total

    combined = league_stats_df.merge(
        player_slot[["zone", "shot_zone_basic", "sub_area", "action_category", "player_share"]],
        on=["zone", "shot_zone_basic", "sub_area", "action_category"],
        how="left",
    ).fillna({"player_share": 0.0})

    above_avg = combined[combined["pts_per_shot"] > league_mix_pts].copy()
    if above_avg.empty:
        return "already well-positioned"

    above_avg["gap"] = above_avg["share"] - above_avg["player_share"]
    best = above_avg.loc[above_avg["gap"].idxmax()]

    if best["gap"] <= 0:
        return "already well-positioned"

    return str(best["slot_label"])


def player_upside(
    diet_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    shots_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Rank players by pts/shot gain from shifting toward the 2030 league-average mix.

    Pass shots_df to attach per-player specific slot recommendations. Omit it
    to run zone-level ranking only (faster, useful for testing or batch sweeps).
    """
    proj = forecast_df[["zone", "projected_pts_per_shot", "projected_share"]]
    league_mix_pts = float((proj["projected_share"] * proj["projected_pts_per_shot"]).sum())

    above_avg_zones = set(proj[proj["projected_pts_per_shot"] > league_mix_pts]["zone"])
    below_avg_zones = set(proj[proj["projected_pts_per_shot"] <= league_mix_pts]["zone"])

    merged = diet_df.merge(proj, on="zone", how="left")

    slot_stats = league_slot_stats(shots_df) if shots_df is not None else None

    rows = []
    for player_name, grp in merged.groupby("player_name"):
        total_attempts = int(grp["attempts"].sum())
        current_mix = float((grp["share"] * grp["projected_pts_per_shot"]).sum())

        low_val = grp[grp["zone"].isin(below_avg_zones)].copy()
        low_val["over_index"] = low_val["share"] - low_val["projected_share"]
        shift_from = (
            low_val.loc[low_val["over_index"].idxmax(), "zone"]
            if not low_val.empty and low_val["over_index"].max() > 0
            else None
        )

        # For shift_to, consider ALL above-avg zones from the forecast — including
        # zones the player has never attempted (treat those as share=0).
        above_avg_proj = proj[proj["zone"].isin(above_avg_zones)].copy()
        player_zone_share = dict(zip(grp["zone"], grp["share"]))
        above_avg_proj = above_avg_proj.copy()
        above_avg_proj["player_share"] = above_avg_proj["zone"].map(player_zone_share).fillna(0.0)
        above_avg_proj["under_index"] = above_avg_proj["projected_share"] - above_avg_proj["player_share"]
        shift_to = (
            above_avg_proj.loc[above_avg_proj["under_index"].idxmax(), "zone"]
            if not above_avg_proj.empty and above_avg_proj["under_index"].max() > 0
            else None
        )

        rec = (
            specific_shot_rec(player_name, shots_df, slot_stats, league_mix_pts)
            if shots_df is not None
            else None
        )

        rows.append({
            "player_name": player_name,
            "total_attempts": total_attempts,
            "current_mix_pts": round(current_mix, 4),
            "league_mix_pts": round(league_mix_pts, 4),
            "opportunity_delta": round(league_mix_pts - current_mix, 4),
            "shift_from_zone": shift_from,
            "shift_to_zone": shift_to,
            "specific_rec": rec,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("opportunity_delta", ascending=False)
        .reset_index(drop=True)
    )
