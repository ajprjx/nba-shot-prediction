import pandas as pd
import pytest
from src.players.analyze import _classify_action, _make_slot_label, _normalize_raw


def _raw(rows):
    return pd.DataFrame(rows, columns=[
        "PLAYER_NAME", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA",
        "ACTION_TYPE", "SHOT_TYPE", "SHOT_MADE_FLAG",
    ])


def test_action_category_attack():
    assert _classify_action("Driving Layup Shot") == "attack"
    assert _classify_action("Cutting Dunk Shot") == "attack"
    assert _classify_action("Driving Floating Jump Shot") == "attack"
    assert _classify_action("Finger Roll Layup Shot") == "attack"


def test_action_category_pull_up():
    assert _classify_action("Pullup Jump shot") == "pull_up"
    assert _classify_action("Step Back Jump shot") == "pull_up"
    assert _classify_action("Fadeaway Jump Shot") == "pull_up"
    assert _classify_action("Turnaround Jump Shot") == "pull_up"


def test_action_category_catch_shoot():
    assert _classify_action("Jump Shot") == "catch_shoot"
    assert _classify_action("Running Jump Shot") == "catch_shoot"


def test_slot_label_self_describing_zone():
    assert _make_slot_label("Left Corner 3", "left", "catch_shoot") == "Left Corner 3 [catch-and-shoot]"
    assert _make_slot_label("Restricted Area", "center", "attack") == "Restricted Area [attack]"


def test_slot_label_sub_area_zone():
    assert _make_slot_label("Above the Break 3", "left wing", "pull_up") == "Above Break 3 (left wing) [pull-up]"
    assert _make_slot_label("Mid-Range", "left", "catch_shoot") == "Mid-Range (left) [catch-and-shoot]"


def test_normalize_drops_backcourt():
    rows = [
        ["Alice", "Restricted Area", "Center(C)", "Layup Shot", "2PT Field Goal", 1],
        ["Alice", "Backcourt",       "Back Court(BC)", "Jump Shot", "3PT Field Goal", 0],
    ]
    out = _normalize_raw(_raw(rows))
    assert len(out) == 1
    assert out.iloc[0]["zone"] == "restricted_area"


def test_normalize_output_columns():
    rows = [["Alice", "Above the Break 3", "Center(C)", "Jump Shot", "3PT Field Goal", 1]]
    out = _normalize_raw(_raw(rows))
    for col in ["player_name", "zone", "shot_zone_basic", "sub_area", "action_category", "made", "pts_value"]:
        assert col in out.columns


def test_normalize_pts_value():
    rows = [
        ["Alice", "Restricted Area",    "Center(C)", "Layup Shot", "2PT Field Goal", 1],
        ["Alice", "Above the Break 3",  "Center(C)", "Jump Shot",  "3PT Field Goal", 0],
    ]
    out = _normalize_raw(_raw(rows))
    assert list(out["pts_value"]) == [2, 3]


def test_normalize_action_classification():
    rows = [
        ["A", "Restricted Area",   "Center(C)",          "Driving Layup Shot", "2PT Field Goal", 1],
        ["A", "Above the Break 3", "Left Side Center(LC)","Pullup Jump shot",   "3PT Field Goal", 0],
        ["A", "Mid-Range",         "Left Side(L)",        "Jump Shot",          "2PT Field Goal", 1],
    ]
    out = _normalize_raw(_raw(rows))
    assert list(out["action_category"]) == ["attack", "pull_up", "catch_shoot"]
