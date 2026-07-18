import numpy as np
from src.data import schema
from src.agent.loaders.kaggle_csv import KaggleCsvLoader, map_basic_zone, derive_zone


def test_map_basic_zone_known_values():
    assert map_basic_zone("Restricted Area") == "restricted_area"
    assert map_basic_zone("In The Paint (Non-RA)") == "paint_non_ra"
    assert map_basic_zone("Mid-Range") == "midrange"
    assert map_basic_zone("Left Corner 3") == "corner_3"
    assert map_basic_zone("Right Corner 3") == "corner_3"
    assert map_basic_zone("Above the Break 3") == "above_break_3"
    assert map_basic_zone("Backcourt") is None


def test_derive_zone_fallback():
    assert derive_zone(is_three=False, shot_distance=2.0, loc_x=0, loc_y=0) == "restricted_area"
    assert derive_zone(is_three=True, shot_distance=23.0, loc_x=10, loc_y=250) == "above_break_3"
    assert derive_zone(is_three=True, shot_distance=22.0, loc_x=220, loc_y=30) == "corner_3"


def test_loader_produces_valid_canonical(synthetic_kaggle_zip):
    loader = KaggleCsvLoader(synthetic_kaggle_zip)
    df = loader.load()
    assert schema.validate_canonical(df) == []
    assert set(df["zone"]) <= set(schema.ZONES)
    assert (df["context"] == "any").all()
    assert (df["coverage"] == "full_history").all()
    # tracking-only metrics are NaN in Plan 1
    assert df["xefg"].isna().all()


def test_loader_computes_shares_and_efg(synthetic_kaggle_zip):
    df = KaggleCsvLoader(synthetic_kaggle_zip).load()
    # season derived from GAME_DATE 2022-12 -> season-ending year 2023
    assert set(df["season"]) == {2023}
    ra = df[df["zone"] == "restricted_area"].iloc[0]
    ab3 = df[df["zone"] == "above_break_3"].iloc[0]
    assert np.isclose(ra["share"], 0.6)      # 60 of 100 attempts
    assert np.isclose(ab3["share"], 0.4)
    assert np.isclose(ra["efg_pct"], 1.0)    # all 60 made 2PT -> fg 1.0, efg 1.0
    assert np.isclose(ab3["efg_pct"], 0.0)   # all missed
