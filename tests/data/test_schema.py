import numpy as np
import pandas as pd
from src.data import schema


def _valid_row(**overrides):
    row = {
        "season": 2024, "zone": "restricted_area", "context": "any",
        "attempts": 1000, "share": 1.0, "fg_pct": 0.6, "efg_pct": 0.6,
        "pts_per_shot": 1.2, "xefg": np.nan, "expected_pts_per_shot": np.nan,
        "shot_making_delta": np.nan, "ft_value": np.nan,
        "coverage": "full_history", "source_id": "kaggle.nba_shots",
    }
    row.update(overrides)
    return row


def test_empty_canonical_has_exact_columns():
    df = schema.empty_canonical()
    assert list(df.columns) == schema.CANONICAL_COLUMNS
    assert len(df) == 0


def test_valid_frame_passes():
    # two zones in one season whose shares sum to 1.0
    df = pd.DataFrame([
        _valid_row(zone="restricted_area", share=0.4),
        _valid_row(zone="above_break_3", share=0.6),
    ])
    assert schema.validate_canonical(df) == []


def test_bad_zone_is_reported():
    df = pd.DataFrame([_valid_row(zone="halfcourt")])
    errors = schema.validate_canonical(df)
    assert any("zone" in e for e in errors)


def test_shares_must_sum_to_one_within_context_any():
    df = pd.DataFrame([
        _valid_row(zone="restricted_area", share=0.4),
        _valid_row(zone="above_break_3", share=0.4),  # sums to 0.8
    ])
    errors = schema.validate_canonical(df)
    assert any("sum to 1" in e for e in errors)


def test_duplicate_grain_is_reported():
    df = pd.DataFrame([_valid_row(share=0.5), _valid_row(share=0.5)])
    errors = schema.validate_canonical(df)
    assert any("duplicate" in e.lower() for e in errors)


def test_missing_column_is_reported():
    df = schema.empty_canonical().drop(columns=["pts_per_shot"])
    errors = schema.validate_canonical(df)
    assert any("pts_per_shot" in e for e in errors)
