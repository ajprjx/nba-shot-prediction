import zipfile
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_kaggle_zip(tmp_path):
    """A tiny archive.zip whose single CSV mimics the Kaggle NBA-shots schema."""
    rows = []
    # season 2023 (ending year): GAME_DATE month >= 10 rolls into next season
    for _ in range(60):
        rows.append(["2022-12-01", "Restricted Area", "2PT Field Goal", 2.0, 0, 0, True])
    for _ in range(40):
        rows.append(["2022-12-01", "Above the Break 3", "3PT Field Goal", 25.0, 10, 250, False])
    df = pd.DataFrame(rows, columns=[
        "GAME_DATE", "BASIC_ZONE", "SHOT_TYPE", "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_MADE",
    ])
    csv_bytes = df.to_csv(index=False).encode()
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("nba_shots.csv", csv_bytes)
    return str(zip_path)


@pytest.fixture
def synthetic_canonical():
    """Multi-season canonical frame with a clean upward 3PT trend (for forecast tests)."""
    from src.data import schema
    recs = []
    for i, season in enumerate(range(2015, 2025)):
        three = 0.30 + 0.02 * i          # rising above-break-3 share
        rim = 1.0 - three
        recs.append(dict(season=season, zone="restricted_area", context="any",
                         attempts=1000, share=rim, fg_pct=0.62, efg_pct=0.62,
                         pts_per_shot=1.24 + 0.005 * i, xefg=np.nan,
                         expected_pts_per_shot=np.nan, shot_making_delta=np.nan,
                         ft_value=np.nan, coverage="full_history", source_id="synthetic"))
        recs.append(dict(season=season, zone="above_break_3", context="any",
                         attempts=1000, share=three, fg_pct=0.36, efg_pct=0.54,
                         pts_per_shot=1.05 + 0.004 * i, xefg=np.nan,
                         expected_pts_per_shot=np.nan, shot_making_delta=np.nan,
                         ft_value=np.nan, coverage="full_history", source_id="synthetic"))
    return pd.DataFrame(recs)[schema.CANONICAL_COLUMNS]
