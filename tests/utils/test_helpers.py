import pandas as pd
from src.utils import helpers
from src.data import schema


def test_write_then_read_roundtrip(synthetic_canonical, tmp_path):
    path = str(tmp_path / "canon.parquet")
    helpers.write_canonical(synthetic_canonical, path)
    back = helpers.read_canonical(path)
    assert list(back.columns) == schema.CANONICAL_COLUMNS
    assert len(back) == len(synthetic_canonical)


def test_validate_data_delegates(synthetic_canonical):
    assert helpers.validate_data(synthetic_canonical) == []


def test_log_message_runs(capsys):
    helpers.log_message("hello", level="INFO")
    assert "hello" in capsys.readouterr().out
