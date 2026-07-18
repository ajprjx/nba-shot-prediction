"""Shared IO, logging, and validation helpers."""
from __future__ import annotations

import logging

import pandas as pd

from src.data import schema

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_logger = logging.getLogger("nba_shot_prediction")


def log_message(message: str, level: str = "INFO") -> None:
    print(message)  # user-facing progress
    _logger.log(getattr(logging, level.upper(), logging.INFO), message)


def write_canonical(df: pd.DataFrame, path: str) -> None:
    df[schema.CANONICAL_COLUMNS].to_parquet(path, index=False)


def read_canonical(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)[schema.CANONICAL_COLUMNS]


def validate_data(df: pd.DataFrame) -> list[str]:
    return schema.validate_canonical(df)
