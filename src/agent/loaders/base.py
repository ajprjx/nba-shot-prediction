"""Loader abstraction: every data source implements fetch + normalize and
returns rows conforming to the canonical schema."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    source_id: str = "unknown"

    @abstractmethod
    def fetch(self) -> object:
        """Return raw source data (network/disk). Cache-friendly."""

    @abstractmethod
    def normalize(self, raw: object) -> pd.DataFrame:
        """Transform raw data into a canonical-schema DataFrame."""

    def load(self) -> pd.DataFrame:
        return self.normalize(self.fetch())
