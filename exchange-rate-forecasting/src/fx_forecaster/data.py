from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "exchange_rate"}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV dataset."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate dates found, e.g. {dupes[:5]}")

    numeric_candidates = [c for c in df.columns if c != "date"]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if (df["exchange_rate"] <= 0).any():
        raise ValueError("exchange_rate must be strictly positive for log-return modeling")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def numeric_predictor_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c != "date" and pd.api.types.is_numeric_dtype(df[c])
    ]
