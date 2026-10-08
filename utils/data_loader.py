"""Dataset loading, validation, and profiling (no Streamlit dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.config import DATA_PATH, FEATURES, TARGET


class DataError(RuntimeError):
    """Raised when the dataset is missing, unreadable, or malformed."""


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and return a validated, correctly typed DataFrame."""
    if not path.is_file():
        raise DataError(
            f"Dataset not found at '{path}'. Place loan_dataset_20000.csv "
            "in the project's data/ folder."
        )
    try:
        raw = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"The dataset file could not be parsed: {exc}") from exc
    return validate_dataset(raw)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Check required columns and types; return only the columns the model uses."""
    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise DataError(f"Dataset is missing required columns: {', '.join(missing_columns)}.")

    clean = df[required].copy()
    for column in required:
        converted = pd.to_numeric(clean[column], errors="coerce")
        invalid = converted.isna() & clean[column].notna()
        if invalid.any():
            raise DataError(
                f"Column '{column}' contains {int(invalid.sum())} non-numeric values."
            )
        clean[column] = converted

    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=[TARGET])
    if not set(clean[TARGET].unique()) <= {0, 1}:
        raise DataError(f"Target column '{TARGET}' must contain only 0 and 1.")
    if clean[TARGET].nunique() < 2:
        raise DataError(f"Target column '{TARGET}' must contain both classes.")

    clean[TARGET] = clean[TARGET].astype(int)
    return clean.reset_index(drop=True)


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise data quality: missing values, duplicates, outliers, balance."""
    features = df[FEATURES]
    q1, q3 = features.quantile(0.25), features.quantile(0.75)
    iqr = q3 - q1
    is_outlier = (features < q1 - 1.5 * iqr) | (features > q3 + 1.5 * iqr)
    positives = int((df[TARGET] == 1).sum())

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing": {column: int(count) for column, count in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "positive_rate": float(df[TARGET].mean()),
        "class_counts": {"0": int(len(df)) - positives, "1": positives},
        "iqr_outliers": {column: int(is_outlier[column].sum()) for column in FEATURES},
        "open_exceeds_total_rows": int((df["open_acc"] > df["total_acc"]).sum()),
    }
