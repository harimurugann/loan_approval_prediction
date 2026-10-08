"""Dataset aggregations used by the analytics page (no plotting here)."""

from __future__ import annotations

import pandas as pd

from utils.config import TARGET

RateSeries = tuple[list[str], list[float], list[int]]


def binned_rate(df: pd.DataFrame, column: str, bins: int = 10, decimals: int = 0) -> RateSeries:
    """Repayment rate per quantile bin of ``column``: (labels, rates, counts)."""
    subset = df[[column, TARGET]].dropna()
    groups = pd.qcut(subset[column], q=bins, duplicates="drop")
    stats = subset.groupby(groups, observed=True)[TARGET].agg(["mean", "size"])
    labels = [
        f"{interval.left:,.{decimals}f}–{interval.right:,.{decimals}f}"
        for interval in stats.index
    ]
    return labels, stats["mean"].astype(float).tolist(), stats["size"].astype(int).tolist()


def rate_by_term(df: pd.DataFrame) -> RateSeries:
    """Repayment rate for each loan term: (labels, rates, counts)."""
    stats = df.groupby("term")[TARGET].agg(["mean", "size"]).sort_index()
    labels = [f"{int(term)} months" for term in stats.index]
    return labels, stats["mean"].astype(float).tolist(), stats["size"].astype(int).tolist()
