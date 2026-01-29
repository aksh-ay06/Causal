"""Feature engineering for causal analysis."""

import numpy as np
import pandas as pd


def create_age_bins(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Add ``age_bin`` column (equal‑width bins)."""
    df = df.copy()
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=False)
    return df


def create_income_quantiles(df: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    """Add ``income_q`` column (quantile‑based)."""
    df = df.copy()
    df["income_q"] = pd.qcut(df["income"], q=q, labels=False, duplicates="drop")
    return df


def create_interaction_terms(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Create multiplicative interaction columns for heterogeneity analysis."""
    if pairs is None:
        pairs = [("age", "education"), ("female", "education")]
    df = df.copy()
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature‑engineering pipeline."""
    df = create_age_bins(df)
    df = create_income_quantiles(df)
    df = create_interaction_terms(df)
    return df
