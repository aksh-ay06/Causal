"""Data cleaning and encoding utilities."""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing treatment/outcome, impute covariate NaNs."""
    required = ["smoking", "health_score", "cancer"]
    present = [c for c in required if c in df.columns]
    df = df.dropna(subset=present).copy()

    # Numeric covariates → median imputation
    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Categorical covariates → mode imputation
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One‑hot encode the ``race`` column; leave ordinals as‑is."""
    if "race" not in df.columns:
        return df
    dummies = pd.get_dummies(df["race"], prefix="race", drop_first=True, dtype=int)
    df = pd.concat([df.drop(columns=["race"]), dummies], axis=1)
    return df


def scale_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler on selected numeric columns (returns scaler for inverse)."""
    if columns is None:
        columns = ["age", "income"]
    columns = [c for c in columns if c in df.columns]
    scaler = StandardScaler()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def clean_data(df: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
    """Full cleaning pipeline: missing → encode → (optional) scale."""
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    if scale:
        df, _ = scale_features(df)
    return df
