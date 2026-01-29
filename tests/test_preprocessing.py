"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_loader import generate_synthetic_brfss, load_dataset
from src.preprocessing.cleaning import (
    clean_data,
    encode_categoricals,
    handle_missing_values,
)
from src.preprocessing.feature_engineering import (
    create_age_bins,
    create_income_quantiles,
    create_interaction_terms,
    engineer_features,
)


# ── Data loader ────────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_returns_dataframe(self):
        df = generate_synthetic_brfss(n=500, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self):
        df = generate_synthetic_brfss(n=1000, seed=0)
        assert df.shape[0] == 1000
        assert "smoking" in df.columns
        assert "health_score" in df.columns
        assert "cancer" in df.columns

    def test_treatment_is_binary(self):
        df = generate_synthetic_brfss(n=2000, seed=0)
        assert set(df["smoking"].unique()).issubset({0, 1})

    def test_cancer_is_binary(self):
        df = generate_synthetic_brfss(n=2000, seed=0)
        assert set(df["cancer"].unique()).issubset({0, 1})

    def test_deterministic(self):
        df1 = generate_synthetic_brfss(n=200, seed=42)
        df2 = generate_synthetic_brfss(n=200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_both_treatment_groups_present(self):
        df = generate_synthetic_brfss(n=5000, seed=0)
        assert df["smoking"].sum() > 0
        assert (df["smoking"] == 0).sum() > 0

    def test_load_dataset_fallback(self):
        df = load_dataset(path=None, n=100, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100


# ── Cleaning ───────────────────────────────────────────────────────────────────

class TestCleaning:
    @pytest.fixture()
    def raw_df(self):
        return generate_synthetic_brfss(n=500, seed=0)

    def test_encode_categoricals_drops_race(self, raw_df):
        encoded = encode_categoricals(raw_df)
        assert "race" not in encoded.columns
        assert any(c.startswith("race_") for c in encoded.columns)

    def test_handle_missing_drops_na_treatment(self):
        df = pd.DataFrame({
            "smoking": [1, np.nan, 0],
            "health_score": [70, 80, 90],
            "cancer": [0, 1, 0],
            "age": [30, 40, 50],
        })
        cleaned = handle_missing_values(df)
        assert len(cleaned) == 2

    def test_clean_data_pipeline(self, raw_df):
        cleaned = clean_data(raw_df)
        assert "race" not in cleaned.columns
        assert cleaned.isna().sum().sum() == 0


# ── Feature engineering ────────────────────────────────────────────────────────

class TestFeatureEngineering:
    @pytest.fixture()
    def clean_df(self):
        df = generate_synthetic_brfss(n=500, seed=0)
        return clean_data(df)

    def test_age_bins(self, clean_df):
        result = create_age_bins(clean_df)
        assert "age_bin" in result.columns

    def test_income_quantiles(self, clean_df):
        result = create_income_quantiles(clean_df)
        assert "income_q" in result.columns

    def test_interaction_terms(self, clean_df):
        result = create_interaction_terms(clean_df, [("age", "education")])
        assert "age_x_education" in result.columns

    def test_engineer_features_pipeline(self, clean_df):
        result = engineer_features(clean_df)
        assert "age_bin" in result.columns
        assert "income_q" in result.columns
