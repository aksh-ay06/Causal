"""Tests for causal estimation models."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_loader import generate_synthetic_brfss
from src.preprocessing.cleaning import clean_data
from src.causal_models.propensity_score import PropensityScoreMatching
from src.causal_models.ipw import IPWEstimator
from src.utils.config import TRUE_ATE_HEALTH


@pytest.fixture(scope="module")
def prepared_data():
    """Shared synthetic dataset for all model tests."""
    df = generate_synthetic_brfss(n=5000, seed=42)
    df = clean_data(df)
    covs = ["age", "female", "education", "income", "race_black", "race_hispanic", "race_other"]
    covs = [c for c in covs if c in df.columns]
    X = df[covs].values
    T = df["smoking"].values
    Y = df["health_score"].values
    return X, T, Y, df


# ── Propensity Score Matching ──────────────────────────────────────────────────

class TestPSM:
    def test_fit_produces_scores(self, prepared_data):
        X, T, Y, _ = prepared_data
        psm = PropensityScoreMatching(seed=42)
        psm.fit(X, T)
        assert psm.propensity_scores_ is not None
        assert len(psm.propensity_scores_) == len(T)
        assert np.all((psm.propensity_scores_ >= 0) & (psm.propensity_scores_ <= 1))

    def test_match_returns_indices(self, prepared_data):
        X, T, Y, _ = prepared_data
        psm = PropensityScoreMatching(seed=42, caliper=None)
        psm.fit(X, T)
        t_idx, c_idx = psm.match(T)
        assert len(t_idx) > 0
        assert len(c_idx) > 0
        # All treated indices should actually be treated
        assert np.all(T[t_idx] == 1)
        assert np.all(T[c_idx] == 0)

    def test_ate_has_correct_keys(self, prepared_data):
        X, T, Y, _ = prepared_data
        psm = PropensityScoreMatching(seed=42)
        psm.fit(X, T)
        psm.match(T)
        result = psm.estimate_ate(Y, T, n_bootstrap=100)
        assert "ate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["ate"] <= result["ci_upper"]

    def test_ate_direction(self, prepared_data):
        """Smoking should have a negative effect on health (true ATE = -5)."""
        X, T, Y, _ = prepared_data
        psm = PropensityScoreMatching(seed=42)
        psm.fit(X, T)
        psm.match(T)
        result = psm.estimate_ate(Y, T, n_bootstrap=100)
        assert result["ate"] < 0, "Smoking ATE on health should be negative"


# ── IPW ────────────────────────────────────────────────────────────────────────

class TestIPW:
    def test_fit_produces_weights(self, prepared_data):
        X, T, Y, _ = prepared_data
        ipw = IPWEstimator(seed=42)
        ipw.fit(X, T)
        assert ipw.weights_ is not None
        assert len(ipw.weights_) == len(T)
        assert np.all(ipw.weights_ > 0)

    def test_propensity_scores_in_range(self, prepared_data):
        X, T, Y, _ = prepared_data
        ipw = IPWEstimator(seed=42)
        ipw.fit(X, T)
        ps = ipw.propensity_scores_
        assert np.all((ps > 0) & (ps < 1))

    def test_ate_has_correct_keys(self, prepared_data):
        X, T, Y, _ = prepared_data
        ipw = IPWEstimator(seed=42)
        ipw.fit(X, T)
        result = ipw.estimate_ate(Y, T, n_bootstrap=100)
        assert "ate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_ate_direction(self, prepared_data):
        X, T, Y, _ = prepared_data
        ipw = IPWEstimator(seed=42)
        ipw.fit(X, T)
        result = ipw.estimate_ate(Y, T, n_bootstrap=100)
        assert result["ate"] < 0, "Smoking ATE on health should be negative"

    def test_att_has_correct_keys(self, prepared_data):
        X, T, Y, _ = prepared_data
        ipw = IPWEstimator(seed=42)
        ipw.fit(X, T)
        result = ipw.estimate_att(Y, T, n_bootstrap=100)
        assert "att" in result
        assert "ci_lower" in result
        assert "ci_upper" in result


# ── Estimate quality (loose tolerance) ─────────────────────────────────────────

class TestEstimateQuality:
    """Check that estimates are within a reasonable range of the true ATE."""

    def test_psm_within_tolerance(self, prepared_data):
        X, T, Y, _ = prepared_data
        psm = PropensityScoreMatching(seed=42)
        psm.fit(X, T)
        psm.match(T)
        result = psm.estimate_ate(Y, T, n_bootstrap=200)
        assert abs(result["ate"] - TRUE_ATE_HEALTH) < 3.0, (
            f"PSM ATE {result['ate']:.2f} too far from true {TRUE_ATE_HEALTH}"
        )

    def test_ipw_within_tolerance(self, prepared_data):
        X, T, Y, _ = prepared_data
        ipw = IPWEstimator(seed=42)
        ipw.fit(X, T)
        result = ipw.estimate_ate(Y, T, n_bootstrap=200)
        assert abs(result["ate"] - TRUE_ATE_HEALTH) < 3.0, (
            f"IPW ATE {result['ate']:.2f} too far from true {TRUE_ATE_HEALTH}"
        )
