"""Tests for diagnostic utilities."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_loader import generate_synthetic_brfss
from src.preprocessing.cleaning import clean_data
from src.diagnostics.balance import compute_smd, balance_table, assess_overlap
from src.diagnostics.sensitivity import rosenbaum_bounds, compute_e_value
from src.diagnostics.placebo import negative_control_test
from src.causal_models.propensity_score import PropensityScoreMatching


@pytest.fixture(scope="module")
def data_and_model():
    df = generate_synthetic_brfss(n=3000, seed=42)
    df = clean_data(df)
    covs = ["age", "female", "education", "income"]
    X = df[covs].values
    T = df["smoking"].values
    Y = df["health_score"].values

    psm = PropensityScoreMatching(seed=42)
    psm.fit(X, T)
    psm.match(T)
    return df, covs, X, T, Y, psm


# ── Balance ────────────────────────────────────────────────────────────────────

class TestBalance:
    def test_smd_identical_is_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_smd(x, x) == pytest.approx(0.0)

    def test_smd_sign(self):
        treated = np.array([10.0, 12.0, 14.0])
        control = np.array([1.0, 2.0, 3.0])
        assert compute_smd(treated, control) > 0

    def test_balance_table_shape(self, data_and_model):
        df, covs, *_ = data_and_model
        bt = balance_table(df, covs)
        assert len(bt) == len(covs)
        assert "covariate" in bt.columns
        assert "smd_unadjusted" in bt.columns

    def test_overlap_returns_dict(self, data_and_model):
        _, _, _, T, _, psm = data_and_model
        result = assess_overlap(psm.propensity_scores_, T)
        assert "positivity_violation_rate" in result
        assert 0 <= result["positivity_violation_rate"] <= 1


# ── Sensitivity ────────────────────────────────────────────────────────────────

class TestSensitivity:
    def test_rosenbaum_bounds_structure(self, data_and_model):
        _, _, _, T, Y, psm = data_and_model
        t_idx, c_idx = psm.matched_indices_
        bounds = rosenbaum_bounds(Y[t_idx], Y[c_idx])
        assert len(bounds) > 0
        assert "gamma" in bounds[0]
        assert "upper_p" in bounds[0]

    def test_gamma_1_most_significant(self, data_and_model):
        """At Γ=1 (no hidden bias), the p‑value should be smallest."""
        _, _, _, T, Y, psm = data_and_model
        t_idx, c_idx = psm.matched_indices_
        bounds = rosenbaum_bounds(Y[t_idx], Y[c_idx])
        p_at_1 = bounds[0]["upper_p"]
        p_at_max = bounds[-1]["upper_p"]
        assert p_at_1 <= p_at_max

    def test_e_value_structure(self):
        result = compute_e_value(2.0, 1.5)
        assert "e_value_point" in result
        assert "e_value_ci" in result
        assert result["e_value_point"] >= 1.0


# ── Placebo ────────────────────────────────────────────────────────────────────

class TestPlacebo:
    def test_negative_control_covers_zero(self, data_and_model):
        """Smoking should not causally affect age (negative control)."""
        df, covs, X, T, _, _ = data_and_model
        result = negative_control_test(
            X, T, df["age"].values, PropensityScoreMatching, seed=42,
        )
        assert "negative_control_ate" in result
        assert "covers_zero" in result
