"""Inverse Probability Weighting (IPTW) estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.config import RANDOM_SEED


class IPWEstimator:
    """Estimate causal effects via Inverse Probability of Treatment Weighting.

    Parameters
    ----------
    trim_quantile : float
        Symmetric quantile at which to trim extreme propensity scores
        (e.g. 0.01 trims the bottom 1 % and top 99 %).
    """

    def __init__(
        self,
        trim_quantile: float = 0.01,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.trim_quantile = trim_quantile
        self.seed = seed
        self._ps_model: LogisticRegression | None = None
        self.propensity_scores_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
    ) -> "IPWEstimator":
        """Estimate propensity scores and compute IPW weights."""
        self._ps_model = LogisticRegression(
            max_iter=1000, random_state=self.seed, solver="lbfgs"
        )
        self._ps_model.fit(X, treatment)
        ps = self._ps_model.predict_proba(X)[:, 1]
        ps = self._trim(ps)
        self.propensity_scores_ = ps

        treatment = np.asarray(treatment)
        # ATE weights: treated get 1/ps, control get 1/(1-ps)
        self.weights_ = treatment / ps + (1 - treatment) / (1 - ps)
        return self

    def _trim(self, ps: np.ndarray) -> np.ndarray:
        lo = np.quantile(ps, self.trim_quantile)
        hi = np.quantile(ps, 1 - self.trim_quantile)
        return np.clip(ps, lo, hi)

    # ── Estimation ──────────────────────────────────────────────────────────
    def estimate_ate(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        n_bootstrap: int = 500,
    ) -> dict:
        """Horvitz–Thompson ATE estimator with bootstrap CI."""
        outcome = np.asarray(outcome, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        ps = self.propensity_scores_

        ate = self._ht_ate(outcome, treatment, ps)

        # Bootstrap CI
        rng = np.random.default_rng(self.seed)
        n = len(outcome)
        boot_ates = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            boot_ates.append(self._ht_ate(outcome[idx], treatment[idx], ps[idx]))
        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))

        return {"ate": float(ate), "ci_lower": ci_lower, "ci_upper": ci_upper}

    def estimate_att(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        n_bootstrap: int = 500,
    ) -> dict:
        """ATT via IPW (weight the controls by ps/(1-ps))."""
        outcome = np.asarray(outcome, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        ps = self.propensity_scores_

        att = self._att(outcome, treatment, ps)

        rng = np.random.default_rng(self.seed)
        n = len(outcome)
        boot_atts = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            boot_atts.append(self._att(outcome[idx], treatment[idx], ps[idx]))
        ci_lower = float(np.percentile(boot_atts, 2.5))
        ci_upper = float(np.percentile(boot_atts, 97.5))

        return {"att": float(att), "ci_lower": ci_lower, "ci_upper": ci_upper}

    # ── Internal ────────────────────────────────────────────────────────────
    @staticmethod
    def _ht_ate(y: np.ndarray, t: np.ndarray, ps: np.ndarray) -> float:
        """Horvitz–Thompson ATE."""
        weighted_treated = np.sum(t * y / ps) / np.sum(t / ps)
        weighted_control = np.sum((1 - t) * y / (1 - ps)) / np.sum((1 - t) / (1 - ps))
        return weighted_treated - weighted_control

    @staticmethod
    def _att(y: np.ndarray, t: np.ndarray, ps: np.ndarray) -> float:
        """ATT via weighting controls by ps/(1-ps)."""
        n_treated = t.sum()
        if n_treated == 0:
            return 0.0
        mean_treated = (t * y).sum() / n_treated
        w_control = (1 - t) * ps / (1 - ps)
        mean_control = (w_control * y).sum() / w_control.sum()
        return mean_treated - mean_control
