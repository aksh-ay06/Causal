"""Inverse Probability Weighting (IPTW) estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        self._scaler: StandardScaler | None = None
        self._X: np.ndarray | None = None
        self.propensity_scores_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
    ) -> "IPWEstimator":
        """Estimate propensity scores and compute IPW weights."""
        self._X = np.asarray(X, dtype=float)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(self._X)
        self._ps_model = LogisticRegression(
            max_iter=1000, random_state=self.seed, solver="lbfgs"
        )
        self._ps_model.fit(X_scaled, treatment)
        ps = self._ps_model.predict_proba(X_scaled)[:, 1]
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
        n_bootstrap: int = 200,
    ) -> dict:
        """Horvitz–Thompson ATE estimator with bootstrap CI."""
        outcome = np.asarray(outcome, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        ps = self.propensity_scores_

        ate = self._ht_ate(outcome, treatment, ps)

        if n_bootstrap == 0:
            return {"ate": float(ate), "ci_lower": np.nan, "ci_upper": np.nan}

        # Bootstrap CI – re-estimate propensity scores on each resample
        rng = np.random.default_rng(self.seed)
        n = len(outcome)
        X_arr = self._X
        boot_ates = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            X_b, T_b, Y_b = X_arr[idx], treatment[idx], outcome[idx]
            scaler_b = StandardScaler()
            X_bs = scaler_b.fit_transform(X_b)
            ps_model_b = LogisticRegression(
                max_iter=1000, random_state=self.seed, solver="lbfgs"
            )
            ps_model_b.fit(X_bs, T_b)
            ps_b = self._trim(ps_model_b.predict_proba(X_bs)[:, 1])
            boot_ates.append(self._ht_ate(Y_b, T_b, ps_b))
        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))

        return {"ate": float(ate), "ci_lower": ci_lower, "ci_upper": ci_upper}

    def estimate_att(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        n_bootstrap: int = 200,
    ) -> dict:
        """ATT via IPW (weight the controls by ps/(1-ps))."""
        outcome = np.asarray(outcome, dtype=float)
        treatment = np.asarray(treatment, dtype=float)
        ps = self.propensity_scores_

        att = self._att(outcome, treatment, ps)

        if n_bootstrap == 0:
            return {"att": float(att), "ci_lower": np.nan, "ci_upper": np.nan}

        # Bootstrap CI – re-estimate propensity scores on each resample
        rng = np.random.default_rng(self.seed)
        n = len(outcome)
        X_arr = self._X
        boot_atts = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            X_b, T_b, Y_b = X_arr[idx], treatment[idx], outcome[idx]
            scaler_b = StandardScaler()
            X_bs = scaler_b.fit_transform(X_b)
            ps_model_b = LogisticRegression(
                max_iter=1000, random_state=self.seed, solver="lbfgs"
            )
            ps_model_b.fit(X_bs, T_b)
            ps_b = self._trim(ps_model_b.predict_proba(X_bs)[:, 1])
            boot_atts.append(self._att(Y_b, T_b, ps_b))
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
