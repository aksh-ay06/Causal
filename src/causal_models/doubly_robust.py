"""Doubly Robust (AIPW) estimator with cross-fitting — no econml dependency."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold

from src.utils.config import RANDOM_SEED


class DoublyRobustEstimator:
    """Augmented Inverse Probability Weighting (AIPW) estimator.

    Doubly robust: consistent if *either* the outcome model or the
    propensity model is correctly specified.  Uses K-fold cross-fitting
    to avoid over-fitting bias.

    Parameters
    ----------
    n_folds : int
        Number of cross-fitting folds.
    """

    def __init__(
        self,
        n_folds: int = 5,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.n_folds = n_folds
        self.seed = seed
        self._outcome: np.ndarray | None = None
        self._treatment: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._mu0: np.ndarray | None = None  # E[Y|X, T=0]
        self._mu1: np.ndarray | None = None  # E[Y|X, T=1]
        self._ps: np.ndarray | None = None   # P(T=1|X)
        self._scores: np.ndarray | None = None  # individual AIPW scores

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        X: np.ndarray | pd.DataFrame,
        W: np.ndarray | pd.DataFrame | None = None,
    ) -> "DoublyRobustEstimator":
        """Fit nuisance models with K-fold cross-fitting.

        Parameters
        ----------
        outcome : array (Y)
        treatment : array (T, binary)
        X : array (covariates / effect modifiers)
        W : ignored (kept for API compatibility)
        """
        Y = np.asarray(outcome, dtype=float)
        T = np.asarray(treatment, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        n = len(Y)

        self._outcome = Y
        self._treatment = T
        self._X = X_arr

        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        ps_hat = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for train_idx, test_idx in kf.split(X_arr):
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            Y_tr, T_tr = Y[train_idx], T[train_idx]

            # Propensity model
            ps_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=self.seed
            )
            ps_model.fit(X_tr, T_tr)
            ps_hat[test_idx] = np.clip(ps_model.predict_proba(X_te)[:, 1], 0.01, 0.99)

            # Outcome model for controls: E[Y|X, T=0]
            ctrl_mask = T_tr == 0
            if ctrl_mask.sum() > 0:
                m0 = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=self.seed
                )
                m0.fit(X_tr[ctrl_mask], Y_tr[ctrl_mask])
                mu0_hat[test_idx] = m0.predict(X_te)

            # Outcome model for treated: E[Y|X, T=1]
            trt_mask = T_tr == 1
            if trt_mask.sum() > 0:
                m1 = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=self.seed
                )
                m1.fit(X_tr[trt_mask], Y_tr[trt_mask])
                mu1_hat[test_idx] = m1.predict(X_te)

        self._mu0 = mu0_hat
        self._mu1 = mu1_hat
        self._ps = ps_hat

        # AIPW individual scores
        score1 = mu1_hat + T * (Y - mu1_hat) / ps_hat
        score0 = mu0_hat + (1 - T) * (Y - mu0_hat) / (1 - ps_hat)
        self._scores = score1 - score0

        return self

    # ── Estimation ──────────────────────────────────────────────────────────
    def estimate_ate(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> dict:
        """Population ATE with 95% CI."""
        scores = self._scores
        ate = float(scores.mean())
        se = float(scores.std() / np.sqrt(len(scores)))
        return {
            "ate": ate,
            "ci_lower": ate - 1.96 * se,
            "ci_upper": ate + 1.96 * se,
            "se": se,
        }

    def estimate_att(
        self,
        X_treated: np.ndarray | pd.DataFrame | None = None,
    ) -> dict:
        """ATT approximated as mean AIPW score over treated units."""
        treated_mask = self._treatment == 1
        scores_t = self._scores[treated_mask]
        att = float(scores_t.mean())
        se = float(scores_t.std() / np.sqrt(len(scores_t)))
        return {
            "att": att,
            "ci_lower": att - 1.96 * se,
            "ci_upper": att + 1.96 * se,
        }

    def estimate_cate(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Individual-level CATE via mu1(X) - mu0(X) from the fitted models."""
        return self._mu1 - self._mu0
