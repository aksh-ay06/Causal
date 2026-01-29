"""Propensity Score Matching estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression

from src.utils.config import RANDOM_SEED


class PropensityScoreMatching:
    """Estimate causal effects via nearest‑neighbour propensity score matching.

    Parameters
    ----------
    n_neighbors : int
        Number of control matches per treated unit.
    caliper : float or None
        Maximum allowed distance in propensity‑score space.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        caliper: float | None = 0.05,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.seed = seed
        self._ps_model: LogisticRegression | None = None
        self.propensity_scores_: np.ndarray | None = None
        self.matched_indices_: tuple[np.ndarray, np.ndarray] | None = None

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
    ) -> "PropensityScoreMatching":
        """Estimate propensity scores using logistic regression."""
        self._ps_model = LogisticRegression(
            max_iter=1000, random_state=self.seed, solver="lbfgs"
        )
        self._ps_model.fit(X, treatment)
        self.propensity_scores_ = self._ps_model.predict_proba(X)[:, 1]
        return self

    # ── Matching ────────────────────────────────────────────────────────────
    def match(
        self,
        treatment: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest‑neighbour matching on propensity scores.

        Returns indices of (treated, matched_control) pairs.
        """
        treatment = np.asarray(treatment)
        ps = self.propensity_scores_
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        tree = KDTree(ps[control_idx].reshape(-1, 1))
        distances, indices = tree.query(
            ps[treated_idx].reshape(-1, 1), k=self.n_neighbors
        )

        if self.n_neighbors == 1:
            distances = distances.ravel()
            indices = indices.ravel()

        # Apply caliper
        if self.caliper is not None:
            mask = distances <= self.caliper if self.n_neighbors == 1 else distances.max(axis=1) <= self.caliper
            treated_matched = treated_idx[mask]
            control_matched = control_idx[indices[mask].ravel()] if self.n_neighbors == 1 else control_idx[indices[mask].ravel()]
        else:
            treated_matched = np.repeat(treated_idx, self.n_neighbors) if self.n_neighbors > 1 else treated_idx
            control_matched = control_idx[indices.ravel()]

        self.matched_indices_ = (treated_matched, control_matched)
        return treated_matched, control_matched

    # ── Estimation ──────────────────────────────────────────────────────────
    def estimate_ate(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        n_bootstrap: int = 500,
    ) -> dict:
        """ATE via matched‑pair mean difference with bootstrap CI."""
        if self.matched_indices_ is None:
            self.match(treatment)

        outcome = np.asarray(outcome)
        t_idx, c_idx = self.matched_indices_
        diff = outcome[t_idx] - outcome[c_idx]
        ate = float(diff.mean())

        # Bootstrap CI
        rng = np.random.default_rng(self.seed)
        boot_ates = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(diff), len(diff))
            boot_ates.append(diff[idx].mean())
        ci_lower, ci_upper = float(np.percentile(boot_ates, 2.5)), float(np.percentile(boot_ates, 97.5))

        return {"ate": ate, "ci_lower": ci_lower, "ci_upper": ci_upper, "n_matched": len(t_idx)}

    def estimate_att(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        n_bootstrap: int = 500,
    ) -> dict:
        """ATT – for PSM with matching on treated, ATT ≈ ATE of matched sample."""
        return self.estimate_ate(outcome, treatment, n_bootstrap)

    # ── Convenience ─────────────────────────────────────────────────────────
    def get_matched_data(
        self,
        df: pd.DataFrame,
        treatment_col: str = "smoking",
    ) -> pd.DataFrame:
        """Return DataFrame restricted to the matched sample."""
        if self.matched_indices_ is None:
            self.match(df[treatment_col].values)
        t_idx, c_idx = self.matched_indices_
        all_idx = np.concatenate([t_idx, c_idx])
        return df.iloc[all_idx].copy().reset_index(drop=True)
