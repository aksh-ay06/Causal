"""Propensity Score Matching estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        self._scaler: StandardScaler | None = None
        self._X: np.ndarray | None = None
        self.propensity_scores_: np.ndarray | None = None
        self.matched_indices_: tuple[np.ndarray, np.ndarray] | None = None

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray | pd.Series,
    ) -> "PropensityScoreMatching":
        """Estimate propensity scores using logistic regression."""
        self._X = np.asarray(X, dtype=float)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(self._X)
        self._ps_model = LogisticRegression(
            max_iter=1000, random_state=self.seed, solver="lbfgs"
        )
        self._ps_model.fit(X_scaled, treatment)
        self.propensity_scores_ = self._ps_model.predict_proba(X_scaled)[:, 1]
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
            control_matched = control_idx[indices[mask].ravel()]
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
        n_bootstrap: int = 200,
    ) -> dict:
        """ATE via matched‑pair mean difference with bootstrap CI."""
        if self.matched_indices_ is None:
            self.match(treatment)

        outcome = np.asarray(outcome)
        treatment = np.asarray(treatment)
        t_idx, c_idx = self.matched_indices_
        diff = outcome[t_idx] - outcome[c_idx]
        ate = float(diff.mean())

        if n_bootstrap == 0:
            return {"ate": ate, "ci_lower": np.nan, "ci_upper": np.nan, "n_matched": len(t_idx)}

        # Bootstrap CI – re-fit propensity model and re-match on each resample
        rng = np.random.default_rng(self.seed)
        n = len(outcome)
        X_arr = self._X
        boot_ates = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            X_b, T_b, Y_b = X_arr[idx], treatment[idx], outcome[idx]
            psm_b = PropensityScoreMatching(
                n_neighbors=self.n_neighbors,
                caliper=self.caliper,
                seed=self.seed,
            )
            psm_b.fit(X_b, T_b)
            psm_b.match(T_b)
            t_b, c_b = psm_b.matched_indices_
            if len(t_b) == 0:
                continue
            boot_ates.append(float((Y_b[t_b] - Y_b[c_b]).mean()))
        ci_lower, ci_upper = float(np.percentile(boot_ates, 2.5)), float(np.percentile(boot_ates, 97.5))

        return {"ate": ate, "ci_lower": ci_lower, "ci_upper": ci_upper, "n_matched": len(t_idx)}

    def estimate_att(
        self,
        outcome: np.ndarray | pd.Series,
        treatment: np.ndarray | pd.Series,
        n_bootstrap: int = 200,
    ) -> dict:
        """ATT – for PSM with matching on treated, ATT ≈ ATE of matched sample."""
        result = self.estimate_ate(outcome, treatment, n_bootstrap)
        result["att"] = result.pop("ate")
        return result

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
