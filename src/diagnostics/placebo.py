"""Placebo / falsification tests for causal estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.config import RANDOM_SEED


def placebo_treatment_test(
    X: np.ndarray | pd.DataFrame,
    outcome: np.ndarray | pd.Series,
    real_treatment: np.ndarray | pd.Series,
    estimator_cls,
    n_permutations: int = 200,
    seed: int = RANDOM_SEED,
) -> dict:
    """Permutation‑based placebo test.

    Randomly reassign treatment labels and re‑estimate the ATE.
    If the real ATE is extreme relative to the placebo distribution,
    the finding is unlikely due to chance.

    Parameters
    ----------
    estimator_cls
        A class with ``.fit(X, treatment)`` and
        ``.estimate_ate(outcome, treatment)`` methods.
    """
    X_arr = np.asarray(X)
    outcome_arr = np.asarray(outcome)
    treatment_arr = np.asarray(real_treatment)

    # Real ATE (point estimate only — no bootstrap CI needed)
    est = estimator_cls(seed=seed)
    est.fit(X_arr, treatment_arr)
    if hasattr(est, "match"):
        est.match(treatment_arr)
    real_result = est.estimate_ate(outcome_arr, treatment_arr, n_bootstrap=0)
    real_ate = real_result["ate"]

    # Placebo distribution
    rng = np.random.default_rng(seed)
    placebo_ates: list[float] = []
    for _ in range(n_permutations):
        fake_treatment = rng.permutation(treatment_arr)
        est_p = estimator_cls(seed=seed)
        est_p.fit(X_arr, fake_treatment)
        if hasattr(est_p, "match"):
            est_p.match(fake_treatment)
        try:
            res = est_p.estimate_ate(outcome_arr, fake_treatment, n_bootstrap=0)
            placebo_ates.append(res["ate"])
        except Exception:
            continue

    placebo_ates_arr = np.array(placebo_ates)
    p_value = float(np.mean(np.abs(placebo_ates_arr) >= np.abs(real_ate)))

    return {
        "real_ate": real_ate,
        "placebo_mean": float(placebo_ates_arr.mean()),
        "placebo_std": float(placebo_ates_arr.std()),
        "p_value": p_value,
        "n_permutations": len(placebo_ates),
    }


def negative_control_test(
    X: np.ndarray | pd.DataFrame,
    treatment: np.ndarray | pd.Series,
    negative_outcome: np.ndarray | pd.Series,
    estimator_cls,
    seed: int = RANDOM_SEED,
) -> dict:
    """Estimate the effect of treatment on a negative‑control outcome.

    A negative‑control outcome should **not** be causally affected by
    treatment.  A large estimated effect signals residual confounding.
    """
    X_arr = np.asarray(X)
    treatment_arr = np.asarray(treatment)
    negative_arr = np.asarray(negative_outcome)

    est = estimator_cls(seed=seed)
    est.fit(X_arr, treatment_arr)
    if hasattr(est, "match"):
        est.match(treatment_arr)
    result = est.estimate_ate(negative_arr, treatment_arr)

    return {
        "negative_control_ate": result["ate"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "covers_zero": result["ci_lower"] <= 0 <= result["ci_upper"],
    }


def run_falsification_suite(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    outcome_col: str,
    estimator_cls,
    n_permutations: int = 200,
    seed: int = RANDOM_SEED,
) -> dict:
    """Run the full falsification battery: placebo + negative control.

    Uses ``age`` as a negative‑control outcome (treatment should not
    cause age in cross‑sectional data).
    """
    X = df[covariates].values
    treatment = df[treatment_col].values
    outcome = df[outcome_col].values

    placebo = placebo_treatment_test(
        X, outcome, treatment, estimator_cls, n_permutations, seed
    )

    negative_control = None
    if "age" in df.columns and "age" not in [treatment_col, outcome_col]:
        negative_control = negative_control_test(
            X, treatment, df["age"].values, estimator_cls, seed
        )

    return {
        "placebo_test": placebo,
        "negative_control_test": negative_control,
    }
