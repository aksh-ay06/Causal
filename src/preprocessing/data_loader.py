"""Data loading utilities – synthetic BRFSS generator & CSV loader."""

import numpy as np
import pandas as pd

from src.utils.config import (
    DEFAULT_N_SAMPLES,
    N_STATES,
    RANDOM_SEED,
    TRUE_ATE_CANCER_LOGODDS,
    TRUE_ATE_HEALTH,
)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_brfss(
    n: int = DEFAULT_N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic BRFSS‑like data with *known* causal effects.

    The data‑generating process deliberately introduces confounding so that
    a naïve comparison of smokers vs. non‑smokers will be biased.  The true
    treatment effects are defined in ``config.py`` and can be used to
    validate causal estimators.

    Returns
    -------
    pd.DataFrame
        Columns: age, female, race, education, income, state_id, urban,
        smoking (treatment), health_score (continuous outcome),
        cancer (binary outcome).
    """
    rng = np.random.default_rng(seed)

    # ── Confounders ────────────────────────────────────────────────────────
    age = rng.uniform(18, 85, n)
    female = rng.binomial(1, 0.52, n)
    race = rng.choice(
        ["white", "black", "hispanic", "other"],
        size=n,
        p=[0.60, 0.20, 0.13, 0.07],
    )
    education = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.05, 0.10, 0.25, 0.30, 0.20, 0.10])
    income = np.clip(
        30_000 + 5_000 * education + 200 * age + rng.normal(0, 15_000, n),
        10_000,
        200_000,
    )
    state_id = rng.integers(0, N_STATES, n)
    urban = rng.binomial(1, 0.75, n)

    # ── Treatment assignment (confounded) ──────────────────────────────────
    race_num = np.where(
        race == "white", 0.0,
        np.where(race == "black", 0.3,
                 np.where(race == "hispanic", 0.1, 0.2)),
    )
    logit_t = (
        -1.5
        + 0.015 * age
        - 0.4 * female
        - 0.25 * education
        + 0.000005 * income
        + race_num
    )
    prob_smoking = _expit(logit_t)
    smoking = rng.binomial(1, prob_smoking)

    # ── Outcome: health_score (continuous) ─────────────────────────────────
    health_score = (
        80
        - 0.3 * age
        + 3.0 * female
        + 2.0 * education
        + 0.00005 * income
        + TRUE_ATE_HEALTH * smoking
        + rng.normal(0, 5, n)
    )

    # ── Outcome: cancer (binary) ───────────────────────────────────────────
    cancer_logit = (
        -4.0
        + 0.03 * age
        - 0.1 * female
        - 0.05 * education
        + TRUE_ATE_CANCER_LOGODDS * smoking
    )
    cancer = rng.binomial(1, _expit(cancer_logit))

    df = pd.DataFrame(
        {
            "age": age,
            "female": female,
            "race": race,
            "education": education,
            "income": income,
            "state_id": state_id,
            "urban": urban,
            "smoking": smoking,
            "health_score": health_score,
            "cancer": cancer,
        }
    )
    return df


def load_dataset(path: str | None = None, **kwargs) -> pd.DataFrame:
    """Load a CSV dataset or fall back to synthetic data.

    Parameters
    ----------
    path : str or None
        Path to a CSV file.  If *None*, synthetic data is generated.
    **kwargs
        Forwarded to ``generate_synthetic_brfss`` when *path* is None.
    """
    if path is not None:
        return pd.read_csv(path)
    return generate_synthetic_brfss(**kwargs)
