"""Sensitivity analysis for unobserved confounding."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.utils.config import FIGURES_DIR


def rosenbaum_bounds(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_range: np.ndarray | None = None,
) -> list[dict]:
    """Wilcoxon signed‑rank Rosenbaum bounds for matched‑pair data.

    For each Γ (odds‑of‑treatment ratio), compute upper‑bound p‑value.
    If p < 0.05 even at high Γ, the result is insensitive to hidden bias.

    Parameters
    ----------
    treated_outcomes, control_outcomes : array
        Outcomes from matched treated and control units (same length).
    gamma_range : array or None
        Array of Γ values.  Default ``[1.0, 1.25, 1.5, ..., 3.0]``.

    Returns
    -------
    list[dict]
        Each dict: ``{"gamma": float, "upper_p": float}``.
    """
    if gamma_range is None:
        gamma_range = np.arange(1.0, 3.25, 0.25)

    diff = np.asarray(treated_outcomes, dtype=float) - np.asarray(control_outcomes, dtype=float)
    # Align sign so that we always test positive ranks (standard Rosenbaum formulation).
    # If the observed effect is negative, flip the sign of differences.
    if diff.mean() < 0:
        diff = -diff

    abs_diff = np.abs(diff)
    ranks = np.argsort(np.argsort(abs_diff)) + 1.0  # average‑ranks approximation

    # Observed test statistic: sum of ranks for positive differences
    T_obs = float(ranks[diff > 0].sum())

    results = []
    for gamma in gamma_range:
        # Worst‑case upper bound: probability that a pair is positive
        p_plus = gamma / (1 + gamma)
        # Expectation and variance of T under this worst case
        E_T = float(np.sum(ranks * p_plus))
        V_T = float(np.sum(ranks ** 2 * p_plus * (1 - p_plus)))
        if V_T == 0:
            results.append({"gamma": float(gamma), "upper_p": 1.0})
            continue
        z = (T_obs - E_T) / np.sqrt(V_T)
        # One‑sided upper p‑value (standard normal approximation)
        upper_p = float(norm.sf(z))
        results.append({"gamma": float(gamma), "upper_p": upper_p})

    return results


def compute_e_value(point_estimate: float, ci_bound: float) -> dict:
    """Compute the E‑value for a risk‑ratio‑scale point estimate.

    The E‑value is the minimum strength of association on the risk‑ratio
    scale that an unmeasured confounder would need to have with *both*
    treatment and outcome to explain away the observed effect.

    Parameters
    ----------
    point_estimate : float
        Observed risk ratio (or approximation thereof).
    ci_bound : float
        Lower (or upper, depending on direction) confidence limit.
    """
    def _e(rr: float) -> float:
        rr = max(rr, 1 / rr)  # always ≥ 1
        return float(rr + np.sqrt(rr * (rr - 1)))

    return {
        "e_value_point": _e(point_estimate),
        "e_value_ci": _e(ci_bound),
    }


def sensitivity_plot(
    bounds: list[dict],
    alpha: float = 0.05,
    save: bool = True,
    filename: str = "rosenbaum_bounds.png",
) -> None:
    """Plot Rosenbaum‑bounds p‑values as a function of Γ."""
    gammas = [b["gamma"] for b in bounds]
    pvals = [b["upper_p"] for b in bounds]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gammas, pvals, "o-", color="#4C72B0")
    ax.axhline(alpha, color="red", linestyle="--", label=f"α = {alpha}")
    ax.set_xlabel("Γ (sensitivity parameter)")
    ax.set_ylabel("Upper‑bound p‑value")
    ax.set_title("Rosenbaum Sensitivity Analysis")
    ax.legend()
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
