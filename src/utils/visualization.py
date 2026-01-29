"""Shared plotting helpers for the causal inference pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import FIGURES_DIR


def plot_propensity_distribution(
    ps: np.ndarray,
    treatment: np.ndarray,
    save: bool = True,
    filename: str = "propensity_overlap.png",
) -> None:
    """Overlapping histograms of propensity scores by treatment group."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ps[treatment == 1], bins=50, alpha=0.5, label="Treated", density=True)
    ax.hist(ps[treatment == 0], bins=50, alpha=0.5, label="Control", density=True)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distribution by Treatment Group")
    ax.legend()
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_treatment_effects(
    results: dict[str, dict],
    true_effect: float | None = None,
    save: bool = True,
    filename: str = "treatment_effects.png",
) -> None:
    """Bar chart comparing ATE point estimates across methods.

    Parameters
    ----------
    results : dict
        ``{method_name: {"ate": float, "ci_lower": float, "ci_upper": float}}``
    true_effect : float or None
        If provided, draw a horizontal reference line.
    """
    methods = list(results.keys())
    ates = [results[m]["ate"] for m in methods]
    ci_lo = [results[m]["ci_lower"] for m in methods]
    ci_hi = [results[m]["ci_upper"] for m in methods]
    errors = [[a - lo for a, lo in zip(ates, ci_lo)],
              [hi - a for a, hi in zip(ates, ci_hi)]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(methods, ates, xerr=errors, capsize=4, color="#4C72B0", alpha=0.8)
    if true_effect is not None:
        ax.axvline(true_effect, color="red", linestyle="--", label=f"True ATE = {true_effect}")
        ax.legend()
    ax.set_xlabel("Estimated ATE")
    ax.set_title("Comparison of Causal Estimators")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forest(
    results: dict[str, dict],
    true_effect: float | None = None,
    save: bool = True,
    filename: str = "forest_plot.png",
) -> None:
    """Forest plot of treatment effect estimates with CIs."""
    methods = list(results.keys())
    ates = [results[m]["ate"] for m in methods]
    ci_lo = [results[m]["ci_lower"] for m in methods]
    ci_hi = [results[m]["ci_upper"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(methods))
    ax.errorbar(ates, y, xerr=[[a - lo for a, lo in zip(ates, ci_lo)],
                                [hi - a for a, hi in zip(ates, ci_hi)]],
                fmt="o", color="#4C72B0", capsize=5, markersize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(methods)
    if true_effect is not None:
        ax.axvline(true_effect, color="red", linestyle="--", label=f"True ATE = {true_effect}")
        ax.legend()
    ax.set_xlabel("ATE Estimate")
    ax.set_title("Forest Plot — Treatment Effect Estimates")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_covariate_distributions(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "smoking",
    save: bool = True,
    filename: str = "covariate_distributions.png",
) -> None:
    """KDE plots of covariates split by treatment group."""
    n = len(covariates)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, cov in enumerate(covariates):
        sns.kdeplot(data=df, x=cov, hue=treatment_col, ax=axes[i], fill=True, common_norm=False)
        axes[i].set_title(cov)

    for j in range(len(covariates), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Covariate Distributions by Treatment Group", fontsize=14)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
