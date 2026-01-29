"""Covariate balance diagnostics for causal inference."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import FIGURES_DIR


def compute_smd(
    treated: np.ndarray,
    control: np.ndarray,
) -> float:
    """Standardised Mean Difference between two groups."""
    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


def balance_table(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "smoking",
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute SMD for each covariate, optionally weighted.

    Returns a DataFrame with columns: covariate, smd_unadjusted, (smd_adjusted).
    """
    treated_mask = (df[treatment_col] == 1).values
    rows = []
    for cov in covariates:
        t_vals = df.loc[treated_mask, cov].values.astype(float)
        c_vals = df.loc[~treated_mask, cov].values.astype(float)
        smd_raw = compute_smd(t_vals, c_vals)
        row = {"covariate": cov, "smd_unadjusted": round(smd_raw, 4)}

        if weights is not None:
            w_t = weights[treated_mask]
            w_c = weights[~treated_mask]
            wm_t = np.average(t_vals, weights=w_t)
            wm_c = np.average(c_vals, weights=w_c)
            wv_t = np.average((t_vals - wm_t) ** 2, weights=w_t)
            wv_c = np.average((c_vals - wm_c) ** 2, weights=w_c)
            pooled = np.sqrt((wv_t + wv_c) / 2)
            row["smd_adjusted"] = round((wm_t - wm_c) / pooled if pooled > 0 else 0.0, 4)

        rows.append(row)
    return pd.DataFrame(rows)


def love_plot(
    bal: pd.DataFrame,
    threshold: float = 0.1,
    save: bool = True,
    filename: str = "love_plot.png",
) -> None:
    """Love plot showing SMD before and (optionally) after adjustment."""
    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * len(bal))))
    y = range(len(bal))
    ax.scatter(bal["smd_unadjusted"].abs(), y, marker="x", color="red", s=80, label="Unadjusted")
    if "smd_adjusted" in bal.columns:
        ax.scatter(bal["smd_adjusted"].abs(), y, marker="o", color="blue", s=80, label="Adjusted")
    ax.axvline(threshold, color="grey", linestyle="--", label=f"Threshold = {threshold}")
    ax.set_yticks(list(y))
    ax.set_yticklabels(bal["covariate"])
    ax.set_xlabel("|Standardised Mean Difference|")
    ax.set_title("Love Plot — Covariate Balance")
    ax.legend()
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def assess_overlap(
    ps: np.ndarray,
    treatment: np.ndarray,
    min_ps: float = 0.05,
    max_ps: float = 0.95,
) -> dict:
    """Check propensity score overlap / positivity."""
    treated_ps = ps[treatment == 1]
    control_ps = ps[treatment == 0]
    violation_rate = float(np.mean((ps < min_ps) | (ps > max_ps)))
    return {
        "treated_ps_mean": float(treated_ps.mean()),
        "control_ps_mean": float(control_ps.mean()),
        "treated_ps_range": (float(treated_ps.min()), float(treated_ps.max())),
        "control_ps_range": (float(control_ps.min()), float(control_ps.max())),
        "positivity_violation_rate": violation_rate,
    }
