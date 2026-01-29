"""Central configuration for the causal inference project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Variable names ─────────────────────────────────────────────────────────────
TREATMENT_COL = "smoking"
OUTCOME_HEALTH = "health_score"
OUTCOME_CANCER = "cancer"

COVARIATE_COLS = [
    "age",
    "female",
    "race_hispanic",
    "race_other",
    "race_white",
    "education",
    "income",
    "state_id",
    "urban",
]

# ── True causal effects (for synthetic data validation) ────────────────────────
TRUE_ATE_HEALTH = -5.0
TRUE_ATE_CANCER_LOGODDS = 0.80  # log‑odds scale in DGP

# ── Synthetic data defaults ────────────────────────────────────────────────────
DEFAULT_N_SAMPLES = 10_000
N_STATES = 20
