# Causal Impact Analysis of Public Policy Interventions

Estimates the **causal effect of smoking on health outcomes** using observational data and rigorous causal inference methods.

## Causal Question

> What is the causal impact of smoking on health scores and cancer prevalence?

This project uses synthetic BRFSS-like data with **known treatment effects**, allowing validation of causal estimators against ground truth.

## Methods Implemented

| Method | Module | Estimands |
|--------|--------|-----------|
| Propensity Score Matching | `src/causal_models/propensity_score.py` | ATE, ATT |
| Inverse Probability Weighting (IPTW) | `src/causal_models/ipw.py` | ATE, ATT |
| Doubly Robust (AIPW via EconML) | `src/causal_models/doubly_robust.py` | ATE, ATT, CATE |

## Diagnostics

- Covariate balance (Standardised Mean Differences, Love plots)
- Propensity score overlap / positivity checks
- Rosenbaum bounds sensitivity analysis
- E-values for unmeasured confounding
- Placebo treatment permutation tests
- Negative control outcomes (falsification)

## Quickstart

```bash
# Setup
conda env create -f environment.yml
conda activate causal-inference

# Run notebooks
cd notebooks
jupyter notebook

# Run tests
pytest tests/ -v
```

## Project Structure

```
data/               Raw and processed datasets
notebooks/          4-notebook analysis pipeline
  01_data_exploration_and_dag.ipynb
  02_causal_estimation.ipynb
  03_diagnostics_and_sensitivity.ipynb
  04_results_summary.ipynb
src/
  preprocessing/    Data loading, cleaning, feature engineering
  causal_models/    PSM, IPW, Doubly Robust estimators
  diagnostics/      Balance, sensitivity, falsification tests
  utils/            DAG, config, visualization helpers
tests/              Unit tests for all modules
figures/            Generated plots
reports/            Analysis reports
```

## Causal Framework

- **Framework**: Rubin Potential Outcomes Model
- **Identification**: Backdoor adjustment via observed confounders
- **Estimands**: ATE (Average Treatment Effect), ATT (Average Treatment on Treated), CATE (Conditional ATE)
- **DAG**: Defined in `src/utils/dag.py`, visualised in Notebook 01

## Key Assumptions

1. **Conditional exchangeability** — no unmeasured confounding given covariates
2. **Positivity** — all covariate strata have both treated and control units
3. **SUTVA** — no interference between units
4. **Correct model specification** — validated via doubly-robust estimation
