# Competitive Performance Prediction System (HEMA)

## Overview

This project builds a **pre-match win probability model** for competitive HEMA (Historical European Martial Arts) tournaments using publicly available fighter ratings and match histories.

The goal is to estimate:

> **P(fighter wins | information available *before* the match)**

The project emphasizes **temporal correctness**, **leakage prevention**, and **interpretability**, mirroring real-world applied machine learning constraints rather than benchmark-style modeling.

---

## Problem Statement

Given two fighters scheduled to compete in a tournament bout, predict the probability that the focal fighter wins using only pre-match information.

Key challenges addressed:

* Ratings are updated monthly, not per match
* Fighters may have long periods of inactivity
* Many competitors appear with little or no prior history (cold start)

---

## Data Collection

Match and rating data were collected programmatically from publicly accessible sources.

To ensure reliable and respectful data acquisition, the scraping pipeline was designed with:

* Rate-limited HTTP requests
* Exponential backoff retry logic for transient failures
* Idempotent requests to allow safe restarts

This allowed the data pipeline to run unattended and consistently without overwhelming the source.

---

## Data Sources

* **Tournament match history**: individual bouts with fighter IDs, opponents, divisions, stages, and outcomes
* **Rating history**: monthly snapshots of fighter ratings and confidence values published by a third-party rating system

All rating joins are performed **backward in time** to ensure no future information is used.

---

## Dataset Construction

The dataset is built through a reproducible pipeline:

1. Load raw match and rating history data
2. Normalize and sort all data chronologically
3. Join ratings to matches using backward-in-time temporal joins
4. Track per-fighter state (experience and recency) in strict match order
5. Generate pre-match features
6. Drop matches without valid pre-match information
7. Freeze the dataset for modeling

Final dataset:

* **28,684 matches**
* Temporal train/test split based on match date (no random shuffling)

---

## Feature Engineering

### Design Principles

* Only information available before the match is used
* Temporal causality is strictly enforced
* Missing history is handled explicitly (never encoded as zero)

### Pre-Match Features

| Feature                           | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `ratings_diff`                    | Rating difference between fighter and opponent at match time |
| `experience_diff`                 | Difference in number of prior matches                        |
| `fighter_days_since_last_fought`  | Days since fighter’s previous match                          |
| `opponent_days_since_last_fought` | Days since opponent’s previous match                         |
| `days_since_last_fought_diff`     | Relative recency advantage                                   |
| `fighter_first_match`             | Fighter has no prior recorded matches                        |
| `opponent_first_match`            | Opponent has no prior recorded matches                       |

Cold-start cases are handled via explicit flags rather than misleading numeric encodings.

---

## Modeling

### Baseline Model: Logistic Regression

A logistic regression model was used as an interpretable baseline.

**Results (held-out future data):**

* ROC-AUC: ~0.79
* Accuracy: ~71%

Coefficient inspection shows:

* Rating difference is the dominant predictor
* Experience provides a modest secondary advantage
* Cold-start effects are asymmetric but meaningful
* Recency effects are present but smaller, consistent with rating decay already encoding inactivity

---

### Nonlinear Model: LightGBM

A tree-based LightGBM model was trained using the same features and temporal split.

**Outcome:**

* Performance (ROC-AUC and accuracy) closely matched logistic regression

**Interpretation:**
This indicates that the engineered features capture most of the predictive signal in a near-linear form. Additional nonlinear modeling does not materially improve performance, validating the feature design and confirming the suitability of a simple, interpretable model for this problem.

---

## Model Comparison Summary

* Logistic regression and LightGBM achieve comparable performance
* No evidence of strong nonlinear interactions beyond engineered features
* Feature engineering quality dominates model choice

This comparison was used as a validation step rather than a performance-chasing exercise.

---

## Project Structure

```
HEMARATINGSANALYSIS/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── fighter_state.py
│   ├── build_dataset.py
│   ├── scraper.py
│   └── models/
│       ├── logistic_regression.py
│       └── lightgbm.py
├── tests/
│   ├── build_dataset_test.py
│   └── scraper_test.py
|
├── notebooks/
│   └── feature_sanity_check.ipynb
└── README.md
```

---

## Why This Project Matters

This project demonstrates:

* Leakage-safe temporal feature engineering
* Cold-start handling in real competition data
* End-to-end ML pipeline construction
* Model comparison driven by insight, not metrics chasing
* Interpretable results aligned with domain expectations

It reflects production-style applied ML rather than benchmark optimization.

---

## Current Status

* Dataset construction complete (v1 frozen)
* Logistic regression baseline evaluated
* Nonlinear model comparison completed

---

## Next Steps

* Probability calibration analysis
* Feature ablation study
* Division- and stage-specific modeling
* Model monitoring across eras

---

## Notes

This project is intended as a demonstration of applied machine learning methodology and engineering judgment, not as a commercial betting or forecasting system.
