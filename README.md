# Competitive Performance Prediction System (HEMA)

## Overview

This project builds a **pre-match win probability model** for competitive HEMA (Historical European Martial Arts) tournaments, using publicly available fighter ratings and match histories.

The goal is to predict the probability that a fighter wins a match **using only information that would have been available before the match occurred**, with careful attention to temporal correctness and data leakage prevention.

---

## Problem Statement

Given two fighters scheduled to compete in a tournament bout, estimate:

> **P(fighter wins | pre-match information)**

This is a binary classification problem with real-world constraints:

* Ratings are updated monthly
* Fighters may have long periods of inactivity
* Many competitors have limited or no prior match history (cold start)

---

## Data Sources

* **Match history dataset**
  Individual tournament bouts with fighter IDs, opponents, divisions, stages, and outcomes.

* **Rating history dataset**
  Monthly snapshots of fighter ratings and confidence scores published by a third-party rating system.

All joins between matches and ratings are performed **backward in time** to ensure that no future information is used.

---

## Feature Engineering

### Core Principles

* **Strict temporal ordering** (no leakage)
* **Causal feature construction**
* **Explicit handling of missing / cold-start cases**

### Pre-Match Features

| Feature                           | Description                                                     |
| --------------------------------- | --------------------------------------------------------------- |
| `ratings_diff`                    | Difference between fighter and opponent rating at time of match |
| `experience_diff`                 | Difference in total prior matches                               |
| `fighter_days_since_last_fought`  | Days since fighter’s previous match                             |
| `opponent_days_since_last_fought` | Days since opponent’s previous match                            |
| `days_since_last_fought_diff`     | Relative recency advantage                                      |
| `fighter_first_match`             | Flag indicating fighter has no prior match history              |
| `opponent_first_match`            | Flag indicating opponent has no prior match history             |

Cold-start cases are handled explicitly via flags rather than misleading numeric encodings.

---

## Dataset Construction

The dataset is generated through a reproducible pipeline:

1. Load raw match and rating history data
2. Normalize and sort by date
3. Join ratings **backward in time** using temporal joins
4. Track per-fighter state (experience, recency) chronologically
5. Generate pre-match features
6. Freeze the dataset for modeling

Final dataset size:

* **28,684 matches**
* **Temporal train/test split** (no random shuffling)

---

## Baseline Model

### Model

* **Logistic Regression** (interpretable baseline)

### Features Used

* Ratings difference
* Experience difference
* Recency features
* First-match flags

### Evaluation Setup

* Train/test split based on match date
* Metrics reported on held-out future data

### Results

* **ROC-AUC:** 0.79
* **Accuracy:** ~71%

The learned coefficients align with domain expectations:

* Rating difference is the dominant predictor
* Experience provides a secondary advantage
* First-match fighters are slightly disadvantaged
* Recency effects are present but smaller, as expected given rating decay

---

## Why This Project Matters

This project demonstrates:

* Temporal data engineering
* Leakage-safe feature construction
* Cold-start handling
* Interpretable modeling
* End-to-end ML pipeline design

It mirrors real production ML constraints more closely than typical benchmark datasets.

---

## Project Structure

```
src/
├── data/
│   ├── raw/
│   └── processed/
├── state/
│   └── fighter_state.py
├── build_dataset.py
├── train_baseline.py
└── README.md
```

---

## Current Status

* Dataset construction complete (v1 frozen)
* Logistic regression baseline implemented and evaluated

---

## Next Steps

* Probability calibration analysis
* Feature ablation study
* Nonlinear models (tree-based)
* Division-specific modeling
* Comparison against published win probabilities

---

## Notes

This project is intended as a **demonstration of applied machine learning**, not as a commercial betting system.

---

If you want, next I can:

* Tighten this for a **portfolio site**
* Add a **“Results” section with plots**
* Rewrite it for **GitHub vs recruiter PDF**
* Help you turn this into **resume bullets**

Just tell me where you want to aim.
