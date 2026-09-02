# 30-Day Machine Learning Engineering Bootcamp

A self-directed, 30-day intensive covering classical machine learning fundamentals
end to end — from regression basics through unsupervised learning, ensemble
boosting, hyperparameter tuning, and production-ready pipeline packaging. Every
day was completed with real, verified code and honestly reported results,
including the ones that weren't flattering.

This repository is the first phase of a longer 6-month journey toward a
hireable ML Engineer role. The second phase, covering deep learning, MLOps,
NLP, and LLMs, continues in a separate weekend curriculum repository.

## How This Bootcamp Was Structured

Each day followed a fixed five-section format:

- **Theory** — core concepts explained in plain language with real-world examples
- **Mathematics** — the exact formulas behind each concept, traced by hand
- **Coding** — working implementations, run and verified, not just described
- **Practical** — a hands-on task applying the day's concepts to real or realistic data
- **Revision** — active recall questions to confirm the concepts actually stuck

## Roadmap Overview

| Days | Focus |
|---|---|
| 1–13 | Python and math foundations, regression fundamentals |
| 14–21 | Regression pipelines, classification fundamentals, model evaluation |
| 22–25 | Unsupervised learning: K-Means, hierarchical clustering, DBSCAN, PCA |
| 26–27 | Ensemble boosting (AdaBoost, Gradient Boosting, XGBoost), cross-validation, hyperparameter tuning |
| 28–30 | Feature engineering pipelines, production packaging, portfolio and career prep |

## Day-by-Day Index

| Day | Topic |
|---|---|
| 22 | Unsupervised Learning Foundations |
| 23 | K-Means Clustering & Silhouette Analysis |
| 24 | Hierarchical Clustering & DBSCAN |
| 25 | Principal Component Analysis (PCA) |
| 26 | Ensemble Boosting: AdaBoost, Gradient Boosting & XGBoost |
| 27 | Cross Validation & Hyperparameter Tuning |
| 28 | Feature Engineering & Scikit-Learn Pipelines |
| 29 | End-to-End ML Pipeline & Project Documentation |
| 30 | Mock Interview, Portfolio Review & Career Roadmap |

Each day has its own folder containing a README, working code, and any
generated visualizations or datasets used for that day's practical work.

## Key Projects

- **Customer Segmentation (Day 23)** — K-Means clustering to sort customers into
  behavioral personas, with cluster count validated using both the Elbow Method
  and Silhouette Score.
- **Non-Spherical Clustering Comparison (Day 24)** — direct benchmark of K-Means
  versus DBSCAN on curved, moon-shaped data, demonstrating exactly where
  center-based clustering fails and density-based clustering succeeds.
- **Dimensionality Reduction on Diagnostic Data (Day 25)** — PCA applied to a
  30-feature breast cancer dataset, compressed to 2 components while preserving
  visible class separation.
- **Boosting Benchmark (Day 26)** — Random Forest vs XGBoost, 5-fold
  cross-validated, comparing both accuracy and training speed.
- **Hyperparameter Tuning Study (Day 27)** — GridSearchCV vs RandomizedSearchCV
  on XGBoost, including an honest result where tuning did not improve on
  Random Forest's sensible defaults.
- **End-to-End Customer Purchase Pipeline (Days 28–29)** — a leak-free,
  fully modular, documented, reproducible prediction pipeline, restructured
  from notebook code into a real `src/`-based project with measured inference
  latency and batch throughput.

## Tech Stack

Python, NumPy, Pandas, Scikit-Learn, XGBoost, SciPy, Matplotlib

## Principles Followed Throughout

- Every reported result came from code that was actually run — no fabricated
  or idealized outputs.
- Unflattering results were kept and taught from, not hidden. Day 27's tuned
  Random Forest scoring slightly below its own defaults is documented exactly
  as it happened.
- Every new tool or formula was explained with a real-world example before
  being used in code.
- Every pipeline was built to be leak-free: preprocessing statistics were
  calculated only from training data, never from data the model would later
  be evaluated on.

## What's Next

This repository closes the classical machine learning foundation. The next
phase covers PyTorch and deep learning, convolutional neural networks,
FastAPI and Docker for model serving, and hands-on work with large language
models and retrieval-augmented generation — the specific gap between a strong
classical ML practitioner and a genuinely hireable ML Engineer in the current
job market.

## Author

Muhammad Huzaifa Naeem — Artificial Intelligence student, Air University.
Documenting this journey publicly on LinkedIn throughout.
