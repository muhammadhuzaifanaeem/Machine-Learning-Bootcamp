# Day-14 Real Estate Price Prediction

An end-to-end regression pipeline that predicts median house values from demographic and geographic features, built and evaluated across three linear model variants (Linear, Ridge, Lasso Regression).

## Problem Statement

Real estate pricing is a continuous prediction problem with direct business value: buyers, sellers, and lenders all need reliable price estimates. This project builds a reproducible pipeline that takes raw housing data and outputs a trained, saved model capable of predicting house values on new, unseen data.

## Dataset

California Housing Dataset (via `sklearn.datasets.fetch_california_housing`)

- **Samples:** ~20,640 housing districts
- **Features:** Median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, longitude
- **Target:** Median house value (in $100,000s)

## Approach

1. **Exploratory Data Analysis** — examined target distribution, checked for outliers, and used a correlation heatmap to identify median income as the strongest predictor of house value before any modeling began.
2. **Train/Test Split** — data was split (80/20) *before* any preprocessing to prevent data leakage.
3. **Preprocessing Pipeline** — numeric features were standardized using `StandardScaler`, wrapped inside a single `sklearn.pipeline.Pipeline` so preprocessing and modeling always run together, in the same order, on any new data.
4. **Model Comparison** — trained and benchmarked three regression approaches: Linear Regression, Ridge Regression, and Lasso Regression, to evaluate whether regularization improves generalization on this dataset.
5. **Evaluation** — assessed all models using MAE, RMSE, and R², rather than a single metric, since each captures a different failure mode (typical error size, sensitivity to large misses, and overall explained variance, respectively).
6. **Model Persistence** — the best-performing pipeline (preprocessing + model together) was serialized with `joblib`, and reloaded to confirm it produces correct predictions independent of the original training session.



## Project Structure

```
real-estate-price-prediction/
├── main.py                    # Full pipeline: load → EDA → train → evaluate → save
├── real_estate_model.joblib   # Saved, trained model pipeline
├── requirements.txt
└── README.md


This will run the complete pipeline end-to-end: load data, generate EDA plots, train all three models, print the evaluation comparison table, and save the best model to `real_estate_model.joblib`.

The saved pipeline includes preprocessing, so raw feature data can be passed directly without manual scaling.

## Key Learnings

- Data leakage prevention: fitting scalers only on training data, never on test data
- Why bundling preprocessing and modeling into a single `Pipeline` object matters for reproducibility
- Why a single metric (e.g. accuracy or R² alone) is insufficient to judge a regression model's real-world reliability

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib