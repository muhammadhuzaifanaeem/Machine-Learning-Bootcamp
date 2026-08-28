import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# We create 500 samples and 50 features.
#
# Only 10 features are actually informative.
# The remaining features are not informative.
#
# This gives us a good environment to understand
# Lasso feature selection.

X, y = make_regression(
    n_samples=500,
    n_features=50,
    n_informative=10,
    noise=10,
    random_state=42
)

print("Dataset created successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feature_names)

df["Target"] = y

print("\nFirst 5 rows:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum())


X = df.drop("Target", axis=1)

y = df["Target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Regularization is sensitive to feature scale.
#
# For example:
#
# Feature A values: 0 - 1
# Feature B values: 0 - 100000
#
# The large-scale feature could receive an unfair
# influence from the regularization penalty.
#
# Therefore, we standardize the features.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_pred = linear_model.predict(
    X_test_scaled
)


def evaluate_model(model_name, y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    return mae, mse, rmse, r2


linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

# Ridge uses L2 regularization.
#
# alpha controls regularization strength.
#
# Larger alpha = stronger regularization.

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train_scaled,
    y_train
)

ridge_pred = ridge_model.predict(
    X_test_scaled
)


ridge_results = evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_pred
)


# Lasso uses L1 regularization.
#
# One special property:
#
# Lasso can make some coefficients exactly zero.
#
# This can perform feature selection.

lasso_model = Lasso(
    alpha=0.1,
    max_iter=10000
)

lasso_model.fit(
    X_train_scaled,
    y_train
)

lasso_pred = lasso_model.predict(
    X_test_scaled
)


lasso_results = evaluate_model(
    "Lasso Regression",
    y_test,
    lasso_pred
)


# Elastic Net combines:
#
# L1 regularization
# +
# L2 regularization
#
# l1_ratio:
#
# 1.0 = mostly L1
# 0.0 = mostly L2
#
# 0.5 = equal balance

elastic_model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    max_iter=10000
)

elastic_model.fit(
    X_train_scaled,
    y_train
)

elastic_pred = elastic_model.predict(
    X_test_scaled
)


elastic_results = evaluate_model(
    "Elastic Net",
    y_test,
    elastic_pred
)


results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge",
        "Lasso",
        "Elastic Net"
    ],

    "MAE": [
        linear_results[0],
        ridge_results[0],
        lasso_results[0],
        elastic_results[0]
    ],

    "MSE": [
        linear_results[1],
        ridge_results[1],
        lasso_results[1],
        elastic_results[1]
    ],

    "RMSE": [
        linear_results[2],
        ridge_results[2],
        lasso_results[2],
        elastic_results[2]
    ],

    "R2": [
        linear_results[3],
        ridge_results[3],
        lasso_results[3],
        elastic_results[3]
    ]
})

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results.round(4))

coefficient_comparison = pd.DataFrame({

    "Feature": feature_names,

    "Linear_Regression": linear_model.coef_,

    "Ridge": ridge_model.coef_,

    "Lasso": lasso_model.coef_,

    "Elastic_Net": elastic_model.coef_

})

print("\nCoefficient Comparison:")
print(coefficient_comparison)


lasso_coefficients = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": lasso_model.coef_

})

lasso_coefficients["Absolute_Coefficient"] = (
    lasso_coefficients["Coefficient"].abs()
)

lasso_coefficients = lasso_coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\nLasso Coefficients:")
print(
    lasso_coefficients[
        ["Feature", "Coefficient"]
    ].to_string(index=False)
)

zero_coefficients = np.sum(
    lasso_model.coef_ == 0
)

nonzero_coefficients = np.sum(
    lasso_model.coef_ != 0
)

print("\nLasso Feature Selection")
print("-----------------------")

print(
    "Zero coefficients:",
    zero_coefficients
)

print(
    "Non-zero coefficients:",
    nonzero_coefficients
)


selected_features = [
    feature_names[i]
    for i, coefficient in enumerate(lasso_model.coef_)
    if coefficient != 0
]

print("\nSelected Features by Lasso:")

for feature in selected_features:
    print(feature)

plt.figure(figsize=(15, 7))

plt.plot(
    feature_names,
    linear_model.coef_,
    marker="o",
    label="Linear Regression"
)

plt.plot(
    feature_names,
    ridge_model.coef_,
    marker="o",
    label="Ridge"
)

plt.plot(
    feature_names,
    lasso_model.coef_,
    marker="o",
    label="Lasso"
)

plt.plot(
    feature_names,
    elastic_model.coef_,
    marker="o",
    label="Elastic Net"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xticks(
    rotation=90
)

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.title(
    "Comparison of Model Coefficients"
)

plt.legend()

plt.tight_layout()

plt.show()


alphas = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]

ridge_coefficients = []

for alpha in alphas:

    model = Ridge(
        alpha=alpha
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    ridge_coefficients.append(
        model.coef_
    )

ridge_coefficients = np.array(
    ridge_coefficients
)


plt.figure(figsize=(15, 8))

for feature_index in range(X.shape[1]):

    plt.plot(
        alphas,
        ridge_coefficients[:, feature_index],
        marker="o",
        label=feature_names[feature_index]
    )

plt.xscale("log")

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Regularization Strength (Alpha)"
)

plt.ylabel(
    "Coefficient Value"
)

plt.title(
    "Ridge Coefficient Shrinkage as Alpha Increases"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

plt.show()


lasso_coefficients_path = []

for alpha in alphas:

    model = Lasso(
        alpha=alpha,
        max_iter=10000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    lasso_coefficients_path.append(
        model.coef_
    )

lasso_coefficients_path = np.array(
    lasso_coefficients_path
)


plt.figure(figsize=(15, 8))

for feature_index in range(X.shape[1]):

    plt.plot(
        alphas,
        lasso_coefficients_path[:, feature_index],
        marker="o",
        label=feature_names[feature_index]
    )

plt.xscale("log")

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Regularization Strength (Alpha)"
)

plt.ylabel(
    "Coefficient Value"
)

plt.title(
    "Lasso Coefficient Shrinkage as Alpha Increases"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

plt.show()


print("\nLasso Feature Selection Across Alpha Values")
print("=" * 60)

for i, alpha in enumerate(alphas):

    coefficients = lasso_coefficients_path[i]

    zero_count = np.sum(
        coefficients == 0
    )

    nonzero_count = np.sum(
        coefficients != 0
    )

    print(
        f"Alpha: {alpha:<7} | "
        f"Zero coefficients: {zero_count:<3} | "
        f"Non-zero coefficients: {nonzero_count}"
    )


zero_counts = []

nonzero_counts = []

for coefficients in lasso_coefficients_path:

    zero_counts.append(
        np.sum(coefficients == 0)
    )

    nonzero_counts.append(
        np.sum(coefficients != 0)
    )


plt.figure(figsize=(10, 6))

plt.plot(
    alphas,
    zero_counts,
    marker="o",
    label="Zero Coefficients"
)

plt.plot(
    alphas,
    nonzero_counts,
    marker="o",
    label="Non-Zero Coefficients"
)

plt.xscale("log")

plt.xlabel(
    "Regularization Strength (Alpha)"
)

plt.ylabel(
    "Number of Features"
)

plt.title(
    "Lasso Feature Selection as Alpha Increases"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.show()


# Here we use Lasso predictions.

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    lasso_pred
)

# Ideal prediction line

minimum = min(
    y_test.min(),
    lasso_pred.min()
)

maximum = max(
    y_test.max(),
    lasso_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Values"
)

plt.ylabel(
    "Predicted Values"
)

plt.title(
    "Lasso: Actual vs Predicted"
)

plt.show()


print("\n")
print("=" * 70)
print("DAY 12 SUMMARY")
print("=" * 70)

print("""
Regularization helps control model complexity and reduce overfitting.

Ridge:
- Uses L2 regularization
- Shrinks coefficients
- Usually keeps coefficients non-zero

Lasso:
- Uses L1 regularization
- Shrinks coefficients
- Can make coefficients exactly zero
- Can perform feature selection

Elastic Net:
- Combines L1 and L2 regularization
- Can perform feature selection
- Can be useful with correlated features

Alpha:
- Controls regularization strength
- Larger alpha means stronger regularization
""")

print("Day 12 practical completed successfully!")
