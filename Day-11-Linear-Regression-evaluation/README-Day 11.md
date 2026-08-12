# Day 11 — Scikit-Learn Linear Regression & Model Evaluation

## 🚀 30-Day Machine Learning Bootcamp

Today I moved from implementing Linear Regression and Gradient Descent from scratch to using **Scikit-Learn's professional implementation** and evaluating a model properly on unseen data.

The goal was not just to train a model, but to understand the complete workflow:

**Dataset → Train/Test Split → Model → Training → Prediction → Evaluation → Interpretation**

---

## 🎯 Learning Objectives

By the end of Day 11, I learned:

* How Scikit-Learn implements Linear Regression
* The purpose of `fit()` and `predict()`
* Training vs testing data
* Model coefficients and intercept
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score
* Actual vs predicted visualization
* How to evaluate regression models
* How to compare a custom implementation with Scikit-Learn

---

# 🧠 1. Linear Regression in Scikit-Learn

Instead of manually implementing the optimization process, Scikit-Learn provides an optimized implementation:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### `fit()`

`fit()` trains the model using the training data.

The model learns the parameters:

* Weights / coefficients
* Intercept / bias

### `predict()`

`predict()` uses the learned parameters to generate predictions for new observations.

---

# 📐 2. Linear Regression Equation

For a single feature:

[
\hat{y}=wx+b
]

For multiple features:

[
\hat{y}=w_1x_1+w_2x_2+\dots+w_nx_n+b
]

Where:

* `x` = input features
* `w` = learned coefficients
* `b` = intercept
* `ŷ` = predicted target

---

# 🔀 3. Train/Test Split

The dataset was divided into training and testing sets.

The training data is used to learn the model.

The testing data is kept separate so that we can evaluate performance on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The model was trained only on:

```text
X_train, y_train
```

and evaluated using:

```text
X_test, y_test
```

---

# 📊 4. Regression Evaluation Metrics

A machine learning model should not be judged only by looking at predictions.

We need quantitative evaluation metrics.

## MAE — Mean Absolute Error

[
MAE=
\frac{1}{n}
\sum |y_i-\hat{y_i}|
]

MAE measures the average absolute prediction error.

Lower MAE generally indicates better performance.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

---

## MSE — Mean Squared Error

[
MSE=
\frac{1}{n}
\sum(y_i-\hat{y_i})^2
]

MSE squares the errors, which means larger errors receive greater penalty.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

---

## RMSE — Root Mean Squared Error

[
RMSE=\sqrt{MSE}
]

RMSE brings the error back to approximately the same unit as the target.

```python
import numpy as np

rmse = np.sqrt(mse)
```

Lower RMSE generally indicates better predictive performance.

---

## R² — Coefficient of Determination

R² measures how well the model explains variation in the target relative to a simple mean-based baseline.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

An R² closer to 1 generally indicates stronger explanatory performance.

R² can also be negative on unseen data when the model performs worse than the mean baseline.

---

# 🌍 5. Dataset

For today's practical work, I used the **Diabetes regression dataset available through Scikit-Learn**.

The dataset contains:

* 442 observations
* 10 numerical features
* 1 continuous target

```python
from sklearn.datasets import load_diabetes

data = load_diabetes()

X = data.data
y = data.target
```

This dataset was used as a regression benchmark for learning model training and evaluation.

---

# 🧪 6. Complete Workflow

The practical workflow was:

```text
Load Dataset
     ↓
Inspect Dataset
     ↓
Separate X and y
     ↓
Train/Test Split
     ↓
Create Linear Regression Model
     ↓
Train with fit()
     ↓
Generate Predictions
     ↓
Calculate MAE
     ↓
Calculate MSE
     ↓
Calculate RMSE
     ↓
Calculate R²
     ↓
Visualize Actual vs Predicted
     ↓
Interpret Results
```

---

# 📈 7. Actual vs Predicted Visualization

An actual-vs-predicted plot was created to visually inspect model performance.

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")

plt.show()
```

Predictions closer to the ideal relationship between actual and predicted values indicate better model behavior.

---

# 🔎 8. Model Coefficients

After training, the learned coefficients can be inspected using:

```python
print(model.coef_)
```

The intercept can be accessed using:

```python
print(model.intercept_)
```

These parameters describe the learned Linear Regression relationship.

---

# 🆚 9. Comparison With Day 10

An important part of today's work was connecting today's knowledge with Day 10.

### Day 10

I implemented Linear Regression using Gradient Descent from scratch.

```text
Prediction
↓
Calculate Error
↓
Calculate Gradient
↓
Update Parameters
↓
Repeat
```

### Day 11

I used Scikit-Learn:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The important lesson is that using a library does not mean we should treat the algorithm as a black box.

I first learned the underlying mathematics and implementation and then moved to the professional library implementation.

---

# 🧠 Key Takeaways

* `fit()` learns model parameters.
* `predict()` generates predictions using learned parameters.
* Training and testing data must be separated.
* MAE measures average absolute error.
* MSE penalizes large errors more strongly.
* RMSE provides error in approximately the target's scale.
* R² measures performance relative to a mean-based baseline.
* Coefficients describe the learned relationship between features and target.
* Visualization helps us understand model behavior.
* A model should be evaluated on unseen data.
* Understanding an algorithm from scratch makes library implementations easier to understand.

---

# 🧪 Practical Work Completed

* [x] Loaded a regression dataset
* [x] Inspected the dataset
* [x] Created train/test split
* [x] Trained Linear Regression using Scikit-Learn
* [x] Generated predictions
* [x] Calculated MAE
* [x] Calculated MSE
* [x] Calculated RMSE
* [x] Calculated R²
* [x] Inspected coefficients
* [x] Inspected intercept
* [x] Created actual-vs-predicted visualization
* [x] Connected Scikit-Learn implementation with the Day 10 from-scratch implementation

---

# 📁 Project Structure

```text
Day_11/
│
├── README.md
├── notes.md
├── assignment.md
├── quiz.md
│
└── practical/
    └── linear_regression_sklearn.ipynb
```

---

# 🚀 Bootcamp Progress

**Day:** 11 / 30

**Topic:** Scikit-Learn Linear Regression & Model Evaluation

**Status:** Completed ✅

### Learning Philosophy

> Understand the theory → Understand the mathematics → Implement from scratch → Use professional libraries → Evaluate → Interpret → Apply to real-world problems.

---

## 🔜 Next

The next stage is **Regularization**, where I will learn how Ridge, Lasso, and Elastic Net can control model complexity and reduce overfitting.
