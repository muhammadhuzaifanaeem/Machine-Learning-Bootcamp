# Day 10 — Gradient Descent for Linear Regression

## 🚀 30-Day Machine Learning Bootcamp

Today I moved from understanding Linear Regression mathematically to implementing the learning process using Gradient Descent from scratch.

The main goal was to understand how a model actually learns its parameters instead of treating machine learning algorithms as black boxes.

---

## 🎯 Learning Objectives

By the end of Day 10, I focused on:

- Understanding Gradient Descent
- Understanding the learning rate
- Understanding iterative parameter updates
- Understanding Batch, Mini-Batch, and Stochastic Gradient Descent
- Deriving gradients for Linear Regression
- Implementing Linear Regression from scratch
- Tracking the cost function during training
- Understanding convergence
- Applying the custom model to a real regression dataset
- Comparing my implementation with Scikit-learn

---

# 🧠 1. What is Gradient Descent?

Gradient Descent is an optimization algorithm used to find parameter values that minimize a cost function.

For Linear Regression, our model is:

ŷ = Xw + b

Where:

- X = input features
- w = model weights
- b = bias
- ŷ = predicted value

The goal is to find values of `w` and `b` that produce predictions as close as possible to the actual values.

---

# 📉 2. Cost Function

For Linear Regression, we used Mean Squared Error style cost:

J(w,b) = (1 / 2m) Σ(ŷ - y)²

Where:

- J = cost
- m = number of training examples
- ŷ = predicted value
- y = actual value

A lower cost means the predictions are generally closer to the actual values.

---

# 🔄 3. Gradient Descent Process

The basic learning loop is:

1. Initialize weights and bias
2. Make predictions
3. Calculate the error
4. Calculate gradients
5. Update weights and bias
6. Calculate the cost
7. Repeat

Conceptually:

X
↓
Prediction
↓
Error
↓
Gradient
↓
Parameter Update
↓
Lower Cost
↓
Repeat

---

# 📐 4. Gradient Equations

Weight gradient:

dw = (1/m) Xᵀ(ŷ - y)

Bias gradient:

db = (1/m) Σ(ŷ - y)

Parameter updates:

w = w - αdw

b = b - αdb

Where:

- α = learning rate
- dw = gradient of weights
- db = gradient of bias

---

# ⚙️ 5. Learning Rate

The learning rate controls how large each update step is.

A very small learning rate can make training slow.

A very large learning rate can cause unstable training or prevent convergence.

I experimented with different learning rates to observe their effect on the cost function and model performance.

---

# 🧪 6. Linear Regression From Scratch

I implemented my own:

`LinearRegressionGD`

The class contains:

- `__init__()`
- `fit()`
- `predict()`

The model performs:

- prediction
- error calculation
- gradient calculation
- parameter updates
- cost calculation
- cost history tracking

This implementation was built without using Scikit-learn's Linear Regression model for the actual learning process.

---

# 📊 7. Synthetic Dataset

I first tested the implementation using a simple dataset following approximately:

y = 2x + 1

This was useful because the expected relationship was already known.

The model was expected to learn approximately:

w ≈ 2

b ≈ 1

This allowed me to verify whether my Gradient Descent implementation was working correctly.

---

# 🌍 8. Real Dataset

After testing the algorithm on synthetic data, I applied the custom model to the Scikit-learn Diabetes regression dataset.

The dataset contains:

- 442 samples
- 10 numerical features
- 1 continuous target

The dataset was used as a benchmark for practicing regression rather than as a medical diagnostic system.

---

# 🔀 9. Train/Test Split

The dataset was divided into:

- Training data
- Testing data

The training set was used to learn the model parameters.

The test set was kept separate to evaluate how well the trained model performs on unseen data.

---

# 📏 10. Feature Scaling

Before Gradient Descent training, the features were standardized.

Standardization follows:

z = (x - μ) / σ

This puts features on comparable scales and can help Gradient Descent converge more effectively.

Importantly, the scaler was fitted using training data and then applied to the test data.

---

# 📈 11. Model Evaluation

I evaluated the regression model using:

### MAE
Mean Absolute Error

### MSE
Mean Squared Error

### RMSE
Root Mean Squared Error

### R²
Coefficient of Determination

These metrics provide different ways of understanding model prediction performance.

---

# 📉 12. Visualizations

I created visualizations including:

- Gradient Descent cost curve
- Regression line on synthetic data
- Actual vs predicted values
- Learning-rate experiments

The cost curve was particularly important because it allowed me to visually observe whether the optimization process was converging.

---

# 🧪 13. Experiments

I experimented with different learning rates, including:

- 0.001
- 0.01
- 0.1

I also experimented with different numbers of iterations.

The purpose was to understand:

- Slow convergence
- Faster convergence
- Instability
- The relationship between learning rate and optimization

---

# 🏆 14. Scikit-learn Comparison

After implementing Linear Regression from scratch, I compared my model with Scikit-learn's Linear Regression implementation.

The comparison included:

- Model predictions
- R²
- Regression coefficients
- Overall behavior

This helped verify that my implementation was producing sensible results.

---

# 💡 Key Lessons

Today's most important lesson was:

Machine Learning is not simply about calling:

`model.fit(X, y)`

Understanding what happens inside `fit()` is essential.

I learned that a Linear Regression model can learn by repeatedly:

Prediction → Error → Gradient → Update → Lower Cost → Repeat

---

# 🧠 Key Takeaways

- Gradient Descent is an optimization algorithm.
- The learning rate controls the size of parameter updates.
- Gradients tell us the direction in which the cost changes.
- Parameters are updated in the opposite direction of the gradient.
- Feature scaling can improve Gradient Descent behavior.
- The cost curve helps us understand convergence.
- Building algorithms from scratch makes ML concepts much easier to understand.
- Benchmarking against established implementations is useful for validating our implementation.

---

# 📂 Project Structure

Day_10/

├── README.md

├── notes.md

├── assignment.md

├── quiz.md

└── practical/

    ├── 01_linear_regression_gd_from_scratch.ipynb
    └── 02_diabetes_benchmark.ipynb

---

# 🚀 What's Next?

The next stage of the bootcamp will continue building Linear Regression knowledge and then move toward more advanced supervised learning algorithms.

The goal is not just to learn how to use ML libraries.

The goal is to understand:

**What the algorithm does → Why it works → How the mathematics works → How to implement it → How to evaluate it → When to use it in a real project.**

---

## 📅 Machine Learning Bootcamp

Day: 10 / 30

Focus: Gradient Descent for Linear Regression

Status: In Progress 🚀