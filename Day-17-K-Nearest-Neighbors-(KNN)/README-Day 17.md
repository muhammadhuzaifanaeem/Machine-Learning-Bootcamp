# Day 17 — K-Nearest Neighbors (KNN)

## Objective

Master distance-based, non-parametric, lazy classification using
K-Nearest Neighbors, and understand how the choice of K affects the
trade-off between bias and variance.

## Topics Covered

- Instance-based (Lazy) vs Model-based (Eager) Learning Algorithms
- Distance Metric Choice and Its Impact on Classification
- Critical Feature Scaling Requirement for Distance-Based Models
- Hyperparameter K Selection
- Bias-Variance Trade-off in KNN
- Euclidean Distance
- Manhattan Distance
- Minkowski Distance
- Manual KNN Implementation
- Scikit-learn KNeighborsClassifier
- Decision Boundary Comparison Across K Values

## Key Formulas

Euclidean Distance: d(p,q) = sqrt(sum((p_i - q_i)^2))

Manhattan Distance: d(p,q) = sum(|p_i - q_i|)

Minkowski Distance: d(p,q) = (sum(|p_i - q_i|^m))^(1/m)

## Practical Work

Implemented a K-Nearest Neighbors prediction function manually in
NumPy and verified it produced identical results to Scikit-learn's
KNeighborsClassifier. Trained models across K = 1, 5, 15, and 50, and
plotted their decision boundaries to visually compare overfitting at
low K against underfitting at high K, alongside train and test
accuracy scores confirming the same pattern numerically.

## Tools

- Python
- NumPy
- Scikit-learn
- Matplotlib

## Key Learning

KNN is a lazy learner that stores the entire training dataset and
performs all its computation at prediction time by finding the K
closest points and taking a majority vote, in contrast to eager
learners like Logistic Regression that learn a fixed set of
parameters during training. Feature scaling is non-negotiable for
KNN, since its predictions depend entirely on distance calculations,
and unscaled features would let larger-magnitude columns dominate
the result. Small values of K lead to high variance and overfitting,
while large values of K lead to high bias and oversmoothed
predictions, making K a critical hyperparameter to tune carefully
rather than choose arbitrarily.