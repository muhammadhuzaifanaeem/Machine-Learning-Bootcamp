# Day 19 — Random Forest & Ensemble Bagging

## Objective

Master Bootstrap Aggregation (Bagging) and Random Forest ensembles,
and understand how combining many Decision Trees reduces the
overfitting seen in a single tree.

## Topics Covered

- Ensemble Theory: Reducing Variance via Multi-Model Aggregation
- Bootstrap Sampling (Bagging)
- Out-Of-Bag (OOB) Error Estimation
- Random Subspace Method: Random Feature Subsets at Split Nodes
- Ensemble Variance Reduction
- Scikit-learn RandomForestClassifier
- Gini-based Feature Importances
- Model Stability Comparison

## Key Formulas

Ensemble Variance Reduction: Var(X_bar) = rho * sigma^2 + ((1-rho)/B) * sigma^2

Default Classification Feature Subspace Size: m = sqrt(p)

## Practical Work

Trained a single Decision Tree and a Random Forest Classifier on the
same dataset and compared their train-test accuracy gaps, confirming
the Random Forest showed noticeably less overfitting. Extracted and
visualized Gini-based feature importances to identify which features
contributed most across all trees in the forest. Computed the
Out-Of-Bag score as a built-in validation estimate, and tested model
stability by running both models across multiple random train-test
splits, confirming the Random Forest produced more consistent
accuracy scores than the single tree.

## Tools

- Python
- Scikit-learn
- Matplotlib

## Key Learning

A Random Forest reduces the high variance of a single Decision Tree
by training many trees on different bootstrap samples of the data and
restricting each split to a random subset of features, which
decorrelates the trees from one another. Averaging predictions across
many decorrelated trees cancels out individual overfitting, producing
a model that is both more accurate and more stable than any single
tree, while Out-Of-Bag scoring provides an honest performance estimate
without requiring a separate held-out test set.