# Day 15 — Classification Fundamentals & Confusion Matrix

## Objective

Understand the foundations of classification problems and learn
how to diagnose a classifier's performance using the confusion
matrix and the metrics derived from it.

## Topics Covered

- Binary, Multi-class, and Multi-label Classification
- Decision Thresholds
- Probability Calibration
- Confusion Matrix Structure
- True Positives, True Negatives
- False Positives, False Negatives
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Confusion Matrix Visualization

## Key Formulas

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

## Practical Work

Generated a confusion matrix using Scikit-learn and visualized it
using a Seaborn heatmap. Manually calculated Accuracy, Precision,
Recall, and F1-Score from raw TP/TN/FP/FN counts to verify
understanding before relying on built-in functions.

## Tools

- Python
- Scikit-learn
- Seaborn
- Matplotlib

## Key Learning

Accuracy alone can be misleading on imbalanced datasets, since a
model can score high accuracy while still failing to catch the
minority class. Precision and Recall expose the specific kinds of
mistakes a model makes, and the confusion matrix is the tool that
makes those mistakes visible rather than hidden behind a single
summary number.