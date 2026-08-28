# Day 16 — Logistic Regression & Binary Cross-Entropy

## Objective

Master classification using Sigmoid mapping and understand how
Logistic Regression converts a linear equation into a probability,
optimized through the Log Loss cost function.

## Topics Covered

- Linear Decision Boundaries in Classification Space
- Odds and Log-Odds (Logit Function)
- Sigmoid Activation Function
- Binary Cross-Entropy (Log Loss) Derivation
- Parameter Gradient Update for Logistic Regression
- Manual Sigmoid and Binary Cross-Entropy Implementation
- Scikit-learn LogisticRegression
- Decision Boundary Visualization

## Key Formulas

Sigmoid Function: sigma(z) = 1 / (1 + e^(-z))

Binary Cross-Entropy Loss: L = -[y * log(y_hat) + (1-y) * log(1-y_hat)]

Odds = P / (1 - P)

Log-Odds (Logit) = log(P / (1-P))

## Practical Work

Implemented the Sigmoid function and Binary Cross-Entropy loss
manually in NumPy to understand the underlying math before using
Scikit-learn's LogisticRegression. Trained a Logistic Regression
model on 2D data and plotted the resulting decision boundary to
visualize how the model separates the two classes using a straight
line.

## Tools

- Python
- NumPy
- Scikit-learn
- Matplotlib

## Key Learning

Logistic Regression works by passing a linear equation through the
Sigmoid function to produce a probability between 0 and 1, then
applying a decision threshold to reach a final class prediction.
It uses Binary Cross-Entropy instead of Mean Squared Error as its
cost function because Cross-Entropy keeps the optimization surface
smooth and convex even after the Sigmoid transformation, which
allows Gradient Descent to reliably reach the true minimum.