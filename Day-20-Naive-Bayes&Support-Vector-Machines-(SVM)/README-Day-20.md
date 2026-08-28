# Day 20 — Naive Bayes & Support Vector Machines (SVM)

## Objective

Master probabilistic classification using Naive Bayes and
maximum-margin classification using Support Vector Machines,
including how the Kernel Trick enables non-linear decision
boundaries.

## Topics Covered

- Naive Bayes Class-Conditional Independence Assumption
- SVM Maximum Margin Hyperplane Principle
- Support Vectors
- Kernel Trick Intuition: RBF and Polynomial Kernels
- Naive Bayes Rule
- SVM Margin Width Derivation
- Hinge Loss
- GaussianNB and MultinomialNB
- Scikit-learn SVC with Linear and RBF Kernels

## Key Formulas

Naive Bayes Rule: P(Y|X1..Xn) is proportional to P(Y) * product of P(Xi|Y)

SVM Margin Width: Margin = 2 / ||w||

Hinge Loss: L(y, f(x)) = max(0, 1 - y*f(x))

## Practical Work

Trained GaussianNB on continuous numeric data and MultinomialNB on
text data using CountVectorizer. Trained SVM with both linear and RBF
kernels, comparing performance on linearly-separable data versus
circular, non-linearly-separable data to observe how the RBF kernel
solves problems a linear kernel structurally cannot. Benchmarked
Naive Bayes against SVM on a high-dimensional text classification
task, comparing both accuracy and training speed.

## Tools

- Python
- Scikit-learn

## Key Learning

Naive Bayes assumes all features are independent given the class,
which is rarely true in reality but still produces strong, fast
classifiers because ranking classes correctly matters more than
having perfectly calibrated probabilities. SVM instead finds the
decision boundary that maximizes the margin between classes, and the
Kernel Trick allows it to solve problems that are not linearly
separable in the original feature space by implicitly mapping data
into a higher-dimensional space where a straight boundary becomes
possible. Naive Bayes trains significantly faster than SVM, making it
a strong first baseline for high-dimensional text classification
tasks.