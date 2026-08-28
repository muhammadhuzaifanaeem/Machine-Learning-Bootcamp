# Day 12 — Regularization: Ridge, Lasso & Elastic Net

## 📌 30-Day Machine Learning Bootcamp

**Day:** 12  
**Topic:** Regularization — Ridge, Lasso & Elastic Net  
**Learning Focus:** Controlling overfitting, reducing model complexity, and performing feature selection.

---

## 🎯 Learning Objectives

By the end of Day 12, I learned:

- Why machine learning models overfit
- What regularization is and why it is needed
- The Bias-Variance Tradeoff
- L1 Regularization
- L2 Regularization
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- How regularization affects model coefficients
- How `alpha` controls regularization strength
- How Lasso can perform feature selection
- Why feature scaling is important for regularized models
- How to compare Linear Regression, Ridge, Lasso, and Elastic Net
- How to analyze model performance using MAE, MSE, RMSE and R²

---

# 1. What is Overfitting?

Overfitting happens when a machine learning model learns the training data too closely, including noise and unnecessary patterns.

For example:

```text
Training Performance → Excellent
Testing Performance  → Poor