# Day 21 — Project: Customer Churn Prediction & ROC-AUC

## Objective

Execute a complete classification project on a real dataset with
imbalanced classes, comparing multiple models using ROC-AUC and
translating the results into a business-facing recommendation.

## Topics Covered

- Class Imbalance Strategies: Oversampling (SMOTE), Undersampling,
  Class Weights
- ROC Curve: True Positive Rate vs False Positive Rate
- AUC Metric Interpretation
- Business Trade-offs: False Positive Cost vs False Negative Cost
- Multi-model Training and Comparison
- Confusion Matrix Analysis on a Real Dataset

## Key Formulas

True Positive Rate: TPR = TP / (TP + FN)

False Positive Rate: FPR = FP / (FP + TN)

AUC: Area Under the ROC Curve, computed across all decision
thresholds

## Practical Work

Used the real IBM Telco Customer Churn dataset (7,043 customers) and
built a complete pipeline: cleaned a known data quality issue in the
TotalCharges column, explored churn patterns by contract type and
tenure, applied class weighting and SMOTE to address a 26.5 percent
churn rate imbalance, and trained Logistic Regression, Random Forest,
and SVM classifiers. Plotted overlaid ROC curves and compared AUC
scores across all three models, then selected the best model based
on Recall and AUC rather than Accuracy alone, since missing a real
churner is more costly to the business than a false alarm.

## Tools

- Python
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

## Key Learning

Accuracy can be a misleading metric for model selection on imbalanced
data. In this project, Random Forest achieved the highest accuracy
but the lowest Recall, meaning it missed more than half of the actual
churners, while Logistic Regression caught significantly more real
churners at a small cost to precision. Since a missed churner
represents lost revenue while a false alarm only costs a wasted
retention offer, the model with higher Recall and AUC was the more
defensible business choice, even though it did not have the highest
raw accuracy.