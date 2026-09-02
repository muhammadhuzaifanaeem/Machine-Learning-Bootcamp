# Day 27: Cross Validation and Hyperparameter Tuning

## Objective
Implement robust cross-validation schemes and systematic hyperparameter search strategies by understanding why a single train-test split cannot be trusted alone, learning how Stratified K-Fold and TimeSeriesSplit solve specific real-world failure modes, comparing Grid Search against Random Search on a real model, and honestly evaluating whether tuning actually improves performance over sensible defaults.

## Topics Covered
K-Fold, Stratified K-Fold, and TimeSeriesSplit cross-validation, including the specific failure modes each one solves
Grid Search versus Random Search versus Bayesian Optimization strategies for searching hyperparameter space
Designing effective hyperparameter search spaces based on sensible, well-informed ranges rather than blind guessing
The cross-validation mean performance estimate formula, and why standard deviation across folds matters as much as the average
Hyperparameter grid search space combinatorics, and why the number of combinations multiplies rather than adds as more hyperparameters are tuned
Implementing a StratifiedKFold evaluation loop in Scikit-Learn
Executing GridSearchCV and RandomizedSearchCV on an XGBoost model
Tuning Random Forest and XGBoost hyperparameters to optimize cross-validated F1-score, and comparing tuned results honestly against untuned baselines

## Key Formulas
Cross-validation mean performance estimate: E equals one divided by K, multiplied by the sum across every fold k of the performance score E sub k
Hyperparameter grid search space combinatorics: the total number of combinations equals the product of the number of values chosen for each individual hyperparameter

## Practical Work
Demonstrated plain KFold's real failure mode on a small imbalanced toy dataset, finding that two out of three folds contained zero examples of the minority class while the third fold contained only minority class examples, then confirmed StratifiedKFold correctly preserved the true class balance in every fold. Implemented a full StratifiedKFold evaluation loop on the real breast cancer diagnostic dataset, finding a mean accuracy of ninety five point six one percent with a standard deviation of only one point two three percent across folds, confirming consistent, trustworthy performance. Ran GridSearchCV on an XGBoost model across a small eight-combination grid, finding a best F1 score of zero point nine seven three seven. Ran RandomizedSearchCV across a much wider, continuous search space with only twenty sampled combinations, finding a slightly better F1 score of zero point nine seven five one by exploring parameter values that did not exist anywhere in the fixed grid. Directly compared baseline versus tuned F1 scores for both Random Forest and XGBoost, finding that XGBoost improved meaningfully with tuning while Random Forest's tuned score came out very slightly lower than its already well-chosen default settings, an honest and instructive result rather than a guaranteed improvement.

## Tools
Scikit learn's KFold, StratifiedKFold, and TimeSeriesSplit
Scikit learn's GridSearchCV and RandomizedSearchCV
Scikit learn's RandomForestClassifier and cross_val_score
XGBoost's XGBClassifier
Scipy's stats module, specifically randint and uniform, for defining continuous random search distributions
Matplotlib for the baseline versus tuned F1 score comparison chart

## Key Learning
A single train-test split score is a sample of a model's true performance, not proof of it, which is why cross-validation and its standard deviation across folds matter more than any single reported number. Stratified K-Fold and TimeSeriesSplit exist to prevent two very specific, very real failure modes: imbalanced classes accidentally landing unevenly across folds, and future information accidentally leaking backward into training on time-ordered data. Hyperparameter tuning is a genuine search for improvement, not a guaranteed upgrade over sensible defaults, and the only way to know whether tuning actually helped is to always keep and honestly compare the untuned baseline score, exactly as demonstrated when Random Forest's tuned result came out slightly worse than its defaults on this dataset.
