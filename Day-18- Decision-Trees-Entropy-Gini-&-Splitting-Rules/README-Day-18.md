# Day 18 — Decision Trees: Entropy, Gini & Splitting Rules

## Objective

Understand recursive binary splitting and tree impurity measures, and
learn how a Decision Tree decides which question to ask at each step
of its structure.

## Topics Covered

- Decision Tree Structure: Root Nodes, Internal Splits, and Leaves
- Greedy Top-Down Recursive Binary Splitting Strategy
- Overfitting Characteristics in Deep Trees
- Pruning Strategies
- Entropy
- Gini Impurity
- Information Gain
- Scikit-learn DecisionTreeClassifier
- Visual Tree Diagram Rendering

## Key Formulas

Entropy: H(S) = -sum(p_i * log2(p_i))

Gini Impurity: G(S) = 1 - sum(p_i^2)

Information Gain: IG(S, A) = H(S) - sum((|S_v|/|S|) * H(S_v))

## Practical Work

Trained a Decision Tree Classifier and compared an unpruned, fully
grown tree against a pruned tree limited by max_depth and
min_samples_leaf, confirming that the unpruned tree overfits with a
large gap between train and test accuracy. Manually calculated Gini
Impurity and Information Gain for a sample split by hand, then
verified the logic against a rendered visual tree diagram showing how
impurity decreases from the root node down to the leaves.

## Tools

- Python
- Scikit-learn
- Matplotlib

## Key Learning

A Decision Tree builds its structure by greedily choosing, at each
step, the question that produces the highest Information Gain,
without considering whether a different choice might lead to a
better tree several steps later. This greedy behavior means trees
are not guaranteed to be globally optimal. Left unrestricted, a tree
will keep splitting until it perfectly memorizes the training data,
which is why pruning controls such as max_depth and min_samples_leaf
are essential for building a tree that generalizes well to unseen
data rather than overfitting to noise.