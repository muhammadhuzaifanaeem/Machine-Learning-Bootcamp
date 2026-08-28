# Day 22: Unsupervised Learning Foundations

## Objective
Understand the foundations of unsupervised learning, including how it differs from supervised learning, where it is used in the real world, how model quality is judged without labeled answers, and the distance-based math that clustering algorithms are built on.

## Topics Covered
Supervised versus unsupervised learning paradigm differences
Core applications of unsupervised learning: customer segmentation, anomaly detection, and pattern discovery
Evaluating model performance without ground truth labels
Distance matrix calculation in multi dimensional space
Intra cluster variance versus inter cluster separation
Feature correlation matrix and summary statistics as pre clustering exploratory analysis
Mapping real business scenarios to the correct unsupervised learning approach

## Key Formulas
Euclidean distance between two points extends the Pythagorean theorem across any number of features: distance equals the square root of the sum of the squared differences between each matching feature of two data points
Correlation measures the strength and direction of the relationship between two features, ranging from negative one to positive one

## Practical Work
Loaded the Iris flower dataset with its species labels removed, to simulate a true unlabeled dataset. Calculated a pairwise Euclidean distance matrix across the first five flowers to see how distance quantifies similarity between data points. Computed a full feature correlation matrix to identify which features move together, finding that petal length and petal width were almost perfectly correlated at zero point nine six. Generated summary statistics for all four features to understand each one's typical range and spread, revealing that petal length has a much larger spread than sepal width, which explains why feature scaling matters before clustering.

## Tools
Pandas for data loading and summary statistics
Scipy spatial distance module, specifically pdist and squareform, for pairwise distance calculation
Scikit learn's Iris dataset loader
Correlation and describe functions from Pandas

## Key Learning
Unsupervised learning solves a fundamentally different problem than supervised learning because there is no answer key to check against, so every technique in this space is ultimately built on measuring how similar or different data points are to each other using distance and correlation. Today's work made clear that scaling features before computing distance is not optional, since features with larger natural ranges will silently dominate the distance calculation and distort any grouping built on top of it.