# Day 25: Principal Component Analysis (PCA)

## Objective
Master variance-maximizing linear dimensionality reduction by understanding why high-dimensional data becomes a problem, learning PCA's core strategy of capturing maximum variance through orthogonal directions, computing PCA by hand using both the covariance matrix and Singular Value Decomposition, and applying it to compress a real 30-feature medical dataset down to 2 dimensions while preserving meaningful class structure.

## Topics Covered
Dimensionality reduction goals and the Curse of Dimensionality, and why adding more features can make models worse rather than better
Variance maximization and orthogonal projection as PCA's core strategy for deciding which information to keep
Eigenvalues, Eigenvectors, and Principal Components explained through geometric intuition, connecting stretching directions to captured variance
Sample Covariance Matrix as the table that captures how every feature relates to every other feature at once
The Eigenvalue equation as the precise mathematical definition of a non-rotating stretching direction
Explained Variance Ratio as the method for converting a raw eigenvalue into a meaningful percentage of total information captured
Computing PCA step by step using both the Covariance Matrix method and Singular Value Decomposition in NumPy, and confirming both methods agree
Training scikit learn's PCA and plotting Cumulative Explained Variance to decide how many components to keep
Reducing a 30 feature dataset down to 2 dimensions and visualizing class separation

## Key Formulas
Sample Covariance Matrix: Sigma equals one divided by (n minus 1), multiplied by the centered data matrix transposed and multiplied by itself
Eigenvalue equation: Sigma v equals lambda v, where v is an eigenvector and lambda is its corresponding eigenvalue
Explained Variance Ratio: EVR for component i equals lambda for component i, divided by the sum of every eigenvalue

## Practical Work
Built a small five sample toy dataset and manually computed its covariance matrix, confirming the result matched NumPy's built in covariance function exactly. Calculated eigenvalues and eigenvectors of that covariance matrix, finding the first principal component captured ninety six point nine nine percent of all variance in the toy data. Verified this result independently using Singular Value Decomposition, confirming both methods produced identical eigenvalues and directions, differing only by an expected and harmless sign flip. Applied scikit learn's PCA to the real breast cancer diagnostic dataset containing thirty features across five hundred sixty nine patients, finding that ten principal components were sufficient to capture ninety five percent of the dataset's total variance. Reduced the same dataset down to just two principal components, capturing sixty three point two four percent of total variance, and visualized the result, finding that malignant and benign tumor classes formed visibly distinct regions even though PCA was never given the class labels during fitting.

## Tools
NumPy for manual covariance matrix calculation, eigenvalue decomposition, and Singular Value Decomposition
Scikit learn's PCA and StandardScaler
Scikit learn's breast cancer diagnostic dataset
Matplotlib for the cumulative explained variance curve and the two dimensional class separation scatter plot

## Key Learning
PCA does not delete columns from a dataset arbitrarily. It mathematically searches for new directions, each a weighted combination of every original feature, that capture the maximum possible spread of information while remaining completely independent of every other chosen direction. The Curse of Dimensionality explains why this compression is often necessary rather than optional, since too many raw features can make distance based calculations and model training unreliable even when every individual feature seems informative on its own. A good looking low dimensional PCA visualization is a genuinely useful sign of real structure in the data, but it is not on its own proof that a predictive model will succeed, since PCA is fit completely independently of any class labels.
