# Day 23: K-Means Clustering and Silhouette Analysis

## Objective
Master partition based clustering by implementing K-Means from scratch, understanding its mathematical foundations, applying scikit learn's production version, and learning two independent methods for choosing the correct number of clusters.

## Topics Covered
K-Means iterative centroid optimization algorithm, covering the assign then recalculate loop until convergence
Sensitivity of K-Means to random initial centroid placement and how K-Means plus plus smart seeding solves it
Determining the optimal number of clusters using the Elbow Method and the Silhouette Score
Within Cluster Sum of Squares, also called Inertia, as the mathematical measure of cluster tightness
The centroid update formula, which is simply the average position of all points in a group
The Silhouette Coefficient formula, which scores how well an individual point fits its assigned group versus the next nearest group
Building K-Means entirely from scratch in NumPy to see every algorithm step directly in code
Training scikit learn's KMeans model and extracting its inertia score
Plotting Elbow Method and Silhouette Score graphs across a range of candidate K values
Segmenting a customer dataset into named, business ready personas

## Key Formulas
Within Cluster Sum of Squares, J, equals the sum across every cluster of the sum across every point in that cluster of the squared distance between the point and its cluster's centroid
Centroid update: the new centroid of a cluster equals one divided by the number of points in that cluster, multiplied by the sum of every point in that cluster
Silhouette Coefficient: s equals b minus a, divided by the maximum of a and b, where a is the average distance to points in the same cluster and b is the average distance to points in the nearest neighboring cluster

## Practical Work
Built a synthetic customer dataset with annual spending and visit frequency containing three natural underlying groups, without revealing that structure to the algorithm. Implemented K-Means entirely from scratch in NumPy and confirmed it converged in five iterations to three clearly separated centroids. Trained scikit learn's KMeans model on the same data and confirmed it reached identical centroids in only two iterations due to K-Means plus plus smart seeding. Ran both the Elbow Method and Silhouette Score across K values from two through eight, and found that both methods independently agreed that three clusters produced the best result, with a peak Silhouette Score of zero point eight one five. Used this confirmed cluster count to segment the customer dataset into three named personas, Budget Browsers, Steady Regulars, and Premium Loyalists, each with a distinct average spending and visit pattern.

## Tools
NumPy for the from scratch K-Means implementation and distance calculations
Pandas for dataset handling
Scikit learn's KMeans and silhouette_score functions
Matplotlib for the Elbow curve, Silhouette score graph, and final persona scatter plot

## Key Learning
K-Means is not a mysterious black box. It is a simple repeating loop of assigning points to their nearest centroid and then recalculating that centroid as the average of its assigned points, continuing until nothing changes. The real skill in using it correctly is not the algorithm itself but choosing the right number of clusters, which is why the Elbow Method and Silhouette Score should always be checked together rather than relying on either one alone. Independent agreement between both methods gives real confidence in the final cluster count before any business decision is built on top of it.