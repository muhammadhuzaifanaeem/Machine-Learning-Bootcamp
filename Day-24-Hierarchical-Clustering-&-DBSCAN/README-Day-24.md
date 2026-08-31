# Day 24: Hierarchical Clustering and DBSCAN

# Objective
Master two clustering approaches that solve the two core weaknesses of K-Means: agglomerative hierarchical clustering, which removes the need to decide the number of groups in advance, and DBSCAN, which groups by density rather than distance to a center point, allowing it to correctly handle non-spherical cluster shapes and identify genuine outliers.

# Topics Covered
Agglomerative versus Divisive hierarchical clustering concepts, and why agglomerative bottom up merging is the far more commonly used direction in practice
Linkage criteria comparison: Single, Complete, Average, and Ward linkage, and how each produces meaningfully different cluster shapes
DBSCAN density metrics: Core points, Border points, and Noise points, and how density based grouping differs fundamentally from center based grouping
Ward's minimum variance linkage distance formula, which selects merges based on minimizing the increase in overall cluster tightness
Epsilon neighborhood definition, the mathematical foundation behind how DBSCAN decides which points count as neighbors
Performing Agglomerative Clustering and plotting a dendrogram using scipy
Training scikit learn's DBSCAN model and isolating noise points from real clusters
Comparing K-Means and DBSCAN performance directly on non-spherical, moon-shaped geometric data

# Key Formulas
Ward's linkage selects, at every merge step, whichever pair of clusters would produce the smallest possible increase in total Within-Cluster Sum of Squares if merged
Epsilon neighborhood: N sub eps of p equals the set of every point q in the dataset such that the distance between p and q is less than or equal to eps

# Practical Work
Built a small eight customer dataset containing three obviously tight groups and used scipy's linkage function with Ward linkage to construct a full merge tree, confirming through the dendrogram that the three groups merged internally at very small distances before merging with each other at much larger distances. Trained scikit learn's DBSCAN model on a moon shaped dataset with ten deliberately scattered outlier points added, and confirmed it correctly identified both true crescent shaped groups while correctly labeling all ten outliers as noise rather than forcing them into a cluster. Directly compared K-Means and DBSCAN on the same moon shaped data, confirming visually and numerically that K-Means incorrectly sliced straight through both curved shapes due to its reliance on distance to a center point, while DBSCAN correctly traced both true crescent shapes using density based chaining instead.

# Tools
Scipy's cluster hierarchy module, specifically linkage and dendrogram, for agglomerative clustering
Scikit learn's AgglomerativeClustering and DBSCAN models
Scikit learn's make_moons function to generate non-spherical test data
Matplotlib for the dendrogram and side by side cluster comparison visualizations

# Key Learning
K-Means is not a universally correct clustering tool. It structurally assumes every cluster is roughly round because it groups purely by distance to a center point, which means it fails visibly on curved or irregularly shaped real world data such as coastlines, rivers, or winding delivery routes. Hierarchical clustering solves the different problem of not knowing the right number of groups in advance by building a full tree of relationships you can cut at any level after the fact, while DBSCAN solves the different problem of non-spherical shapes and genuine outliers by grouping based on local crowdedness instead of distance to a center. Choosing the right clustering algorithm depends entirely on the actual shape of the data, not on which tool is generally considered more advanced.