# Day 28: Feature Engineering and Scikit-Learn Pipelines

## Objective
Build automated, leak-free feature preprocessing pipelines by learning how to transform numeric features, encode categorical features correctly based on whether a natural order exists, and chain every preprocessing step together with a final model using Scikit-Learn's Pipeline architecture, specifically to prevent data leakage between training and test data.

## Topics Covered
Feature transformations: StandardScaler, MinMaxScaler, and Power Transforms, and when each one is the correct choice based on a feature's distribution shape
Categorical encoding techniques: One-Hot Encoding for categories with no natural order, Ordinal Encoding for categories with a genuine order, and Target Encoding for high-cardinality categorical columns
Scikit-Learn Pipeline architecture and how it structurally prevents data leakage by ensuring every preprocessing step is fit only on training data
StandardScaler normalization formula and MinMaxScaler transformation formula, traced by hand and verified against Scikit-Learn's own implementation
Constructing a ColumnTransformer to apply different preprocessing steps to different columns of the same dataset
Chaining data transformers and a classifier into a single sklearn.pipeline.Pipeline object

## Key Formulas
StandardScaler normalization: z equals (x minus the mean) divided by the standard deviation
MinMaxScaler transformation: x scaled equals (x minus the minimum) divided by (the maximum minus the minimum)

## Practical Work
Built a deliberately messy twelve row customer dataset containing missing values across both numeric and categorical columns, along with one categorical column with a genuine natural order and one without. Traced StandardScaler and MinMaxScaler formulas by hand on a small numeric example and confirmed both matched Scikit-Learn's built in implementations exactly. Constructed a ColumnTransformer routing numeric features through median imputation and scaling, nominal categorical features through most-frequent imputation and One-Hot Encoding, and the ordinal education column through Ordinal Encoding with an explicit Bachelors-Masters-PhD order. Chained this preprocessor with a Random Forest classifier into a single Pipeline and trained it successfully on the messy dataset. Directly demonstrated the real danger of data leakage by fitting a scaler on the full dataset before splitting versus fitting it only on training data after splitting, finding a meaningful gap of over nine thousand in the calculated mean for the income feature between the two approaches.

## Tools
Scikit learn's StandardScaler, MinMaxScaler, and SimpleImputer
Scikit learn's OneHotEncoder and OrdinalEncoder
Scikit learn's ColumnTransformer and Pipeline
Scikit learn's RandomForestClassifier and train_test_split

## Key Learning
A Scikit-Learn Pipeline is not a convenience feature, it is a structural safeguard. Calculating any preprocessing statistic, such as a mean for scaling or a most-frequent value for imputation, using data that includes the test set is a genuine form of data leakage that makes a model look better during development than it will actually perform on real, unseen data. Bundling every preprocessing step and the final model into a single Pipeline object, fit only on training data, makes this specific mistake structurally impossible rather than something that has to be remembered and manually avoided every single time.
