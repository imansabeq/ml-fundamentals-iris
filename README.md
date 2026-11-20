ML Fundamentals on the Iris Dataset

Two core machine learning techniques applied to the classic Iris dataset (150 samples, 4 features, 3 species):

Linear Regression from scratch using Gradient Descent (NumPy only)
K-Means Clustering (unsupervised) with proper evaluation
What's inside
File	Description
iris_analysis.py	Full pipeline: training, evaluation and plots
iris.csv	Dataset
images/	Generated figures
Part 1: Linear Regression with Gradient Descent

Predicts petal.length from sepal.length.

Implemented the MSE cost function and the gradient updates manually, with no ML library for training.
Standardized the feature so gradient descent converges quickly, then converted the parameters back to the original scale.
Validated the result against scikit-learn's LinearRegression: both give the same slope and intercept.

Results

Equation: petal.length = 1.858 * sepal.length - 7.101
Cost (MSE): 17.22 before training, 0.74 after
R² = 0.76

Cost vs iterations Regression line

Part 2: K-Means Clustering
Removed the labels and scaled the features with StandardScaler.
Chose k using the elbow method and the silhouette score.
Compared the clusters with the real species using the Adjusted Rand Index and a crosstab.

Results (k = 3)

Silhouette score: 0.46
Adjusted Rand Index vs true species: 0.62
Setosa is separated perfectly. Versicolor and Virginica partially overlap, which is expected since they are close in feature space.
The silhouette score is actually highest at k = 2, because Setosa is far from the other two species, which are close to each other.

Choosing k Clusters vs species

Run it
bash
pip install -r requirements.txt
python iris_analysis.py
Tech stack

Python, NumPy, Pandas, Matplotlib, scikit-learn

What I learned
How gradient descent works under the hood, and why feature scaling and the learning rate matter for convergence.
How to evaluate clustering when labels exist (ARI) and when they don't (silhouette, elbow).
