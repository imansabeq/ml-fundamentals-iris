<h1 align="center">🌸 ML Fundamentals on the Iris Dataset</h1>

<p align="center">
  Linear Regression from scratch with Gradient Descent + K-Means Clustering
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-11557c" alt="Matplotlib">
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Part 1: Linear Regression with Gradient Descent](#-part-1-linear-regression-with-gradient-descent)
- [Part 2: K-Means Clustering](#-part-2-k-means-clustering)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Key Takeaways](#-key-takeaways)

---

## 📌 Overview

This project applies two core machine learning techniques to the classic Iris dataset:

1. **Supervised learning:** Linear Regression implemented **from scratch** using Gradient Descent (NumPy only, no ML library for training).
2. **Unsupervised learning:** K-Means Clustering with proper evaluation (elbow method, silhouette score, Adjusted Rand Index).

## 📊 Dataset

The [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) contains 150 samples of three species (Setosa, Versicolor, Virginica) with four features each.

| Feature | Description |
|---------|-------------|
| `sepal.length` | Sepal length (cm) |
| `sepal.width` | Sepal width (cm) |
| `petal.length` | Petal length (cm) |
| `petal.width` | Petal width (cm) |
| `variety` | Species (used only for evaluation in Part 2) |

---

## 📈 Part 1: Linear Regression with Gradient Descent

**Goal:** predict `petal.length` from `sepal.length`.

**Approach**
- Implemented the MSE cost function and the gradient updates manually.
- Standardized the feature so gradient descent converges quickly, then converted the parameters back to the original scale.
- Validated the result against scikit-learn's `LinearRegression`: both give the same slope and intercept.

**Results**

| Metric | Value |
|--------|-------|
| Final equation | `petal.length = 1.858 × sepal.length − 7.101` |
| Cost (MSE) before training | 17.22 |
| Cost (MSE) after training | 0.74 |
| R² score | 0.76 |

<p align="center">
  <img src="images/cost_vs_iterations.png" width="45%" alt="Cost vs iterations">
  <img src="images/regression_line.png" width="45%" alt="Regression line">
</p>

---

## 🔵 Part 2: K-Means Clustering

**Goal:** group the flowers into clusters without using the species labels.

**Approach**
- Removed the labels and scaled the features with `StandardScaler`.
- Chose `k` using the elbow method and the silhouette score.
- Compared the clusters with the real species using the Adjusted Rand Index and a crosstab.

**Results (k = 3)**

| Metric | Value |
|--------|-------|
| Silhouette score | 0.46 |
| Adjusted Rand Index vs true species | 0.62 |

| Species | Cluster 0 | Cluster 1 | Cluster 2 |
|---------|:---------:|:---------:|:---------:|
| Setosa | 0 | 50 | 0 |
| Versicolor | 39 | 0 | 11 |
| Virginica | 14 | 0 | 36 |

> 💡 Setosa is separated perfectly. Versicolor and Virginica partially overlap, which is expected since they are close in feature space. The silhouette score is actually highest at k = 2, because Setosa is far from the other two species.

<p align="center">
  <img src="images/choose_k.png" width="90%" alt="Choosing k">
</p>
<p align="center">
  <img src="images/clusters_vs_species.png" width="90%" alt="Clusters vs species">
</p>

---

## 📂 Project Structure

```
ml-fundamentals-iris/
├── images/              # Generated figures
├── iris.csv             # Dataset
├── iris_analysis.py     # Full pipeline: training, evaluation, plots
├── requirements.txt     # Dependencies
└── README.md
```

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ml-fundamentals-iris.git
cd ml-fundamentals-iris

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python iris_analysis.py
```

The figures are saved automatically in the `images/` folder.

## 🎯 Key Takeaways

- How gradient descent works under the hood, and why feature scaling and the learning rate matter for convergence.
- How to evaluate clustering when labels exist (ARI) and when they don't (silhouette, elbow).
- Validating a from-scratch implementation against a trusted library.

## 🛠️ Tech Stack

Python · NumPy · Pandas · Matplotlib · scikit-learn
