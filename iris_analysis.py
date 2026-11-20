"""
ML Fundamentals on the Iris Dataset
-----------------------------------
Part 1: Linear Regression implemented from scratch with Gradient Descent (NumPy only)
Part 2: K-Means Clustering (unsupervised) with proper evaluation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

IMG_DIR = "images"

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
df = pd.read_csv("iris.csv")
print("Dataset shape:", df.shape)
print(df.head(), "\n")

# ======================================================================
# PART 1 - Linear Regression with Gradient Descent
# Predict petal.length from sepal.length
# ======================================================================
x = df["sepal.length"].values
y = df["petal.length"].values
n = len(x)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Standardize x so gradient descent converges quickly with a larger learning rate
x_mean, x_std = x.mean(), x.std()
x_scaled = (x - x_mean) / x_std

m, b = 0.0, 0.0
learning_rate = 0.1
iterations = 1000
costs = []

print("Cost before training:", round(mse(y, m * x_scaled + b), 4))

for _ in range(iterations):
    y_pred = m * x_scaled + b
    dm = (-2 / n) * np.sum(x_scaled * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)
    m -= learning_rate * dm
    b -= learning_rate * db
    costs.append(mse(y, y_pred))

# Convert parameters back to the original (unscaled) feature space
m_orig = m / x_std
b_orig = b - m * x_mean / x_std
y_hat = m_orig * x + b_orig

print("Cost after training: ", round(mse(y, y_hat), 4))
print(f"Final equation: petal.length = {m_orig:.3f} * sepal.length + {b_orig:.3f}")
print("R^2 score:", round(r2_score(y, y_hat), 4))

# Sanity check against scikit-learn
sk = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"scikit-learn : slope={sk.coef_[0]:.3f}, intercept={sk.intercept_:.3f}")

pred_at_mean = m_orig * x.mean() + b_orig
print(f"Predicted petal.length at mean sepal.length ({x.mean():.2f}): {pred_at_mean:.3f}\n")

# Plot: cost vs iterations
plt.figure(figsize=(6, 4))
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent: Cost vs Iterations")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/cost_vs_iterations.png", dpi=150)
plt.close()

# Plot: regression line
plt.figure(figsize=(6, 4))
plt.scatter(x, y, alpha=0.7, label="Data")
xs = np.linspace(x.min(), x.max(), 100)
plt.plot(xs, m_orig * xs + b_orig, color="red", label="Regression line")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Linear Regression on Iris Data")
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/regression_line.png", dpi=150)
plt.close()

# ======================================================================
# PART 2 - K-Means Clustering
# ======================================================================
features = df.drop(columns=["variety"])
X_scaled = StandardScaler().fit_transform(features)

# Choose k: elbow (inertia) + silhouette score
ks = range(2, 9)
inertias, sil_scores = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertias, marker="o")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil_scores, marker="o", color="green")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Score")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/choose_k.png", dpi=150)
plt.close()
print("Silhouette scores by k:", {k: round(s, 3) for k, s in zip(ks, sil_scores)})

# Final model with k = 3 (matches the 3 species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df["cluster"] = kmeans.labels_

sil = silhouette_score(X_scaled, kmeans.labels_)
ari = adjusted_rand_score(df["variety"], df["cluster"])
print(f"\nk=3 silhouette score: {sil:.3f}")
print(f"Adjusted Rand Index vs true species: {ari:.3f}")
print("\nClusters vs real species:")
print(pd.crosstab(df["variety"], df["cluster"]))

# Plot: clusters vs true species side by side
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(df["petal.length"], df["petal.width"], c=df["cluster"], cmap="viridis")
ax[0].set_title("K-Means Clusters (k=3)")
ax[1].scatter(df["petal.length"], df["petal.width"],
              c=df["variety"].astype("category").cat.codes, cmap="viridis")
ax[1].set_title("True Species")
for a in ax:
    a.set_xlabel("Petal Length (cm)")
    a.set_ylabel("Petal Width (cm)")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/clusters_vs_species.png", dpi=150)
plt.close()
