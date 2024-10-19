
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=300, n_features=5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Reconstruction and error calculation
X_reconstructed = pca.inverse_transform(X_pca)
reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
threshold = np.percentile(reconstruction_error, 95)
outliers = reconstruction_error > threshold

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=~outliers, cmap='coolwarm', edgecolor='k')
plt.scatter(X_pca[outliers, 0], X_pca[outliers, 1], color='red', label='Anomalies')
plt.legend()
plt.show()
