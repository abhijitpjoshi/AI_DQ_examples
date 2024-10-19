
# Machine Learning Anomaly Detection Examples

This repository contains multiple examples of machine learning techniques used for anomaly detection and clustering. Each example demonstrates a different approach to handling anomalies or clustering data points. The examples include:

1. **Random Forest for Outlier Detection**
2. **Principal Component Analysis (PCA) for Anomaly Detection**
3. **Autoencoder for Real-Time Anomaly Detection**
4. **DBSCAN Clustering for Outlier Detection**

## Overview

These examples cover a range of techniques, including both supervised and unsupervised learning methods. The goal of this repository is to provide a practical understanding of how to apply different anomaly detection approaches to various datasets.

### Libraries Used
- `numpy` for data manipulation
- `scikit-learn` for machine learning models and dataset generation
- `matplotlib` for data visualization
- `tensorflow` for deep learning (Autoencoder example)

To install all the required libraries, use:

```sh
pip install numpy scikit-learn matplotlib tensorflow
```

## 1. Random Forest for Outlier Detection

Random Forest can be used to identify anomalies by fitting a classifier and examining the accuracy of predictions. Here's a simplified version:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

**Explanation**: This code trains a Random Forest classifier to detect anomalies in the dataset and then evaluates the accuracy.

## 2. PCA for Anomaly Detection

Principal Component Analysis (PCA) can reduce dimensionality and help identify anomalies by calculating reconstruction error:

```python
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
```

**Explanation**: PCA reduces data to two dimensions, calculates reconstruction errors, and flags data points with high errors as anomalies.

## 3. Autoencoder for Real-Time Anomaly Detection

An Autoencoder can be used to reconstruct data and identify anomalies by measuring reconstruction error:

```python
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=300, n_features=5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Autoencoder model
input_dim = X_scaled.shape[1]
encoding_dim = 2
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=20, batch_size=16, validation_split=0.2, verbose=0)

# Reconstruction errors
X_reconstructed = autoencoder.predict(X_scaled)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
threshold = np.percentile(reconstruction_error, 90)
outliers = reconstruction_error > threshold

# Visualize
plt.hist(reconstruction_error, bins=30, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()
```

**Explanation**: This Autoencoder compresses data and reconstructs it. Higher reconstruction errors indicate potential anomalies.

## 4. DBSCAN Clustering for Outlier Detection

DBSCAN is a density-based clustering algorithm that can identify clusters and outliers:

```python
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Generate synthetic data
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering Results')
plt.show()
```

**Explanation**: DBSCAN groups data points into clusters and marks outliers with a label of `-1`.

## Running the Examples

To run these examples, clone the repository and execute the corresponding Python scripts:

```sh
git clone <repository_url>
cd ml_anomaly_detection_examples
python <script_name>.py
```

Each script generates plots or results to help visualize clustering or anomaly detection.

## Notes

- These examples provide a variety of approaches to detecting anomalies.
- The right approach depends on the type of data and the characteristics of anomalies you are trying to detect.

## License

This project is licensed under the MIT License.

## Contributing

Feel free to fork this repository and contribute by adding more examples or improving the existing ones.
