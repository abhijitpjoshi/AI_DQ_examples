
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
