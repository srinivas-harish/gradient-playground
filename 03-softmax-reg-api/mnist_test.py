# mnist_test.py

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.model import SoftmaxRegression

# Load MNIST dataset (10k samples)
print("Loading MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X, y = X[:10000], y[:10000]  # slice for speed

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode
encoder = OneHotEncoder(sparse_output=False, categories='auto')
Y = encoder.fit_transform(y.reshape(-1, 1))

# TTS
X_train, X_test, Y_train, Y_test, y_train_raw, y_test_raw = train_test_split(
    X, Y, y, test_size=0.2, random_state=42
)

# Initialize and train
K = Y.shape[1]
n = X.shape[1]
model = SoftmaxRegression(K=K, n=n, lr=0.1)

print("Training...")
model.train(X_train, Y_train, epochs=1000, print_interval=100)

# Evaluate
print("Evaluating...")
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test_raw.astype(int))

print(f"Test Accuracy: {accuracy:.4f}")
