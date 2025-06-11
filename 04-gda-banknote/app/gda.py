import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class GDAClassifier:
    def __init__(self):
        self.class_priors = {}
        self.class_means = {}
        self.class_covariances = {}
        self.classes = []

    def _compute_mean(self, X):
        return np.mean(X, axis=0)

    def _compute_covariance(self, X, mean):
        m = X.shape[0]
        centered = X - mean
        cov = (centered.T @ centered) / m  # n x n
        return cov

    def _gaussian_pdf(self, x, mean, cov):
        """Multivariate Gaussian PDF """
        n = x.shape[0]
        diff = x - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_const = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(det_cov))
        exponent = -0.5 * (diff.T @ inv_cov @ diff)
        return norm_const * np.exp(exponent)

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            self.class_means[c] = self._compute_mean(X_c)
            self.class_covariances[c] = self._compute_covariance(X_c, self.class_means[c])

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prior = self.class_priors[c]
                mean = self.class_means[c]
                cov = self.class_covariances[c]
                likelihood = self._gaussian_pdf(x, mean, cov)
                class_probs[c] = prior * likelihood
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
