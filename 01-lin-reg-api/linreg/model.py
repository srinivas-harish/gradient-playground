# linreg/model.py

import torch

class LinearRegressionManualSGD:
    def __init__(self, lr=1e-6, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.theta = None

    def train(self, X, y):
        n_features = X.shape[1]
        self.theta = torch.ones(n_features) * 0.5

        for epoch in range(self.epochs):
            for i in range(len(X)):
                pred = X[i] @ self.theta
                err = pred - y[i]
                grad = err * X[i]
                self.theta -= self.lr * grad

        return self.theta

    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model is not trained yet. Call `train`.")
        return X @ self.theta


def train(X, y, lr=1e-6, epochs=100):
    model = LinearRegressionManualSGD(lr=lr, epochs=epochs)
    return model.train(X, y)
