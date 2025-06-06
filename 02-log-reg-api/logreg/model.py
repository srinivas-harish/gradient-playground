# logreg/model.py

import torch

class LogisticRegressionMiniBatchGD:
    def __init__(self, lr=1e-6, epochs=100, batch_size = 32):
        self.lr = lr
        self.epochs = epochs
        self.theta = None
        self.batch_size = batch_size

    def train(self, X, y):
        m, n = X.shape
        self.theta = torch.ones(n) * 0.5

        for epoch in range(self.epochs):
            # Shuffle data each epoch
            indices = torch.randperm(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                logits = xb @ self.theta               
                y_hat = torch.sigmoid(logits)          

                eps = 1e-9
                y_hat = torch.clamp(y_hat, eps, 1 - eps)

                # vectorized binary cross-entropy loss gradient
                # dL/dtheta = (1/B) * X^T (y_hat - y)
                error = y_hat - yb                     
                grad = (error.unsqueeze(1) * xb).mean(0)  

                self.theta -= self.lr * grad

        return self.theta


    def predict(self, X):
        if self.theta is None:
            raise ValueError("Model is not trained yet. Call `train`.")
        probs = torch.sigmoid(X @ self.theta)
        return probs


def train(X, y, lr=1e-6, epochs=100, batch_size=32):
    model = LogisticRegressionMiniBatchGD(lr=lr, epochs=epochs, batch_size = batch_size)
    return model.train(X, y)
