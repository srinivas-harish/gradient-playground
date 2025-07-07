# model.py
# Reimplemented based on the Stanford CS229 softmax regression derivation

import numpy as np

class SoftmaxRegression:
    def __init__(self, K, n, lr=0.01):
        # K: number of classes
        # n: number of input features
        # theta: weight matrix of shape (K x n)
        # lr: learning rate

        self.theta = np.full((K,n),0.5)
        self.lr = lr 
        self.K = K
        self.n = n

    def softmax(self, logits): 
        # logits.shape = (m,K), logits = [logit_0, logit_1, ..., logit_K]
        # softmax_probs.shape = (m, K), predicted probabilities for each class per training example
        # probs_k = exp(logit k) / sum_j=1^K [(exp(logit j)]

        softmax_probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        softmax_probs /= softmax_probs.sum(axis=1, keepdims=True)
        return softmax_probs

    def forward(self, X):
        # forward(X): H_θ(X) = softmax(Xθ^T)
        # X: design set, shape.X = (m,n)
 
        logits = X @ self.theta.T       # shape: (m, K)
        return self.softmax(logits)     # shape: (m, K)

    def compute_loss(self, P, Y): #for reporting
        # cross-entropy loss (MLE)
        # P: prediction/model probabilities
        # Y: true values
        # m: number of examples

        m = Y.shape[0]  
        eps = 1e-15     # to avoid log(0)

        # to avoid log(0)
        P = np.clip(P, eps, 1 - eps)

        # Cross-entropy: -1/m * sum(Y * log(P))
        loss = -np.sum(Y * np.log(P)) / m
        return loss      

    def backward(self, X, P, Y):
        # update theta using gradient descent
        # grad: gradient of cost function (cross-entropy loss)

        m = Y.shape[0]
        grad = (1/m) * (P - Y).T @ X
        self.theta -= self.lr * grad

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            P = self.forward(X)
            self.backward(X,P,Y)

            if epoch%10==0: # print loss, interval of 10 epochs
                print(f"Epoch {epoch}, Loss: {self.compute_loss(P,Y):.4f}")

    def predict(self, X):
        # pick the highest probability class per example

        return np.argmax(self.forward(X), axis=1)