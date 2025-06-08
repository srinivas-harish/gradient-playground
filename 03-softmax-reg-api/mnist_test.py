# mnist_test.py
# NEW: Script to train/test softmax on MNIST

import torch
import math
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from typing import List
import random

# ---- SoftmaxRegression model, slightly modified ----

class SoftmaxRegression:
    def __init__(self, input_dim: int, num_classes: int):
        self.theta = torch.randn(num_classes, input_dim, dtype=torch.float32) * 0.01
        self.theta.requires_grad_()   



    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.theta @ x  # [K, n] @ [n] = [K]

    def softmax(self, z: torch.Tensor) -> torch.Tensor:
        exp_z = torch.tensor([math.exp(v.item()) for v in z], dtype=torch.float32)
        tot_sum = torch.sum(exp_z)
        return exp_z / tot_sum

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        z = self.logits(x)
        return self.softmax(z)

    def predict(self, x: torch.Tensor) -> int:
        probs = self.predict_proba(x)
        return torch.argmax(probs).item()

    def loss(self, x: torch.Tensor, y: int) -> torch.Tensor:
        z = self.logits(x)



        z_max = torch.max(z)
        exp_z = torch.tensor([math.exp((v - z_max).item()) for v in z], dtype=torch.float32)
        log_sum_exp = z_max + torch.log(torch.sum(exp_z))




        return log_sum_exp - z[y]

    def train(self, X: List[torch.Tensor], Y: List[int], lr: float = 0.1, epochs: int = 10):
        for epoch in range(epochs):
            total_loss = 0.0

            for x, y in zip(X, Y):
                z = self.logits(x)
 
                z_max = torch.max(z)
                exp_z = torch.tensor([math.exp((v - z_max).item()) for v in z], dtype=torch.float32)
 
                tot_sum = torch.sum(exp_z)
                probs = exp_z / tot_sum

                loss = torch.log(tot_sum) - z[y]
                total_loss += loss.item()

                grad_z = probs.clone()
                grad_z[y] -= 1.0
                grad_theta = torch.outer(grad_z, x)

                with torch.no_grad():
                    self.theta -= lr * grad_theta 

            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# ---- MNIST test harness ----

def prepare_data(limit=1000):
    train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())

    # Limit dataset to first `limit` samples (faster?)
    X_train = []
    Y_train = []

    for i in range(limit):
        img, label = train_dataset[i]
        img = img.view(-1).float() / 255.0  # Flatten and normalize to [0, 1]
        X_train.append(img)
        Y_train.append(label)

    X_test = []
    Y_test = []

    for i in range(200):
        img, label = test_dataset[i]
        img = img.view(-1).float() / 255.0
        X_test.append(img)
        Y_test.append(label)

    return X_train, Y_train, X_test, Y_test

def evaluate(model, X: List[torch.Tensor], Y: List[int]):
    correct = 0
    for x, y in zip(X, Y):
        pred = model.predict(x)
        if pred == y:
            correct += 1
    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    input_dim = 28 * 28
    num_classes = 10

    X_train, Y_train, X_test, Y_test = prepare_data(limit=10000)

    model = SoftmaxRegression(input_dim=input_dim, num_classes=num_classes)
    model.train(X_train, Y_train, lr=0.05, epochs=30)

    print("Evaluating on test set...")
    evaluate(model, X_test, Y_test)
