import torch
import math
from typing import List

class SoftmaxRegression:
    def __init__(self, input_dim: int, num_classes: int):
        
        self.theta = torch.randn(num_classes, input_dim, dtype=torch.float32, requires_grad=True)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute raw scores z = θx
        """
        return self.theta @ x  # [K, n] @ [n] = [K]

    def softmax(self, z: torch.Tensor) -> torch.Tensor:
        """
        Manually compute softmax using exp(z_k) / sum(exp(z_j))
        """
        exp_z = torch.tensor([math.exp(v.item()) for v in z], dtype=torch.float32)
        tot_sum = torch.sum(exp_z)
        return exp_z / tot_sum

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities
        """
        z = self.logits(x)
        return self.softmax(z)

    def predict(self, x: torch.Tensor) -> int:
        """
        Predict the class with highest probability
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs).item()

    def loss(self, x: torch.Tensor, y: int) -> torch.Tensor:
        """
        Cross-entropy loss for a single example
        """
        z = self.logits(x)
        log_sum_exp = torch.log(torch.sum(torch.tensor([math.exp(v.item()) for v in z])))
        return log_sum_exp - z[y]  # neg log likelihood

    def train(self, X: List[torch.Tensor], Y: List[int], lr: float = 0.1, epochs: int = 100):
        """
        Trains the softmax regression model using gradient descent.
        X: list of input tensors (each of shape [input_dim])
        Y: list of integer class labels (each in [0, num_classes-1])
        lr: learning rate
        epochs: number of training iterations
        """
        for epoch in range(epochs):
            total_loss = 0.0

            for x, y in zip(X, Y):
                # Forward pass
                z = self.logits(x)
                exp_z = torch.tensor([math.exp(v.item()) for v in z], dtype=torch.float32)
                tot_sum = torch.sum(exp_z)
                probs = exp_z / tot_sum

                # Loss
                loss = torch.log(tot_sum) - z[y]
                total_loss += loss.item()

                # Backward pass manually: dL/dz_k = softmax_k - 1(y = k)
                grad_z = probs.clone()
                grad_z[y] -= 1.0  # subtract 1 for the correct class

                # Convert gradient w.r.t. z into gradient w.r.t. theta
                grad_theta = torch.outer(grad_z, x)  # [K, 1] * [1, n] = [K, n]

                # Manual gradient descent step
                with torch.no_grad():
                    self.theta -= lr * grad_theta
                    self.theta.requires_grad = True  # re-enable gradients if needed later

            # Print average loss per epoch
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
