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
