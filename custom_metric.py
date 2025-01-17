import numpy as np
import torch
from torchmetrics import Metric


class AccuracySamples(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.sigmoid(pred)
        pred = (pred >= self.threshold).int()
        self.correct += torch.all(target == pred, dim=1).sum()
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)


def binary_cross_entropy(pred_proba: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-12
    pred_proba = np.clip(pred_proba, eps, 1 - eps)
    bce = - np.mean(target * np.log(pred_proba) + (1 - target) * np.log(1 - pred_proba), axis=1)
    return bce