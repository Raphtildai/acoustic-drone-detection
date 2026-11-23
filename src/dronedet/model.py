"""The detection network (0.47 M parameters), its loss, and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import DEFAULT, FrontEnd


class DetectionCNN(nn.Module):
    """Binary drone / no-drone classifier.

    Four conv blocks (3x3 conv, batch norm, ReLU, 2x2 max pool; 3->32->64->128->256 channels),
    a frequency-attention module (two 1x1 convs and a softmax over frequency), global average
    pooling and a Linear(256,256)-ReLU-Dropout(0.4)-Linear(256,2) head.
    Input (B, 3, 64, T), output logits (B, 2).
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(self._block(3, 32), self._block(32, 64),
                                     self._block(64, 128), self._block(128, 256))
        self.freq_attn = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1)), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)), nn.Softmax(dim=2))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2))

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = (feat * self.freq_attn(feat)).sum(dim=2, keepdim=True)
        return self.classifier(self.gap(feat).flatten(1))


class FocalLoss(nn.Module):
    """Focal loss (Lin et al.) with class weight alpha for the drone class and label smoothing."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.6, label_smoothing: float = 0.02):
        super().__init__()
        self.gamma, self.alpha, self.label_smoothing = gamma, alpha, label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        focal = ((1 - torch.exp(-ce)) ** self.gamma) * ce
        if self.alpha is not None:
            focal = focal * torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return focal.mean()


def save_checkpoint(path, model: nn.Module, threshold: float, fe: FrontEnd = DEFAULT, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "best_threshold": float(threshold),
                "frontend": fe.to_dict(), "format": "dronedet-1", **extra}, str(path))


def load_checkpoint(path, device: str = "cpu"):
    """-> (model in eval mode, validation threshold, FrontEnd). Also reads checkpoints of the
    original thesis code (which store only `model_state` and `best_threshold`)."""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model = DetectionCNN().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    fe = FrontEnd(**ck["frontend"]) if "frontend" in ck else DEFAULT
    return model, float(ck.get("best_threshold", 0.5)), fe


@torch.no_grad()
def predict_proba(model: nn.Module, X, batch_size: int = 64, device: str = "cpu"):
    """Feature tensors (N, 3, F, T) -> drone probability (N,)."""
    import numpy as np
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = [torch.softmax(model(X[i:i + batch_size].to(device)), 1)[:, 1].cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)
