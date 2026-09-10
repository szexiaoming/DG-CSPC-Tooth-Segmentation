"""
Dice Loss for binary segmentation.

Dice Loss = 1 - Dice Coefficient
Useful for handling foreground-background class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Supports both sigmoid-activated and raw logit inputs via the `from_logits` parameter.
    """

    def __init__(self, from_logits: bool = True, smooth: float = 1e-5) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_per_sample = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )

        return 1.0 - dice_per_sample.mean()
