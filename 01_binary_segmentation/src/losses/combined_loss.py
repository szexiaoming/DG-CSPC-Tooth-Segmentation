"""
Combined loss function for tooth segmentation.

Total Loss = BCE Loss + Dice Loss + lambda_boundary * Boundary Loss

Supports YAML configuration for weight parameters.
"""

import torch
import torch.nn as nn
from typing import Optional

from .dice_loss import DiceLoss
from .boundary_loss import BoundaryLoss


class CombinedLoss(nn.Module):
    """
    Combined segmentation loss.

    Total = w_bce * BCE + w_dice * Dice + w_boundary * Boundary

    All components are optional and can be individually disabled by setting
    the corresponding weight to 0.0.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.1,
        boundary_radius: int = 3,
        boundary_factor: float = 5.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.from_logits = from_logits

        # BCE loss
        self.bce = nn.BCEWithLogitsLoss() if from_logits else nn.BCELoss()

        # Dice loss
        self.dice = DiceLoss(from_logits=from_logits) if dice_weight > 0 else None

        # Boundary loss
        self.boundary = (
            BoundaryLoss(
                boundary_radius=boundary_radius,
                boundary_weight_factor=boundary_factor,
                from_logits=from_logits,
            )
            if boundary_weight > 0
            else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss, optionally weighted by per-pixel confidence.

        """
        losses = {}

        # Confidence-weighted BCE Loss
        if confidence is not None:
            # L = -[c * y*log(p) + c * (1-y)*log(1-p)] / mean(c)
            bce_per_pixel = nn.functional.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            )
            weighted_bce = (bce_per_pixel * confidence).sum()
            weight_sum = confidence.sum() + 1e-8
            bce_val = weighted_bce / weight_sum
        else:
            bce_val = self.bce(pred, target)

        losses["bce"] = bce_val.item()
        total = self.bce_weight * bce_val

        # Confidence-weighted Dice Loss
        if self.dice is not None:
            if confidence is not None:
                dice_val = self._weighted_dice(pred, target, confidence)
            else:
                dice_val = self.dice(pred, target)
            total = total + self.dice_weight * dice_val
            losses["dice"] = dice_val.item()
        else:
            losses["dice"] = None

        # Boundary Loss (confidence scales boundary weights)
        if self.boundary is not None:
            if confidence is not None:
                boundary_val = self.boundary(pred, target, confidence)
            else:
                boundary_val = self.boundary(pred, target)
            total = total + self.boundary_weight * boundary_val
            losses["boundary"] = boundary_val.item()
        else:
            losses["boundary"] = None

        losses["loss"] = total
        return losses

    def _weighted_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dice loss weighted by per-pixel confidence.
        
        """
        smooth = 1e-5
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # Weighted intersection and union
        intersection = (pred * target * confidence).sum(dim=(1, 2, 3))
        pred_sum = (pred * confidence).sum(dim=(1, 2, 3))
        target_sum = (target * confidence).sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        return (1.0 - dice).mean()


def build_loss_from_config(loss_config: dict) -> CombinedLoss:
    """
    Build a CombinedLoss from a configuration dictionary.

    """
    return CombinedLoss(
        bce_weight=loss_config.get("bce_weight", 1.0),
        dice_weight=loss_config.get("dice_weight", 1.0),
        boundary_weight=loss_config.get("boundary_weight", 0.1),
        boundary_radius=loss_config.get("boundary_radius", 3),
        boundary_factor=loss_config.get("boundary_factor", 5.0),
        from_logits=loss_config.get("from_logits", True),
    )
