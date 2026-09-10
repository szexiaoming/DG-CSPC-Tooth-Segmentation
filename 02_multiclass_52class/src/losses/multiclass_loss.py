"""
Multi-class combined loss for tooth instance segmentation.

"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """
    Multi-class Dice loss with optional class weighting.

    """

    def __init__(
        self,
        num_classes: int = 53,
        smooth: float = 1e-5,
        ignore_bg: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        # Per-class Dice
        intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.ignore_bg:
            # Average over tooth classes only (skip class 0 = background)
            dice_per_class = dice_per_class[1:]

        return 1.0 - dice_per_class.mean()


class CombinedMulticlassLoss(nn.Module):
    """
    CrossEntropy + multi-class Dice loss.

    """

    def __init__(
        self,
        num_classes: int = 53,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_bg_dice: bool = True,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            ignore_index=-1,  # Don't ignore any class in loss
        )
        self.dice_loss = MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_bg=ignore_bg_dice,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice

        return {
            "loss": total,
            "ce_loss": ce.detach(),
            "dice_loss": dice.detach(),
        }


def compute_class_weights(
    dataloader,
    num_classes: int = 53,
    device: str = "cpu",
    hard_class_boost: dict = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a dataloader.

    """
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    total_pixels = 0

    for batch in dataloader:
        masks = batch["mask"]  # (B, H, W)
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()
        total_pixels += masks.numel()

    # Avoid division by zero: add 1 to each count
    class_counts = class_counts + 1.0
    weights = total_pixels / (num_classes * class_counts)

    # Apply hard class boost
    if hard_class_boost:
        for c, factor in hard_class_boost.items():
            if c < len(weights):
                weights[c] *= factor

    weights = weights / weights.mean()  # Normalize to mean=1.0

    return weights.to(device)


def build_multiclass_loss(config: dict, device: str = "cpu") -> nn.Module:
    """
    Build CombinedMulticlassLoss from config.
    
    """
    loss_cfg = config.get("loss", {})
    num_classes = config.get("model", {}).get("num_classes", 53)

    class_weights = None
    if loss_cfg.get("use_class_weights", False):
        # Will be computed later after dataloader is created
        class_weights = None

    return CombinedMulticlassLoss(
        num_classes=num_classes,
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        class_weights=class_weights,
        ignore_bg_dice=True,
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )
