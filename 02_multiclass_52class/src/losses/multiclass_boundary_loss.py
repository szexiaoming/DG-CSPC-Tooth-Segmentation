"""
Multi-class boundary loss for tooth instance segmentation.

"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def extract_multiclass_boundary(
    mask: np.ndarray,
    num_classes: int = 53,
    radius: int = 2,
) -> np.ndarray:
    """
    Extract boundary regions from a multi-class mask.

    """
    boundary = np.zeros(mask.shape, dtype=np.uint8)

    for c in range(num_classes):
        # For each class, find its boundary with everything else
        class_mask = (mask == c).astype(np.uint8)
        if class_mask.sum() == 0:
            continue

        # Dilate the class mask
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        dilated = ndimage.binary_dilation(class_mask, structure=kernel, iterations=radius)
        eroded = ndimage.binary_erosion(class_mask, structure=kernel, iterations=radius)

        # Boundary = pixels in dilated but not in eroded
        class_boundary = dilated.astype(np.uint8) - eroded.astype(np.uint8)
        boundary = np.maximum(boundary, class_boundary)

    return boundary.astype(np.float32)


def build_boundary_weight_map(
    mask: torch.Tensor,
    num_classes: int = 53,
    boundary_radius: int = 2,
    boundary_weight: float = 5.0,
) -> torch.Tensor:
    """
    Build per-pixel weight map emphasizing boundary regions.

    """
    B = mask.shape[0]
    mask_np = mask.detach().cpu().numpy()
    weight_maps = []

    for b in range(B):
        boundary = extract_multiclass_boundary(
            mask_np[b], num_classes=num_classes, radius=boundary_radius
        )
        # Dilate boundary to create a band
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary_band = ndimage.binary_dilation(
            boundary.astype(np.uint8), structure=kernel, iterations=boundary_radius
        ).astype(np.float32)

        weight = 1.0 + boundary_band * (boundary_weight - 1.0)
        weight_maps.append(weight)

    weight_map = np.stack(weight_maps)[:, np.newaxis, :, :]  # (B, 1, H, W)
    return torch.from_numpy(weight_map).to(mask.device)


def _build_boundary_weight_gpu(
    target: torch.Tensor,
    boundary_radius: int = 2,
    boundary_weight: float = 5.0,
) -> torch.Tensor:
    """
    GPU-only boundary weight map builder using max-pooling for speed.

    """
    B, H, W = target.shape
    device = target.device

    # Binary mask: tooth vs background
    binary = (target > 0).float().unsqueeze(1)  # (B, 1, H, W)

    kernel_size = 2 * boundary_radius + 1
    padding = boundary_radius

    # Dilation  max_pool on binary mask
    dilated = F.max_pool2d(
        binary,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    # Erosion  1 - max_pool(1 - binary)
    eroded = 1.0 - F.max_pool2d(
        1.0 - binary,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    # Boundary band = dilated - eroded
    boundary_band = (dilated - eroded).clamp(0, 1)  # (B, 1, H, W)

    # Expand boundary to create a wider band
    if boundary_radius > 2:
        boundary_band = F.max_pool2d(
            boundary_band,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    # Weight map: 1.0 for interior, boundary_weight for edges
    weight_map = 1.0 + boundary_band * (boundary_weight - 1.0)

    return weight_map


class MulticlassBoundaryLoss(nn.Module):
    """
    Boundary-aware multi-class cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int = 53,
        boundary_radius: int = 2,
        boundary_weight: float = 5.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.boundary_radius = boundary_radius
        self.boundary_weight = boundary_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        weight_map = _build_boundary_weight_gpu(
            target,
            boundary_radius=self.boundary_radius,
            boundary_weight=self.boundary_weight,
        )  # (B, 1, H, W)

        # Per-pixel CE loss
        ce_per_pixel = F.cross_entropy(
            logits, target,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )  # (B, H, W)

        # Weight by boundary map
        weighted_ce = ce_per_pixel * weight_map.squeeze(1)  # (B, H, W)

        # Average
        loss = weighted_ce.mean()

        # Compute fraction of boundary pixels (for monitoring)
        boundary_frac = ((weight_map > 1.0).float().mean()).detach()

        return {
            "loss": loss,
            "boundary_fraction": boundary_frac,
        }


class BoundaryEnhancedMulticlassLoss(nn.Module):
    """
    Combined loss: CE + Dice + Boundary.

    """

    def __init__(
        self,
        num_classes: int = 53,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        boundary_radius: int = 2,
        boundary_emphasis: float = 5.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            ignore_index=-1,
        )
        self.dice_loss = _MulticlassDiceLoss(num_classes=num_classes, ignore_bg=True)
        self.boundary_loss = MulticlassBoundaryLoss(
            num_classes=num_classes,
            boundary_radius=boundary_radius,
            boundary_weight=boundary_emphasis,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = self.ce_loss(logits, target)
        dice_out = self.dice_loss(logits, target)
        dice = dice_out["loss"] if isinstance(dice_out, dict) else dice_out
        bnd_out = self.boundary_loss(logits, target)
        bnd = bnd_out["loss"]

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.boundary_weight * bnd
        )

        return {
            "loss": total,
            "ce_loss": ce.detach(),
            "dice_loss": dice.detach(),
            "boundary_loss": bnd.detach(),
            "boundary_fraction": bnd_out.get("boundary_fraction", torch.tensor(0.0)),
        }


class _MulticlassDiceLoss(nn.Module):
    """
    Internal multi-class Dice loss (same as original but self-contained).
    
    """

    def __init__(self, num_classes: int = 53, smooth: float = 1e-5, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.ignore_bg:
            dice_per_class = dice_per_class[1:]

        return {"loss": 1.0 - dice_per_class.mean()}


def build_boundary_multiclass_loss(config: dict, device: str = "cpu") -> nn.Module:
    """
    Build BoundaryEnhancedMulticlassLoss from config.
    
    """
    loss_cfg = config.get("loss", {})
    num_classes = config.get("model", {}).get("num_classes", 53)

    class_weights = None
    if loss_cfg.get("use_class_weights", False):
        class_weights = None  # Computed later from dataloader

    return BoundaryEnhancedMulticlassLoss(
        num_classes=num_classes,
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        boundary_weight=loss_cfg.get("boundary_weight", 0.5),
        class_weights=class_weights,
        boundary_radius=loss_cfg.get("boundary_radius", 2),
        boundary_emphasis=loss_cfg.get("boundary_emphasis", 5.0),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )
