"""
Boundary-aware loss functions for improving tooth boundary segmentation.

Implements:
- Boundary Loss (distance-transform-based) via MONAI
- Simple Boundary BCE: weighted BCE focusing on boundary regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import numpy as np
from scipy import ndimage


def extract_boundary(mask: np.ndarray, dilation_radius: int = 2) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask using morphological gradient.

    Boundary = dilation(mask) - erosion(mask)

    """
    kernel = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=np.uint8)
    dilated = ndimage.binary_dilation(mask, structure=kernel)
    eroded = ndimage.binary_erosion(mask, structure=kernel)
    boundary = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    return boundary.astype(np.float32)


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    Compute normalized distance transform for boundary loss.

    For each pixel, computes the signed distance to the nearest boundary:
    - Positive inside the object (tooth)
    - Negative outside the object (background)
    Values are normalized to roughly [-1, 1] range.

    """
    mask_bool = mask.astype(np.bool_)

    # Distance to background (inside object is positive)
    dist_to_bg = ndimage.distance_transform_edt(mask_bool)
    # Distance to object (outside object is positive)
    dist_to_obj = ndimage.distance_transform_edt(~mask_bool)

    # Signed distance: positive inside, negative outside
    signed_dist = dist_to_bg - dist_to_obj

    # Normalize to roughly [-1, 1]
    max_dist = max(dist_to_bg.max(), dist_to_obj.max())
    if max_dist > 0:
        signed_dist = signed_dist / max_dist

    return signed_dist.astype(np.float32)


class BoundaryLoss(nn.Module):
    """
    Boundary loss that encourages predictions to match ground truth near tooth boundaries.

    Uses a boundary weight map where pixels near the boundary have higher weight.
    Can be combined with BCE or Dice loss.

    Loss = mean( boundary_weight * BCE(pred, target) )

    where boundary_weight is larger for pixels near the tooth boundary.
    """

    def __init__(
        self,
        boundary_radius: int = 3,
        boundary_weight_factor: float = 5.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.boundary_radius = boundary_radius
        self.boundary_weight_factor = boundary_weight_factor
        self.from_logits = from_logits

    def _compute_weight_map(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create a per-pixel weight map emphasizing boundary regions.

        """
        # Work with numpy for morphological ops
        mask_np = mask.detach().cpu().numpy()
        batch_size = mask_np.shape[0]

        weight_maps = []
        for i in range(batch_size):
            m = mask_np[i].squeeze()  # (H, W)
            boundary = extract_boundary(m, dilation_radius=self.boundary_radius)

            # Dilate boundary to create a band
            kernel = np.ones(
                (self.boundary_radius * 2 + 1, self.boundary_radius * 2 + 1),
                dtype=np.uint8,
            )
            boundary_band = ndimage.binary_dilation(
                boundary.astype(np.uint8), structure=kernel
            ).astype(np.float32)

            # Base weight = 1.0, higher near boundaries
            weight = 1.0 + boundary_band * (self.boundary_weight_factor - 1.0)
            weight_maps.append(weight)

        weight_map = np.stack(weight_maps)  # (N, H, W)
        if mask.dim() == 4:
            weight_map = weight_map[:, np.newaxis, :, :]  # (N, 1, H, W)

        return torch.from_numpy(weight_map).to(mask.device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.from_logits:
            pred_sigmoid = torch.sigmoid(pred)
        else:
            pred_sigmoid = pred

        # Compute per-pixel BCE
        bce = F.binary_cross_entropy(pred_sigmoid, target, reduction="none")

        # Compute boundary weight map
        weight_map = self._compute_weight_map(target)

        # Combine boundary weight with confidence (if provided)
        if confidence is not None:
            effective_weight = weight_map * confidence
            weighted_bce = bce * effective_weight
            # Normalize by mean weight to keep scale consistent
            return weighted_bce.sum() / (effective_weight.sum() + 1e-8)
        else:
            weighted_bce = bce * weight_map
            return weighted_bce.mean()
