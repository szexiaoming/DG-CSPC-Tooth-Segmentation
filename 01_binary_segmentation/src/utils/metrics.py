"""
Evaluation metrics for tooth segmentation.

"""

import numpy as np
from typing import Tuple, Optional


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Compute Dice coefficient between two binary masks.

    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return float(dice)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Compute Intersection over Union (Jaccard index).

    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def precision_recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> Tuple[float, float]:
    """
    Compute precision and recall.

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)

    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return float(precision), float(recall)


def boundary_f1_score(
    pred: np.ndarray,
    target: np.ndarray,
    dilation_radius: int = 2,
    tolerance: int = 2,
) -> float:
    """
    Compute Boundary F1 score by extracting contours and measuring overlap
    within a tolerance band.

    """
    from scipy import ndimage

    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    # Extract boundaries via morphological gradient
    kernel = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=np.uint8)

    pred_dilated = ndimage.binary_dilation(pred, structure=kernel).astype(np.uint8)
    pred_eroded = ndimage.binary_erosion(pred, structure=kernel).astype(np.uint8)
    pred_boundary = pred_dilated - pred_eroded

    target_dilated = ndimage.binary_dilation(target, structure=kernel).astype(np.uint8)
    target_eroded = ndimage.binary_erosion(target, structure=kernel).astype(np.uint8)
    target_boundary = target_dilated - target_eroded

    # Create tolerance band around target boundary
    tolerance_kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), dtype=np.uint8)
    target_tolerance = ndimage.binary_dilation(
        target_boundary, structure=tolerance_kernel
    ).astype(np.uint8)
    pred_tolerance = ndimage.binary_dilation(
        pred_boundary, structure=tolerance_kernel
    ).astype(np.uint8)

    # True positives: prediction boundary within tolerance of target boundary
    tp = np.logical_and(pred_boundary, target_tolerance).sum()
    fp = np.logical_and(pred_boundary, 1 - target_tolerance).sum()
    fn = np.logical_and(target_boundary, 1 - pred_tolerance).sum()

    if tp + fp + fn == 0:
        return 1.0  # Both boundaries are empty

    precision_boundary = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_boundary = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision_boundary + recall_boundary == 0:
        return 0.0

    bf1 = 2 * precision_boundary * recall_boundary / (precision_boundary + recall_boundary)
    return float(bf1)


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute all standard metrics for a single prediction-mask pair.
    
    """
    # Threshold if needed
    if pred.dtype != np.bool_:
        pred = (pred > 0.5).astype(np.uint8)
    if target.dtype != np.bool_:
        target = (target > 0.5).astype(np.uint8)

    prec, rec = precision_recall(pred, target)
    bf1 = boundary_f1_score(pred, target)

    return {
        "dice": dice_coefficient(pred, target),
        "iou": iou_score(pred, target),
        "precision": prec,
        "recall": rec,
        "boundary_f1": bf1,
    }
