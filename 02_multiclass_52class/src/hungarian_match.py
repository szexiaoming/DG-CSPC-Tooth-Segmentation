"""
Hungarian algorithm for globally optimal FDI tooth assignment.

"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional


def assign_fdi_hungarian(
    instances: np.ndarray,
    fdi_priors: dict,
    max_cost: float = 100.0,
    quadrant_weight: float = 0.3,
    size_weight: float = 0.2,
) -> np.ndarray:
    """
    Assign FDI class labels to each instance using Hungarian algorithm.

    """
    multiclass = np.zeros(instances.shape, dtype=np.uint8)

    unique_labels = sorted(np.unique(instances))
    unique_labels = [l for l in unique_labels if l > 0]
    n_inst = len(unique_labels)

    if n_inst == 0:
        return multiclass

    fdi_classes = sorted(fdi_priors.keys())
    n_fdi = len(fdi_classes)

    # Compute instance features: centroid and size
    inst_features = {}
    for label_id in unique_labels:
        ys, xs = np.where(instances == label_id)
        inst_features[label_id] = {
            "centroid_y": ys.mean(),
            "centroid_x": xs.mean(),
            "area": len(ys),
            "bbox": (ys.min(), xs.min(), ys.max(), xs.max()),
        }

    # Build cost matrix: rows=instances, columns=FDI classes
    # Add dummy columns to make it square (for unmatched instances)
    n_total = max(n_inst, n_fdi)
    cost_matrix = np.full((n_total, n_total), fill_value=max_cost, dtype=np.float64)

    for i, label_id in enumerate(unique_labels):
        feat = inst_features[label_id]
        iy, ix = feat["centroid_y"], feat["centroid_x"]
        area = feat["area"]
        # Determine quadrant: upper=Q1+Q2 (y<256), lower=Q3+Q4 (y>=256)
        # left=Q2+Q3 (x<256), right=Q1+Q4 (x>=256) for 512x512
        inst_quadrant = _get_quadrant(iy, ix)

        for j, c in enumerate(fdi_classes):
            info = fdi_priors[c]

            # 1. Spatial (Mahalanobis) distance
            dy = (iy - info["mean_y"]) / max(info["std_y"], 5.0)
            dx = (ix - info["mean_x"]) / max(info["std_x"], 5.0)
            spatial_cost = np.sqrt(dy**2 + dx**2)

            # 2. Size mismatch penalty
            if "mean_area" in info and info.get("std_area", 0) > 0:
                da = (area - info["mean_area"]) / max(info["std_area"], 1.0)
                size_cost = abs(da) * size_weight
            else:
                size_cost = 0.0

            # 3. Quadrant mismatch penalty
            fdi_quadrant = _fdi_to_quadrant(c)
            quad_cost = quadrant_weight if inst_quadrant != fdi_quadrant else 0.0

            # Total cost
            cost_matrix[i, j] = spatial_cost + size_cost + quad_cost

    # For dummy rows (if more FDI classes than instances), use 0 cost
    # so they match with nothing (handled by max_cost check later)

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Assign based on results
    for i, j in zip(row_ind, col_ind):
        if i >= n_inst or j >= n_fdi:
            continue  # Dummy match
        if cost_matrix[i, j] >= max_cost:
            continue  # Cost too high, reject assignment

        label_id = unique_labels[i]
        fdi_class = fdi_classes[j]
        multiclass[instances == label_id] = fdi_class

    return multiclass


def _get_quadrant(y: float, x: float, h: int = 512, w: int = 512) -> int:
    """Determine image quadrant: 1=UR, 2=UL, 3=LL, 4=LR."""
    if y < h / 2:
        return 1 if x >= w / 2 else 2  # Upper Right=Q1, Upper Left=Q2
    else:
        return 3 if x < w / 2 else 4   # Lower Left=Q3, Lower Right=Q4


def _fdi_to_quadrant(fdi: int) -> int:
    """Map FDI tooth number to expected quadrant.

    FDI numbering: Q1=11-18 (UR), Q2=21-28 (UL), Q3=31-38 (LL), Q4=41-48 (LR)
    Deciduous: Q5=51-55 (UR), Q6=61-65 (UL), Q7=71-75 (LL), Q8=81-85 (LR)

    """
    if 11 <= fdi <= 18:
        return 1  # Upper Right (permanent)
    elif 21 <= fdi <= 28:
        return 2  # Upper Left (permanent)
    elif 31 <= fdi <= 38:
        return 3  # Lower Left (permanent)
    elif 41 <= fdi <= 48:
        return 4  # Lower Right (permanent)
    elif 51 <= fdi <= 55:
        return 1  # Upper Right (deciduous, analog to Q1)
    elif 61 <= fdi <= 65:
        return 2  # Upper Left (deciduous, analog to Q2)
    elif 71 <= fdi <= 75:
        return 3  # Lower Left (deciduous, analog to Q3)
    elif 81 <= fdi <= 85:
        return 4  # Lower Right (deciduous, analog to Q4)
    return 0


def compute_fdi_priors(multiclass_dir) -> dict:
    """
    Compute comprehensive FDI priors from labeled multi-class masks.

    Includes centroid (mean, std), area (mean, std), and tooth count
    for each FDI class.
    """
    from pathlib import Path
    from PIL import Image
    import numpy as np

    multiclass_dir = Path(multiclass_dir)
    class_data = {c: {"centroids": [], "areas": []} for c in range(1, 53)}

    for mask_path in sorted(multiclass_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path))
        for c in range(1, 53):
            ys, xs = np.where(mask == c)
            if len(ys) > 10:
                class_data[c]["centroids"].append((ys.mean(), xs.mean()))
                class_data[c]["areas"].append(len(ys))

    result = {}
    for c in range(1, 53):
        data = class_data[c]
        if data["centroids"]:
            cys = [p[0] for p in data["centroids"]]
            cxs = [p[1] for p in data["centroids"]]
            areas = data["areas"]
            result[c] = {
                "mean_y": float(np.mean(cys)),
                "mean_x": float(np.mean(cxs)),
                "std_y": float(np.std(cys)) + 15.0,
                "std_x": float(np.std(cxs)) + 10.0,
                "mean_area": float(np.mean(areas)),
                "std_area": float(np.std(areas)) + 50.0,
                "count": len(data["centroids"]),
            }

    return result


def compare_assignments(
    instances: np.ndarray,
    greedy_result: np.ndarray,
    hungarian_result: np.ndarray,
    fdi_priors: dict,
) -> dict:
    """
    Compare greedy vs Hungarian assignment quality.

    """
    greedy_classes = set(np.unique(greedy_result)) - {0}
    hungarian_classes = set(np.unique(hungarian_result)) - {0}

    # Count how many unique FDI classes each method assigned
    # More classes = better coverage
    return {
        "greedy_n_classes": len(greedy_classes),
        "hungarian_n_classes": len(hungarian_classes),
        "hungarian_improvement": len(hungarian_classes) - len(greedy_classes),
        "greedy_classes": sorted(greedy_classes),
        "hungarian_classes": sorted(hungarian_classes),
    }
