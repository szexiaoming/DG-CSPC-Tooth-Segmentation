"""
Generate pseudo multi-class masks for unlabeled images.

"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_labeled_positions(multiclass_dir: Path) -> dict:
    """
    Compute average centroid for each FDI class from labeled multi-class masks.

    """
    class_centroids = {c: [] for c in range(1, 53)}

    for mask_path in sorted(multiclass_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path))
        for c in range(1, 53):
            ys, xs = np.where(mask == c)
            if len(ys) > 5:  # At least a few pixels
                class_centroids[c].append((ys.mean(), xs.mean()))

    # Compute mean and std per class
    result = {}
    for c, points in class_centroids.items():
        if points:
            ys = [p[0] for p in points]
            xs = [p[1] for p in points]
            result[c] = {
                "mean_y": np.mean(ys),
                "mean_x": np.mean(xs),
                "std_y": np.std(ys) + 20,  # Add margin
                "std_x": np.std(xs) + 15,
                "count": len(points),
            }

    print(f"Computed centroids for {len(result)} tooth classes "
          f"(from {len(list(multiclass_dir.glob('*.png')))} labeled images)")
    return result


def split_instances(binary_mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Split a binary tooth mask into individual tooth instances.

    Uses distance transform + watershed.
    """
    # Close small gaps
    se = ndimage.generate_binary_structure(2, 2)
    closed = ndimage.binary_closing(binary_mask, structure=se, iterations=2)
    opened = ndimage.binary_opening(closed, structure=se, iterations=1)

    if not opened.any():
        return np.zeros_like(binary_mask, dtype=np.int32)

    # Distance transform
    distance = ndimage.distance_transform_edt(opened)

    # Find local maxima as seeds
    coords = peak_local_max(
        distance,
        min_distance=15,
        exclude_border=5,
        threshold_abs=3.0,
    )

    if len(coords) == 0:
        # Fallback: single instance
        return opened.astype(np.int32)

    # Create seed markers
    markers = np.zeros(distance.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1

    # Watershed
    labels = watershed(-distance, markers, mask=opened)

    # Filter tiny fragments
    for label_id in range(1, labels.max() + 1):
        if (labels == label_id).sum() < min_area:
            labels[labels == label_id] = 0

    # Renumber remaining labels to be sequential
    unique_labels = sorted(np.unique(labels))
    unique_labels = [l for l in unique_labels if l > 0]
    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(unique_labels, 1):
        new_labels[labels == old_id] = new_id

    return new_labels


def assign_fdi_labels(instances: np.ndarray, centroids: dict) -> np.ndarray:
    """
    Assign FDI class labels to each instance based on spatial proximity.

    """
    multiclass = np.zeros(instances.shape, dtype=np.uint8)

    unique_labels = sorted(np.unique(instances))
    unique_labels = [l for l in unique_labels if l > 0]

    if not unique_labels:
        return multiclass

    # Compute centroid of each instance
    inst_centroids = {}
    for label_id in unique_labels:
        ys, xs = np.where(instances == label_id)
        inst_centroids[label_id] = (ys.mean(), xs.mean())

    # Track which FDI classes have been assigned
    assigned_fdi = set()

    for label_id, (iy, ix) in inst_centroids.items():
        best_class = None
        best_score = float("inf")

        for c, info in centroids.items():
            if c in assigned_fdi:
                continue
            # Mahalanobis-like distance normalized by std
            dy = (iy - info["mean_y"]) / info["std_y"]
            dx = (ix - info["mean_x"]) / info["std_x"]
            dist = np.sqrt(dy**2 + dx**2)

            if dist < best_score:
                best_score = dist
                best_class = c

        # Only assign if within reasonable range
        if best_class is not None and best_score < 4.0:
            multiclass[instances == label_id] = best_class
            assigned_fdi.add(best_class)

    return multiclass


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Paths
    probs_dir = project_root / "data" / "pseudo_probs"         # SSL v2 soft probs
    multiclass_labeled_dir = project_root / "data" / "processed" / "masks_multiclass"
    image_dir = project_root / "data" / "processed" / "images"
    output_dir = project_root / "data" / "pseudo_masks_multiclass"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits to exclude labeled images
    splits_dir = project_root / "data" / "splits"
    labeled_names = set()
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        with open(splits_dir / split_file, "r") as f:
            labeled_names.update(line.strip() for line in f if line.strip())

    # Load FDI position priors from labeled data
    print("Loading FDI centroids from labeled data...")
    fdi_centroids = load_labeled_positions(multiclass_labeled_dir)

    # Process unlabeled images
    prob_files = sorted(probs_dir.glob("*.npy"))
    print(f"\nProcessing {len(prob_files)} unlabeled images...")

    stats = {"total": 0, "with_teeth": 0, "teeth_found": []}

    for prob_path in tqdm(prob_files, desc="Generating pseudo multi-class"):
        name = prob_path.stem
        if name in labeled_names:
            continue

        # Load soft probability map
        prob = np.load(prob_path)  # (512, 512), float16

        # Threshold to binary
        binary = (prob > 0.5).astype(np.uint8)

        if binary.sum() < 500:  # No meaningful tooth region
            # Save empty mask
            empty = np.zeros((512, 512), dtype=np.uint8)
            Image.fromarray(empty).save(output_dir / f"{name}.png")
            continue

        stats["with_teeth"] += 1

        # Split into individual teeth
        instances = split_instances(binary, min_area=300)

        if instances.max() == 0:
            empty = np.zeros((512, 512), dtype=np.uint8)
            Image.fromarray(empty).save(output_dir / f"{name}.png")
            continue

        # Assign FDI labels
        multiclass = assign_fdi_labels(instances, fdi_centroids)

        n_teeth = len(np.unique(multiclass)) - 1
        stats["teeth_found"].append(n_teeth)
        stats["total"] += 1

        # Save
        Image.fromarray(multiclass).save(output_dir / f"{name}.png")

    # Report
    print(f"\n=== Results ===")
    print(f"Total processed: {stats['total']}")
    print(f"Images with teeth detected: {stats['with_teeth']}")
    if stats["teeth_found"]:
        print(f"Average teeth per image: {np.mean(stats['teeth_found']):.1f} "
              f"(range: {min(stats['teeth_found'])}-{max(stats['teeth_found'])})")
        print(f"Median teeth per image: {np.median(stats['teeth_found']):.0f}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
