"""
YOLO + SAM pseudo label pipeline.

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hungarian_match import compute_fdi_priors

# YOLO 1-class detector: class 0 = tooth. FDI assigned by Hungarian matching.
YOLO_TO_FDI = {0: 0}  # 0 = unknown FDI, Hungarian will reassign


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo-model", type=str, required=True)
    p.add_argument("--sam-checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--yolo-conf", type=float, default=0.25,
                   help="YOLO confidence threshold")
    p.add_argument("--yolo-iou", type=float, default=0.45,
                   help="YOLO NMS IoU threshold")
    return p.parse_args()


def detect_teeth_yolo(yolo_model, image_bgr, conf=0.25, iou=0.45) -> List[Dict]:
    """
    Run YOLO detection on one image.

    """
    results = yolo_model(image_bgr, conf=conf, iou=iou, verbose=False)

    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())

            detections.append({
                "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                "yolo_class": cls_id,
                "fdi_class": YOLO_TO_FDI.get(cls_id, 0),
                "confidence": conf,
            })

    return detections


def sam_refine_boxes(
    sam_model,
    image_rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    SAM box-prompt refinement. Returns one mask per box.
    """
    from segment_anything import SamPredictor

    if not boxes:
        return []

    predictor = SamPredictor(sam_model)
    predictor.set_image(image_rgb)

    input_boxes = torch.tensor(boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        input_boxes, image_rgb.shape[:2]
    )

    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    result = []
    for i in range(masks.shape[0]):
        m = masks[i, 0].cpu().numpy().astype(np.uint8)
        result.append(m)

    return result


def merge_yolo_sam(
    detections: List[Dict],
    sam_masks: List[np.ndarray],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Merge SAM masks with YOLO FDI predictions into multi-class mask.

    """
    h, w = image_shape
    # Each tooth gets its FDI class initially from YOLO
    multiclass = np.zeros((h, w), dtype=np.int32)

    for det, mask in zip(detections, sam_masks):
        fdi = det["fdi_class"]
        if fdi == 0:
            continue
        # Only assign pixels within this mask
        multiclass[mask > 0] = fdi

    return multiclass


def build_instances_from_masks(sam_masks: List[np.ndarray], h: int, w: int) -> np.ndarray:
    """Convert SAM masks to instance labels (1, 2, 3, ...)."""
    instances = np.zeros((h, w), dtype=np.int32)
    for i, mask in enumerate(sam_masks):
        instances[mask > 0] = i + 1
    return instances


def classify_and_assign(instances: np.ndarray, fdi_priors: dict) -> np.ndarray:
    """
    Two-stage FDI assignment: classify each instance as permanent/deciduous,
    then run Hungarian separately for each group.

    Classification: for each instance, compare best spatial-match cost against
    permanent (FDI 1-32) vs deciduous (FDI 33-52) priors. Smaller area biases toward deciduous.
    """
    from src.hungarian_match import assign_fdi_hungarian

    h, w = instances.shape

    # Split priors
    permanent_priors = {k: v for k, v in fdi_priors.items() if 1 <= k <= 32}
    deciduous_priors = {k: v for k, v in fdi_priors.items() if 33 <= k <= 52}

    # Classify each instance
    unique_labels = sorted([lab for lab in np.unique(instances) if lab > 0])

    if not unique_labels:
        return np.zeros((h, w), dtype=np.uint8)

    perm_labels = []
    decid_labels = []

    for label_id in unique_labels:
        ys, xs = np.where(instances == label_id)
        area = len(ys)
        iy, ix = ys.mean(), xs.mean()

        # Best spatial cost against permanent priors
        best_perm = float("inf")
        for c, info in permanent_priors.items():
            dy = (iy - info["mean_y"]) / max(info["std_y"], 5.0)
            dx = (ix - info["mean_x"]) / max(info["std_x"], 5.0)
            cost = np.sqrt(dy**2 + dx**2)
            if cost < best_perm:
                best_perm = cost

        # Best spatial cost against deciduous priors
        best_decid = float("inf")
        for c, info in deciduous_priors.items():
            dy = (iy - info["mean_y"]) / max(info["std_y"], 5.0)
            dx = (ix - info["mean_x"]) / max(info["std_x"], 5.0)
            cost = np.sqrt(dy**2 + dx**2)
            if cost < best_decid:
                best_decid = cost

        # Area heuristic: very small teeth → bias toward deciduous
        if area < 800:
            best_decid *= 0.7

        if best_perm <= best_decid:
            perm_labels.append(label_id)
        else:
            decid_labels.append(label_id)

    # Build separate instance maps
    perm_map = np.zeros((h, w), dtype=np.int32)
    for new_id, old_id in enumerate(perm_labels, 1):
        perm_map[instances == old_id] = new_id

    decid_map = np.zeros((h, w), dtype=np.int32)
    for new_id, old_id in enumerate(decid_labels, 1):
        decid_map[instances == old_id] = new_id

    # Hungarian for each group
    result = np.zeros((h, w), dtype=np.uint8)

    if perm_labels:
        perm_result = assign_fdi_hungarian(
            perm_map, permanent_priors,
            max_cost=5.0, quadrant_weight=0.3, size_weight=0.15,
        )
        result[perm_result > 0] = perm_result[perm_result > 0]

    if decid_labels:
        decid_result = assign_fdi_hungarian(
            decid_map, deciduous_priors,
            max_cost=5.0, quadrant_weight=0.3, size_weight=0.15,
        )
        result[decid_result > 0] = decid_result[decid_result > 0]

    return result


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    image_dir = project_root / "data" / "processed" / "images"
    labeled_dir = project_root / "data" / "processed" / "masks_multiclass"
    output_dir = project_root / "data" / "pseudo_masks_yolo_sam"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    splits_dir = project_root / "data" / "splits"
    labeled_names = set()
    for sf in ["train.txt", "val.txt", "test.txt"]:
        with open(splits_dir / sf, "r") as f:
            labeled_names.update(line.strip() for line in f if line.strip())

    # Load YOLO
    print("Loading YOLO model...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        return

    yolo = YOLO(args.yolo_model)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    yolo.to(device)

    # Load SAM
    print("Loading SAM model...")
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam.to(device)
    sam.eval()

    # Load FDI priors for Hungarian refinement
    print("Loading FDI priors...")
    fdi_priors = compute_fdi_priors(labeled_dir)

    # Get unlabeled images
    all_images = sorted(p.stem for p in image_dir.glob("*.png"))
    unlabeled = [n for n in all_images if n not in labeled_names]
    print(f"\nProcessing {len(unlabeled)} unlabeled images...")

    quality_scores = {}
    stats = {"total": 0, "teeth_found": []}

    for name in tqdm(unlabeled, desc="YOLO+SAM"):
        img_path = image_dir / f"{name}.png"
        if not img_path.exists():
            img_path = image_dir / f"{name}.jpg"
        if not img_path.exists():
            continue

        # Load image
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_bgr = cv2.resize(image_bgr, (512, 512))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Step 1: YOLO detection
        detections = detect_teeth_yolo(yolo, image_bgr, args.yolo_conf, args.yolo_iou)

        if not detections:
            empty = np.zeros((512, 512), dtype=np.uint8)
            Image.fromarray(empty).save(output_dir / f"{name}.png")
            quality_scores[name] = {"quality": 0.0, "n_teeth": 0}
            continue

        # Step 2: SAM refinement
        boxes = [d["bbox"] for d in detections]
        try:
            sam_masks = sam_refine_boxes(sam, image_rgb, boxes, device)
        except Exception as e:
            print(f"  SAM error on {name}: {e}")
            sam_masks = []

        if not sam_masks:
            empty = np.zeros((512, 512), dtype=np.uint8)
            Image.fromarray(empty).save(output_dir / f"{name}.png")
            quality_scores[name] = {"quality": 0.0, "n_teeth": 0}
            continue

        # Step 3: Build instances and assign FDI
        h, w = 512, 512
        instances = build_instances_from_masks(sam_masks, h, w)

        # Two-stage: classify permanent/deciduous → Hungarian per group
        multiclass = classify_and_assign(instances, fdi_priors)

        n_teeth = len(np.unique(multiclass)) - 1
        stats["total"] += 1
        stats["teeth_found"].append(n_teeth)

        # Quality score (up to 52 teeth for mixed dentition)
        if 20 <= n_teeth <= 52:
            q = 0.7 + 0.3 * min(1.0, n_teeth / 48.0)
        elif 14 <= n_teeth < 20:
            q = 0.4 + 0.3 * min(1.0, n_teeth / 48.0)
        else:
            q = max(0.1, n_teeth / 20.0)

        quality_scores[name] = {
            "quality": round(q, 3),
            "n_teeth": n_teeth,
            "method": "yolo_sam_hungarian",
        }

        Image.fromarray(multiclass).save(output_dir / f"{name}.png")

    # Save quality scores
    with open(output_dir / "quality_scores.json", "w") as f:
        json.dump(quality_scores, f, indent=2)

    # Report
    print(f"\n=== YOLO+SAM Pseudo Label Results ===")
    print(f"Total: {stats['total']}")
    if stats["teeth_found"]:
        print(f"Avg teeth/img: {np.mean(stats['teeth_found']):.1f} "
              f"(median: {np.median(stats['teeth_found']):.0f})")

    good = sum(1 for s in quality_scores.values() if s["quality"] >= 0.6)
    ok = sum(1 for s in quality_scores.values() if 0.4 <= s["quality"] < 0.6)
    bad = sum(1 for s in quality_scores.values() if s["quality"] < 0.4)
    print(f"Quality: Good={good}, OK={ok}, Bad={bad}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
