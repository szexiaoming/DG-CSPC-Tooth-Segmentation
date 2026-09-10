"""
Convert multi-class tooth masks to YOLO detection format.

For each labeled mask (mask values 1-32 -> YOLO class 0-31):
1. Extract each tooth instance (connected component per class)
2. Compute bounding box
3. Save as YOLO format: class_id xc yc w h (normalized 0-1)

Also creates a YOLO dataset.yaml config.

"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mask_to_yolo_boxes(mask: np.ndarray, img_h: int, img_w: int) -> list:
    """
    Extract YOLO-format bounding boxes from multi-class mask.

    The mask uses pixel values 1-32 (corresponding to FDI 11-48 in order).
    YOLO class = mask_value - 1 (i.e., mask value 1 -> class 0 for FDI 11).

    If the mask dimensions differ from img_h/img_w, the mask is resized first.

    """
    mask_h, mask_w = mask.shape[:2]

    # Resize mask to match image dimensions if they differ
    if mask_h != img_h or mask_w != img_w:
        mask = np.array(
            Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
        )
        mask_h, mask_w = img_h, img_w

    boxes = []
    for mask_val in np.unique(mask):
        if mask_val == 0:
            continue
        ys, xs = np.where(mask == mask_val)
        if len(ys) < 20:
            continue

        # All teeth -> YOLO class 0 (1-class detector: "tooth" only)
        # Hungarian matching assigns FDI later based on spatial position
        yolo_cls = 0

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # Expand bbox slightly for YOLO context (clamped to image bounds)
        pad = 5
        y1 = max(0, y1 - pad)
        x1 = max(0, x1 - pad)
        y2 = min(mask_h, y2 + pad)
        x2 = min(mask_w, x2 + pad)

        # Normalize by image dimensions
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        xc = ((x1 + x2) / 2.0) / img_w
        yc = ((y1 + y2) / 2.0) / img_h

        # Safety clamp to [0, 1]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        # Skip degenerate boxes
        if bw <= 0.001 or bh <= 0.001:
            continue

        boxes.append((yolo_cls, xc, yc, bw, bh))

    return boxes


def main():
    project_root = Path(__file__).resolve().parent.parent

    mask_dir = project_root / "data" / "processed" / "masks_multiclass"
    image_dir = project_root / "data" / "processed" / "images"
    splits_dir = project_root / "data" / "splits"

    # YOLO dataset dirs
    yolo_dir = project_root / "data" / "yolo_dataset"
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (yolo_dir / sub).mkdir(parents=True, exist_ok=True)

    # Load splits
    train_names = set()
    val_names = set()
    with open(splits_dir / "train.txt") as f:
        train_names = {line.strip() for line in f if line.strip()}
    with open(splits_dir / "val.txt") as f:
        val_names = {line.strip() for line in f if line.strip()}

    # Also add test set to training (YOLO needs more data)
    with open(splits_dir / "test.txt") as f:
        test_names = {line.strip() for line in f if line.strip()}

    # Use train+val+test for YOLO training (only 30 images total)
    all_labeled = train_names | val_names | test_names
    # Reserve 5 for val
    val_subset = set(list(all_labeled)[:5])
    train_subset = all_labeled - val_subset

    print(f"YOLO train: {len(train_subset)}, val: {len(val_subset)}")

    # FDI class names: all teeth → class "tooth" (1-class detector)
    fdi_names = ["tooth"]

    for split, names in [("train", train_subset), ("val", val_subset)]:
        for name in tqdm(names, desc=f"Converting {split}"):
            # Copy image
            img_src = image_dir / f"{name}.png"
            if not img_src.exists():
                img_src = image_dir / f"{name}.jpg"
            img = Image.open(img_src)
            img_h, img_w = img.height, img.width
            img.save(yolo_dir / "images" / split / f"{name}.png")

            # Convert mask
            mask_path = mask_dir / f"{name}.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                boxes = mask_to_yolo_boxes(mask, img_h, img_w)
            else:
                boxes = []

            # Write YOLO labels
            label_path = yolo_dir / "labels" / split / f"{name}.txt"
            with open(label_path, "w") as f:
                for cls_id, xc, yc, bw, bh in boxes:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # Create dataset.yaml
    yaml_content = f

    for i, name in enumerate(fdi_names):
        yaml_content += f"  {i}: {name}\n"

    with open(yolo_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content.strip())

    print(f"\nDataset created at: {yolo_dir}")
    print(f"Train images: {len(train_subset)}, Val images: {len(val_subset)}")
    print(f"Classes: 1 (tooth detector, FDI assigned by Hungarian)")
    print(f"\nNext: python src/train_yolo.py")


if __name__ == "__main__":
    main()
