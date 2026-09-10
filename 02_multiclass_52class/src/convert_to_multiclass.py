"""
Convert JSON polygon annotations to multi-class instance masks.

"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fdi_to_class(fdi_label: str) -> int:
    """Convert FDI tooth label like '11' or '51' to class index 0-51."""
    label = int(fdi_label)
    quadrant = label // 10  # 1-4 (permanent) or 5-8 (deciduous)
    position = label % 10   # 1-8 (permanent) or 1-5 (deciduous)
    if quadrant <= 4:
        # Permanent: quadrants 1-4, positions 1-8
        if position < 1 or position > 8:
            raise ValueError(f"Invalid FDI label: {fdi_label}")
        return (quadrant - 1) * 8 + (position - 1)
    else:
        # Deciduous: quadrants 5-8, positions 1-5
        if position < 1 or position > 5:
            raise ValueError(f"Invalid FDI label: {fdi_label}")
        return 32 + (quadrant - 5) * 5 + (position - 1)


def json_to_multiclass_mask(json_path: str, image_size: tuple = (512, 512)) -> np.ndarray:
    """
    Convert a JSON annotation file to a multi-class mask.

    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine original image size for scaling
    orig_h = data.get("imageHeight", image_size[0])
    orig_w = data.get("imageWidth", image_size[1])
    scale_x = image_size[1] / orig_w
    scale_y = image_size[0] / orig_h

    mask = np.zeros(image_size, dtype=np.uint8)


    shapes = data.get("shapes", [])
    for shape in shapes:
        label = shape.get("label", "")
        points = shape.get("points", [])

        if not label or not points:
            continue

        try:
            class_id = fdi_to_class(label) + 1  # 1-32, leaving 0 for background
        except ValueError:
            continue  # Skip non-FDI labels

        # Scale points to target size
        scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in points]

        # Create a single-tooth mask
        tooth_mask = Image.new("L", (image_size[1], image_size[0]), 0)
        draw = ImageDraw.Draw(tooth_mask)
        draw.polygon(scaled_points, fill=class_id)
        tooth_arr = np.array(tooth_mask, dtype=np.uint8)

        mask[tooth_arr > 0] = tooth_arr[tooth_arr > 0]

    return mask


def main():
    project_root = Path(__file__).resolve().parent.parent
    json_dir = project_root.parent / "STS24-2DXray" / "Train-Labeled" / "Masks"
    output_dir = project_root / "data" / "processed" / "masks_multiclass"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON annotation files")

    # Build FDI → class mapping for reference
    fdi_map = {}
    # Permanent (quadrants 1-4, positions 1-8)
    for quad in range(1, 5):
        for pos in range(1, 9):
            fdi = f"{quad}{pos}"
            fdi_map[fdi] = fdi_to_class(fdi) + 1
    # Deciduous (quadrants 5-8, positions 1-5)
    for quad in range(5, 9):
        for pos in range(1, 6):
            fdi = f"{quad}{pos}"
            fdi_map[fdi] = fdi_to_class(fdi) + 1

    print(f"FDI -> Class mapping (52 classes):")
    for quad in [1, 2, 3, 4]:
        teeth = [f"{quad}{pos}" for pos in range(1, 9)]
        cls = [fdi_map[t] for t in teeth]
        print(f"  Q{quad} (permanent): {teeth} → classes {cls}")
    for quad in [5, 6, 7, 8]:
        teeth = [f"{quad}{pos}" for pos in range(1, 6)]
        cls = [fdi_map[t] for t in teeth]
        print(f"  Q{quad} (deciduous): {teeth} → classes {cls}")

    all_classes = set()
    for jf in tqdm(json_files, desc="Converting"):
        mask = json_to_multiclass_mask(str(jf))
        out_path = output_dir / f"{jf.stem.replace('_Mask', '')}.png"
        Image.fromarray(mask).save(out_path)
        classes_present = set(np.unique(mask).tolist())
        all_classes.update(classes_present)

    print(f"\nAll classes present in dataset: {sorted(all_classes)}")
    print(f"Total unique tooth classes: {len(all_classes) - 1}")  # minus background
    print(f"Masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
