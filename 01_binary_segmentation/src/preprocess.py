"""
Preprocess raw 2D panoramic X-ray images and LabelMe JSON masks.

Converts:
- RGB X-ray images -> grayscale normalized PNG images
- LabelMe JSON polygon masks -> binary PNG masks (255=tooth, 0=background)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw X-ray images and JSON masks."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="STS24-2DXray",
        help="Path to the directory containing raw dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed images and masks.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Target size (height width) for resizing. Default: 512 512",
    )
    parser.add_argument(
        "--labeled_only",
        action="store_true",
        help="Only process labeled data (skip Train-Unlabeled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    return parser.parse_args()


def json_to_mask(
    json_path: str, image_height: int, image_width: int
) -> np.ndarray:
    """
    Convert a LabelMe JSON annotation to a binary segmentation mask.

    All tooth polygons (regardless of FDI label) are merged into a single foreground class (255=tooth, 0=background) for semantic segmentation.
    """
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    if not shapes:
        print(f"  Warning: No shapes found in {json_path}, returning empty mask.")

    for shape in shapes:
        points = np.array(shape["points"], dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)

    return mask


def preprocess_image(
    img_path: str,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Load, convert to grayscale, normalize, and resize an X-ray image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # Resize with bilinear interpolation
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    return img


def preprocess_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a binary mask using nearest-neighbor interpolation to preserve label boundaries.
    """
    mask = cv2.resize(
        mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST
    )
    return mask


def process_labeled_split(
    raw_dir: Path,
    out_dir: Path,
    image_size: Tuple[int, int],
    split_name: str,
    overwrite: bool,
) -> List[str]:
    """
    Process a split that contains Images/ and Masks/ subdirectories.
    """
    images_dir = raw_dir / "Images"
    masks_dir = raw_dir / "Masks"
    out_img_dir = out_dir / "images"
    out_mask_dir = out_dir / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"  Images directory not found: {images_dir}, skipping.")
        return []

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    processed_names = []

    print(f"Processing {split_name}: {len(image_files)} images")

    for img_path in tqdm(image_files, desc=split_name):
        name = img_path.stem  # e.g., "STS24_Train_Labeled_0001"
        out_img_path = out_img_dir / f"{name}.png"
        out_mask_path = out_mask_dir / f"{name}.png"

        if out_img_path.exists() and out_mask_path.exists() and not overwrite:
            processed_names.append(name)
            continue

        # Preprocess image
        try:
            img = preprocess_image(str(img_path), image_size)
            Image.fromarray(img).save(out_img_path)
        except Exception as e:
            print(f"  Error processing image {img_path}: {e}")
            continue

        # Preprocess mask
        json_files = list(masks_dir.glob(f"{name}*.json"))
        if len(json_files) > 0:
            json_path = str(json_files[0])
            # Read original dimensions from JSON or image
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            orig_h = data.get("imageHeight", image_size[0])
            orig_w = data.get("imageWidth", image_size[1])

            mask_orig = json_to_mask(json_path, orig_h, orig_w)
            mask = preprocess_mask(mask_orig, image_size)
            Image.fromarray(mask).save(out_mask_path)
        else:
            # No mask JSON found — create empty mask
            print(f"  Warning: No mask JSON for {name}, creating empty mask.")
            mask = np.zeros(image_size, dtype=np.uint8)
            Image.fromarray(mask).save(out_mask_path)

        processed_names.append(name)

    print(f"  Processed {len(processed_names)} images from {split_name}")
    return processed_names


def process_unlabeled_split(
    raw_dir: Path,
    out_img_dir: Path,
    image_size: Tuple[int, int],
    overwrite: bool,
) -> List[str]:
    """
    Process unlabeled images (no masks available).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png"))
    processed_names = []

    print(f"Processing unlabeled: {len(image_files)} images")

    for img_path in tqdm(image_files, desc="Unlabeled"):
        name = img_path.stem
        out_img_path = out_img_dir / f"{name}.png"

        if out_img_path.exists() and not overwrite:
            processed_names.append(name)
            continue

        try:
            img = preprocess_image(str(img_path), image_size)
            Image.fromarray(img).save(out_img_path)
            processed_names.append(name)
        except Exception as e:
            print(f"  Error processing image {img_path}: {e}")
            continue

    print(f"  Processed {len(processed_names)} unlabeled images")
    return processed_names


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    image_size = tuple(args.image_size)  # (H, W)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing with target size: {image_size}")
    print(f"Raw data: {raw_dir}")
    print(f"Output:   {out_dir}")

    # Process Train-Labeled (has Images/ and Masks/)
    labeled_dir = raw_dir / "Train-Labeled"
    if labeled_dir.exists():
        process_labeled_split(
            labeled_dir, out_dir, image_size, "Train-Labeled", args.overwrite
        )
    else:
        print(f"Train-Labeled not found at {labeled_dir}")

    # Process Validation-Public (has images but no masks in Testset)
    val_dir = raw_dir / "Validation-Public"
    if val_dir.exists():
        val_images_dir = val_dir
        out_img_dir = out_dir / "images"
        process_unlabeled_split(
            val_images_dir, out_img_dir, image_size, args.overwrite
        )
    else:
        print(f"Validation-Public not found at {val_dir}")

    # Process Train-Unlabeled
    if not args.labeled_only:
        unlabeled_dir = raw_dir / "Train-Unlabeled"
        if unlabeled_dir.exists():
            out_img_dir = out_dir / "images"
            process_unlabeled_split(
                unlabeled_dir, out_img_dir, image_size, args.overwrite
            )
        else:
            print(f"Train-Unlabeled not found at {unlabeled_dir}")
    else:
        print("Skipping unlabeled data (--labeled_only flag set).")

    print("\nPreprocessing complete!")
    print(f"Processed images: {out_dir / 'images'}")
    print(f"Processed masks:  {out_dir / 'masks'}")


if __name__ == "__main__":
    main()
