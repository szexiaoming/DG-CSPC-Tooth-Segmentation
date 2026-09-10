"""
Create train/validation/test splits from the labeled dataset.

Splits are saved as text files listing image names (one per line, without extension).
Supports random split with a fixed seed, and optional patient-level splitting.
"""

import argparse
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data splits for tooth segmentation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed images/ and masks/.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/splits",
        help="Output directory for split text files.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (must sum to 1.0). Default: 0.7 0.15 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--patient_level",
        action="store_true",
        help="Split by patient ID if possible.",
    )
    return parser.parse_args()


def extract_patient_id(name: str) -> str:
    """
    Extract a patient-level identifier from an image name.

    """
    # For STS 2024 format: STS24_Train_Labeled_XXXX
    # The numeric suffix can serve as patient ID
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]  # Use the number as patient ID
    return name


def make_splits(
    names: List[str],
    ratios: List[float],
    seed: int,
    patient_level: bool,
) -> tuple:
    """
    Create random train/val/test splits.

    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {ratios}"

    random.seed(seed)
    names_sorted = sorted(names)

    if patient_level:
        # Group by patient ID
        patient_to_images = {}
        for name in names_sorted:
            pid = extract_patient_id(name)
            patient_to_images.setdefault(pid, []).append(name)

        patient_ids = sorted(patient_to_images.keys())
        random.shuffle(patient_ids)

        n_total = len(patient_ids)
        n_train = max(1, round(n_total * ratios[0]))
        n_val = max(1, round(n_total * ratios[1]))
        n_test = n_total - n_train - n_val

        train_pids = patient_ids[:n_train]
        val_pids = patient_ids[n_train : n_train + n_val]
        test_pids = patient_ids[n_train + n_val :]

        train_names = [n for pid in train_pids for n in patient_to_images[pid]]
        val_names = [n for pid in val_pids for n in patient_to_images[pid]]
        test_names = [n for pid in test_pids for n in patient_to_images[pid]]

        print(f"Patient-level split: {n_total} patients")
        print(f"  Train patients: {n_train}, images: {len(train_names)}")
        print(f"  Val patients:   {n_val}, images: {len(val_names)}")
        print(f"  Test patients:  {n_test}, images: {len(test_names)}")
    else:
        # Random image-level split
        names_shuffled = names_sorted.copy()
        random.shuffle(names_shuffled)

        n_total = len(names_shuffled)
        n_train = max(1, round(n_total * ratios[0]))
        n_val = max(1, round(n_total * ratios[1]))
        n_test = n_total - n_train - n_val

        train_names = sorted(names_shuffled[:n_train])
        val_names = sorted(names_shuffled[n_train : n_train + n_val])
        test_names = sorted(names_shuffled[n_train + n_val :])

        print(f"Image-level split: {n_total} total images")
        print(f"  Train: {n_train}")
        print(f"  Val:   {n_val}")
        print(f"  Test:  {n_test}")

    return train_names, val_names, test_names


def save_split(names: List[str], filepath: Path) -> None:
    """Save a list of names to a text file, one per line."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"  Saved {len(names)} names → {filepath}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    masks_dir = data_dir / "masks"
    out_dir = Path(args.out_dir)

    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Masks directory not found: {masks_dir}\n"
            f"Please run preprocess.py first."
        )

    # Get all image names that have masks
    mask_files = sorted(masks_dir.glob("*.png"))
    names = [f.stem for f in mask_files]

    if len(names) == 0:
        raise ValueError(f"No mask files found in {masks_dir}")

    print(f"Found {len(names)} labeled images with masks.")

    # Create splits
    train, val, test = make_splits(names, args.ratios, args.seed, args.patient_level)

    # Save split files
    print("\nSaving split files:")
    save_split(train, out_dir / "train.txt")
    save_split(val, out_dir / "val.txt")
    save_split(test, out_dir / "test.txt")

    print("\nDone! Splits saved to", out_dir)


if __name__ == "__main__":
    main()
