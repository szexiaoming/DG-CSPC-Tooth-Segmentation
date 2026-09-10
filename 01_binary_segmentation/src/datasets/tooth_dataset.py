"""
PyTorch Dataset for 2D Panoramic X-ray Tooth Segmentation.

Reads preprocessed PNG images and masks from the data/processed/ directory.
Supports Albumentations-based data augmentation for training.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class ToothSegmentationDataset(Dataset):
    """
    Dataset for tooth semantic segmentation.

    Expects the following directory structure:
        data/processed/images/   -> preprocessed PNG images
        data/processed/masks/    -> preprocessed PNG binary masks (0=bg, 255=tooth)

    The split file is a text file with one image filename (without extension) per line.

    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        split_file: Optional[str] = None,
        image_names: Optional[list] = None,
        transform: Optional[object] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size

        # Read image names from split file or use provided list
        if image_names is not None:
            self.image_names = list(image_names)
        elif split_file is not None:
            with open(split_file, "r", encoding="utf-8") as f:
                self.image_names = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Either split_file or image_names must be provided.")

        if len(self.image_names) == 0:
            raise ValueError("No image names found.")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.image_names[idx]

        # Load image and mask as NumPy arrays
        img_path = self.image_dir / f"{name}.png"
        mask_path = self.mask_dir / f"{name}.png"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = np.array(Image.open(img_path))  # (H, W) grayscale or (H, W, C)
        mask = np.array(Image.open(mask_path))  # (H, W) values 0 or 255

        # Ensure 2D
        if image.ndim == 3:
            # Convert RGB to grayscale by taking first channel
            image = image[:, :, 0] if image.shape[2] >= 1 else image.squeeze(-1)

        # Binarize mask: 0 or 1
        mask = (mask > 127).astype(np.float32)

        # Ensure image is float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Apply Albumentations transform
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Add channel dimension if needed: (H, W) → (1, H, W)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)  # (H, W) → (1, H, W)

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "name": name,
        }


def get_transforms_train(image_size: Tuple[int, int] = (512, 512)) -> object:
    """
    Create Albumentations training augmentations.

    """
    import albumentations as A

    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(0.0, 0.05),
                rotate=(-5, 5),
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(std_range=(0.0, 0.02), p=0.3),
        ],
        is_check_shapes=False,
    )


def get_transforms_val(image_size: Tuple[int, int] = (512, 512)) -> object:
    """
    Create Albumentations validation transforms (no random augmentation).

    """
    import albumentations as A

    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
        ],
        is_check_shapes=False,
    )
