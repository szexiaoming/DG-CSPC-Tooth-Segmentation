"""
Exploratory binary SSL v2 dataset.

Uses teacher probability maps for weighted pseudo-label training.

Note: the archived weight

    w = 1 - 2 * |p - 0.5|

is highest near p = 0.5, so it behaves as an uncertainty weight rather than
a confidence weight. This experiment is not included in the final dissertation.

"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SoftPseudoDataset(Dataset):
    """
    Dataset that loads images and their soft pseudo-label probability maps.

    """

    def __init__(
        self,
        image_dir: str,
        prob_dir: str,
        image_names: Optional[List[str]] = None,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.prob_dir = Path(prob_dir)
        self.transform = transform or A.Compose([A.Resize(height=image_size[1], width=image_size[0])])
        self.image_size = image_size

        if image_names is not None:
            self.image_names = list(image_names)
        else:
            # All images that have corresponding probability maps
            prob_files = set(f.stem for f in self.prob_dir.glob("*.npy"))
            img_files = set(f.stem for f in self.image_dir.glob("*.png"))
            self.image_names = sorted(prob_files & img_files)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.image_names[idx]

        # Load image
        img_path = self.image_dir / f"{name}.png"
        img = np.array(Image.open(img_path))
        if img.ndim == 3:
            img = img[:, :, 0]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Load soft probability map
        prob_path = self.prob_dir / f"{name}.npy"
        prob = np.load(prob_path).astype(np.float32)

        # Resize probability map to match if needed
        if prob.shape != (self.image_size[0], self.image_size[1]):
            prob_img = Image.fromarray((prob * 255).astype(np.uint8))
            prob_img = prob_img.resize((self.image_size[0], self.image_size[1]), Image.NEAREST)
            prob = np.array(prob_img).astype(np.float32) / 255.0

        # Apply transform (same for image and probability map)
        transformed = self.transform(image=img, mask=prob)
        img_t = transformed["image"]
        prob_t = transformed["mask"].astype(np.float32)

        # Add channel dimension
        if img_t.ndim == 2:
            img_t = img_t[np.newaxis, ...]
        if prob_t.ndim == 2:
            prob_t = prob_t[np.newaxis, ...]

        # Hard mask: threshold at 0.5
        hard_mask = (prob_t > 0.5).astype(np.float32)

        # Confidence map: how far from 0.5 (peak confidence at extremes)
        confidence = 1.0 - 2.0 * np.abs(prob_t - 0.5)
        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            "image": torch.from_numpy(img_t.copy()).float(),
            "soft_target": torch.from_numpy(prob_t.copy()).float(),
            "mask": torch.from_numpy(hard_mask.copy()).float(),
            "confidence": torch.from_numpy(confidence.copy()).float(),
            "name": name,
        }
