# Detection-Guided Semi-Supervised 52-Class Tooth Segmentation in Panoramic X-rays

Semi-supervised pipeline that segments all 52 teeth (32 permanent FDI 11–48 + 20 deciduous FDI 51–85)
in panoramic X-rays from only 21 labelled images. A YOLO + SAM + Hungarian-matching stage generates
pseudo-labels on unlabelled images, which then drive iterative self-training of a lightweight U-Net.
The final **DG-CSPC** model reaches **Dice 0.610** on the 5-image test set (98% over the supervised
baseline); at inference only the U-Net is used — no detector or SAM.

---

## Quick start

### 1. Get the data and weights

The dataset and model weights (~3.3 GB) are **not** in this package. Download them from Google Drive
and copy them back into the matching folders:

> **Google Drive:** https://drive.google.com/drive/folders/1BNadgVK_0DBEhDUBDYqWFbiSfhMMRVR3

| Drive folder | copy into |
|---|---|
| `thesis_data_weights/01_binary_segmentation/` (data + experiments) | `01_binary_segmentation/` |
| `thesis_data_weights/02_multiclass_52class/` (data + experiments) | `02_multiclass_52class/` |
| `thesis_data_weights/auxiliary/` (SAM + YOLO weights) | project root (SAM) / `--model` (YOLO) |

The train/val/test split (21/4/5 images) is already packaged as `04_splits/` — copy it into each
project's `data/splits/` (no download needed).

### 2. Install

```bash
# CUDA GPU required for training
pip install -r requirements.txt

# Windows only — avoid OpenMP runtime clash (torch + cv2/skimage):
set KMP_DUPLICATE_LIB_OK=TRUE
```

### 3. Run

**52-class (DG-CSPC, main result):**

```bash
cd 02_multiclass_52class
python src/train_multiclass_v3.py --config configs/ours_multiclass_v3_selective.yaml
# training auto-evaluates the test set and writes experiments/ours_multiclass_v3_selective/summary.json
```

**Binary segmentation:**

```bash
cd 01_binary_segmentation
python src/train.py --config configs/unet.yaml
python src/evaluate.py --config configs/unet.yaml   # writes results/<name>_summary_test.csv
```

**Per-sample / diagnostic metrics:**

```bash
cd 03_evaluation
python reproduce_per_sample.py      # per-sample Dice (Table 4.5)
python eval_diagnostic_metrics.py   # precision / recall / F1 (Table 4.7)
```

---

## What it does

Two sub-projects:

- **01_binary_segmentation** — tooth region vs background; compares 6 models (U-Net, U-Net++,
  DeepLabV3+, SegFormer-B0, our lightweight U-Net, and +boundary loss).
- **02_multiclass_52class** — the main contribution: 52-class FDI instance segmentation via a
  detection-guided semi-supervised pipeline.

The 52-class pipeline (only the final U-Net runs at inference):

1. **YOLO detection + SAM segmentation** produce per-tooth masks on unlabelled images.
2. **Two-stage Hungarian matching** assigns each mask its FDI number.
3. **Iterative self-training** (rounds R1/R2) trains the U-Net on these pseudo-labels.
4. **DG-CSPC** applies class-selective pseudo-label construction: weak deciduous classes keep the
   detection-guided (YOLO+SAM) labels, strong permanent classes use the model's own predictions,
   with per-class boosting on hard classes.

## Results

### 52-class FDI segmentation (test set, Dice)

| Method | Dice (teeth) | Dice (macro) |
|---|---:|---:|
| Supervised (21 labelled) | 0.308 | 0.321 |
| Old pipeline (greedy FDI) | 0.248 | 0.261 |
| YOLO+SAM+Hungarian | 0.387 | 0.398 |
| Self-train R1 | 0.505 | 0.514 |
| Self-train R2 | 0.415 | 0.426 |
| **DG-CSPC (ours)** | **0.610** | **0.598** |

### Binary segmentation (test set)

| Model | Dice | Params |
|---|---:|---:|
| U-Net++ | 0.878 | 26.07M |
| Ours Lightweight (MobileNetV2) | 0.876 | 6.63M |

---

## Directory layout

```
raw_code_submission/
├── 01_binary_segmentation/   # binary tooth segmentation (configs / src / results / experiments)
├── 02_multiclass_52class/    # 52-class FDI pipeline (configs / src / results / per_class_dice / experiments)
├── 03_evaluation/            # per-sample & diagnostic metric scripts
├── 04_splits/                # train.txt (21) / val.txt (4) / test.txt (5)
├── 05_qualitative/           # Chapter 4 figures + test-sample overlays
└── requirements.txt
```

## Reproducing each dissertation method

Each row of the results table maps to a config + weight + result file under `02_multiclass_52class/`:

| Method | config | weight (`experiments/…/best_model.pth`) | result (`results/…_summary.json`) |
|---|---|---|---|
| Supervised | `ours_multiclass_v3_supervised_baseline.yaml` | `ours_multiclass_v3_supervised` | `01_supervised` |
| Old pipeline | `ours_multiclass_v3.yaml` | `ours_multiclass_v3` | `02_old_pipeline` |
| YOLO+SAM+Hungarian | `ours_multiclass_v3_iter2.yaml` | `ours_multiclass_v3_iter2` | `03_yolo_sam_hungarian` |
| Self-train R1 | `ours_multiclass_v3_iter3.yaml` | `ours_multiclass_v3_iter3` | `04_self_train_r1` |
| Self-train R2 | `ours_multiclass_v3_iter4.yaml` | `ours_multiclass_v3_iter4` | `05_self_train_r2` |
| DG-CSPC | `ours_multiclass_v3_selective.yaml` | `ours_multiclass_v3_selective` | `06_dg_cspc` |

Per-class Dice for every method is in `02_multiclass_52class/per_class_dice/` (files `01`–`06`,
52 rows each: FDI 11–48 permanent + FDI 51–85 deciduous).
