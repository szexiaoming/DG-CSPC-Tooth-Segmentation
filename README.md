# Raw Code & Data — Thesis Reproducibility Submission Package

This folder collects the raw code, configs, and result files behind **every numerical result** in the dissertation; **model weights and datasets are NOT included in this package** (size exceeds the limit, see Section 8).
Each dissertation table maps line-by-line to a script and data file here, clearly named and reproducible.

> Data source: local backup `server_backup/dental_xray_project/` (full archive downloaded from the server `/root/autodl-tmp/dental_backup.tar`, sha256 = `cd8e8706…0739a8` verified) and the binary segmentation project `tooth-xray-segmentation/`.

> **This package does NOT contain the dataset (`data/`) or the model weights (`.pth` / `.pt`)** — together ~3.3 GB, exceeding the 230 MB upload limit;
> they are uploaded to a **Google Drive** folder named `thesis_data_weights/` (see Section 8 for the link and the full mapping).
> **Before reproducing, copy the files back from `thesis_data_weights/` into the named folders inside this package** — what to download and where to place it is in the table below (the full subdirectory-level mapping is in **Section 8** at the end):

| What to download | Path in the drive folder | Named folder to place it in this package |
|---|---|---|
| Binary segmentation data (2400 X-rays + 30 binary masks) | `thesis_data_weights/01_binary_segmentation/data/processed/` | `01_binary_segmentation/data/processed/` |
| Binary segmentation weights (6 reported + 1 excluded `best_model.pth`) | `thesis_data_weights/01_binary_segmentation/experiments/<exp_name>/best_model.pth` | `01_binary_segmentation/experiments/<exp_name>/best_model.pth` |
| 52-class data (images + `masks_multiclass/` + 8 `pseudo_masks_*` + `yolo_model/` `yolo_dataset/`) | `thesis_data_weights/02_multiclass_52class/data/` | `02_multiclass_52class/data/` |
| 52-class weights (6 reported methods + 5 ablation variants `best_model.pth`) | `thesis_data_weights/02_multiclass_52class/experiments/<exp_name>/best_model.pth` | `02_multiclass_52class/experiments/<exp_name>/best_model.pth` |
| SAM + YOLO auxiliary weights | `thesis_data_weights/auxiliary/` | project root (SAM) / `--model` (YOLO base) |

> `data/splits/` (the three-layer split) is already packaged as `04_splits/` in this package — **no download needed**; just copy it to each project's `data/splits/`.

---

## 1. Directory structure

```
raw_code_submission/
├── README.md                        # this file
├── requirements.txt                 # environment dependencies
├── 01_binary_segmentation/          # Table 4.1 (binary segmentation, 6 reported models + 1 excluded exploratory experiment)
│   ├── src/                         # train.py / evaluate.py / models / losses / datasets / utils
│   ├── configs/                     # 6 reported + 1 excluded .yaml (unet / unetpp / deeplabv3plus / segformer_b0 / ours_*)
│   ├── results/                     # 6 reported + 1 excluded *_summary_test.csv (metric,mean,std,min,max)
│   ├── experiments/                 # 6 reported + 1 excluded <exp_name>/best_model.pth (not packaged, see Section 8)
│   └── data/                        # processed images/masks, pseudo_masks, pseudo_probs (not packaged, see Section 8)
├── 02_multiclass_52class/           # Table 4.2–4.4, 4.6, A.1, A.2 (52-class FDI, 6 methods)
│   ├── src/                         # train_multiclass_v3.py / self_train_* / yolo_sam_pseudo / hungarian / train_yolo / generate_pseudo_multiclass / plot_fig4_* …
│   ├── configs/                     # 12 .yaml (6 methods + 6 variants, see Section 4)
│   ├── results/                     # 01–06 _summary.json + _results_test.csv
│   ├── per_class_dice/              # 01–06 _per_class.csv (source of Table A.1/A.2 per-class Dice)
│   ├── experiments/                 # 6 reported methods + ablation variants <exp_name>/best_model.pth + sam/ssl/yolo pretrained weights (not packaged, see Section 8)
│   └── data/                        # processed + 8 pseudo_masks_* + yolo_model/yolo_dataset (not packaged, see Section 8)
├── 03_evaluation/                   # evaluation scripts & results for Table 4.5, 4.7
│   ├── eval_per_sample_macro.py     # per-sample macro Dice (source of the Table 4.5 baseline + DG-CSPC columns)
│   ├── eval_per_sample.py           # per-sample micro Dice (NOT the Table 4.5 source, see below)
│   ├── eval_selective_persample.py  # per-sample micro Dice (NOT the Table 4.5 source, see below)
│   ├── reproduce_per_sample.py      # Table 4.5 per-sample Dice script (writes per_sample_dice.csv)
│   ├── per_sample_dice.csv          # Table 4.5 per-sample Dice results (actual server inference, macro)
│   ├── eval_diagnostic_metrics.py   # diagnostic metrics P/R/F1 (Table 4.7)
│   ├── eval52_all.py                # 52-class aggregate evaluation
│   └── diagnostic_metrics.csv       # Table 4.7 diagnostic metrics (P/R/F1, 6 methods)
├── 04_splits/                       # train.txt(21) / val.txt(4) / test.txt(5)
└── 05_qualitative/                  # Chapter 4 qualitative figures + 5 test-sample overlays (see Section 7)
```

---

## 2. Dissertation tables ↔ code/data mapping (core)

### Table 4.1 — binary segmentation: 6 reported models + 1 excluded exploratory experiment (`01_binary_segmentation/`)

| Model | config | result file | weights |
|---|---|---|---|
| U-Net | `configs/unet.yaml` | `results/unet_summary_test.csv` | `experiments/unet/best_model.pth` |
| U-Net++ | `configs/unetpp.yaml` | `results/unetpp_summary_test.csv` | `experiments/unetpp/best_model.pth` |
| DeepLabV3+ | `configs/deeplabv3plus.yaml` | `results/deeplabv3plus_summary_test.csv` | `experiments/deeplabv3plus/best_model.pth` |
| SegFormer-B0 | `configs/segformer_b0.yaml` | `results/segformer_b0_summary_test.csv` | `experiments/segformer_b0/best_model.pth` |
| Ours Lightweight | `configs/ours_lightweight.yaml` | `results/ours_lightweight_summary_test.csv` | `experiments/ours_lightweight/best_model.pth` |
| Ours + Boundary | `configs/ours_lightweight_boundary.yaml` | `results/ours_lightweight_boundary_summary_test.csv` | `experiments/ours_lightweight_boundary/best_model.pth` |

Column mapping: `Dice` ← the `mean` column of the `dice` row in the csv; `IoU` ← `iou`; `BF1` ← `boundary_f1`.
The `Params (M)` column is **not code-generated** — it is the model parameter count (literature/manual); see the notes in Section 5.

### Excluded exploratory experiment

`Ours SSL v2` is preserved as an exploratory binary segmentation experiment
but is not reported in the final dissertation. A post-hoc audit identified an
inverted confidence/uncertainty weighting in its pseudo-label loss.
This issue does not affect the 52-class DG-CSPC framework.

Archived (kept but marked excluded; not counted among Table 4.1's 6 reported models):
`configs/ours_lightweight_ssl_v2.yaml` · `results/ours_lightweight_ssl_v2_summary_test.csv` · `experiments/ours_lightweight_ssl_v2/best_model.pth`

### Table 4.2 — 52-class summary (`02_multiclass_52class/results/`)

6 methods (dissertation order = number prefix):

| Dissertation method | # | summary.json | Dice(macro) | Dice(teeth) | IoU |
|---|---|---|---|---|---|
| Supervised | 01 | `01_supervised_summary.json` | 0.3210 | 0.3084 | **0.2071** |
| Old pipeline | 02 | `02_old_pipeline_summary.json` | 0.2614 | 0.2478 | 0.1786 |
| YOLO+SAM+Hungarian | 03 | `03_yolo_sam_hungarian_summary.json` | 0.3979 | 0.3868 | **0.2853** |
| Self-train R1 | 04 | `04_self_train_r1_summary.json` | 0.5136 | 0.5047 | **0.3976** |
| Self-train R2 | 05 | `05_self_train_r2_summary.json` | 0.4257 | 0.4151 | **0.3244** |
| DG-CSPC (Ours) | 06 | `06_dg_cspc_summary.json` | 0.5981 | 0.6096 | 0.4548 |

Field mapping: `Dice(macro)` ← `test_dice_macro`; `Dice(teeth)` ← `test_dice_teeth`; `IoU` ← `test_iou`.
(`test_iou` = macro IoU over 53 classes including background; defined in `test_eval()` of `src/train_multiclass_v3.py`.)

**Naming note**: the dissertation's "YOLO+SAM+Hungarian" corresponds to experiment directory `ours_multiclass_v3_iter2`
(its config `ours_multiclass_v3_iter2.yaml` has `pseudo_mask_dir = data/pseudo_masks_yolo_sam`,
i.e. round-2 training using YOLO+SAM pseudo-labels). The repo also contains an `ours_multiclass_v3_yolo_sam` experiment
(Dice 0.5841), which is an **intermediate result not included in the dissertation** and unrelated to this row.

### Table 4.3 — DG-CSPC per quadrant (computed from `per_class_dice/06_dg_cspc_per_class.csv`)

Per-class Dice of the 52 classes grouped by quadrant and averaged:

| Quadrant | FDI range | # classes |
|---|---|---|
| Q1 (UR, permanent) | 11–18 | 8 |
| Q2 (UL, permanent) | 21–28 | 8 |
| Q3 (LL, permanent) | 31–38 | 8 |
| Q4 (LR, permanent) | 41–48 | 8 |
| Q5 (UR, deciduous) | 51–55 | 5 |
| Q6 (UL, deciduous) | 61–65 | 5 |
| Q7 (LL, deciduous) | 71–75 | 5 |
| Q8 (LR, deciduous) | 81–85 | 5 |

Activated/Inactive = number of classes in that quadrant with Dice > 0.

### Table 4.4 — best/worst FDI classes (same `06_dg_cspc_per_class.csv`)

Sort by `dice` descending; take the top 5 / bottom 5.

### Table 4.5 — per-sample Dice (`03_evaluation/`)

- Baseline + DG-CSPC columns ← `eval_per_sample_macro.py` (per-sample **macro** Dice, consistent with the
  `dice_teeth` used by `test_eval()`: per-class `2·I/(P+G)` averaged over the 52 teeth)
- The 5 test samples = `04_splits/test.txt` (0001/0004/0009/0021/0024)
- Reproduction script: `03_evaluation/reproduce_per_sample.py` (reads the config's `test_split` + two weights, outputs the exact Dice)
- Reproduced results: `03_evaluation/per_sample_dice.csv` (actual server inference, exact values)

> Note: `eval_per_sample.py` and `eval_selective_persample.py` compute **micro** Dice
> (`2·ΣI/Σ(P+G)`), whose values differ from the macro convention used in Table 4.5 (e.g. 0001 DG-CSPC: micro=0.273,
> macro=0.800); **do not use them as the Table 4.5 source**.

### Table 4.6 — self-training vs DG-CSPC (summary + activation counts)

- Dice(teeth) ← `test_dice_teeth` in each `summary.json`
- Decid. Activated ← number of deciduous classes (FDI 51–85) with Dice > 0 in the corresponding `per_class_dice/` CSV

### Table 4.7 — diagnostic metrics (`03_evaluation/diagnostic_metrics.csv`)

Computed from macro precision/recall via the dissertation footnote `F1 = 2·P·R/(P+R)`, for 6 methods:

| Method | precision_macro | recall_macro | F1 = 2PR/(P+R) | dice_teeth |
|---|---|---|---|---|
| Supervised | 0.3299 | 0.3463 | 0.338 | 0.3084 |
| Old pipeline | 0.6680 | 0.2630 | 0.377 | 0.2478 |
| YOLO+SAM+Hungarian | 0.3915 | 0.4365 | 0.413 | 0.3868 |
| Self-train R1 | 0.5246 | 0.5840 | 0.553 | 0.5047 |
| Self-train R2 | 0.4063 | 0.4401 | 0.423 | 0.4151 |
| DG-CSPC | 0.6762 | 0.6402 | 0.658 | 0.6096 |

> `Self-train R2`'s F1 from the 4-decimal P/R (0.4063/0.4401) equals 0.4225, rounded to 0.423;
> the dissertation shows 0.422 — a 0.001 rounding-boundary difference (see Section 5).

### Table A.1 / A.2 — per-class Dice (`02_multiclass_52class/per_class_dice/`)

6 files `01_supervised_per_class.csv` … `06_dg_cspc_per_class.csv`, each with 52 rows
(FDI 11–48 permanent 32 classes + FDI 51–85 deciduous 20 classes). Columns `class_id,fdi,dice` (06 uses `class,fdi,dice`).

---

## 3. Reproduction

> **Before reproducing**: first restore `data/` and `experiments/` from the backup per Section 8 (dataset + model weights + pseudo-labels),
> otherwise the commands below fail for missing data/weights.

52-class (DG-CSPC as the example):

```bash
# 1) environment
pip install -r requirements.txt

# 2) train DG-CSPC (config points to data/ and pseudo_masks_selective)
cd 02_multiclass_52class
python src/train_multiclass_v3.py --config configs/ours_multiclass_v3_selective.yaml

# 3) training auto-runs test_eval() and writes experiments/<name>/summary.json
#    (contains test_dice_macro / test_dice_teeth / test_iou)

# 4) per-sample and diagnostic metrics
cd ../03_evaluation
python reproduce_per_sample.py   # Table 4.5 per-sample Dice (writes per_sample_dice.csv)
python eval_diagnostic_metrics.py    # Table 4.7
```

> The full 6-method pipeline (pseudo-label generation → self-training → DG-CSPC) involves
> `generate_pseudo_multiclass.py → convert_to_yolo.py → train_yolo.py → yolo_sam_pseudo.py → hungarian_match.py → self_train_iter*.py → self_train_selective.py`;
> see the `pseudo_mask_dir` dependency chain in each config under `02_multiclass_52class/configs/`.
> Note: training/inference requires a CUDA GPU (the code uses `model.cuda()`).

Binary (Table 4.1):

```bash
cd 01_binary_segmentation
python src/train.py --config configs/unet.yaml        # per model
python src/evaluate.py --config configs/unet.yaml     # writes summary_test.csv
```

---

## 4. Naming conventions

- The `01`–`06` prefixes = the order of the 6 methods in Table 4.2 (Supervised → DG-CSPC).
- The `01`–`06` prefixes (binary weights) = the order of the 6 reported models in Table 4.1; `07` = the excluded exploratory experiment SSL v2.
- Only `best_model.pth` (the final best) is kept as weights; the mid-training `checkpoint_epoch_*.pth` snapshots are **not packaged**
  (in the original repo these intermediate checkpoints total ~20 GB in the binary project alone; they are not needed for reproduction).
- Intermediate-experiment code/config not in the dissertation's main tables (multiclass v1–v2, exploratory plotting/ablation scripts, binary copies, etc.)
  has been removed from `02_multiclass_52class/`; only the 6 variant configs are kept (`iter3_lowlr` / `iter4_v2` / `lite` /
  `sel_v2` / `selective_r2` / `selective_t60`), corresponding to the intermediate results 0.476 / 0.425 etc. mentioned in the text.

---

## 5. Data-audit notes (important: 2 places where the dissertation differs from the code)

After auditing every table in the dissertation PDF, **the vast majority of numbers match the code exactly**, but 2 places were found where the dissertation's values differ from the code's actual output.
This package **keeps the code's actual output**; please correct the dissertation accordingly (code is authoritative):

| Location | Dissertation (wrong) | Code actual value | Source file |
|---|---|---|---|
| Table 4.2 IoU — Supervised | 0.1915 | **0.2071** | `results/01_supervised_summary.json` |
| Table 4.2 IoU — YOLO+SAM | 0.2522 | **0.2853** | `results/03_yolo_sam_hungarian_summary.json` |
| Table 4.2 IoU — Self-train R1 | 0.3572 | **0.3976** | `results/04_self_train_r1_summary.json` |
| Table 4.2 IoU — Self-train R2 | 0.2798 | **0.3244** | `results/05_self_train_r2_summary.json` |
| Table 4.3 Q4 Mean Dice | 0.4978 | **0.4977** | `per_class_dice/06_dg_cspc_per_class.csv` (mean of FDI 41–48) |
| Table 4.3 Deciduous total | 0.5424 | **0.5418** | same (mean of the 20 deciduous teeth, ≈0.542 consistent with the text) |

Everything else (all of Table 4.1, both Dice columns of Table 4.2, Tables 4.4–4.7, and the 312 per-class Dice of Table A.1/A.2)
has been checked value-by-value with no discrepancy.

Also requiring manual confirmation (not decidable from code):

- **Table 4.1 `Params (M)` column**: model parameter counts, not code output. SegFormer-B0 is written as 5.6 (MiT-B0);
  please re-check against the dissertation's cited source.
- **Table 4.7 F1 column**: the authoritative source is `03_evaluation/diagnostic_metrics.csv` (`F1 = 2·P·R/(P+R)`, see Section 2).
  `Self-train R2`'s F1 = 0.4225 falls on a rounding boundary; the dissertation shows 0.422, exact value is 0.423.

---

## 6. Data-file inventory

> **The `data/` and `experiments/` files in the table below are NOT packaged in this submission** (together ~3.3 GB, exceeding the 230 MB limit);
> they are stored in the server backup + cloud drive; see Section 8 for how to obtain them.

| File | Description | Dissertation correspondence | Packaged |
|---|---|---|---|
| `04_splits/train.txt` (21) / `val.txt` (4) / `test.txt` (5) | three-layer data split | the 21/4/5 description | ✅ packaged |
| `01_binary_segmentation/data/processed/` | binary input X-rays + masks | Table 4.1 training input | ❌ not packaged |
| `02_multiclass_52class/data/processed/` | input X-rays `images/` + 52-class ground-truth `masks_multiclass/` | training input | ❌ not packaged |
| `02_multiclass_52class/data/pseudo_masks_*` | pseudo-label masks of each round | each method's pseudo-labels | ❌ not packaged |
| `01_binary_segmentation/data/pseudo_probs/` | soft probabilities of unlabeled samples (1.2G) | binary SSL v2 self-training (excluded exploratory experiment) | ❌ not packaged / not in drive |
| `01/02 .../experiments/` | binary + multiclass weights (.pth/.pt) | Table 4.1/4.2 reproduction | ❌ not packaged |

---

## 7. Chapter 4 qualitative visualizations (`05_qualitative/`)

The 4 figures of Chapter 4 (Fig 4.1–4.4) were regenerated on the server from their generating scripts using the **DG-CSPC** terminology; the 5 test-sample overlay images are still copies of the existing outputs (not regenerated).

Regenerated content (addressing the review comment, renaming the term `Selective` → `DG-CSPC`):
- **Fig 4.1**: last bar label → `DG-CSPC`; horizontal dashed line labeled `Supervised baseline`.
- **Fig 4.2**: last bar label → `DG-CSPC`; y-axis changed to `Activated deciduous classes (Dice > 0.001)`.
- **Fig 4.3**: now uses `per_class_dice/06_dg_cspc_per_class.csv` (DG-CSPC: Permanent μ=0.652/32, Deciduous μ=0.542/19, Overall Dice(teeth)=0.610) — the old archived PNG was mistakenly Self-train R1, now fixed; legend `Deciduous dead` → `inactive`.
- **Fig 4.4**: column-3 title `Selective model` → `DG-CSPC` (0009 supervised Dice 0.443 unchanged).

**Figure-number ↔ file mapping** (note: the dissertation figure numbers differ from the original filenames, which is easy to confuse):

| Dissertation figure | archived file (`chapter4_figures/`) | generating script |
|---|---|---|
| Fig 4.1 | `Fig4.1_perf_trajectory.png` | `plot_fig4_1_perf_trajectory.py` (Dice-trajectory bar chart) |
| Fig 4.2 | `Fig4.2_decid_activation.png` | `plot_fig4_2_decid_activation.py` (deciduous-activation bar chart) |
| Fig 4.3 | `Fig4.3_52class_per_FDI_dice.png` | `plot_fig4_3_per_fdi_dice.py` (52-class per-FDI Dice) |
| Fig 4.4 | `Fig4.4_qualitative_selective.png` + `.pdf` | `plot_fig4_4_qualitative.py` (DG-CSPC three-column comparison, server GPU) |

The remaining source figures (old numbering / alternative versions, incl. `.pdf`) are kept in `chapter4_figures/other_source_figures/`:
`root_figure_4_2.png`, `server_figure_4_2.png`, `server_figure_4_3.png/.pdf`,
`server_figure_4_4.png/.pdf`, `server_figure_4_2_qualitative_selective.png/.pdf`.

**5 test samples (0001/0004/0009/0021/0024, from `04_splits/test.txt`) overlay images**:

| Subdirectory | Content |
|---|---|
| `test_sample_comparisons/root_compare/` | root `STS24_Train_Labeled_00XX_compare.png` (GT vs prediction comparison) |
| `test_sample_comparisons/tooth_compare/` / `tooth_compare_clean/` / `tooth_compare_v2/` | three server comparison versions (5 each) |
| `fdi_overlays/root_fdi_labels/` and `fdi_overlays/tooth_fdi_overlays/` | FDI-number label overlays (5 each; the two groups have identical md5) |
| `multiclass_overlays/` | 52-class prediction overlays (5 each) |

**Figure-generation scripts** (`02_multiclass_52class/src/`, named correctly by dissertation figure number):

| Dissertation figure | generating script | input | runtime |
|---|---|---|---|
| Fig 4.1 | `plot_fig4_1_perf_trajectory.py` | none (values hardcoded) | local/any, matplotlib |
| Fig 4.2 | `plot_fig4_2_decid_activation.py` | none (values hardcoded) | local/any, matplotlib |
| Fig 4.3 | `plot_fig4_3_per_fdi_dice.py` | `../per_class_dice/06_dg_cspc_per_class.csv` | local/any, matplotlib |
| Fig 4.4 | `plot_fig4_4_qualitative.py` | two `best_model.pth` + test images (`PROJ` points to the server) | CUDA GPU |

> Fig 3.1 (framework flowchart) generating script `gen_framework_flowchart.py` is in the **project root** (not archived in this package).
> The original `plot_chapter4_figures.py` (produced Fig 4.1+4.2 together) was split into `plot_fig4_1_*.py` + `plot_fig4_2_*.py`;
> the original `plot_52class_selective_fig.py` → `plot_fig4_3_per_fdi_dice.py`; the original `plot_figure_4_2_selective_fixed.py` → `plot_fig4_4_qualitative.py`.
> Fig 4.1/4.2 were generated with matplotlib on the server; regenerating locally will differ slightly due to font rendering, so the text still archives the PNG pixel results directly.

---

## 8. How to obtain the data and weights

**The dataset (`data/`) and model weights (`experiments/`) are not packaged in this submission** (together ~3.3 GB, exceeding the 230 MB upload limit). They are uploaded to a **Google Drive** folder named `thesis_data_weights/`, which mirrors this package's layout; copy its contents back into this package per the mapping below.

> **Google Drive folder link**: https://drive.google.com/drive/folders/1XMtzlWz1c9M3w8Bxw0jQEiq08LPU0mRR

The drive folder has three subfolders, each mapping 1:1 onto this package:

| Drive subfolder | contents | restore into this package |
|---|---|---|
| `thesis_data_weights/01_binary_segmentation/` | binary `data/` (images + masks + splits) + `experiments/` (7 `best_model.pth`) | `01_binary_segmentation/` |
| `thesis_data_weights/02_multiclass_52class/` | 52-class `data/` (images + `masks_multiclass/` + 8 `pseudo_masks_*` + `yolo_model/`/`yolo_dataset/` + splits) + `experiments/` (11 `best_model.pth`) | `02_multiclass_52class/` |
| `thesis_data_weights/auxiliary/` | SAM + YOLO auxiliary weights (`sam_vit_b_01ec64.pth`, `yolo26n.pt`, `yolov8n.pt`) | SAM → `--sam-checkpoint`; YOLO base → `--model` |

> Provenance: these files were originally downloaded from the server archive `/root/autodl-tmp/dental_backup.tar` (3.9G, sha256 = `cd8e8706…0739a8`) and the binary project `tooth-xray-segmentation/`; the local copies remain in `server_backup/dental_xray_project/`.

**Per-directory restore mapping (folders to place back into this package after download):**

**① binary `01_binary_segmentation/` ← `thesis_data_weights/01_binary_segmentation/`**

| Path in drive | place back into this package | content |
|---|---|---|
| `data/processed/images/` | `01_binary_segmentation/data/processed/images/` | 2400 input X-rays |
| `data/processed/masks/` | `01_binary_segmentation/data/processed/masks/` | 30 binary masks (Table 4.1 training input) |
| `data/splits/` | `01_binary_segmentation/data/splits/` | train 21 / val 4 / test 5 (also packaged as `04_splits/`) |
| `experiments/<exp_name>/best_model.pth` (7 files) | `01_binary_segmentation/experiments/<exp_name>/best_model.pth` (place back under the same name) | 6 reported (unet / unetpp / deeplabv3plus / segformer_b0 / ours_lightweight / ours_lightweight_boundary) + 1 excluded (ours_lightweight_ssl_v2) |

> `data/pseudo_probs/` (soft probabilities for the **excluded** binary SSL v2 experiment) is **not** in the drive folder — it is not needed to reproduce any reported number.

**② 52-class `02_multiclass_52class/` ← `thesis_data_weights/02_multiclass_52class/`**

| Path in drive | place back into this package | content |
|---|---|---|
| `data/processed/images/` | `02_multiclass_52class/data/processed/images/` | 2400 input X-rays |
| `data/processed/masks_multiclass/` | `02_multiclass_52class/data/processed/masks_multiclass/` | 30 52-class ground-truth masks |
| `data/splits/` | `02_multiclass_52class/data/splits/` | train 21 / val 4 / test 5 (also packaged as `04_splits/`) |
| `data/pseudo_masks_multiclass/` | same name | Old pipeline pseudo-labels (2364 PNG) |
| `data/pseudo_masks_yolo_sam/` | same name | YOLO+SAM+Hungarian pseudo-labels |
| `data/pseudo_masks_iter3/` | same name | Self-train R1 pseudo-labels |
| `data/pseudo_masks_iter4/` | same name | Self-train R2 pseudo-labels |
| `data/pseudo_masks_selective/` | same name | DG-CSPC pseudo-labels |
| `data/pseudo_masks_iter4_v2/` | same name | iter4_v2 ablation-variant pseudo-labels |
| `data/pseudo_masks_selective_r2/` | same name | selective_r2 ablation-variant pseudo-labels |
| `data/pseudo_masks_selective_t60/` | same name | selective_t60 ablation-variant pseudo-labels |
| `data/yolo_model/` + `data/yolo_dataset/` | same name | YOLO detector weights + training set |
| `experiments/<exp_name>/best_model.pth` | place back under the same name | 6 reported methods + 5 ablation variants (mapping below) |

All pseudo-label masks are **PNG** (`<name>.png`) plus one `quality_scores.json` per directory. `pseudo_masks_multiclass/` holds 2364 PNG (the quality filter dropped 6 of the 2370 unlabeled samples); the other 7 directories hold 2370 PNG each.

**Weight ↔ dissertation-method mapping (`best_model.pth` under `02_multiclass_52class/experiments/`, consistent with the hardcoded paths in `03_evaluation/eval_diagnostic_metrics.py`):**

| Dissertation method | weight path |
|---|---|
| Supervised | `experiments/ours_multiclass_v3_supervised/best_model.pth` |
| Old pipeline | `experiments/ours_multiclass_v3/best_model.pth` |
| YOLO+SAM+Hungarian | `experiments/ours_multiclass_v3_iter2/best_model.pth` |
| Self-train R1 | `experiments/ours_multiclass_v3_iter3/best_model.pth` |
| Self-train R2 | `experiments/ours_multiclass_v3_iter4/best_model.pth` |
| DG-CSPC | `experiments/ours_multiclass_v3_selective/best_model.pth` |

The 5 ablation-variant weights (kept because the text cites their intermediate results) are
`ours_multiclass_v3_iter4_v2` / `ours_multiclass_v3_lite` / `ours_multiclass_v3_sel_v2` / `ours_multiclass_v3_selective_r2` / `ours_multiclass_v3_selective_t60`.
The `iter3_lowlr` config has **no** weight in the backup (its result cannot be reproduced from weights).

**Auxiliary weights (`thesis_data_weights/auxiliary/`):**

| File | size | purpose |
|---|---|---|
| `sam_vit_b_01ec64.pth` | 375 MB | SAM ViT-B checkpoint (`yolo_sam_pseudo.py --sam-checkpoint`) |
| `yolov8n.pt` | 6.5 MB | YOLOv8-nano pretrained base (`train_yolo.py --model`) |
| `yolo26n.pt` | 5.5 MB | YOLO detector checkpoint |

The trained detector's own weights (`best.pt` / `last.pt`) are under `02_multiclass_52class/data/yolo_model/`.

> **No download needed**: `data/splits/` (train.txt 21 / val.txt 4 / test.txt 5) is already packaged as `04_splits/` in this package;
> copy `04_splits/` to `01_binary_segmentation/data/splits/` and `02_multiclass_52class/data/splits/`.

**Runtime tip (Windows / Anaconda)**: importing `torch` and then `cv2` / `skimage` in the same process may trigger
`OMP: Error #15` (conflicting OpenMP runtimes `libiomp5md.dll`, causing an immediate crash). Set the environment variable before running:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE        # Windows
export KMP_DUPLICATE_LIB_OK=TRUE     # Linux server
```
