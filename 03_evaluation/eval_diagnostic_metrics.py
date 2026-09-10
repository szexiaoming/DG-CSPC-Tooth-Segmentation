"""
Extract Precision / Recall / F1 for the six 52-class methods (Table 4.7).

"""

import sys
import csv
from pathlib import Path

import torch
import yaml

PROJ = Path(__file__).resolve().parent.parent / '02_multiclass_52class'
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))

from models import create_deep_supervision_model
from train_multiclass_v3 import test_eval

# Paper method name -> (config, checkpoint), in Table 4.2 order.
METHODS = [
    ('Supervised',         'configs/ours_multiclass_v3_supervised_baseline.yaml',
     'experiments/ours_multiclass_v3_supervised/best_model.pth'),
    ('Old pipeline',       'configs/ours_multiclass_v3.yaml',
     'experiments/ours_multiclass_v3/best_model.pth'),
    ('YOLO+SAM+Hungarian', 'configs/ours_multiclass_v3_iter2.yaml',
     'experiments/ours_multiclass_v3_iter2/best_model.pth'),
    ('Self-train R1',      'configs/ours_multiclass_v3_iter3.yaml',
     'experiments/ours_multiclass_v3_iter3/best_model.pth'),
    ('Self-train R2',      'configs/ours_multiclass_v3_iter4.yaml',
     'experiments/ours_multiclass_v3_iter4/best_model.pth'),
    ('DG-CSPC',            'configs/ours_multiclass_v3_selective.yaml',
     'experiments/ours_multiclass_v3_selective/best_model.pth'),
]


def load_model(ckpt_path, cfg_path):
    cfg = yaml.safe_load(open(PROJ / cfg_path))
    mc, ds = cfg['model'], cfg.get('deep_supervision', {})
    model = create_deep_supervision_model(
        mc['name'], mc['encoder'], mc['in_channels'], 53,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1, 2, 3])))
    model.load_state_dict(torch.load(PROJ / ckpt_path, map_location='cuda'))
    model.cuda()
    return model, cfg


def main():
    out_path = Path(__file__).resolve().parent / 'diagnostic_metrics.csv'
    with open(out_path, 'w', newline='') as out_f:
        w = csv.writer(out_f)
        w.writerow(['method', 'precision_macro', 'recall_macro', 'f1_macro', 'dice_teeth'])

        for method, cfg_path, ckpt_path in METHODS:
            if not (PROJ / ckpt_path).exists():
                print(f'SKIP {method}: no checkpoint')
                continue

            model, cfg = load_model(ckpt_path, cfg_path)
            tm = test_eval(model, cfg, 'cuda', 53)

            # Macro precision / recall over the 52 tooth classes (full precision).
            precision_macro = tm['precision']
            recall_macro = tm['recall']
            # F1 = 2PR/(P+R), computed from the unrounded P/R values.
            f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
            dice_teeth = tm['dice_teeth']

            w.writerow([method, f'{precision_macro:.6f}', f'{recall_macro:.6f}',
                        f'{f1_macro:.6f}', f'{dice_teeth:.6f}'])

            print(f'{method:20s}  P={precision_macro:.4f}  R={recall_macro:.4f}  '
                  f'F1={f1_macro:.4f}  Dice(teeth)={dice_teeth:.4f}')

    print(f'\nSaved to {out_path.name}')


if __name__ == '__main__':
    main()
