"""
Evaluate all 52-class checkpoints and save per-class CSV.
"""

import sys, yaml, torch, csv
from pathlib import Path
PROJ = str(Path(__file__).resolve().parent.parent / '02_multiclass_52class')
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + '/src')

from models import create_deep_supervision_model
from train_multiclass_v3 import test_eval, _fdi_names

TASKS = [
    ('ours_multiclass_v3_supervised', 'configs/ours_multiclass_v3_supervised_baseline.yaml'),
    ('ours_multiclass_v3', 'configs/ours_multiclass_v3.yaml'),
    ('ours_multiclass_v3_iter2', 'configs/ours_multiclass_v3_iter2.yaml'),
    ('ours_multiclass_v3_iter3', 'configs/ours_multiclass_v3_iter3.yaml'),
    ('ours_multiclass_v3_iter4', 'configs/ours_multiclass_v3_iter4.yaml'),
]

fdi = _fdi_names()

for exp_name, cfg_path in TASKS:
    ckpt_path = PROJ + '/experiments/' + exp_name + '/best_model.pth'
    if not __import__('os').path.exists(ckpt_path):
        print(f'SKIP {exp_name}: no checkpoint')
        continue

    config = yaml.safe_load(open(PROJ + '/' + cfg_path))
    mc, ds = config['model'], config.get('deep_supervision', {})
    model = create_deep_supervision_model(
        mc['name'], mc['encoder'], mc['in_channels'], 53,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1,2,3])))
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    model.cuda()

    tm = test_eval(model, config, 'cuda', 53)

    print(f'\n=== {exp_name} ===')
    print(f'Dice(teeth)={tm["dice_teeth"]:.4f}  Dice(macro)={tm["dice"]:.4f}')
    for c in range(1, 53):
        print(f'{c:3d}  {fdi[c]:>4s}  Dice={tm["per_class_dice"][c]:.6f}')

    # Save CSV
    csv_path = PROJ + '/experiments/' + exp_name + '/results_test_per_class.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'fdi', 'dice'])
        for c in range(1, 53):
            w.writerow([c, fdi[c], f'{tm["per_class_dice"][c]:.6f}'])
    print(f'Saved {csv_path}')
