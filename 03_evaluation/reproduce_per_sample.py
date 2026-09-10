"""
Reproduce dissertation Table 4.5 per-sample Dice (supervised baseline vs DG-CSPC).

"""
import sys, yaml, torch, csv
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parent.parent / '02_multiclass_52class'
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / 'src'))

from models import create_deep_supervision_model
from src.datasets import MultiClassToothDataset, get_transforms_val
from torch.utils.data import DataLoader

EPS = 1e-6


def load_model(ckpt, cfg_path):
    cfg = yaml.safe_load(open(PROJ / cfg_path))
    mc = cfg['model']
    ds = cfg.get('deep_supervision', {})
    m = create_deep_supervision_model(
        mc['name'], mc['encoder'], mc['in_channels'], 53,
        pretrained=False, aux_scales=tuple(ds.get('aux_scales', [1, 2, 3])))
    m.load_state_dict(torch.load(PROJ / ckpt, map_location='cuda'))
    m.cuda()
    m.eval()
    return m, cfg


def per_sample_dice(model, img, mask):
    with torch.no_grad():
        out = model(img)
        if isinstance(out, dict) and 'main' in out:
            out = out['main']
    pred = out.argmax(dim=1).cpu()
    inter = np.zeros(53, dtype=np.float64)
    t_pred = np.zeros(53, dtype=np.float64)
    t_gt = np.zeros(53, dtype=np.float64)
    for c in range(1, 53):  # exclude background class 0
        pc = (pred == c)
        gc = (mask == c)
        inter[c] = (pc & gc).sum().float().item()
        t_pred[c] = pc.sum().float().item()
        t_gt[c] = gc.sum().float().item()
    per_class_dice = (2 * inter + EPS) / (t_pred + t_gt + EPS)
    macro = float(per_class_dice[1:].mean())          # mean of per-class Dice
    micro = float(2 * inter[1:].sum() / (t_pred[1:].sum() + t_gt[1:].sum() + EPS))
    return macro, micro


def main():
    m_sup, c_sup = load_model(
        'experiments/ours_multiclass_v3_supervised/best_model.pth',
        'configs/ours_multiclass_v3_supervised_baseline.yaml')
    m_sel, c_sel = load_model(
        'experiments/ours_multiclass_v3_selective/best_model.pth',
        'configs/ours_multiclass_v3_selective.yaml')

    dc = c_sel['data']
    sz = tuple(dc['image_size'])
    test_names = [l.strip() for l in open(PROJ / dc['test_split'])]

    dset = MultiClassToothDataset(
        image_dir=str(PROJ / dc['image_dir']),
        mask_dir=str(PROJ / dc['mask_dir']),
        split_file=str(PROJ / dc['test_split']),
        transform=get_transforms_val(sz), image_size=sz, num_classes=53)
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

    rows = []
    # Accumulate to cross-check the aggregate against test_eval's dice_teeth (0.3084 / 0.6096)
    acc_inter_sup = np.zeros(53, dtype=np.float64)
    acc_pred_sup = np.zeros(53, dtype=np.float64)
    acc_gt_sup = np.zeros(53, dtype=np.float64)
    acc_inter_sel = np.zeros(53, dtype=np.float64)
    acc_pred_sel = np.zeros(53, dtype=np.float64)
    acc_gt_sel = np.zeros(53, dtype=np.float64)

    print(f'{"sample_id":<8} {"supervised(macro)":>18} {"dg_cspc(macro)":>16} {"delta":>14}   (micro: sup/sel)')
    print('-' * 90)
    for idx, batch in enumerate(loader):
        img = batch['image'].cuda()
        mask = batch['mask']  # CPU
        name = test_names[idx]
        sid = name[-4:]
        d_sup, mi_sup = per_sample_dice(m_sup, img, mask)
        d_sel, mi_sel = per_sample_dice(m_sel, img, mask)
        delta = d_sel - d_sup
        rows.append([sid, d_sup, d_sel, delta])
        print(f'{sid:<8} {d_sup:18.10f} {d_sel:16.10f} {delta:+14.10f}   (micro {mi_sup:.4f}/{mi_sel:.4f})')

        # accumulate per-class for aggregate check
        with torch.no_grad():
            out_s = m_sup(img)
            if isinstance(out_s, dict) and 'main' in out_s:
                out_s = out_s['main']
            pred_s = out_s.argmax(dim=1).cpu()
            out_l = m_sel(img)
            if isinstance(out_l, dict) and 'main' in out_l:
                out_l = out_l['main']
            pred_l = out_l.argmax(dim=1).cpu()
        for c in range(1, 53):
            ps = (pred_s == c); gs = (mask == c)
            acc_inter_sup[c] += (ps & gs).sum().float().item()
            acc_pred_sup[c] += ps.sum().float().item()
            acc_gt_sup[c] += gs.sum().float().item()
            pl = (pred_l == c); gl = (mask == c)
            acc_inter_sel[c] += (pl & gl).sum().float().item()
            acc_pred_sel[c] += pl.sum().float().item()
            acc_gt_sel[c] += gl.sum().float().item()

    # aggregate macro = test_eval dice_teeth
    pcd_sup = (2 * acc_inter_sup + EPS) / (acc_pred_sup + acc_gt_sup + EPS)
    pcd_sel = (2 * acc_inter_sel + EPS) / (acc_pred_sel + acc_gt_sel + EPS)
    agg_sup = float(pcd_sup[1:].mean())
    agg_sel = float(pcd_sel[1:].mean())

    out_csv = Path(__file__).resolve().parent / 'per_sample_dice.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'supervised_dice', 'dg_cspc_dice', 'delta'])
        for sid, ds, dsel, delta in rows:
            w.writerow([sid, f'{ds:.10f}', f'{dsel:.10f}', f'{delta:.10f}'])

    print()
    print(f'Aggregate MACRO dice_teeth: supervised={agg_sup:.6f} (expect 0.3084)  '
          f'dg_cspc={agg_sel:.6f} (expect 0.6096)')
    print()
    print('=== Rounded to 3 decimal places ===')
    for sid, ds, dsel, delta in rows:
        print(f'{sid}: supervised={ds:.3f}  dg_cspc={dsel:.3f}  delta={delta:+.3f}')
    print()
    print(f'CSV written to {out_csv}')


if __name__ == '__main__':
    main()
