"""
Reproduce thesis Figure 4.3 52-class per-FDI Dice bar chart (DG-CSPC model, Dice(teeth) = 0.610).

"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_csv(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["fdi"]] = float(row["dice"])
    return data


DATA_CSV = Path(__file__).resolve().parent.parent / 'per_class_dice' / '06_dg_cspc_per_class.csv'
data = load_csv(str(DATA_CSV))

# FDI order: permanent 11-48 then deciduous 51-85
perm_order = [f"{q}{p}" for q in range(1, 5) for p in range(1, 9)]
deci_order = [f"{q}{p}" for q in range(5, 9) for p in range(1, 6)]
fdi_order = perm_order + deci_order

vals = np.array([data.get(f, 0.0) for f in fdi_order])
x = np.arange(len(fdi_order))

# Quadrant colors (permanent); deciduous muted orange / grey when dead.
quad_cmap = {1: "#E53935", 2: "#1E88E5", 3: "#43A047", 4: "#FDD835"}
bar_colors = []
for fdi in fdi_order:
    q = int(fdi[0])
    if q <= 4:
        bar_colors.append(quad_cmap[q])
    else:
        bar_colors.append("#FF9800" if data.get(fdi, 0.0) > 0.001 else "#BDBDBD")

# Figure
fig, ax = plt.subplots(figsize=(20, 7))

ax.bar(x, vals, color=bar_colors, edgecolor="white", linewidth=0.3, width=0.75)

# Dead markers (exactly-zero Dice).
for i, fdi in enumerate(fdi_order):
    if data.get(fdi, 0.0) == 0.0:
        ax.scatter(x[i], 0.02, marker='x', color="#C62828", s=22)

# Deciduous separator + subtle region tint.
ax.axvline(x=31.5, color="#333", linewidth=2, linestyle="--")
ax.axvspan(31.5, len(fdi_order) - 1, color="#FF9800", alpha=0.04)

# Stats (single block, top-right — now the only element in that corner).
perm_vals = [v for f, v in zip(fdi_order, vals) if int(f) < 50]
deci_vals = [v for f, v in zip(fdi_order, vals) if int(f) >= 50]
perm_alive = sum(1 for v in perm_vals if v > 0.001)
deci_alive = sum(1 for v in deci_vals if v > 0.001)
deci_dead = len(deci_vals) - deci_alive
overall = float(np.mean(vals))

stats_text = (
    f"Permanent:  \u03bc={np.mean(perm_vals):.3f}  |  {perm_alive}/32 activated\n"
    f"Deciduous:  \u03bc={np.mean(deci_vals):.3f}  |  {deci_alive}/20 activated  ({deci_dead} inactive)\n"
    f"Dice(teeth):  {overall:.3f}"
)
ax.text(0.99, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
        ha="right", va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAFAFA", edgecolor="#CCC"))

# Mean line (value is reported in the stats box, no separate floating label).
ax.axhline(y=overall, color="#333", linestyle="--", alpha=0.5, linewidth=1.5)

# Legend moved below the axis so it no longer crowds the upper band.
from matplotlib.patches import Patch
legend_patches = [
    Patch(facecolor="#E53935", label="UR (Q1)"), Patch(facecolor="#1E88E5", label="UL (Q2)"),
    Patch(facecolor="#43A047", label="LL (Q3)"), Patch(facecolor="#FDD835", label="LR (Q4)"),
    Patch(facecolor="#FF9800", label="Deciduous active"),
    Patch(facecolor="#BDBDBD", label="Deciduous inactive"),
]
ax.legend(handles=legend_patches, fontsize=8, ncol=6, loc="upper center",
          bbox_to_anchor=(0.5, -0.10), framealpha=0.9,
          title="Quadrant / Status", title_fontsize=9)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(fdi_order, fontsize=6, rotation=90)
ax.set_ylabel("Dice Score", fontsize=12)
ax.set_title("Per-class Dice under DG-CSPC (Dice(teeth) = 0.610)",
             fontsize=15, fontweight="bold")
ax.set_ylim(0, 1.10)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
OUT_DIR = Path(__file__).resolve().parent / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / 'per_fdi_dice.png'
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Done → {out}")
