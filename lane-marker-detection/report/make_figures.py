"""Phase 1 training curves for report.tex, from plots/recomputed_metrics.csv.

Run from the project root (lane-marker-detection/):
    uv run --with matplotlib python report/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "plots" / "recomputed_metrics.csv")))
epoch = [int(r["epoch"]) for r in rows]
loss = [float(r["val_loss"]) for r in rows]
iou = [float(r["miou"]) for r in rows]
f1 = [float(r["f1"]) for r in rows]

BLUE, ORANGE, AQUA, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8))
a.plot(epoch, loss, color=BLUE, lw=2)
a.set_title("Validation loss (BCE + Dice)", fontsize=10)
b.plot(epoch, f1, color=ORANGE, lw=2, label="F1")
b.plot(epoch, iou, color=AQUA, lw=2, label="IoU")
b.set_title("Validation IoU and F1", fontsize=10)
b.set_ylim(0.2, 0.5)
b.legend(frameon=False, fontsize=8, loc="center right")
for ax in (a, b):
    ax.set_xlabel("Epoch", color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(ROOT / "report" / "figures" / "phase1_metrics.pdf")
print("best IoU", max(iou), "epoch", epoch[iou.index(max(iou))], "| best F1", max(f1),
      "epoch", epoch[f1.index(max(f1))], "| loss first/min/last", loss[0], min(loss), loss[-1],
      "| final IoU/F1", iou[-1], f1[-1])
