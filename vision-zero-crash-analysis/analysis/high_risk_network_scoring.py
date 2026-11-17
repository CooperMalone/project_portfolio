"""Reference figure for the High Risk Network point system used in
high_risk_network.py: points per risk factor and the score cut-offs for each tier.

Run from the repo root:
    uv run --with matplotlib python analysis/high_risk_network_scoring.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "high_risk_network_scoring.png"

FACTORS = [
    ("Road classification", "0-3", [
        ("Local", 0), ("Collector", 1), ("Major (or Major/x)", 2),
        ("Major/Major or Expressway", 3),
    ]),
    ("Average daily traffic", "0-3", [
        ("Under 5,000", 0), ("5,000-15,000", 1), ("15,000-30,000", 2), ("30,000+", 3),
    ]),
    ("Segment lanes", "0-2", [
        ("2 lanes or one way", 0), ("2 lanes with left turn", 1), ("4+ lanes", 2),
    ]),
    ("Intersection control", "0-2", [
        ("Roundabout", 0), ("Stop / yield / none", 1), ("Signal", 2),
    ]),
]
TIERS = [("High", "6-8", "#eb6834"), ("Moderate", "3-5", "#2a78d6"), ("Low", "0-2", "#a3a29d")]
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#e6e5e0"


def main():
    fig = plt.figure(figsize=(8, 6.2), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.text(4, 95, "High Risk Network point system", fontsize=14, color=INK,
            weight="bold", va="top")
    ax.text(4, 90.5, "Each crash location (LOCKEY) is scored on roadway risk factors only, "
            "not crash history. Total score 0-8.", fontsize=8.5, color=MUTED, va="top")

    # Two columns of factor tables
    col_x = [4, 52]
    top = 83
    for i, (name, rng, rows) in enumerate(FACTORS):
        x = col_x[i % 2]
        y = top - (i // 2) * 30
        ax.text(x, y, name, fontsize=10.5, color=INK, weight="bold", va="top")
        ax.text(x + 44, y, f"{rng} pts", fontsize=8.5, color=MUTED, va="top", ha="right")
        ax.plot([x, x + 44], [y - 3.6, y - 3.6], color=INK, linewidth=0.8)
        for j, (label, pts) in enumerate(rows):
            ry = y - 6 - j * 5
            ax.text(x, ry, label, fontsize=9, color=INK, va="center")
            ax.text(x + 44, ry, str(pts), fontsize=9, color=INK, va="center", ha="right",
                    weight="bold")
            ax.plot([x, x + 44], [ry - 2.5, ry - 2.5], color=RULE, linewidth=0.6)

    ax.text(4, 31, "Segments use lanes and intersections use traffic control, so every location "
            "gets one of the two (max 2 pts).", fontsize=7.5, color=MUTED, va="top", style="italic")

    # Tier bar: score 0..8 split into Low / Moderate / High
    ax.text(4, 19, "Risk tier by total score", fontsize=10.5, color=INK, weight="bold", va="top")
    bar_x0, bar_w, bar_y, bar_h = 4, 92, 6.5, 6
    unit = bar_w / 9
    for name, rng, color in TIERS:
        lo, hi = map(int, rng.split("-"))
        x0 = bar_x0 + lo * unit
        w = (hi - lo + 1) * unit
        ax.add_patch(plt.Rectangle((x0, bar_y), w - 0.4, bar_h, color=color, linewidth=0))
        ax.text(x0 + w / 2, bar_y + bar_h / 2, f"{name}  ({rng})", fontsize=9.5,
                color="white", weight="bold", ha="center", va="center")
    for s in range(9):
        ax.text(bar_x0 + (s + 0.5) * unit, bar_y - 1.5, str(s), fontsize=8, color=MUTED,
                ha="center", va="top")

    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    print(f"Saved {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
