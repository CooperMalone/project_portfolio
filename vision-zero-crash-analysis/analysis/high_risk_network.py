"""High Risk Network (HRN) summary statistics for 2025, modeled on the
Safe Streets Lincoln Vision Zero Action Plan (p. 10).

The plan scores street segments on risk factors that are *not* crash history
(roadway type, traffic volume, posted speed, lanes, truck/bus routes, VRU
presence), sorts them into High / Moderate / Low tiers, and reports each tier's
share of roadway miles and of KSI crashes.

The crash data has no roadway miles, speed limits or truck/bus routes, so this
version scores each location (LOCKEY) on the three risk factors it does carry,
which are constant per location:

  Road classification (0-3)   Local 0 · Collector 1 · Major (or Major/x) 2 ·
                              Major/Major or Expressway 3
  Average daily traffic (0-3) <5k 0 · 5k-15k 1 · 15k-30k 2 · 30k+ 3
  Cross-section / control     Segments: 4+ lanes 2 · 2 lanes w/ left turn 1 ·
  (0-2)                       2 lanes or one way 0
                              Intersections: signal 2 · stop/yield/none 1 ·
                              roundabout 0

Total score 0-8 -> High (6-8), Moderate (3-5), Low (0-2).

Locations are counted instead of roadway miles, and only locations with at
least one 2025 crash appear in the data, so "% of locations" is a share of
crash locations, not of the whole street network.

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib python analysis/high_risk_network.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "high_risk_network.png"
TABLE = ROOT / "figures" / "high_risk_network_summary.csv"

ROAD_CLASS_PTS = {0: 0, 33: 0, 2: 1, 22: 1, 23: 1, 1: 2, 12: 2, 13: 2, 11: 3, 4: 3}
CONTROL_PTS = {
    # segments (lane configuration)
    11: 2, 12: 2, 15: 2, 17: 2, 18: 2, 22: 1, 23: 1, 25: 0, 35: 0,
    # intersections (traffic control)
    1: 2, 2: 1, 3: 1, 4: 1, 5: 0,
}
TIERS = ["High", "Moderate", "Low"]


def adt_pts(adt):
    return 3 if adt >= 30000 else 2 if adt >= 15000 else 1 if adt >= 5000 else 0


def tier(score):
    return "High" if score >= 6 else "Moderate" if score >= 3 else "Low"


def main():
    df = pd.read_excel(ROOT / "2025CrashData.xlsx")
    df["ksi"] = (df["# of Fatalities (1)"] > 0) | (df["# of Suspected Serious Injuries (2)"] > 0)

    loc = df.groupby("LOCKEY").agg(
        road_class=("Road Classification Code", "first"),
        adt=("ADT", "first"),
        control=("Lockey Traffic Control Code", "first"),
        crashes=("ksi", "size"),
        ksi=("ksi", "sum"),
    )
    loc["score"] = (
        loc["road_class"].map(ROAD_CLASS_PTS).fillna(0)
        + loc["adt"].map(adt_pts)
        + loc["control"].map(CONTROL_PTS).fillna(0)
    ).astype(int)
    loc["tier"] = loc["score"].map(tier)

    summary = loc.groupby("tier").agg(
        locations=("score", "size"), crashes=("crashes", "sum"), ksi=("ksi", "sum")
    ).reindex(TIERS)
    summary["pct_locations"] = 100 * summary["locations"] / summary["locations"].sum()
    summary["pct_crashes"] = 100 * summary["crashes"] / summary["crashes"].sum()
    summary["pct_ksi"] = 100 * summary["ksi"] / summary["ksi"].sum()
    summary["ksi_per_100_locations"] = 100 * summary["ksi"] / summary["locations"]

    summary.round(1).to_csv(TABLE, index_label="risk_tier")
    print(summary.round(1).to_string())
    print(f"\nScore distribution:\n{loc['score'].value_counts().sort_index().to_string()}")

    series = [
        ("pct_locations", "% of crash locations", "#a3a29d"),
        ("pct_crashes", "% of all crashes", "#2a78d6"),
        ("pct_ksi", "% of KSI crashes", "#eb6834"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    bar_h = 0.26
    y = range(len(TIERS))
    for i, (col, label, color) in enumerate(series):
        ys = [v + (i - 1) * (bar_h + 0.02) for v in y]
        bars = ax.barh(ys, summary[col], height=bar_h, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in summary[col]],
                     padding=3, fontsize=8, color="#52514e")

    ax.set_yticks(list(y), [f"{t} risk\n({summary.loc[t, 'locations']} locations)" for t in TIERS])
    ax.invert_yaxis()
    ax.set_xlabel("Share of total (%)", color="#52514e")
    ax.set_xlim(0, max(summary[[c for c, _, _ in series]].max()) * 1.12)
    ax.set_title("High Risk Network summary, 2025: risk tier vs. share of crashes",
                 loc="left", fontsize=11, color="#0b0b0b")
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.01,
             "Risk score uses road class, average daily traffic and lanes/control only (no crash history). "
             "KSI = fatal or suspected serious injury.",
             fontsize=6.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    print(f"\nSaved {FIG.relative_to(ROOT)} and {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
