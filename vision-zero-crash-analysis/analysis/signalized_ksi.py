"""Fatal or serious injury (KSI) crashes by traffic control, 2025, compared with
the Safe Streets Lincoln Vision Zero Action Plan's claim that signalized
intersections (4.7% of intersections) hold 55.5% of KSI crashes (2014-2023).

The data carry two control fields, and both are reported:
  - `Intersection Control Code`: control at the crash, from the crash report
    (97 = not applicable, i.e. midblock)
  - `Lockey Traffic Control Code`: control at the location (LOCKEY); codes
    1-5 are intersections, the rest are midblock lane configurations

A location is an intersection when `Between Street (if midblock)` is blank.

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib python analysis/signalized_ksi.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "signalized_ksi.png"
TABLE = ROOT / "figures" / "signalized_ksi.csv"

CITY_SIGNAL_PCT = 55.5
CRASH_CONTROL = {1: "No control", 2: "Signal", 3: "All-way stop", 4: "Partial stop",
                 5: "Yield", 97: "Midblock"}
LOCKEY_CONTROL = {1: "Signal", 2: "Stop", 3: "Yield", 4: "No control", 5: "Roundabout"}


def main():
    df = pd.read_excel(ROOT / "2025CrashData.xlsx")
    ksi = df[(df["# of Fatalities (1)"] > 0) | (df["# of Suspected Serious Injuries (2)"] > 0)]
    n = len(ksi)

    by_crash = ksi["Intersection Control Code"].map(CRASH_CONTROL).fillna("Other / blank")
    by_lockey = ksi["Lockey Traffic Control Code"].map(LOCKEY_CONTROL).fillna("Midblock / blank")
    at_intersection = ksi["Between Street \n(if midblock)"].isna()

    table = pd.DataFrame({
        "measure": [
            "KSI crashes",
            "At an intersection (no Between Street)",
            "Signal, by crash report",
            "Signal, by location (LOCKEY) code",
        ],
        "count": [n, at_intersection.sum(), (by_crash == "Signal").sum(), (by_lockey == "Signal").sum()],
    })
    table["pct_of_ksi"] = (100 * table["count"] / n).round(1)
    table.to_csv(TABLE, index=False)
    print(table.to_string(index=False))
    print(f"\nCity 2014-2023: {CITY_SIGNAL_PCT}% of KSI at signalized intersections")
    print(f"\nKSI by crash-report control:\n{by_crash.value_counts().to_string()}")
    print(f"\nKSI by location control:\n{by_lockey.value_counts().to_string()}")

    counts = by_crash.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    colors = ["#2a78d6" if c == "Signal" else "#a3a29d" for c in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors, height=0.6)
    ax.bar_label(bars, labels=[f"{v} ({100 * v / n:.0f}%)" for v in counts.values],
                 padding=3, fontsize=8, color="#52514e")
    ax.invert_yaxis()
    ax.set_xlabel("Number of KSI crashes", color="#52514e")
    ax.set_title(f"Fatal or serious injury crashes by traffic control, 2025 (n={n})",
                 loc="left", fontsize=11, color="#0b0b0b")
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    fig.text(0.01, 0.01,
             f"Traffic control as recorded on each crash report. For comparison, the City's "
             f"Vision Zero plan found {CITY_SIGNAL_PCT}% of KSI crashes at signals (2014-2023).",
             fontsize=6.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    print(f"\nSaved {FIG.relative_to(ROOT)} and {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
