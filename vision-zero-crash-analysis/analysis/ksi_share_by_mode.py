"""Share of fatal or serious injury (KSI) crashes by mode, 2025 vs. the
Safe Streets Lincoln Vision Zero Action Plan (p. 12, 2014-2023).

Mode is assigned exactly as in `ksi_by_gender_mode.py`. Each 2025 share is
tested against the City's share with a one-sided binomial test: if the true
share were still the City's, how likely is a 2025 count at least this far
from it (in the direction observed)?

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib python analysis/ksi_share_by_mode.py
"""

from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ksi_by_gender_mode import MODE_ORDER, mode_and_gender

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures" / "ksi_share_by_mode.png"
TABLE = ROOT / "figures" / "ksi_share_by_mode.csv"

# Vision Zero Action Plan p. 12, "share of total KSI crashes". Scooter / ATV
# is not reported separately by the City.
CITY_PCT = {"Motor vehicle": 64, "Pedestrian": 11, "Motorcycle": 19, "Bicycle": 6}


def binom_p(k, n, p):
    """One-sided binomial p-value in the direction of k relative to n*p."""
    pmf = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    return sum(pmf[k:]) if k >= n * p else sum(pmf[: k + 1])


def main():
    df = pd.read_excel(ROOT / "2025CrashData.xlsx")
    ksi = df[(df["# of Fatalities (1)"] > 0) | (df["# of Suspected Serious Injuries (2)"] > 0)]
    modes = ksi.apply(lambda row: mode_and_gender(row)[0], axis=1)
    n = len(ksi)

    table = modes.value_counts().reindex(MODE_ORDER, fill_value=0).rename("ksi_2025").to_frame()
    table["pct_2025"] = 100 * table["ksi_2025"] / n
    table["pct_city_2014_2023"] = pd.Series(CITY_PCT)
    table["p_value"] = [
        binom_p(k, n, CITY_PCT[m] / 100) if m in CITY_PCT else float("nan")
        for m, k in table["ksi_2025"].items()
    ]
    table.round({"pct_2025": 1, "p_value": 4}).to_csv(TABLE, index_label="mode")
    print(f"KSI crashes: {n}")
    print(table.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    bar_h = 0.38
    y = range(len(MODE_ORDER))
    series = [
        ("pct_city_2014_2023", "City, 2014-2023", "#a3a29d", -0.5),
        ("pct_2025", f"2025 (n={n})", "#2a78d6", 0.5),
    ]
    for col, label, color, offset in series:
        vals = table[col].fillna(0)
        ys = [v + offset * (bar_h + 0.02) for v in y]
        bars = ax.barh(ys, vals, height=bar_h, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.0f}%" if not pd.isna(v) else "n/a" for v in table[col]],
                     padding=3, fontsize=8, color="#52514e")

    ax.set_yticks(list(y), MODE_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Share of KSI crashes (%)", color="#52514e")
    ax.set_title("Share of fatal or serious injury crashes by mode: City 2014-2023 vs. 2025",
                 loc="left", fontsize=11, color="#0b0b0b")
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.01,
             "City shares: Vision Zero Action Plan p. 12 (Scooter / ATV not reported). "
             "Mode as in ksi_by_gender_mode.py.",
             fontsize=6.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, facecolor="white")
    print(f"\nSaved {FIG.relative_to(ROOT)} and {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
