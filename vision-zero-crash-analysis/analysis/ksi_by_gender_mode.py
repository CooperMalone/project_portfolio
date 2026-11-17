"""Fatal or serious injury (KSI) crashes in 2025, by gender and mode.

A KSI crash has at least one fatality or suspected serious injury.

Mode comes from `Type of NMV Involved`; a blank means only motor vehicles
were involved. Gender is the sex of the road user in that mode:
  - Pedestrian -> `Ped Sex`
  - Bicycle / Motorcycle / Scooter / ATV -> sex of the driver whose vehicle
    type matches that mode
  - Motor vehicle -> `Driver 1 Sex`

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib python analysis/ksi_by_gender_mode.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "ksi_by_gender_mode.png"

MODES = {  # Type of NMV Involved -> (label, matching `Type of Veh N` values)
    "Ped": ("Pedestrian", None),
    "Bike": ("Bicycle", {"BIKE"}),
    "MC": ("Motorcycle", {"MC"}),
    "Scooter": ("Scooter / ATV", {"SCOOT", "SCOOTER"}),
    "ATV": ("Scooter / ATV", {"ATV"}),
}
MODE_ORDER = ["Motor vehicle", "Pedestrian", "Motorcycle", "Bicycle", "Scooter / ATV"]
GENDERS = [("M", "Male", "#2a78d6"), ("F", "Female", "#eb6834"), ("U", "Unknown", "#a3a29d")]


def mode_and_gender(row):
    nmv = row["Type of NMV Involved"]
    if pd.isna(nmv):
        return "Motor vehicle", row["Driver 1 Sex"]
    label, veh_types = MODES.get(nmv, ("Scooter / ATV", None))
    if veh_types is None:
        return label, row["Ped Sex"]
    for n in (1, 2, 3):
        if str(row[f"Type of Veh {n}"]).upper() in veh_types:
            return label, row[f"Driver {n} Sex"]
    return label, None


def main():
    df = pd.read_excel(ROOT / "2025CrashData.xlsx")
    ksi = df[(df["# of Fatalities (1)"] > 0) | (df["# of Suspected Serious Injuries (2)"] > 0)]

    pairs = ksi.apply(mode_and_gender, axis=1, result_type="expand")
    pairs.columns = ["mode", "gender"]
    pairs["gender"] = pairs["gender"].where(pairs["gender"].isin(["M", "F"]), "U")

    counts = pd.crosstab(pairs["mode"], pairs["gender"]).reindex(
        index=MODE_ORDER, columns=[g for g, _, _ in GENDERS], fill_value=0
    )
    print(f"KSI crashes: {len(ksi)}")
    print(counts.assign(Total=counts.sum(axis=1)).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    bar_h = 0.26
    y = range(len(MODE_ORDER))
    for i, (code, label, color) in enumerate(GENDERS):
        ys = [v + (i - 1) * (bar_h + 0.02) for v in y]
        bars = ax.barh(ys, counts[code], height=bar_h, color=color, label=label)
        ax.bar_label(bars, labels=[str(v) if v else "" for v in counts[code]],
                     padding=3, fontsize=8, color="#52514e")

    ax.set_yticks(list(y), [f"{m}  (n={counts.loc[m].sum()})" for m in MODE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Number of crashes", color="#52514e")
    ax.set_title(f"Fatal or serious injury crashes by gender and mode, 2025 (n={len(ksi)})",
                 loc="left", fontsize=11, color="#0b0b0b")
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.01,
             "Gender is of the pedestrian, cyclist or motorcyclist; for motor-vehicle-only crashes, driver 1.",
             fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
