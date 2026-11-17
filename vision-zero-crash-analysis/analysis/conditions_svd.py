"""SVD of external crash conditions at Lincoln's crash hotspots, 2025.

Each row of the matrix is a location (LOCKEY) with at least MIN_CRASHES
crashes, so the top 25 locations from crash_map.py are all included. Each
column is one condition category, and the entry is the share of that
location's crashes that happened in it:

  Weather       Clear, Cloudy, Rain, Snow/sleet/freezing
  Road surface  Dry, Wet, Snow/ice/slush
  Light         Daylight, Dawn/dusk, Dark (lighted), Dark (not lighted)
  Time of day   Overnight, AM peak, Midday, PM peak, Evening
  Day of week   Weekday, Weekend

Shares are taken within each group, leaving out Unknown/Other codes and
rare categories (fog, blowing sand, crosswinds and so on). Columns are
centered on the average location, so a row says how that location differs
from a typical hotspot. Each group's block is then scaled to the same total
variance, so the 5-category time block doesn't outweigh the 2-category day
block.

The right singular vectors (V) are condition patterns; U·Σ gives each
location's score on them. Scores measure *when* a location's crashes happen,
not a rate: we don't know how many hours each place spends wet or dark.

Outputs (in figures/):
  conditions_svd.png            hotspots sorted by their Pattern 1 score,
                                colored by how their crash conditions differ
                                from the average hotspot
  conditions_svd_pattern2.png   the same for Pattern 2
  conditions_svd_explained.png  plain-language slide on reading the two
  conditions_svd_scores.csv     scores, shares and KSI counts per location

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib --with contextily --with folium --with scipy python analysis/conditions_svd.py
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr

from crash_map import ALL_ALLEYS, load, location_counts, nice

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

MIN_CRASHES = 15
MIN_LOADING = 0.15  # conditions weaker than this on a pattern are left off its figure
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"

# Condition group -> (column, {code: category}); unlisted codes are left out.
GROUPS = {
    "Weather": ("Weather Condition Code", {
        3: "Clear", 4: "Cloudy", 7: "Rain",
        2: "Snow/sleet/freezing", 6: "Snow/sleet/freezing", 9: "Snow/sleet/freezing",
        10: "Snow/sleet/freezing"}),
    "Road surface": ("Road Condition Code", {
        1: "Dry", 8: "Wet", 9: "Wet", 2: "Snow/ice/slush", 6: "Snow/ice/slush", 7: "Snow/ice/slush"}),
    "Light": ("Light Condition Code", {
        1: "Daylight", 2: "Dawn/dusk", 3: "Dark, lighted", 4: "Dark, not lighted"}),
    "Time of day": ("hour_bin", None),
    "Day of week": ("Day of Week 1=Sun", {1: "Weekend", 7: "Weekend", **{d: "Weekday" for d in range(2, 7)}}),
}
# SVD signs are arbitrary, so each pattern is oriented to make this condition's
# loading positive. (Weekend and Weekday load equally, so "largest loading
# positive" would flip PC1 on rounding noise.)
ORIENT = {"PC1": ("Day of week", "Weekend"), "PC2": ("Weather", "Clear")}
# Pattern -> (output file, title, rank in the subtitle). Titles follow the
# loadings: PC1 high = weekend/dark, PC2 high = clear/dry/evening.
HEATMAPS = {
    "PC1": ("conditions_svd.png",
            "The SVD sorts Lincoln's crash hotspots from weekend nights to weekday daytime", "biggest"),
    "PC2": ("conditions_svd_pattern2.png",
            "The SVD's second pattern sorts hotspots from clear evenings to cloudy or wet daytime",
            "second-biggest"),
}
PLAIN = {
    "Clear": "Clear weather", "Cloudy": "Cloudy weather", "Dry": "Dry road", "Wet": "Wet road",
    "Snow/ice/slush": "Snowy or icy road", "Dark, lighted": "Dark (lit)",
    "Overnight (0-6)": "Overnight 12-6am", "AM peak (6-9)": "Morning rush 6-9am",
    "Midday (9-15)": "Midday 9am-3pm", "PM peak (15-19)": "Evening rush 3-7pm",
    "Evening (19-24)": "Evening 7pm-12am",
}
HOUR_BINS = [(0, 6, "Overnight (0-6)"), (6, 9, "AM peak (6-9)"), (9, 15, "Midday (9-15)"),
             (15, 19, "PM peak (15-19)"), (19, 24, "Evening (19-24)")]


def hour_bin(t):
    if pd.isna(t) or t == 9999:
        return None
    hour = int(t) // 100
    return next(label for lo, hi, label in HOUR_BINS if lo <= hour < hi)


def condition_shares(m, keys):
    """Share of each location's crashes in each category, group by group."""
    m = m[m["LOCKEY"].isin(keys)].assign(hour_bin=m["Time of Accident"].map(hour_bin))
    blocks = {}
    for group, (col, codes) in GROUPS.items():
        cat = m[col] if codes is None else m[col].map(codes)
        order = ([label for _, _, label in HOUR_BINS] if codes is None
                 else list(dict.fromkeys(codes.values())))
        counts = pd.crosstab(m["LOCKEY"], cat).reindex(index=keys, columns=order, fill_value=0)
        blocks[group] = counts.div(counts.sum(axis=1), axis=0)
    return pd.concat(blocks, axis=1)


def svd(shares):
    X = shares - shares.mean()
    for group in GROUPS:  # equal total variance per group
        X[group] = X[group] / np.sqrt((X[group] ** 2).to_numpy().sum())
    U, s, Vt = np.linalg.svd(X.to_numpy(), full_matrices=False)
    cols = [f"PC{i + 1}" for i in range(len(s))]
    scores = pd.DataFrame(U * s, index=shares.index, columns=cols)
    loadings = pd.DataFrame(Vt.T, index=shares.columns, columns=cols)
    for pc, cond in ORIENT.items():
        if loadings.loc[cond, pc] < 0:
            scores[pc], loadings[pc] = -scores[pc], -loadings[pc]
    return s, scores, loadings


def weekday_daylight(m, keys):
    """Share of each location's crashes that were both on a weekday and in daylight,
    out of crashes with a known light condition."""
    m = m[m["LOCKEY"].isin(keys) & m["Light Condition Code"].isin(GROUPS["Light"][1])]
    both = m["Light Condition Code"].eq(1) & m["Day of Week 1=Sun"].between(2, 6)
    return both.groupby(m["LOCKEY"]).mean().reindex(keys)


def heatmap(table, shares, loadings, var, pc):
    """Hotspots sorted by their score on one pattern; cells are colored by
    percentage points above or below the average hotspot and labeled with the
    hotspot's own share. Sized as a 16:9 slide."""
    filename, title, which = HEATMAPS[pc]
    n_pc = int(pc[2:]) - 1
    pc1 = loadings[pc]
    picks = pc1[pc1.abs() >= MIN_LOADING].sort_values(ascending=False).index
    order = table[pc].sort_values(ascending=False).index
    avg = shares[picks].mean()
    diff = (shares.loc[order, picks] - avg) * 100

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f4f3ef", ORANGE])
    lim = np.ceil(np.abs(diff.to_numpy()).max() / 5) * 5
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_axes((0.222, 0.085, 0.71, 0.78))  # grid plus row labels centered
    im = ax.imshow(diff.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(np.arange(len(picks)),
                  [f"{PLAIN.get(c, c)}\navg {avg[(g, c)]:.0%}" for g, c in picks],
                  fontsize=10 if len(picks) <= 8 else 9, color=INK)
    ax.xaxis.tick_top()
    names = [nice(table.loc[k, "location"]) for k in order]
    ax.set_yticks(np.arange(len(order)), names, fontsize=8, color=INK)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(len(picks) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    actual = shares.loc[order, picks].to_numpy()
    for (i, j), v in np.ndenumerate(actual):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=INK)

    for side in ax.spines.values():
        side.set_visible(False)
    # Divider between conditions that rise with the pattern and those that fall.
    split = (pc1[picks] > 0).sum() - 0.5
    ax.axvline(split, color=INK, linewidth=1.5)

    cax = fig.add_axes((0.417, 0.05, 0.32, 0.012))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Cell color: hotspot vs. the average hotspot (percentage points)", fontsize=10, color=MUTED)
    cbar.ax.tick_params(labelsize=9, colors=MUTED, length=0)
    cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    cbar.outline.set_visible(False)

    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=16, color=INK, fontweight="bold")
    fig.text(0.01, 0.94,
             f"Rows: every Lincoln intersection or road segment with {MIN_CRASHES}+ crashes in 2025 ({len(order)} locations). "
             f"The SVD's {which} pattern accounts for {var[n_pc]:.0%} of how these hotspots differ.",
             fontsize=11, color=MUTED, va="top")
    fig.savefig(FIG / filename, facecolor="white")
    plt.close(fig)


def explainer(table, shares, var):
    """A plain-language slide: what the SVD did, how to read the two heatmaps,
    what each pattern found and what the charts don't say."""
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.text(0.03, 0.94, "How to read the two SVD heatmaps", fontsize=24, color=INK,
             fontweight="bold", va="top")
    wrap = lambda s, w=52: "\n".join(textwrap.fill(p, w) for p in s.split("\n"))
    head = dict(fontsize=15, color=INK, fontweight="bold", va="top")
    body = dict(fontsize=12, color=INK, va="top", linespacing=1.45)
    cols = [0.03, 0.36, 0.69]

    # 1. What we did
    fig.text(cols[0], 0.83, "1. What we did", **head)
    fig.text(cols[0], 0.77, wrap(
        f"We took the {len(table)} Lincoln locations with {MIN_CRASHES} or more crashes in 2025.\n\n"
        "For each one we worked out what % of its crashes happened in each condition: weather, "
        "road surface, light, time of day and weekday vs. weekend.\n\n"
        "The SVD looks for conditions that rise and fall together from one location to the next. "
        "Each group of conditions that moves together is a pattern.\n\n"
        "Every location gets a score on each pattern. The rows of each chart are sorted by that score."),
        **body)

    # 2. How to read a chart, with two sample cells
    fig.text(cols[1], 0.83, "2. How to read a chart", **head)
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f4f3ef", ORANGE])
    norm = TwoSlopeNorm(0, -35, 35)
    (key, w), (key2, w2) = [(k, ("Day of week", "Weekend")) for k in
                            (shares[("Day of week", "Weekend")].idxmax(),
                             shares[("Day of week", "Weekend")].idxmin())]
    avg = shares[w].mean()
    for y, k in [(0.745, key), (0.665, key2)]:
        v = shares.loc[k, w]
        fig.patches.append(plt.Rectangle((cols[1], y - 0.055), 0.07, 0.055, transform=fig.transFigure,
                                         color=cmap(norm((v - avg) * 100))))
        fig.text(cols[1] + 0.035, y - 0.0275, f"{v:.0%}", ha="center", va="center", fontsize=13, color=INK)
        side = "more" if v > avg else "fewer"
        fig.text(cols[1] + 0.08, y - 0.0275,
                 f"{nice(table.loc[k, 'location'])}: {v:.0%} of its crashes\nwere on weekends "
                 f"(average {avg:.0%})\n→ {side} than average, so {'orange' if v > avg else 'blue'}",
                 fontsize=11, color=INK, va="center")
    fig.text(cols[1], 0.585, wrap(
        "Each row is a location; each column is a condition.\n\n"
        "The number is the % of that location's crashes in that condition.\n\n"
        "The color compares it with the average location (shown under each column name): orange is "
        "above average, blue is below.\n\n"
        "The black line splits conditions that rise together (left) from the ones that fall when they "
        "do (right). Going down the rows, the orange shifts from the left side to the right."),
        **body)

    # 3. What the SVD found
    fig.text(cols[2], 0.83, "3. What the SVD found", **head)
    ends = lambda pc: [table[pc].sort_values(ascending=asc).index[0] for asc in (False, True)]
    y = 0.77
    for pc, n, top, bottom in [
            ("PC1", 1, "weekends, after dark and evenings", "weekdays in daylight"),
            ("PC2", 2, "clear, dry evenings", "cloudy, wet or snowy daytime")]:
        hi, lo = ends(pc)
        fig.text(cols[2], y, f"Pattern {n}: {var[n - 1]:.0%} of how the locations differ",
                 fontsize=13, color=INK, fontweight="bold", va="top")
        fig.text(cols[2], y - 0.045, wrap(
            f"Top rows: crashes lean toward {top} (e.g. {nice(table.loc[hi, 'location'])}).\n"
            f"Bottom rows: crashes lean toward {bottom} (e.g. {nice(table.loc[lo, 'location'])}).", 50),
            **body)
        y -= 0.235
    fig.text(cols[2], y + 0.02, wrap(
        f"The two patterns don't overlap. Together they cover {var[0] + var[1]:.0%} of how the "
        "locations differ. Pattern 1 is the bigger of the two.", 50), **body)

    # What the charts don't say
    fig.patches.append(plt.Rectangle((0.03, 0.04), 0.94, 0.16, transform=fig.transFigure,
                                     color="#f4f3ef"))
    fig.text(0.045, 0.185, "What the charts don't tell us", **head)
    fig.text(0.045, 0.135,
             "•  How dangerous a place is. Neither pattern is linked to how severe the crashes are "
             "(killed or seriously injured).\n"
             "•  Why crashes happen then. A daytime location may simply have most of its traffic in the "
             "day; we don't have traffic counts by hour.",
             fontsize=12, color=INK, va="top", linespacing=1.6)
    fig.savefig(FIG / "conditions_svd_explained.png", facecolor="white")
    plt.close(fig)


def main():
    _, mapped = load()
    locs = location_counts(mapped)
    locs["rank"] = range(1, len(locs) + 1)
    locs = locs[(locs["crashes"] >= MIN_CRASHES) & (locs.index != ALL_ALLEYS)]
    shares = condition_shares(mapped, locs.index)
    s, scores, loadings = svd(shares)

    var = s ** 2 / (s ** 2).sum()
    table = locs[["location", "rank", "crashes", "ksi"]].join(scores[["PC1", "PC2", "PC3"]])
    table["ksi_share"] = table["ksi"] / table["crashes"]
    table["weekday_daylight"] = weekday_daylight(mapped, locs.index)
    for pc in HEATMAPS:
        heatmap(table, shares, loadings, var, pc)
    explainer(table, shares, var)

    out = table.join(shares.set_axis([f"{g}: {c}" for g, c in shares.columns], axis=1).round(3))
    out.round({"PC1": 3, "PC2": 3, "PC3": 3, "ksi_share": 3, "weekday_daylight": 3}).to_csv(FIG / "conditions_svd_scores.csv")

    print(f"Locations: {len(table)} (with {MIN_CRASHES}+ crashes), conditions: {shares.shape[1]}")
    print("Variance share: " + ", ".join(f"PC{i + 1} {v:.1%}" for i, v in enumerate(var[:5])))
    for pc in ("PC1", "PC2"):
        top = loadings[pc].sort_values()
        print(f"\n{pc} high: " + ", ".join(f"{c} {v:+.2f}" for (_, c), v in top[::-1].head(4).items()))
        print(f"{pc} low:  " + ", ".join(f"{c} {v:+.2f}" for (_, c), v in top.head(4).items()))
        r, pv = spearmanr(table[pc], table["ksi_share"])
        print(f"{pc} vs. KSI share of crashes: Spearman rho = {r:+.2f} (p = {pv:.2f})")
    print("\nSaved conditions_svd.png, _pattern2.png, _explained.png and _scores.csv in figures/")


if __name__ == "__main__":
    main()
