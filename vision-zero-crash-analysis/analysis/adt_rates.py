"""Traffic-adjusted ranking of Lincoln intersections, 2025.

The top 25 in crash_map.py ranks locations by raw crash count, so it partly
just finds the busiest intersections. This script asks a different question:
which intersections have more crashes than their traffic volume (ADT) and
type of control explain?

Method
  1. Crash rate per million entering vehicles (MEV):
         rate = crashes * 1,000,000 / (365 * ADT)
     Easy to explain, but very noisy where traffic is low, so it is only
     ranked for intersections with at least MIN_CRASHES_FOR_RATE crashes.
  2. Safety performance functions (SPFs), the standard traffic-engineering
     model of expected crashes given traffic, fit separately for signals,
     roundabouts and stop/yield/uncontrolled intersections:
         expected crashes  mu = exp(a) * ADT^b
     Counts are negative binomial (Poisson with extra spread, dispersion
     alpha). Intersections with no 2025 crash never appear in the crash
     file, so each SPF is fit by maximum likelihood on the zero-truncated
     negative binomial, which accounts for the missing zeros.
     b = 1 means crashes are proportional to traffic, so a plain rate is
     fair; b > 1 means busier intersections have higher rates even when
     they are typical, so ranking by rate still favours busy places.
  3. Empirical Bayes (Highway Safety Manual): each intersection's long-run
     crash frequency is a weighted average of its SPF and its own count,
         EB = w * mu + (1 - w) * crashes,   w = 1 / (1 + alpha * mu),
     which pulls small, noisy counts toward the SPF. The ranking uses
     excess = EB - mu, the crashes per year above what a typical
     intersection with the same traffic and control would have.

Which intersections are rated
  - Intersections only: a segment rate needs a length, which the file lacks.
  - Control from `Lockey Traffic Control Code` (1 signal, 2 stop, 3 yield,
    4 none, 5 roundabout), the most common code among a location's crashes.
  - ADT above MIN_ADT. 406 intersections, almost all local/local, have ADT
    of exactly 500, 750, 1,000, 2,000 or 2,500, which look like default
    estimates, not counts. Which street ADT belongs to (cross street, major
    street or total entering) still needs checking with the City.

Outputs (in figures/):
  adt_crashes_vs_traffic.png    crashes against ADT for each control type,
                                with its fitted SPF and the three groups
  adt_excess_top20.png          top 20 by excess: actual crashes against the
                                number expected for their traffic and control
  adt_ranking.csv               every rated intersection: count, rate,
                                expected, EB, excess, ranks and group

Run from the repo root:
    uv run --with pandas --with openpyxl --with matplotlib --with contextily --with folium --with scipy python analysis/adt_rates.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

TOP_N = 25
MIN_CRASHES_FOR_RATE = 10
MIN_ADT = 2500  # at or below: mostly placeholder values on local streets
CONTROL = {1: "Signal", 2: "Stop/yield/none", 3: "Stop/yield/none", 4: "Stop/yield/none",
           5: "Roundabout"}
CONTROL_ORDER = ["Signal", "Stop/yield/none", "Roundabout"]
LABELS_PER_PANEL = {"Signal": 3, "Stop/yield/none": 1, "Roundabout": 2}
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
# First three categorical slots of the reference palette (validated all-pairs).
GROUPS = {
    "Consistent hotspot": "#2a78d6",   # top 25 by count and by excess
    "Hidden risk": "#eb6834",          # top 25 by excess only
    "Busy, not unusual": "#1baf7a",    # top 25 by count only
}


def hessian(f, x, h=1e-4):
    """Central-difference Hessian, for standard errors at the optimum."""
    x, n = np.asarray(x, float), len(x)
    e = np.eye(n) * h
    return np.array([[(f(x + e[i] + e[j]) - f(x + e[i] - e[j])
                       - f(x - e[i] + e[j]) + f(x - e[i] - e[j])) / (4 * h * h)
                      for j in range(n)] for i in range(n)])


def fit_spf(crashes, adt):
    """Zero-truncated negative binomial fit of crashes ~ exp(a) * ADT^b.

    Returns (a, b, alpha, standard errors of (a_centred, b, log alpha)).
    """
    y, x = np.asarray(crashes, float), np.log(np.asarray(adt, float))
    xc = x - x.mean()  # centred for a well-conditioned fit

    def nll(theta):
        a, b, log_alpha = theta
        mu, r = np.exp(a + b * xc), np.exp(-log_alpha)
        log_p0 = r * np.log(r / (r + mu))
        log_p = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + log_p0 + y * np.log(mu / (r + mu))
        return -(log_p - np.log1p(-np.exp(log_p0))).sum()

    res = minimize(nll, x0=[np.log(y.mean()), 0.5, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-8, fatol=1e-10, maxiter=10_000))
    if not res.success:
        raise RuntimeError(f"SPF fit did not converge: {res.message}")
    a_c, b, log_alpha = res.x
    se = np.sqrt(np.diag(np.linalg.inv(hessian(nll, res.x))))
    return a_c - b * x.mean(), b, np.exp(log_alpha), se


def traffic_adjusted(m, locs):
    """Rate, SPF expectation and EB excess for every rated intersection.

    `m` is the crash table and `locs` is crash_map.location_counts(m): one
    row per LOCKEY, in crash-count order. Returns the rated intersections
    sorted by excess, and one row per control type describing its SPF.
    """
    codes = m[m["Lockey Traffic Control Code"].isin(CONTROL)].groupby("LOCKEY")[
        "Lockey Traffic Control Code"].agg(lambda s: s.mode().iloc[0])
    ix = locs.assign(count_rank=range(1, len(locs) + 1),
                     control=codes.reindex(locs.index).map(CONTROL))
    ix = ix[~ix["segment"] & ix["control"].notna() & (ix["adt"] > MIN_ADT)].copy()

    fits = []
    for control in CONTROL_ORDER:
        d = ix["control"] == control
        a, b, alpha, se = fit_spf(ix.loc[d, "crashes"], ix.loc[d, "adt"])
        mu = np.exp(a) * ix.loc[d, "adt"] ** b
        w = 1 / (1 + alpha * mu)
        ix.loc[d, "expected"] = mu
        ix.loc[d, "eb"] = w * mu + (1 - w) * ix.loc[d, "crashes"]
        fits.append(dict(control=control, n=d.sum(), a=a, b=b, se_b=se[1], alpha=alpha))

    ix["rate_mev"] = ix["crashes"] * 1e6 / (365 * ix["adt"])
    ix["excess"] = ix["eb"] - ix["expected"]
    ix["times_expected"] = ix["eb"] / ix["expected"]
    enough = ix["crashes"] >= MIN_CRASHES_FOR_RATE
    ix["rate_rank"] = ix["rate_mev"].where(enough).rank(ascending=False, method="first")
    ix = ix.sort_values("excess", ascending=False)
    ix["excess_rank"] = range(1, len(ix) + 1)

    top_count, top_excess = ix["count_rank"] <= TOP_N, ix["excess_rank"] <= TOP_N
    ix["group"] = np.select([top_count & top_excess, top_excess, top_count], list(GROUPS),
                            default="")
    return ix, pd.DataFrame(fits).set_index("control")


def nice(name):
    from crash_map import nice as crash_map_nice
    return crash_map_nice(name)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def titles(fig, title, note):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, note, fontsize=7, color=MUTED, ha="left", va="bottom", wrap=True)


def scatter_chart(ix, fits):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), dpi=200, sharey=True)
    for ax, control in zip(axes, CONTROL_ORDER):
        d, f = ix[ix["control"] == control], fits.loc[control]
        other = d[d["group"] == ""]
        ax.scatter(other["adt"], other["crashes"], s=10, color=GRAY, alpha=0.5, linewidths=0)
        for g, color in GROUPS.items():
            dg = d[d["group"] == g]
            ax.scatter(dg["adt"], dg["crashes"], s=36, color=color, edgecolors="white",
                       linewidths=1.5, zorder=3)
        xs = np.geomspace(d["adt"].min(), d["adt"].max(), 200)
        ax.plot(xs, np.exp(f["a"]) * xs ** f["b"], color=INK, linewidth=2, zorder=2)
        # Direct labels on the clearest stories only, staggered so close points don't collide.
        for (_, r), dy in zip(d.head(LABELS_PER_PANEL[control]).iterrows(), (8, -1, -10)):
            ax.annotate(nice(r["location"]), (r["adt"], r["crashes"]), xytext=(-8, dy),
                        textcoords="offset points", ha="right", va="center", fontsize=7,
                        color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
                        arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.6, shrinkB=4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{control}  (n={int(f['n'])}, b={f['b']:.2f})", loc="left", fontsize=10,
                     color=INK)
        ax.set_xlabel("Vehicles per day (ADT, log scale)", fontsize=8, color=MUTED)
        ax.set_xticks([3000, 5000, 10000, 20000, 50000])
        ax.set_yticks([1, 2, 5, 10, 20, 50])
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            axis.set_minor_formatter(plt.NullFormatter())
        style(ax)
    axes[0].set_ylabel("Crashes in 2025 (log scale)", fontsize=8, color=MUTED)
    handles = [plt.Line2D([], [], color=INK, linewidth=2)] + [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c, markeredgecolor="white")
        for c in [GRAY, *GROUPS.values()]]
    labels = ["Expected for that traffic (SPF)", "Other intersections",
              *(f"{g} (n={(ix['group'] == g).sum()})" for g in GROUPS)]
    fig.legend(handles, labels, loc="upper right", ncols=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.99, 0.95))
    titles(fig, "Busy or dangerous? Crashes against traffic at Lincoln intersections, 2025",
           f"Each dot is an intersection with at least one 2025 crash and ADT above {MIN_ADT:,} "
           f"(n={len(ix):,}). Curves: zero-truncated negative binomial fit per control type, "
           f"expected crashes = exp(a)·ADT^b. Groups compare the top {TOP_N} by crash count with "
           f"the top {TOP_N} by excess crashes (empirical Bayes estimate minus expected). Where a "
           f"curve sits below the dots (expected under 1 crash), that is because intersections "
           f"with no 2025 crash are not in the data. The roundabout curve rests on only "
           f"{int(fits.loc['Roundabout', 'n'])} locations.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(FIG / "adt_crashes_vs_traffic.png", facecolor="white")
    plt.close(fig)


def excess_chart(ix):
    top = ix.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=200)
    y = np.arange(len(top))
    ax.hlines(y, top["expected"], top["crashes"], color=GRID, linewidth=2, zorder=1)
    ax.scatter(top["expected"], y, s=40, color=GRAY, edgecolors="white", linewidths=1.5, zorder=2)
    ax.scatter(top["crashes"], y, s=64, color=top["group"].map(GROUPS), edgecolors="white",
               linewidths=1.5, zorder=3)
    ax.set_yticks(y, [f"{nice(n)}  ·  {c.lower()}  ·  #{int(k)} by count"
                      for n, c, k in zip(top["location"], top["control"], top["count_rank"])],
                  fontsize=8, color=INK)
    ax.set_xlabel("Crashes in 2025", fontsize=9, color=MUTED)
    ax.set_xlim(0, top["crashes"].max() * 1.08)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=GRAY,
                          markeredgecolor="white")] + [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=GROUPS[g],
                   markeredgecolor="white") for g in GROUPS if (top["group"] == g).any()]
    labels = ["Expected for its traffic and control"] + [
        f"Actual crashes: {g.lower()}" for g in GROUPS if (top["group"] == g).any()]
    ax.legend(handles, labels, loc="lower right", frameon=True, facecolor="white",
              edgecolor="none", fontsize=8)
    style(ax)
    ax.grid(axis="y", visible=False)
    titles(fig, "Intersections with the most crashes above what their traffic predicts, 2025",
           "Top 20 intersections by excess crashes (empirical Bayes estimate minus the SPF "
           "expectation for the same ADT and control type). Gray: crashes expected for a typical "
           "intersection like it. The label gives the rank on the crash-count top 25 list.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIG / "adt_excess_top20.png", facecolor="white")
    plt.close(fig)


def ranking_csv(ix):
    out = ix.reset_index()
    pd.DataFrame({
        "excess_rank": out["excess_rank"],
        "location": out["location"].map(nice),
        "control": out["control"],
        "crashes": out["crashes"],
        "ksi_crashes": out["ksi"],
        "adt": out["adt"].astype(int),
        "rate_per_mev": out["rate_mev"].round(2),
        "expected": out["expected"].round(2),
        "eb_estimate": out["eb"].round(2),
        "excess": out["excess"].round(2),
        "times_expected": out["times_expected"].round(2),
        "count_rank": out["count_rank"],
        f"rate_rank_min{MIN_CRASHES_FOR_RATE}": out["rate_rank"].astype("Int64"),
        "group": out["group"],
        "hrn_tier": out["hrn_tier"],
        "lockey": out["LOCKEY"],
    }).to_csv(FIG / "adt_ranking.csv", index=False)


def main():
    from crash_map import load, location_counts

    _, mapped = load()
    ix, fits = traffic_adjusted(mapped, location_counts(mapped))
    scatter_chart(ix, fits)
    excess_chart(ix)
    ranking_csv(ix)

    print(f"\nRated intersections: {len(ix):,}\nSPFs, expected = exp(a) * ADT^b:")
    print(fits.round(3).to_string())
    print("\n" + ix["group"].value_counts().drop("", errors="ignore").to_string())
    cols = ["location", "control", "crashes", "adt", "expected", "excess", "count_rank"]
    print("\nTop 15 by excess:\n" + ix.head(15)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
