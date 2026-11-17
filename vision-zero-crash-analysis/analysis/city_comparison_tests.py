"""Is each 2025 result a meaningful change from the Safe Streets Lincoln Vision
Zero Action Plan (2014-2023), or within ordinary year-to-year variation?

Each test is one-sided: assuming the City's rate still holds, how likely is a
2025 result at least as far from it, in the direction observed?
  - Shares (k of n): binomial with the City's share as p
  - Total KSI count: Poisson with the City's 10-year annual mean as lambda

Several tests are run, so treat a single small p-value as a strong lead, not
proof.

Run from the repo root:
    uv run --with pandas --with openpyxl python analysis/city_comparison_tests.py
"""

from math import exp, factorial
from pathlib import Path

import pandas as pd

from ksi_by_gender_mode import mode_and_gender
from ksi_share_by_mode import binom_p

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "figures" / "city_comparison_tests.csv"
HRN_TABLE = ROOT / "figures" / "high_risk_network_summary.csv"

CITY_KSI_PER_YEAR = 1305 / 10  # VZAP p. 5: 1,305 KSI crashes, 2014-2023


def poisson_cdf(k, lam):
    return sum(exp(-lam) * lam**i / factorial(i) for i in range(k + 1))


def main():
    df = pd.read_excel(ROOT / "2025CrashData.xlsx")
    ksi = df[(df["# of Fatalities (1)"] > 0) | (df["# of Suspected Serious Injuries (2)"] > 0)]
    n = len(ksi)

    pairs = ksi.apply(mode_and_gender, axis=1, result_type="expand")
    pairs.columns = ["mode", "gender"]
    known = pairs["gender"].isin(["M", "F"])
    male = (pairs["gender"] == "M").sum()
    signal = (ksi["Intersection Control Code"] == 2).sum()
    hrn_high = int(pd.read_csv(HRN_TABLE, index_col="risk_tier").loc["High", "ksi"])

    rows = [
        # (measure, city value, 2025 count, 2025 denominator, city source page)
        ("KSI in High Risk Network tier", 0.581, hrn_high, n, "p. 10"),
        ("KSI involving a pedestrian", 0.11, (pairs["mode"] == "Pedestrian").sum(), n, "p. 12"),
        ("KSI involving a motorcyclist", 0.19, (pairs["mode"] == "Motorcycle").sum(), n, "p. 12"),
        ("KSI involving a bicyclist", 0.06, (pairs["mode"] == "Bicycle").sum(), n, "p. 12"),
        ("KSI with a male road user (known gender)", 0.604, male, known.sum(), "p. 13"),
        ("KSI at a signalized intersection", 0.555, signal, n, "p. 15"),
    ]
    out = pd.DataFrame(
        [(m, 100 * c, k, d, 100 * k / d, binom_p(int(k), int(d), c), "binomial", src)
         for m, c, k, d, src in rows],
        columns=["measure", "city_pct", "count_2025", "n_2025", "pct_2025", "p_value", "test", "city_source"],
    )
    out.loc[len(out)] = ["KSI crashes per year", float("nan"), n, float("nan"), float("nan"),
                         poisson_cdf(n, CITY_KSI_PER_YEAR), f"Poisson, lambda={CITY_KSI_PER_YEAR}", "p. 5"]

    out["p_value"] = out["p_value"].map(lambda p: float(f"{p:.2g}"))
    out.round({"city_pct": 1, "pct_2025": 1}).to_csv(TABLE, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved {TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
