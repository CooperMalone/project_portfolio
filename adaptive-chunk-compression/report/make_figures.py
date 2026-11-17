"""Regenerate every number and figure in report.tex.

Run from the project root (adaptive-chunk-compression/):
    uv run --with matplotlib python report/make_figures.py

Writes report/results.json and report/figures/*.pdf. The real-text results need
data/t8.shakespeare.txt; the synthetic results need no data.
"""
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comparisons.synthetic import (generate_clustered, generate_monotonic_gaps,
                                   generate_sparse_with_outliers, generate_uniform_small)
from compression.bitPacking import BitPackEncoder
from compression.elias import EliasGammaEncoder
from compression.vByte import VByteEncoder
from utils.posting_list import text_to_posting_list

OUT = ROOT / "report"
FIG = OUT / "figures"
CHUNK_SIZES = [50, 100, 200, 500]
SYNTH_CHUNK = 128
TIMING_REPEATS = 3

NAMES = ["vByte", "Elias gamma", "Bit packing", "Adaptive"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

GAP_CODECS = {"vByte": VByteEncoder.encode_gaps, "Elias gamma": EliasGammaEncoder.encode_gaps,
              "Bit packing": BitPackEncoder.encode_gaps}
SEQ_CODECS = {"vByte": VByteEncoder.encode_sequence, "Elias gamma": EliasGammaEncoder.encode_sequence,
              "Bit packing": BitPackEncoder.encode_sequence}


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def real_text(path):
    text = path.read_text(encoding="utf-8")
    lists = [p for docs in text_to_posting_list(text, doc_id=1).values() for p in docs.values()]
    original = 4 * sum(len(p) for p in lists)
    out = {"tokens": len(text.split()), "terms": len(lists),
           "singleton_terms": sum(len(p) == 1 for p in lists), "original_bytes": original, "by_chunk": {}}
    for cs in CHUNK_SIZES:
        sizes = Counter()
        picks = Counter()
        n_chunks = 0
        for p in lists:
            for ch in chunks(p, cs):
                n_chunks += 1
                s = {k: len(f(ch)) for k, f in GAP_CODECS.items()}
                sizes.update(s)
                best = min(s, key=s.get)
                picks[best] += 1
                sizes["Adaptive (no IDs)"] += s[best]
        sizes["Adaptive"] = sizes["Adaptive (no IDs)"] + n_chunks  # 1-byte codec ID per chunk
        sizes["Adaptive (2-bit IDs)"] = sizes["Adaptive (no IDs)"] + math.ceil(n_chunks / 4)
        out["by_chunk"][cs] = {
            "chunks": n_chunks,
            "ratio": {k: v / original for k, v in sizes.items()},
            "picked": {k: v / n_chunks for k, v in picks.items()},
        }

    # Compression time: each codec over the whole index, averaged over chunk sizes and repeats.
    def adaptive_fn(ch):
        encoded = {k: f(ch) for k, f in GAP_CODECS.items()}
        return min(encoded.values(), key=len)

    timers = dict(GAP_CODECS, Adaptive=adaptive_fn)
    times = {k: [] for k in timers}
    for _ in range(TIMING_REPEATS):
        for cs in CHUNK_SIZES:
            for name, fn in timers.items():
                start = time.perf_counter()
                for p in lists:
                    for ch in chunks(p, cs):
                        fn(ch)
                times[name].append(time.perf_counter() - start)
    out["seconds"] = {k: sum(v) / len(v) for k, v in times.items()}
    return out


def synthetic(seed=0):
    random.seed(seed)
    datasets = {"Uniform, small values": generate_uniform_small(),
                "Clustered": generate_clustered(),
                "Sparse with outliers": generate_sparse_with_outliers(),
                "Increasing positions": generate_monotonic_gaps()}
    # Mixed: the four patterns above alternate in blocks of 1,000 values within one list,
    # the setting (dense and sparse regions in the same list) that motivates per-chunk choice.
    gens = [lambda: generate_uniform_small(1000), lambda: generate_clustered(1000),
            lambda: generate_sparse_with_outliers(1000), lambda: generate_monotonic_gaps(1000)]
    datasets["Mixed"] = [v for i in range(10) for v in gens[i % 4]()]
    out = {}
    for name, data in datasets.items():
        whole = {k: len(f(data)) for k, f in SEQ_CODECS.items()}
        chunked = {k: sum(len(f(c)) for c in chunks(data, SYNTH_CHUNK)) for k, f in SEQ_CODECS.items()}
        parts = chunks(data, SYNTH_CHUNK)
        adaptive = sum(min(len(f(c)) for f in SEQ_CODECS.values()) for c in parts) + len(parts)
        out[name] = {"whole": whole, "chunked": chunked, "adaptive": adaptive, "chunks": len(parts)}
    return out


def plot_real(res):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.2
    for i, name in enumerate(NAMES):
        ys = [res["by_chunk"][cs]["ratio"][name] for cs in CHUNK_SIZES]
        xs = [j + (i - 1.5) * w for j in range(len(CHUNK_SIZES))]
        ax.bar(xs, ys, width=w - 0.03, color=COLORS[i], label=name, zorder=2)
    for j, cs in enumerate(CHUNK_SIZES):
        v = res["by_chunk"][cs]["ratio"]["vByte"]
        ax.text(j - 1.5 * w, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(CHUNK_SIZES)), [str(c) for c in CHUNK_SIZES])
    ax.set_xlabel("Chunk size (positions)", color=MUTED)
    ax.set_ylabel("Compressed / original size", color=MUTED)
    ax.set_ylim(0, 0.85)
    style(ax)
    ax.legend(ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_text_ratio.pdf")
    fig.savefig(FIG / "real_text_ratio.png", dpi=160)
    plt.close(fig)


def plot_synthetic(res):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(res)
    w = 0.2
    for i, codec in enumerate(NAMES):
        ys = [res[n]["adaptive"] if codec == "Adaptive" else res[n]["whole"][codec] for n in names]
        xs = [j + (i - 1.5) * w for j in range(len(names))]
        ax.bar(xs, [y / 1000 for y in ys], width=w - 0.03, color=COLORS[i], label=codec, zorder=2)
    ax.set_xticks(range(len(names)), [n.replace(", ", ",\n").replace(" with", "\nwith").replace(" positions", "\npositions")
                                      for n in names], fontsize=8)
    ax.set_ylabel("Compressed size (KB)", color=MUTED)
    style(ax)
    ax.legend(ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_sizes.pdf")
    fig.savefig(FIG / "synthetic_sizes.png", dpi=160)
    plt.close(fig)


def plot_time(res):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    secs = [res["seconds"][n] for n in NAMES]
    ax.bar(range(4), secs, width=0.6, color=COLORS, zorder=2)
    for i, s in enumerate(secs):
        ax.text(i, s + max(secs) * 0.02, f"{s:.2f} s", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), NAMES, fontsize=8)
    ax.set_ylabel("Seconds to compress index", color=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "compression_time.pdf")
    fig.savefig(FIG / "compression_time.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    results = {"synthetic": synthetic()}
    plot_synthetic(results["synthetic"])
    shakespeare = ROOT / "data" / "t8.shakespeare.txt"
    if shakespeare.exists():
        results["shakespeare"] = real_text(shakespeare)
        plot_real(results["shakespeare"])
        plot_time(results["shakespeare"])
    else:
        print(f"Skipping real-text results: {shakespeare} not found")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
