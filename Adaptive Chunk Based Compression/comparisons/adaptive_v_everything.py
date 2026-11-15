"""Compare adaptive chunking against vByte, Elias, and BitPack on Shakespeare."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunk.fixedChunk import ChunkedPostingList
from compression.adaptive import AdaptivePostingListCompressor
from compression.bitPacking import BitPackDecoder, BitPackEncoder
from compression.elias import EliasGammaDecoder, EliasGammaEncoder
from compression.vByte import VByteDecoder, VByteEncoder
from utils.posting_list import text_to_posting_list


def calculate_posting_list_size(posting_list: dict) -> int:
    total = 0
    for doc_positions in posting_list.values():
        for positions in doc_positions.values():
            total += len(positions) * 4
    return total


def calculate_compressed_size(compressed_posting_list: dict) -> int:
    total = 0
    for term_docs in compressed_posting_list.values():
        for chunks in term_docs.values():
            if isinstance(chunks, list):
                for chunk in chunks:
                    if isinstance(chunk, bytes):
                        total += len(chunk)
                    elif isinstance(chunk, dict):
                        total += len(chunk.get("data", b""))
            elif isinstance(chunks, bytes):
                total += len(chunks)
    return total


def compress_with_fixed_algorithm(
    posting_list: dict,
    chunk_size: int,
    encode_fn,
    decode_fn,
    name: str,
) -> tuple[dict, float]:
    chunker = ChunkedPostingList(chunk_size=chunk_size, compress_fn=encode_fn, decompress_fn=decode_fn)
    start = time.perf_counter()
    compressed = chunker.compress(posting_list)
    elapsed = time.perf_counter() - start
    decompressed = chunker.decompress(compressed)
    if decompressed != posting_list:
        raise AssertionError(f"Decompression failed for {name}")
    return compressed, elapsed


def compress_with_adaptive(posting_list: dict) -> tuple[dict, float]:
    compressor = AdaptivePostingListCompressor()
    start = time.perf_counter()
    compressed = compressor.compress_posting_list(posting_list)
    elapsed = time.perf_counter() - start
    decompressed = compressor.decompress_posting_list(compressed)
    if decompressed != posting_list:
        raise AssertionError("Adaptive decompression failed")
    return compressed, elapsed


def render_bar(value: float, max_value: float, width: int = 40) -> str:
    if max_value <= 0:
        return ""
    length = int((value / max_value) * width)
    return "█" * length + " " * (width - length)


def summarize_adaptive(compressed: dict) -> Counter:
    counter = Counter()
    for term_docs in compressed.values():
        for chunks in term_docs.values():
            for chunk in chunks:
                if isinstance(chunk, dict):
                    counter[chunk["algorithm"]] += 1
    return counter


def run_comparison() -> None:
    shakespeare_path = Path("data/t8.shakespeare.txt")
    if not shakespeare_path.exists():
        raise FileNotFoundError(f"Missing Shakespeare dataset at {shakespeare_path}")

    text = shakespeare_path.read_text(encoding="utf-8")
    posting_list = text_to_posting_list(text, doc_id=1)
    original_size = calculate_posting_list_size(posting_list)
    total_positions = sum(len(positions) for doc_positions in posting_list.values() for positions in doc_positions.values())

    print("\n=== Shakespeare Adaptive Compression Comparison ===")
    print(f"Terms: {len(posting_list)}")
    print(f"Total positions: {total_positions}")
    print(f"Original posting list size: {original_size:,} bytes\n")

    chunk_size = 128
    results: Dict[str, Dict[str, Any]] = {}

    print(f"Running fixed chunk compression with chunk_size={chunk_size}...\n")
    fixed_algorithms = [
        ("vByte", VByteEncoder.encode_gaps, VByteDecoder.decode_gaps),
        ("Elias", EliasGammaEncoder.encode_gaps, EliasGammaDecoder.decode_gaps),
        ("BitPack", BitPackEncoder.encode_gaps, BitPackDecoder.decode_gaps),
    ]

    for name, encode_fn, decode_fn in fixed_algorithms:
        compressed, elapsed = compress_with_fixed_algorithm(
            posting_list, chunk_size, encode_fn, decode_fn, name
        )
        size = calculate_compressed_size(compressed)
        compression_pct = (1 - size / original_size) * 100 if original_size > 0 else 0.0
        results[name] = {
            "size": size,
            "compression_pct": compression_pct,
            "time": elapsed,
            "compressed": compressed,
        }
        print(f"{name}: {size:,} bytes  compression={compression_pct:.2f}%  time={elapsed:.3f}s")

    print("\nRunning adaptive compression...\n")
    adaptive_compressed, adaptive_elapsed = compress_with_adaptive(posting_list)
    adaptive_size = calculate_compressed_size(adaptive_compressed)
    adaptive_compression_pct = (1 - adaptive_size / original_size) * 100 if original_size > 0 else 0.0
    results["Adaptive"] = {
        "size": adaptive_size,
        "compression_pct": adaptive_compression_pct,
        "time": adaptive_elapsed,
        "compressed": adaptive_compressed,
    }

    algorithm_counts = summarize_adaptive(adaptive_compressed)
    print(f"Adaptive: {adaptive_size:,} bytes  compression={adaptive_compression_pct:.2f}%  time={adaptive_elapsed:.3f}s")
    print("Adaptive algorithm usage:", dict(algorithm_counts))

    print("\nSummary:\n")
    max_size = max(result["size"] for result in results.values())
    print(f"{'Name':<10} {'Size':>12} {'Compression':>12} {'Time(s)':>9} {'Graph':<42}")
    print("" + "-" * 92)
    for name, result in results.items():
        bar = render_bar(result["size"], max_size)
        print(
            f"{name:<10} {result['size']:>12,} {result['compression_pct']:>11.2f}% {result['time']:>9.3f} {bar}"
        )

    print("\nDone. Run this file directly with `python3 comparisons/adaptive_v_everything.py`.")


if __name__ == "__main__":
    run_comparison()
