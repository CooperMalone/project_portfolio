import random
import time

from compression.vByte import *
from compression.bitPacking import *
from compression.elias import *

from chunk.adaptiveChunk import AdaptiveChunkedCompressor, FixedChunker


# =========================
# Dataset Generators (unchanged)
# =========================
def generate_uniform_small(n=10000, max_val=10):
    return [random.randint(0, max_val) for _ in range(n)]


def generate_clustered(n=10000, base=1000, noise=5):
    return [base + random.randint(0, noise) for _ in range(n)]


def generate_sparse_with_outliers(n=10000):
    data = []
    for _ in range(n):
        if random.random() < 0.95:
            data.append(random.randint(1, 5))
        else:
            data.append(random.randint(1000, 100000))
    return data


def generate_monotonic_gaps(n=10000):
    pos = 0
    result = []
    for _ in range(n):
        gap = random.randint(1, 5)
        pos += gap
        result.append(pos)
    return result


# =========================
# Size Helpers
# =========================
def get_size(obj):
    if isinstance(obj, bytes):
        return len(obj)
    elif isinstance(obj, list):
        return len(obj) * 4
    else:
        return len(obj)


def adaptive_size(chunks):
    total = 0
    metadata = 0

    for c in chunks:
        data = c["data"]

        total += get_size(data)
        metadata += 1  # 1 byte per chunk

    return total + metadata, metadata


# =========================
# Core Test Function
# =========================
def test_dataset(name, data, chunk_size=128):
    print(f"\n=== {name} ===")

    # ===== Fixed (no chunking) =====
    vbyte = VByteEncoder.encode_sequence(data)
    bitpack = BitPackEncoder.encode_sequence(data)
    elias = EliasGammaEncoder.encode_sequence(data)

    print(f"\n--- Fixed ---")
    print(f"vByte size: {len(vbyte):,}")
    print(f"BitPack size: {len(bitpack):,}")
    print(f"Elias size: {len(elias):,}")

    # ===== Adaptive (chunked) =====
    chunker = FixedChunker(chunk_size)

    codecs = {
        "vbyte": (VByteEncoder.encode_sequence, VByteDecoder.decode_sequence),
        "bitpack": (BitPackEncoder.encode_sequence, BitPackDecoder.decode_sequence),
        "elias": (EliasGammaEncoder.encode_sequence, EliasGammaDecoder.decode_sequence),
    }

    adaptive = AdaptiveChunkedCompressor(chunker, codecs)

    # ---- Compression timing ----
    start = time.perf_counter()
    compressed_chunks = adaptive.compress_list(data)
    compress_time = time.perf_counter() - start

    # ---- Decompression timing ----
    start = time.perf_counter()
    decompressed = adaptive.decompress_list(compressed_chunks)
    decompress_time = time.perf_counter() - start

    assert decompressed == data

    # ---- Size ----
    total_size, metadata_size = adaptive_size(compressed_chunks)

    print(f"\n--- Adaptive (chunk_size={chunk_size}) ---")
    print(f"Total size: {total_size:,}")
    print(f"Metadata: {metadata_size:,} bytes ({100 * metadata_size / total_size:.2f}%)")

    print(f"Compression time: {compress_time:.6f} sec")
    print(f"Decompression time: {decompress_time:.6f} sec")

    # ===== Winner =====
    fixed_sizes = {
        "vByte": len(vbyte),
        "BitPack": len(bitpack),
        "Elias": len(elias),
        "Adaptive": total_size
    }

    winner = min(fixed_sizes, key=fixed_sizes.get)
    print(f"\n🏆 Winner: {winner}")


# =========================
# Run All Tests
# =========================
def run_all_tests():
    test_dataset("Uniform Small", generate_uniform_small())
    test_dataset("Clustered", generate_clustered())
    test_dataset("Sparse w/ Outliers", generate_sparse_with_outliers())
    test_dataset("Monotonic Gaps", generate_monotonic_gaps())
