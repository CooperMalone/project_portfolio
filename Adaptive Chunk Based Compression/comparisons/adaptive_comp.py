"""
Adaptive Compression Comparison (Shakespeare)

Compares:
- vByte (fixed chunk)
- Elias Gamma (fixed chunk)
- BitPack (fixed chunk)
- Adaptive (chooses best per chunk)

Uses Shakespeare dataset.
"""

from utils.posting_list import text_to_posting_list
from compression.vByte import VByteEncoder, VByteDecoder
from compression.elias import EliasGammaEncoder, EliasGammaDecoder
from compression.bitPacking import BitPackEncoder, BitPackDecoder  # adjust if needed

from chunk.fixedChunk import ChunkedPostingList
from chunk.adaptiveChunk import AdaptiveChunkedPostingList, get_default_codecs


# =========================
# Size Helpers (unchanged)
# =========================
def calculate_posting_list_size(posting_list):
    total = 0
    for doc_pos in posting_list.values():
        for positions in doc_pos.values():
            total += len(positions) * 4
    return total

# def calculate_compressed_size(compressed_posting_list):
#     total = 0
#     metadata_bytes = 0
#
#     for term_docs in compressed_posting_list.values():
#         for chunks in term_docs.values():
#
#             for chunk in chunks:
#                 if isinstance(chunk, dict):
#                     data = chunk["data"]
#                     codec = chunk["codec"]
#
#                     # --- DATA SIZE ---
#                     if isinstance(data, bytes):
#                         total += len(data)
#                     elif isinstance(data, list):
#                         total += len(data) * 4
#                     else:
#                         total += len(data)
#
#                     # --- METADATA SIZE ---
#                     # Assume 1 byte to store codec ID (realistic encoding)
#                     metadata_bytes += 1
#
#                 else:
#                     # non-adaptive case
#                     if isinstance(chunk, bytes):
#                         total += len(chunk)
#
#     return total + metadata_bytes, metadata_bytes

def calculate_compressed_size(compressed_posting_list):
    total = 0
    for term_docs in compressed_posting_list.values():
        for chunks in term_docs.values():

            if isinstance(chunks, list):
                for chunk in chunks:
                    # Adaptive case
                    if isinstance(chunk, dict):
                        data = chunk["data"]
                    else:
                        data = chunk

                    if isinstance(data, bytes):
                        total += len(data)
                    elif isinstance(data, list):
                        total += len(data) * 4
                    else:
                        total += len(data)

            elif isinstance(chunks, bytes):
                total += len(chunks)

    return total


def print_compression_results(name, original_size, compressed_size, chunk_size, time):
    ratio = compressed_size / original_size if original_size > 0 else 0.0
    reduction = 100 * (1 - ratio)

    print(f"\n{name} (chunk_size={chunk_size})")
    print("-" * 60)
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {ratio:.4f}")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Compression time: {time:.4f}")
    return ratio


# =========================
# Main Comparison
# =========================
def adaptiveComparison():
    print("Reading Shakespeare text...")
    with open("data/t8.shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("Converting to posting list...")
    posting_list = text_to_posting_list(text, doc_id=1)

    total_positions = sum(
        len(positions)
        for doc_pos in posting_list.values()
        for positions in doc_pos.values()
    )
    original_bytes = calculate_posting_list_size(posting_list)

    print("\n" + "=" * 70)
    print("Adaptive Compression Comparison")
    print("=" * 70)
    print(f"Words: {len(text.split())}")
    print(f"Unique terms: {len(posting_list)}")
    print(f"Total positions: {total_positions}")
    print(f"Original size: {original_bytes:,} bytes")

    chunk_sizes = [50, 100, 200, 500]

    results = []

    for chunk_size in chunk_sizes:
        print(f"\n{'='*70}")
        print(f"Chunk Size: {chunk_size}")
        print(f"{'='*70}")

        # ===== Fixed Compressors =====
        vbyte = ChunkedPostingList(
            chunk_size,
            VByteEncoder.encode_gaps,
            VByteDecoder.decode_gaps
        )

        elias = ChunkedPostingList(
            chunk_size,
            EliasGammaEncoder.encode_gaps,
            EliasGammaDecoder.decode_gaps
        )

        bitpack = ChunkedPostingList(
            chunk_size,
            BitPackEncoder.encode_gaps,
            BitPackDecoder.decode_gaps
        )

        # ===== Adaptive =====
        codecs = {
            "vbyte": (VByteEncoder.encode_gaps, VByteDecoder.decode_gaps),
            "elias": (EliasGammaEncoder.encode_gaps, EliasGammaDecoder.decode_gaps),
            "bitpack": (BitPackEncoder.encode_gaps, BitPackDecoder.decode_gaps),
        }

        adaptive = AdaptiveChunkedPostingList(chunk_size, codecs)

        # ===== Run Compression =====
        import time
        start = time.perf_counter()
        vbyte_c = vbyte.compress(posting_list)
        v_t = time.perf_counter() - start

        start = time.perf_counter()
        elias_c = elias.compress(posting_list)
        e_t = time.perf_counter() - start

        start = time.perf_counter()
        bitpack_c = bitpack.compress(posting_list)
        b_t = time.perf_counter() - start

        start = time.perf_counter()
        adaptive_c = adaptive.compress(posting_list)
        a_t = time.perf_counter() - start

        # ===== Verify =====
        assert vbyte.decompress(vbyte_c) == posting_list
        assert elias.decompress(elias_c) == posting_list
        assert bitpack.decompress(bitpack_c) == posting_list
        assert adaptive.decompress(adaptive_c) == posting_list

        # ===== Sizes =====
        vbyte_size = calculate_compressed_size(vbyte_c)
        elias_size = calculate_compressed_size(elias_c)
        bitpack_size = calculate_compressed_size(bitpack_c)
        adaptive_size = calculate_compressed_size(adaptive_c)

        # ===== Print =====
        v_r = print_compression_results("vByte", original_bytes, vbyte_size, chunk_size, v_t)
        e_r = print_compression_results("Elias", original_bytes, elias_size, chunk_size, e_t)
        b_r = print_compression_results("BitPack", original_bytes, bitpack_size, chunk_size, b_t)
        a_r = print_compression_results("Adaptive", original_bytes, adaptive_size, chunk_size, a_t)

        # ===== Winner =====
        sizes = {
            "vByte": vbyte_size,
            "Elias": elias_size,
            "BitPack": bitpack_size,
            "Adaptive": adaptive_size,
        }

        winner = min(sizes, key=sizes.get)
        print(f"\n🏆 Winner: {winner}")

        results.append({
            "chunk_size": chunk_size,
            "vbyte": v_r,
            "elias": e_r,
            "bitpack": b_r,
            "adaptive": a_r,
            "winner": winner
        })

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"{'Chunk':<10} {'vByte':<10} {'Elias':<10} {'BitPack':<10} {'Adaptive':<10} {'Winner':<10}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['chunk_size']:<10} "
            f"{r['vbyte']:.4f}   "
            f"{r['elias']:.4f}   "
            f"{r['bitpack']:.4f}   "
            f"{r['adaptive']:.4f}   "
            f"{r['winner']}"
        )

    print("\n✓ All compressions verified!")
