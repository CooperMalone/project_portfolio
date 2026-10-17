"""
Shakespeare Compression Comparison

Compares vByte and Bit Packing compression on Shakespeare text
using fixed-size chunking.
"""

from utils.posting_list import text_to_posting_list
from compression.vByte import VByteEncoder, VByteDecoder
from compression.bitPacking import BitPackEncoder, BitPackDecoder
from chunk.fixedChunk import ChunkedPostingList


def calculate_posting_list_size(posting_list):
    """Calculate total size of a posting list (4 bytes per position)."""
    total = 0
    for doc_pos in posting_list.values():
        for positions in doc_pos.values():
            total += len(positions) * 4
    return total


def calculate_compressed_size(compressed_posting_list):
    """Calculate total size of compressed posting list."""
    total = 0
    for term_docs in compressed_posting_list.values():
        for chunks in term_docs.values():
            if isinstance(chunks, list):
                for chunk in chunks:
                    if isinstance(chunk, bytes):
                        total += len(chunk)
            elif isinstance(chunks, bytes):
                total += len(chunks)
    return total


def print_compression_results(name, original_size, compressed_size, chunk_size):
    ratio = compressed_size / original_size if original_size > 0 else 0.0
    reduction = 100 * (1 - ratio)

    print(f"\n{name} (chunk_size={chunk_size})")
    print("-" * 60)
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {ratio:.4f}")
    print(f"Size reduction: {reduction:.1f}%")
    return ratio


def vByteBitPackComparison():
    # Read the Shakespeare text
    print("Reading Shakespeare text...")
    with open("data/t8.shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Convert to posting list
    print("Converting to posting list...")
    posting_list = text_to_posting_list(text, doc_id=1)

    # Calculate statistics
    total_positions = sum(
        len(positions)
        for doc_pos in posting_list.values()
        for positions in doc_pos.values()
    )
    original_bytes = calculate_posting_list_size(posting_list)

    print("\n" + "=" * 70)
    print("Shakespeare Compression Comparison (vByte vs BitPack)")
    print("=" * 70)
    print(f"Text size: {len(text)} characters")
    print(f"Words: {len(text.split())}")
    print(f"Unique terms: {len(posting_list)}")
    print(f"Total positions: {total_positions}")
    print(f"Original posting list size: {original_bytes:,} bytes")

    # Define chunk sizes to test
    chunk_sizes = [50, 100, 200, 500]

    results = []

    for chunk_size in chunk_sizes:
        print(f"\n{'='*70}")
        print(f"Chunk Size: {chunk_size}")
        print(f"{'='*70}")

        # --- vByte compression ---
        vbyte_chunker = ChunkedPostingList(
            chunk_size=chunk_size,
            compress_fn=VByteEncoder.encode_gaps,
            decompress_fn=VByteDecoder.decode_gaps
        )
        vbyte_compressed = vbyte_chunker.compress(posting_list)
        vbyte_decompressed = vbyte_chunker.decompress(vbyte_compressed)
        assert vbyte_decompressed == posting_list, "vByte decompression failed"

        # --- BitPack compression ---
        bitpack_chunker = ChunkedPostingList(
            chunk_size=chunk_size,
            compress_fn=BitPackEncoder.encode_gaps,
            decompress_fn=BitPackDecoder.decode_gaps
        )
        bitpack_compressed = bitpack_chunker.compress(posting_list)
        bitpack_decompressed = bitpack_chunker.decompress(bitpack_compressed)
        assert bitpack_decompressed == posting_list, "BitPack decompression failed"

        # Calculate sizes
        vbyte_bytes = calculate_compressed_size(vbyte_compressed)
        bitpack_bytes = calculate_compressed_size(bitpack_compressed)

        # Print results
        vbyte_ratio = print_compression_results("vByte", original_bytes, vbyte_bytes, chunk_size)
        bitpack_ratio = print_compression_results("BitPack", original_bytes, bitpack_bytes, chunk_size)

        # Compare
        if vbyte_bytes < bitpack_bytes:
            diff = 100 * (bitpack_bytes - vbyte_bytes) / bitpack_bytes
            better = f"vByte is {diff:.1f}% better"
        else:
            diff = 100 * (vbyte_bytes - bitpack_bytes) / vbyte_bytes
            better = f"BitPack is {diff:.1f}% better"

        print(f"\n{better}")

        results.append({
            "chunk_size": chunk_size,
            "vbyte_ratio": vbyte_ratio,
            "bitpack_ratio": bitpack_ratio,
            "better": better
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Chunk Size':<15} {'vByte':<15} {'BitPack':<15} {'Winner':<30}")
    print("-" * 70)
    for r in results:
        vbyte = f"{r['vbyte_ratio']:.4f}"
        bitpack = f"{r['bitpack_ratio']:.4f}"
        print(f"{r['chunk_size']:<15} {vbyte:<15} {bitpack:<15} {r['better']:<30}")

    print("\n" + "=" * 70)
    print("✓ All compressions verified and compared!")
    print("=" * 70)

