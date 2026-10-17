"""
Shakespeare Compression Comparison

Compares vByte and Elias Gamma compression on Shakespeare text
using fixed-size chunking.
"""

import os
from pathlib import Path
from utils.data_extraction import extract_newsgroups
from utils.posting_list import text_to_posting_list, documents_to_posting_list
from compression.vByte import VByteEncoder, VByteDecoder
from compression.elias import EliasGammaEncoder, EliasGammaDecoder
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


def load_shakespeare():
    with open("data/t8.shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    posting_list = text_to_posting_list(text, doc_id=1)
    return text, posting_list

def load_newsgroups():
    extract_newsgroups()
    root = Path("data") / "6_newsgroups"
    documents = []

    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            if file_path.name == ".DS_Store":
                continue
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                documents.append(f.read())

    posting_list = documents_to_posting_list(documents)
    return documents, posting_list

def evaluate_dataset(name, posting_list, text_size, word_count):
    total_positions = sum(
        len(positions)
        for doc_pos in posting_list.values()
        for positions in doc_pos.values()
    )
    original_bytes = calculate_posting_list_size(posting_list)

    print("\n" + "=" * 70)
    print(f"{name} Compression Comparison")
    print("=" * 70)
    print(f"Text size: {text_size} characters")
    print(f"Words: {word_count}")
    print(f"Unique terms: {len(posting_list)}")
    print(f"Total positions: {total_positions}")

    # Define chunk sizes to test
    chunk_sizes = [50, 100, 200, 500]

    results = []

    for chunk_size in chunk_sizes:
        print(f"\n{'='*70}")
        print(f"Chunk Size: {chunk_size}")
        print(f"{'='*70}")

        # vByte compression
        vbyte_chunker = ChunkedPostingList(
            chunk_size=chunk_size,
            compress_fn=VByteEncoder.encode_gaps,
            decompress_fn=VByteDecoder.decode_gaps
        )
        vbyte_compressed = vbyte_chunker.compress(posting_list)
        vbyte_decompressed = vbyte_chunker.decompress(vbyte_compressed)
        assert vbyte_decompressed == posting_list, "vByte decompression failed"

        # Elias Gamma compression
        elias_chunker = ChunkedPostingList(
            chunk_size=chunk_size,
            compress_fn=EliasGammaEncoder.encode_gaps,
            decompress_fn=EliasGammaDecoder.decode_gaps
        )
        elias_compressed = elias_chunker.compress(posting_list)
        elias_decompressed = elias_chunker.decompress(elias_compressed)
        assert elias_decompressed == posting_list, "Elias Gamma decompression failed"

        # Calculate sizes
        vbyte_bytes = calculate_compressed_size(vbyte_compressed)
        elias_bytes = calculate_compressed_size(elias_compressed)

        # Print results for this chunk size
        vbyte_ratio = print_compression_results("vByte", original_bytes, vbyte_bytes, chunk_size)
        elias_ratio = print_compression_results("Elias Gamma", original_bytes, elias_bytes, chunk_size)

        # Compare
        if vbyte_bytes < elias_bytes:
            diff = 100 * (elias_bytes - vbyte_bytes) / elias_bytes
            better = f"vByte is {diff:.1f}% better"
        else:
            diff = 100 * (vbyte_bytes - elias_bytes) / vbyte_bytes
            better = f"Elias Gamma is {diff:.1f}% better"

        print(f"\n{better}")
        
        results.append({
            "chunk_size": chunk_size,
            "vbyte_ratio": vbyte_ratio,
            "elias_ratio": elias_ratio,
            "better": better
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Chunk Size':<15} {'vByte':<15} {'Elias Gamma':<15} {'Winner':<30}")
    print("-" * 70)
    for r in results:
        vbyte = f"{r['vbyte_ratio']:.4f}"
        elias = f"{r['elias_ratio']:.4f}"
        print(f"{r['chunk_size']:<15} {vbyte:<15} {elias:<15} {r['better']:<30}")

    print("\n" + "=" * 70)
    print("✓ All compressions verified and compared!")
    print("=" * 70)

def run_shakespeare_comparison():
    text, posting_list = load_shakespeare()
    evaluate_dataset("Shakespeare", posting_list, len(text), len(text.split()))

def run_newsgroups_comparison():
    documents, posting_list = load_newsgroups()
    evaluate_dataset(
        "Newsgroups",
        posting_list,
        sum(len(doc) for doc in documents),
        sum(len(doc.split()) for doc in documents),
    )


def vByteEliasComparison():
    run_shakespeare_comparison()
    run_newsgroups_comparison()