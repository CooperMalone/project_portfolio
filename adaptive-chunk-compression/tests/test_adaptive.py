"""Tests for adaptive chunking and adaptive compression selection."""

from compression.adaptive import AdaptiveChunker, AdaptivePostingListCompressor, choose_best_algorithm


def test_choose_best_algorithm_uniform_small():
    gaps = [1, 2, 1, 3, 4, 2, 5, 1, 2]
    assert choose_best_algorithm(gaps) in {"bitpack", "elias"}


def test_choose_best_algorithm_sparse_outliers():
    gaps = [1, 2, 1, 3, 2, 1000, 1, 2]
    assert choose_best_algorithm(gaps) == "elias"


def test_chunk_positions_splits_large_outliers():
    chunker = AdaptiveChunker(min_chunk_size=2, large_gap_threshold=50, max_chunk_size=10)
    positions = [0, 1, 2, 3, 60, 61, 62]
    chunks = chunker.chunk_positions(positions)
    assert chunks == [[0, 1, 2, 3], [60, 61, 62]]


def test_adaptive_posting_list_roundtrip():
    compressor = AdaptivePostingListCompressor()
    posting_list = {
        "term": {
            1: [0, 1, 2, 10, 11, 12, 1000],
        }
    }

    compressed = compressor.compress_posting_list(posting_list)
    decompressed = compressor.decompress_posting_list(compressed)

    assert decompressed == posting_list


def test_adaptive_chunk_algorithm_metadata():
    compressor = AdaptivePostingListCompressor()
    posting_list = {"term": {1: [0, 1, 2, 3, 4, 5]}}
    compressed = compressor.compress_posting_list(posting_list)
    assert compressed["term"][1]
    assert compressed["term"][1][0]["algorithm"] in {"bitpack", "elias", "vbyte"}
    assert isinstance(compressed["term"][1][0]["data"], bytes)
