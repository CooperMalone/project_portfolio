"""
Comprehensive tests for vByte compression module.
"""

import pytest
from compression.vByte import (
    VByteEncoder,
    VByteDecoder,
    PostingListCompressor,
)


class TestVByteEncoder:
    """Tests for VByteEncoder class."""

    def test_encode_zero(self):
        """Test encoding zero."""
        result = VByteEncoder.encode_integer(0)
        assert result == bytes([0x00])

    def test_encode_single_byte(self):
        """Test encoding integers that fit in a single byte (0-127)."""
        for value in [1, 10, 50, 127]:
            result = VByteEncoder.encode_integer(value)
            assert len(result) == 1
            assert result[0] == value

    def test_encode_two_bytes(self):
        """Test encoding integers that require two bytes (128-16383)."""
        # 128 should be: 0x80, 0x01
        result = VByteEncoder.encode_integer(128)
        assert len(result) == 2
        assert result[0] == 0x80  # 128 & 0x7F = 0, with continuation bit
        assert result[1] == 0x01

    def test_encode_300(self):
        """Test encoding 300 as a specific example."""
        # 300 = 0x12C = 0b100101100
        # Split into 7-bit chunks: 44 (0x2C) and 2 (0x02)
        # Encoded: 0xAC (44 | 0x80), 0x02
        result = VByteEncoder.encode_integer(300)
        assert len(result) == 2
        assert result.hex() == "ac02"

    def test_encode_three_bytes(self):
        """Test encoding integers that require three bytes."""
        result = VByteEncoder.encode_integer(16384)
        assert len(result) == 3

    def test_encode_large_number(self):
        """Test encoding large numbers."""
        result = VByteEncoder.encode_integer(1000000)
        assert len(result) == 3
        assert result.hex() == "c0843d"

    def test_encode_negative_raises_error(self):
        """Test that negative numbers raise ValueError."""
        with pytest.raises(ValueError):
            VByteEncoder.encode_integer(-1)

    def test_encode_sequence(self):
        """Test encoding a sequence of integers."""
        sequence = [10, 20, 35, 200]
        result = VByteEncoder.encode_sequence(sequence)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_empty_sequence(self):
        """Test encoding an empty sequence."""
        result = VByteEncoder.encode_sequence([])
        assert result == bytes()

    def test_encode_gaps_from_positions(self):
        """Test encoding gaps from position list."""
        positions = [0, 3, 5, 10]
        result = VByteEncoder.encode_gaps(positions)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_gaps_single_position(self):
        """Test encoding gaps with single position."""
        result = VByteEncoder.encode_gaps([5])
        assert isinstance(result, bytes)

    def test_encode_gaps_empty(self):
        """Test encoding gaps with empty positions."""
        result = VByteEncoder.encode_gaps([])
        assert result == bytes()


class TestVByteDecoder:
    """Tests for VByteDecoder class."""

    def test_decode_zero(self):
        """Test decoding zero."""
        value, length = VByteDecoder.decode_integer(bytes([0x00]))
        assert value == 0
        assert length == 1

    def test_decode_single_byte(self):
        """Test decoding single-byte integers."""
        for i in range(128):
            encoded = bytes([i])
            value, length = VByteDecoder.decode_integer(encoded)
            assert value == i
            assert length == 1

    def test_decode_two_bytes(self):
        """Test decoding two-byte integers."""
        encoded = bytes([0x80, 0x01])  # 128
        value, length = VByteDecoder.decode_integer(encoded)
        assert value == 128
        assert length == 2

    def test_decode_300(self):
        """Test decoding 300."""
        encoded = bytes([0xac, 0x02])
        value, length = VByteDecoder.decode_integer(encoded)
        assert value == 300
        assert length == 2

    def test_decode_with_offset(self):
        """Test decoding with offset."""
        encoded = bytes([0x00, 0xac, 0x02])  # Prefix byte, then 300
        value, length = VByteDecoder.decode_integer(encoded, offset=1)
        assert value == 300
        assert length == 3

    def test_decode_sequence(self):
        """Test decoding a sequence of integers."""
        encoder = VByteEncoder()
        original = [10, 20, 35, 200]
        encoded = encoder.encode_sequence(original)
        decoded = VByteDecoder.decode_sequence(encoded)
        assert decoded == original

    def test_decode_sequence_with_count(self):
        """Test decoding a specific count of integers."""
        encoder = VByteEncoder()
        original = [10, 20, 35, 200]
        encoded = encoder.encode_sequence(original)
        decoded = VByteDecoder.decode_sequence(encoded, count=2)
        assert len(decoded) == 2
        assert decoded == [10, 20]

    def test_decode_empty_sequence(self):
        """Test decoding empty bytes."""
        result = VByteDecoder.decode_sequence(bytes())
        assert result == []

    def test_decode_gaps_to_positions(self):
        """Test decoding gaps back to positions."""
        encoder = VByteEncoder()
        original_positions = [0, 3, 5, 10]
        encoded_gaps = encoder.encode_gaps(original_positions)
        decoded_positions = VByteDecoder.decode_gaps(encoded_gaps)
        assert decoded_positions == original_positions

    def test_decode_gaps_empty(self):
        """Test decoding empty gap data."""
        result = VByteDecoder.decode_gaps(bytes())
        assert result == []

    def test_decode_insufficient_bytes(self):
        """Test decoding with insufficient bytes."""
        # This should just return what it can decode
        encoded = bytes([0x80])  # Continuation bit set but no follow-up byte
        value, length = VByteDecoder.decode_integer(encoded)
        assert length == 1  # Should consume the byte even if incomplete


class TestRoundTrip:
    """Round-trip tests (encode then decode)."""

    def test_roundtrip_single_integers(self):
        """Test encoding and decoding single integers."""
        test_values = [0, 1, 127, 128, 300, 16383, 16384, 1000000]
        for value in test_values:
            encoded = VByteEncoder.encode_integer(value)
            decoded, _ = VByteDecoder.decode_integer(encoded)
            assert decoded == value

    def test_roundtrip_sequence(self):
        """Test encoding and decoding sequences."""
        test_sequences = [
            [1],
            [1, 2, 3],
            [0, 127, 128, 16383, 16384],
            list(range(100)),
            [1000000, 999999, 123456],
        ]
        for seq in test_sequences:
            encoded = VByteEncoder.encode_sequence(seq)
            decoded = VByteDecoder.decode_sequence(encoded)
            assert decoded == seq

    def test_roundtrip_gaps(self):
        """Test encoding and decoding gaps."""
        test_positions = [
            [0],
            [0, 1],
            [0, 2, 5, 12, 13, 18, 100, 105],
            [0, 100, 101, 102],
            list(range(0, 1000, 10)),
        ]
        for positions in test_positions:
            encoded = VByteEncoder.encode_gaps(positions)
            decoded = VByteDecoder.decode_gaps(encoded)
            assert decoded == positions


class TestPostingListCompressor:
    """Tests for PostingListCompressor class."""

    def test_compress_empty_posting_list(self):
        """Test compressing empty posting list."""
        result = PostingListCompressor.compress_posting_list({})
        assert result == {}

    def test_compress_simple_posting_list(self):
        """Test compressing a simple posting list."""
        posting_list = {"term": {1: [0, 2, 5]}}
        compressed = PostingListCompressor.compress_posting_list(
            posting_list, use_gaps=True
        )
        assert "term" in compressed
        assert 1 in compressed["term"]
        assert isinstance(compressed["term"][1], bytes)

    def test_compress_multiple_terms(self):
        """Test compressing posting list with multiple terms."""
        posting_list = {
            "hello": {1: [0, 5]},
            "world": {1: [1]},
            "test": {1: [2, 3, 7]},
        }
        compressed = PostingListCompressor.compress_posting_list(posting_list)
        assert len(compressed) == 3
        assert all(isinstance(compressed[term][1], bytes) for term in compressed)

    def test_compress_multiple_documents(self):
        """Test compressing posting list with multiple documents."""
        posting_list = {
            "term": {1: [0, 2], 2: [1, 3], 3: [0]},
        }
        compressed = PostingListCompressor.compress_posting_list(posting_list)
        assert compressed["term"][1] != compressed["term"][2]

    def test_decompress_simple_posting_list(self):
        """Test decompressing a compressed posting list."""
        original = {"term": {1: [0, 2, 5]}}
        compressed = PostingListCompressor.compress_posting_list(
            original, use_gaps=True
        )
        decompressed = PostingListCompressor.decompress_posting_list(
            compressed, use_gaps=True
        )
        assert decompressed == original

    def test_roundtrip_complex_posting_list(self):
        """Test compress/decompress roundtrip with complex data."""
        original = {
            "apple": {1: [0, 2, 5], 2: [1]},
            "banana": {1: [1], 2: [0, 3]},
            "cherry": {1: [3, 4], 3: [0, 1, 2]},
        }
        compressed = PostingListCompressor.compress_posting_list(
            original, use_gaps=True
        )
        decompressed = PostingListCompressor.decompress_posting_list(
            compressed, use_gaps=True
        )
        assert decompressed == original

    def test_roundtrip_absolute_encoding(self):
        """Test roundtrip with absolute encoding (no gaps)."""
        original = {"term": {1: [10, 20, 100, 200]}}
        compressed = PostingListCompressor.compress_posting_list(
            original, use_gaps=False
        )
        decompressed = PostingListCompressor.decompress_posting_list(
            compressed, use_gaps=False
        )
        assert decompressed == original

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        original = {"term": {1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}}
        compressed = PostingListCompressor.compress_posting_list(original)
        ratio = PostingListCompressor.get_compression_ratio(original, compressed)
        assert 0 < ratio < 1  # Should be some compression

    def test_compression_ratio_empty(self):
        """Test compression ratio with empty posting list."""
        ratio = PostingListCompressor.get_compression_ratio({}, {})
        assert ratio == 0.0

    def test_compression_ratio_no_compression(self):
        """Test compression ratio when no compression occurs."""
        posting_list = {"a": {1: [0]}}
        compressed = posting_list  # No actual compression
        ratio = PostingListCompressor.get_compression_ratio(
            posting_list, compressed
        )
        assert ratio == 1.0


class TestGapEncodingEfficiency:
    """Tests comparing gap encoding vs absolute encoding efficiency."""

    def test_gap_encoding_is_more_efficient(self):
        """Test that gap encoding is generally more efficient than absolute."""
        posting_list = {"term": {1: list(range(0, 1000, 10))}}

        compressed_gaps = PostingListCompressor.compress_posting_list(
            posting_list, use_gaps=True
        )
        compressed_abs = PostingListCompressor.compress_posting_list(
            posting_list, use_gaps=False
        )

        bytes_gaps = sum(
            len(b)
            for term_bytes in compressed_gaps.values()
            for b in term_bytes.values()
        )
        bytes_abs = sum(
            len(b)
            for term_bytes in compressed_abs.values()
            for b in term_bytes.values()
        )

        assert bytes_gaps <= bytes_abs

    def test_small_gaps_compress_better(self):
        """Test that sequences with small gaps compress better."""
        # Small uniform gaps
        small_gaps = {"term": {1: [0, 1, 2, 3, 4, 5]}}

        # Large scattered positions
        large_gaps = {"term": {1: [0, 1000, 2000, 3000, 4000, 5000]}}

        compressed_small = PostingListCompressor.compress_posting_list(
            small_gaps, use_gaps=True
        )
        compressed_large = PostingListCompressor.compress_posting_list(
            large_gaps, use_gaps=True
        )

        bytes_small = sum(
            len(b)
            for term_bytes in compressed_small.values()
            for b in term_bytes.values()
        )
        bytes_large = sum(
            len(b)
            for term_bytes in compressed_large.values()
            for b in term_bytes.values()
        )

        # Small gaps should compress to fewer bytes
        assert bytes_small < bytes_large


class TestEdgeCases:
    """Edge case tests."""

    def test_very_large_numbers(self):
        """Test encoding/decoding very large numbers."""
        large_num = 2**31 - 1  # Max 32-bit int
        encoded = VByteEncoder.encode_integer(large_num)
        decoded, _ = VByteDecoder.decode_integer(encoded)
        assert decoded == large_num

    def test_long_position_list(self):
        """Test with very long position lists."""
        positions = list(range(10000))
        encoded = VByteEncoder.encode_gaps(positions)
        decoded = VByteDecoder.decode_gaps(encoded)
        assert decoded == positions

    def test_many_documents_in_posting_list(self):
        """Test posting list with many documents."""
        posting_list = {
            "term": {doc_id: [doc_id, doc_id + 100] for doc_id in range(100)}
        }
        compressed = PostingListCompressor.compress_posting_list(posting_list)
        decompressed = PostingListCompressor.decompress_posting_list(compressed)
        assert decompressed == posting_list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
