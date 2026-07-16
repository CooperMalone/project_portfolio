import pytest
from compression.elias import (
    EliasGammaEncoder,
    EliasGammaDecoder,
    EliasGammaCompressor,
)


class TestEliasGammaEncoder:
    def test_encode_zero(self):
        encoded = EliasGammaEncoder.encode_integer(0)
        assert isinstance(encoded, bytes)
        assert encoded == bytes([0x80])

    def test_encode_one(self):
        encoded = EliasGammaEncoder.encode_integer(1)
        assert encoded == bytes([0x40])

    def test_encode_two(self):
        encoded = EliasGammaEncoder.encode_integer(2)
        assert encoded == bytes([0x60])

    def test_encode_decode_values(self):
        values = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64]
        for value in values:
            encoded = EliasGammaEncoder.encode_integer(value)
            decoded, _ = EliasGammaDecoder.decode_integer(encoded)
            assert decoded == value

    def test_encode_sequence(self):
        values = [0, 1, 2, 3, 4, 10, 100]
        encoded = EliasGammaEncoder.encode_sequence(values)
        decoded = EliasGammaDecoder.decode_sequence(encoded)
        assert decoded == values

    def test_encode_empty_sequence(self):
        assert EliasGammaEncoder.encode_sequence([]) == bytes()

    def test_encode_gaps(self):
        positions = [0, 2, 5, 10]
        encoded = EliasGammaEncoder.encode_gaps(positions)
        decoded = EliasGammaDecoder.decode_gaps(encoded)
        assert decoded == positions

    def test_encode_gaps_single_position(self):
        encoded = EliasGammaEncoder.encode_gaps([5])
        decoded = EliasGammaDecoder.decode_gaps(encoded)
        assert decoded == [5]

    def test_encode_gaps_empty(self):
        assert EliasGammaEncoder.encode_gaps([]) == bytes()


class TestEliasGammaDecoder:
    def test_decode_sequence_count(self):
        values = [0, 1, 2, 3, 4, 5]
        encoded = EliasGammaEncoder.encode_sequence(values)
        decoded = EliasGammaDecoder.decode_sequence(encoded, count=3)
        assert decoded == [0, 1, 2]

    def test_decode_incomplete_code_raises(self):
        encoded = bytes([0x00])
        with pytest.raises(IndexError):
            EliasGammaDecoder.decode_integer(encoded)


class TestEliasGammaCompressor:
    def test_compress_decompress_posting_list(self):
        posting_list = {"term": {1: [0, 2, 5, 10]}}
        compressed = EliasGammaCompressor.compress_posting_list(posting_list)
        decompressed = EliasGammaCompressor.decompress_posting_list(compressed)
        assert decompressed == posting_list

    def test_compress_decompress_absolute(self):
        posting_list = {"term": {1: [0, 1, 2, 5]}}
        compressed = EliasGammaCompressor.compress_posting_list(posting_list, use_gaps=False)
        decompressed = EliasGammaCompressor.decompress_posting_list(compressed, use_gaps=False)
        assert decompressed == posting_list

    def test_multiple_terms_and_documents(self):
        posting_list = {
            "a": {1: [0, 2], 2: [1, 3]},
            "b": {1: [0], 2: [2, 4]},
        }
        compressed = EliasGammaCompressor.compress_posting_list(posting_list)
        decompressed = EliasGammaCompressor.decompress_posting_list(compressed)
        assert decompressed == posting_list

    def test_gap_encoding_small_values(self):
        posting_list = {"term": {1: [0, 1, 2, 3, 4, 5]}}
        compressed = EliasGammaCompressor.compress_posting_list(posting_list)
        assert isinstance(compressed["term"][1], bytes)


class TestEliasGammaRoundTrip:
    def test_roundtrip_large_sequence(self):
        values = list(range(50))
        encoded = EliasGammaEncoder.encode_sequence(values)
        decoded = EliasGammaDecoder.decode_sequence(encoded)
        assert decoded == values

    def test_roundtrip_long_positions(self):
        positions = list(range(0, 100, 5))
        encoded = EliasGammaEncoder.encode_gaps(positions)
        decoded = EliasGammaDecoder.decode_gaps(encoded)
        assert decoded == positions


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
