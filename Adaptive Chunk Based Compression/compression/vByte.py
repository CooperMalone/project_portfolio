from typing import List, Tuple


class VByteEncoder:
    @staticmethod
    def encode_integer(value: int) -> bytes:
        if value < 0:
            raise ValueError("vByte only supports non-negative integers")

        result = bytearray()

        # Extract 7-bit chunks
        while value > 127:
            # Add 7 bits with continuation bit set (0x80)
            result.append((value & 0x7F) | 0x80)
            value >>= 7

        # Add final chunk without continuation bit
        result.append(value & 0x7F)

        return bytes(result)

    @staticmethod
    def encode_sequence(integers: List[int]) -> bytes:
        result = bytearray()
        for value in integers:
            result.extend(VByteEncoder.encode_integer(value))
        return bytes(result)

    @staticmethod
    def encode_gaps(positions: List[int]) -> bytes:
        if not positions:
            return bytes()

        gaps = [positions[0]]  # First position
        for i in range(1, len(positions)):
            gaps.append(positions[i] - positions[i - 1])

        return VByteEncoder.encode_sequence(gaps)


class VByteDecoder:
    @staticmethod
    def decode_integer(data: bytes, offset: int = 0) -> Tuple[int, int]:
        result = 0
        shift = 0

        while offset < len(data):
            byte = data[offset]
            offset += 1

            result |= (byte & 0x7F) << shift

            if byte & 0x80 == 0:
                break

            shift += 7

        return result, offset

    @staticmethod
    def decode_sequence(data: bytes, count: int | None = None) -> List[int]:
        result = []
        offset = 0

        while offset < len(data):
            if count is not None and len(result) >= count:
                break

            value, offset = VByteDecoder.decode_integer(data, offset)
            result.append(value)

        return result

    @staticmethod
    def decode_gaps(data: bytes) -> List[int]:
        gaps = VByteDecoder.decode_sequence(data)
        if not gaps:
            return []

        # Reconstruct positions from gaps
        positions = [gaps[0]]
        for gap in gaps[1:]:
            positions.append(positions[-1] + gap)

        return positions


class PostingListCompressor:
    @staticmethod
    def compress_posting_list(
        posting_list: dict, use_gaps: bool = True
    ) -> dict:
        compressed = {}

        for term, doc_positions in posting_list.items():
            compressed[term] = {}

            for doc_id, positions in doc_positions.items():
                if use_gaps:
                    encoded = VByteEncoder.encode_gaps(positions)
                else:
                    encoded = VByteEncoder.encode_sequence(positions)

                compressed[term][doc_id] = encoded

        return compressed

    @staticmethod
    def decompress_posting_list(
        compressed_list: dict, use_gaps: bool = True
    ) -> dict:
        decompressed = {}

        for term, doc_positions in compressed_list.items():
            decompressed[term] = {}

            for doc_id, encoded_bytes in doc_positions.items():
                if use_gaps:
                    positions = VByteDecoder.decode_gaps(encoded_bytes)
                else:
                    positions = VByteDecoder.decode_sequence(encoded_bytes)

                decompressed[term][doc_id] = positions

        return decompressed

    @staticmethod
    def get_compression_ratio(
        original_posting_list: dict, compressed_posting_list: dict
    ) -> float:
        # Original size (4 bytes per integer)
        original_size = 0
        for term_positions in original_posting_list.values():
            for positions in term_positions.values():
                original_size += len(positions) * 4

        # Compressed size: support both bytes and raw integer lists.
        compressed_size = 0
        for term_bytes in compressed_posting_list.values():
            for encoded in term_bytes.values():
                if isinstance(encoded, bytes):
                    compressed_size += len(encoded)
                elif isinstance(encoded, (list, tuple)):
                    compressed_size += len(encoded) * 4
                else:
                    compressed_size += len(encoded)

        if original_size == 0:
            return 0.0

        return compressed_size / original_size
