from typing import List, Tuple


class EliasGammaEncoder:
    @staticmethod
    def _write_bits(buffer: bytearray, bit_buffer: int, bit_count: int, value: int, length: int) -> tuple[bytearray, int, int]:
        while length > 0:
            can_write = min(8 - bit_count, length)
            shift = length - can_write
            chunk = (value >> shift) & ((1 << can_write) - 1)
            bit_buffer = (bit_buffer << can_write) | chunk
            bit_count += can_write
            length -= can_write

            if bit_count == 8:
                buffer.append(bit_buffer)
                bit_buffer = 0
                bit_count = 0

        return buffer, bit_buffer, bit_count

    @staticmethod
    def encode_integer(value: int) -> bytes:
        if value < 0:
            raise ValueError("Elias gamma only supports non-negative integers")

        n = value + 1
        length = n.bit_length()
        buffer = bytearray()
        bit_buffer = 0
        bit_count = 0

        # Unary prefix: (length - 1) zeros, then one
        if length > 1:
            buffer, bit_buffer, bit_count = EliasGammaEncoder._write_bits(
                buffer, bit_buffer, bit_count, 0, length - 1
            )
        buffer, bit_buffer, bit_count = EliasGammaEncoder._write_bits(
            buffer, bit_buffer, bit_count, 1, 1
        )

        # Binary suffix: value without leading 1
        if length > 1:
            suffix = n & ((1 << (length - 1)) - 1)
            buffer, bit_buffer, bit_count = EliasGammaEncoder._write_bits(
                buffer, bit_buffer, bit_count, suffix, length - 1
            )

        if bit_count > 0:
            buffer.append(bit_buffer << (8 - bit_count))

        return bytes(buffer)

    @staticmethod
    def encode_sequence(integers: List[int]) -> bytes:
        result = bytearray()
        bit_buffer = 0
        bit_count = 0

        for value in integers:
            n = value + 1
            length = n.bit_length()

            # Unary prefix: (length - 1) zeros, then one
            if length > 1:
                result, bit_buffer, bit_count = EliasGammaEncoder._write_bits(
                    result, bit_buffer, bit_count, 0, length - 1
                )
            result, bit_buffer, bit_count = EliasGammaEncoder._write_bits(
                result, bit_buffer, bit_count, 1, 1
            )

            # Binary suffix: value without leading 1
            if length > 1:
                suffix = n & ((1 << (length - 1)) - 1)
                result, bit_buffer, bit_count = EliasGammaEncoder._write_bits(
                    result, bit_buffer, bit_count, suffix, length - 1
                )

        if bit_count > 0:
            result.append(bit_buffer << (8 - bit_count))

        return bytes(result)

    @staticmethod
    def encode_gaps(positions: List[int]) -> bytes:
        if not positions:
            return bytes()

        gaps = [positions[0]]
        for i in range(1, len(positions)):
            gaps.append(positions[i] - positions[i - 1])

        return EliasGammaEncoder.encode_sequence(gaps)


class EliasGammaDecoder:
    @staticmethod
    def _read_bit(data: bytes, bit_index: int) -> int:
        byte_index = bit_index // 8
        bit_in_byte = 7 - (bit_index % 8)
        return (data[byte_index] >> bit_in_byte) & 1

    @staticmethod
    def decode_integer(data: bytes, bit_offset: int = 0) -> Tuple[int, int]:
        total_bits = len(data) * 8
        if bit_offset >= total_bits:
            raise IndexError("Bit offset out of range")

        bit_index = bit_offset
        zeros = 0

        while bit_index < total_bits:
            if EliasGammaDecoder._read_bit(data, bit_index) == 1:
                bit_index += 1
                break
            zeros += 1
            bit_index += 1

        if bit_index > total_bits:
            raise IndexError("Incomplete Elias gamma code")

        value = 1
        for _ in range(zeros):
            if bit_index >= total_bits:
                raise IndexError("Incomplete Elias gamma code suffix")
            value = (value << 1) | EliasGammaDecoder._read_bit(data, bit_index)
            bit_index += 1

        return value - 1, bit_index - bit_offset

    @staticmethod
    def decode_sequence(data: bytes, count: int | None = None) -> List[int]:
        result: List[int] = []
        bit_offset = 0
        total_bits = len(data) * 8

        while bit_offset < total_bits:
            if count is not None and len(result) >= count:
                break

            try:
                value, consumed = EliasGammaDecoder.decode_integer(data, bit_offset)
            except IndexError:
                break

            result.append(value)
            bit_offset += consumed

        return result

    @staticmethod
    def decode_gaps(data: bytes) -> List[int]:
        values = EliasGammaDecoder.decode_sequence(data)
        if not values:
            return []

        positions = [values[0]]
        for gap in values[1:]:
            positions.append(positions[-1] + gap)

        return positions


class EliasGammaCompressor:
    @staticmethod
    def compress_posting_list(posting_list: dict, use_gaps: bool = True) -> dict:
        compressed = {}
        for term, doc_positions in posting_list.items():
            compressed[term] = {}
            for doc_id, positions in doc_positions.items():
                if use_gaps:
                    encoded = EliasGammaEncoder.encode_gaps(positions)
                else:
                    encoded = EliasGammaEncoder.encode_sequence(positions)
                compressed[term][doc_id] = encoded
        return compressed

    @staticmethod
    def decompress_posting_list(compressed_list: dict, use_gaps: bool = True) -> dict:
        decompressed = {}
        for term, doc_positions in compressed_list.items():
            decompressed[term] = {}
            for doc_id, encoded_bytes in doc_positions.items():
                if use_gaps:
                    positions = EliasGammaDecoder.decode_gaps(encoded_bytes)
                else:
                    positions = EliasGammaDecoder.decode_sequence(encoded_bytes)
                decompressed[term][doc_id] = positions
        return decompressed
