from typing import List, Tuple
import math


class BitPackEncoder:
    @staticmethod
    def _bits_required(value: int) -> int:
        if value == 0:
            return 1
        return value.bit_length()

    @staticmethod
    def encode_sequence(integers: List[int]) -> bytes:
        if not integers:
            return bytes()

        min_val = min(integers)
        deltas = [x - min_val for x in integers]

        max_delta = max(deltas)
        bit_width = BitPackEncoder._bits_required(max_delta)

        result = bytearray()

        # --- Header ---
        # 4 bytes: min value
        result.extend(min_val.to_bytes(4, byteorder="little"))

        # 1 byte: bit width
        result.append(bit_width)

        # 4 bytes: number of integers
        result.extend(len(integers).to_bytes(4, byteorder="little"))

        # --- Bit packing ---
        buffer = 0
        bits_in_buffer = 0

        for value in deltas:
            buffer |= (value << bits_in_buffer)
            bits_in_buffer += bit_width

            while bits_in_buffer >= 8:
                result.append(buffer & 0xFF)
                buffer >>= 8
                bits_in_buffer -= 8

        # Flush remaining bits
        if bits_in_buffer > 0:
            result.append(buffer & 0xFF)

        return bytes(result)

    @staticmethod
    def encode_gaps(positions: List[int]) -> bytes:
        if not positions:
            return bytes()

        gaps = [positions[0]]
        for i in range(1, len(positions)):
            gaps.append(positions[i] - positions[i - 1])

        return BitPackEncoder.encode_sequence(gaps)


class BitPackDecoder:
    @staticmethod
    def decode_sequence(data: bytes) -> List[int]:
        if not data:
            return []

        offset = 0

        # --- Header ---
        min_val = int.from_bytes(data[offset:offset+4], "little")
        offset += 4

        bit_width = data[offset]
        offset += 1

        count = int.from_bytes(data[offset:offset+4], "little")
        offset += 4

        result = []
        buffer = 0
        bits_in_buffer = 0

        mask = (1 << bit_width) - 1

        while len(result) < count:
            while bits_in_buffer < bit_width and offset < len(data):
                buffer |= data[offset] << bits_in_buffer
                bits_in_buffer += 8
                offset += 1

            value = buffer & mask
            buffer >>= bit_width
            bits_in_buffer -= bit_width

            result.append(value + min_val)

        return result

    @staticmethod
    def decode_gaps(data: bytes) -> List[int]:
        gaps = BitPackDecoder.decode_sequence(data)
        if not gaps:
            return []

        positions = [gaps[0]]
        for gap in gaps[1:]:
            positions.append(positions[-1] + gap)

        return positions
