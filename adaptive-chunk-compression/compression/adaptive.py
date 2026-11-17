from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from compression.bitPacking import BitPackDecoder, BitPackEncoder
from compression.elias import EliasGammaDecoder, EliasGammaEncoder
from compression.vByte import VByteDecoder, VByteEncoder

AlgorithmName = str
ChunkMetadata = Dict[str, Any]
CompressionAlgorithm = Tuple[Callable[[List[int]], bytes], Callable[[bytes], List[int]]]

COMPRESSORS: Dict[AlgorithmName, CompressionAlgorithm] = {
    "vbyte": (VByteEncoder.encode_gaps, VByteDecoder.decode_gaps),
    "elias": (EliasGammaEncoder.encode_gaps, EliasGammaDecoder.decode_gaps),
    "bitpack": (BitPackEncoder.encode_gaps, BitPackDecoder.decode_gaps),
}

SEQUENCE_ENCODERS: Dict[AlgorithmName, Callable[[List[int]], bytes]] = {
    "vbyte": VByteEncoder.encode_sequence,
    "elias": EliasGammaEncoder.encode_sequence,
    "bitpack": BitPackEncoder.encode_sequence,
}


def positions_to_gaps(positions: List[int]) -> List[int]:
    if not positions:
        return []

    gaps = [positions[0]]
    for index in range(1, len(positions)):
        gaps.append(positions[index] - positions[index - 1])
    return gaps


def choose_best_algorithm(gaps: List[int]) -> AlgorithmName:
    """Choose the smallest compressor by actual encoded size on the raw gap sequence."""
    if not gaps:
        return "vbyte"

    sizes = {
        name: len(encode_sequence(gaps))
        for name, encode_sequence in SEQUENCE_ENCODERS.items()
    }

    return min(sizes, key=sizes.get)


def best_encoded_size(gaps: List[int]) -> int:
    """Return the compressed size of the best algorithm for this gap sequence."""
    if not gaps:
        return 0
    return min(len(encode_sequence(gaps)) for encode_sequence in SEQUENCE_ENCODERS.values())


class AdaptiveChunker:
    """Create chunks by detecting gap-distribution boundaries."""

    def __init__(
        self,
        min_chunk_size: int = 4,
        max_chunk_size: int = 128,
        large_gap_threshold: int = 64,
        small_gap_limit: int = 10,
        small_gap_ratio_threshold: float = 0.75,
        large_gap_factor: float = 16.0,
    ) -> None:
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.large_gap_threshold = large_gap_threshold
        self.small_gap_limit = small_gap_limit
        self.small_gap_ratio_threshold = small_gap_ratio_threshold
        self.large_gap_factor = large_gap_factor

    def chunk_positions(self, positions: List[int]) -> List[List[int]]:
        """Split a sorted posting list into adaptive chunks using gap heuristics."""
        if not positions:
            return []

        chunks: List[List[int]] = []
        current_chunk: List[int] = [positions[0]]
        current_gaps: List[int] = []
        gap_sum = 0
        small_gap_count = 0
        last_position = positions[0]

        for position in positions[1:]:
            gap = position - last_position
            split = False

            if len(current_chunk) >= self.max_chunk_size:
                split = True
            elif len(current_gaps) >= self.min_chunk_size:
                average_gap = gap_sum / len(current_gaps)
                small_ratio = small_gap_count / len(current_gaps)

                if gap > self.large_gap_threshold and small_ratio >= self.small_gap_ratio_threshold:
                    split = True
                elif average_gap > 0 and gap / average_gap >= self.large_gap_factor:
                    split = True

            if split:
                chunks.append(current_chunk)
                current_chunk = [position]
                current_gaps = []
                gap_sum = 0
                small_gap_count = 0
            else:
                current_chunk.append(position)
                current_gaps.append(gap)
                gap_sum += gap
                if gap <= self.small_gap_limit:
                    small_gap_count += 1

            last_position = position

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _small_gap_ratio(self, gaps: List[int]) -> float:
        if not gaps:
            return 0.0
        small_count = sum(1 for gap in gaps if gap <= self.small_gap_limit)
        return small_count / len(gaps)


class AdaptivePostingListCompressor:
    """Compress posting lists using adaptive chunking and per-chunk algorithm choice."""

    def __init__(self, chunker: AdaptiveChunker | None = None) -> None:
        self.chunker = chunker or AdaptiveChunker()

    def compress_posting_list(self, posting_list: dict) -> dict:
        compressed: dict = {}

        for term, doc_positions in posting_list.items():
            compressed[term] = {}
            for doc_id, positions in doc_positions.items():
                compressed[term][doc_id] = [
                    self._compress_chunk(chunk)
                    for chunk in self.chunker.chunk_positions(positions)
                ]

        return compressed

    def _compress_chunk(self, positions: List[int]) -> ChunkMetadata:
        gaps = positions_to_gaps(positions)
        best_algorithm = choose_best_algorithm(gaps)
        encode, _ = COMPRESSORS[best_algorithm]
        return {
            "algorithm": best_algorithm,
            "data": encode(positions),
        }


    def decompress_posting_list(self, compressed_list: dict) -> dict:
        decompressed: dict = {}

        for term, doc_positions in compressed_list.items():
            decompressed[term] = {}
            for doc_id, chunks in doc_positions.items():
                decoded_positions: List[int] = []
                for chunk in chunks:
                    algorithm = chunk["algorithm"]
                    encoded_data = chunk["data"]
                    if algorithm not in COMPRESSORS:
                        raise ValueError(f"Unknown compression algorithm: {algorithm}")
                    _, decode = COMPRESSORS[algorithm]
                    decoded_positions.extend(decode(encoded_data))
                decompressed[term][doc_id] = decoded_positions

        return decompressed
