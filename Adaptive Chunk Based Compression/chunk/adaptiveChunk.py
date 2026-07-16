from typing import List, Dict, Any


# =========================
# Fixed Chunker (unchanged)
# =========================
class FixedChunker:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, data: List[int]) -> List[List[int]]:
        return [
            data[i:i + self.chunk_size]
            for i in range(0, len(data), self.chunk_size)
        ]

    def unchunk(self, chunks: List[List[int]]) -> List[int]:
        result = []
        for chunk in chunks:
            result.extend(chunk)
        return result


# =========================
# Adaptive Compressor
# =========================
class AdaptiveChunkedCompressor:
    def __init__(self, chunker: FixedChunker, codecs: Dict[str, tuple]):
        """
        codecs = {
            "vbyte": (compress_fn, decompress_fn),
            "bitpack": (...),
            "elias": (...)
        }
        """
        self.chunker = chunker
        self.codecs = codecs

    # ---------- Utility ----------
    def _get_size(self, obj: Any) -> int:
        """Normalize size measurement across different formats."""
        if isinstance(obj, bytes):
            return len(obj)
        elif isinstance(obj, list):
            return len(obj) * 4  # assume 4 bytes per int fallback
        elif isinstance(obj, bytearray):
            return len(obj)
        else:
            try:
                return len(obj)
            except:
                return 0

    # ---------- Core Logic ----------
    def choose_codec(self, chunk: List[int]) -> str:
        """Try all codecs and pick smallest."""
        best_codec = None
        best_size = float("inf")

        for name, (compress_fn, _) in self.codecs.items():
            try:
                compressed = compress_fn(chunk)
                size = self._get_size(compressed)

                if size < best_size:
                    best_size = size
                    best_codec = name

            except Exception:
                # Skip codecs that fail
                continue

        # Fallback safety
        return best_codec if best_codec is not None else "vbyte"

    def compress_list(self, data: List[int]) -> List[Dict]:
        chunks = self.chunker.chunk(data)
        result = []

        for chunk in chunks:
            codec_name = self.choose_codec(chunk)
            compress_fn, _ = self.codecs[codec_name]

            compressed = compress_fn(chunk)

            result.append({
                "codec": codec_name,
                "data": compressed
            })

        return result

    def decompress_list(self, compressed_chunks: List[Dict]) -> List[int]:
        result_chunks = []

        for item in compressed_chunks:
            codec_name = item["codec"]
            data = item["data"]

            _, decompress_fn = self.codecs[codec_name]
            chunk = decompress_fn(data)

            result_chunks.append(chunk)

        return self.chunker.unchunk(result_chunks)


# =========================
# Posting List Wrapper
# =========================
class AdaptiveChunkedPostingList:
    def __init__(self, chunk_size: int, codecs: Dict[str, tuple]):
        self.chunker = FixedChunker(chunk_size)
        self.pipeline = AdaptiveChunkedCompressor(
            self.chunker,
            codecs
        )

    def compress(self, posting_list: dict) -> dict:
        result = {}

        for term, docs in posting_list.items():
            result[term] = {}

            for doc_id, positions in docs.items():
                result[term][doc_id] = self.pipeline.compress_list(positions)

        return result

    def decompress(self, compressed: dict) -> dict:
        result = {}

        for term, docs in compressed.items():
            result[term] = {}

            for doc_id, chunks in docs.items():
                result[term][doc_id] = self.pipeline.decompress_list(chunks)

        return result


# =========================
# Example Codec Wrappers
# =========================
# You plug YOUR implementations here

def vbyte_compress(chunk: List[int]) -> bytes:
    from vbyte import VByteEncoder
    return VByteEncoder.encode_gaps(chunk)


def vbyte_decompress(data: bytes) -> List[int]:
    from vbyte import VByteDecoder
    return VByteDecoder.decode_gaps(data)


# Replace these with your real implementations
def bitpack_compress(chunk: List[int]):
    from bitpack import compress  # your function
    return compress(chunk)


def bitpack_decompress(data):
    from bitpack import decompress
    return decompress(data)


def elias_compress(chunk: List[int]):
    from elias import encode  # your function
    return encode(chunk)


def elias_decompress(data):
    from elias import decode
    return decode(data)


# =========================
# Helper to Build Codec Map
# =========================
def get_default_codecs():
    return {
        "vbyte": (vbyte_compress, vbyte_decompress),
        "bitpack": (bitpack_compress, bitpack_decompress),
        "elias": (elias_compress, elias_decompress),
    }
