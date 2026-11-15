class FixedChunker:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def chunk(self, data):
        return [
            data[i:i + self.chunk_size]
            for i in range(0, len(data), self.chunk_size)
        ]

    def unchunk(self, chunks):
        result = []
        for chunk in chunks:
            result.extend(chunk)
        return result

class ChunkedCompressor:
    def __init__(self, chunker, compress_fn, decompress_fn):
        self.chunker = chunker
        self.compress = compress_fn
        self.decompress = decompress_fn

    def compress_list(self, data):
        chunks = self.chunker.chunk(data)
        return [self.compress(chunk) for chunk in chunks]

    def decompress_list(self, compressed_chunks):
        chunks = [self.decompress(c) for c in compressed_chunks]
        return self.chunker.unchunk(chunks)

class ChunkedPostingList:
    def __init__(self, chunk_size, compress_fn, decompress_fn):
        self.chunker = FixedChunker(chunk_size)
        self.pipeline = ChunkedCompressor(
            self.chunker,
            compress_fn,
            decompress_fn
        )

    def compress(self, posting_list):
        result = {}

        for term, docs in posting_list.items():
            result[term] = {}

            for doc_id, positions in docs.items():
                result[term][doc_id] = self.pipeline.compress_list(positions)

        return result

    def decompress(self, compressed):
        result = {}

        for term, docs in compressed.items():
            result[term] = {}

            for doc_id, chunks in docs.items():
                result[term][doc_id] = self.pipeline.decompress_list(chunks)

        return result
