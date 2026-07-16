import time
class CompressionAlgo:
    def __init__(self, name, compress, decompress):
        self.name = name
        self.compress = compress
        self.decompress = decompress

def benchmark(algo, input_data):
    start = time.perf_counter()
    compressed = algo.compress(input_data)
    mid = time.perf_counter()

    decompressed = algo.decompress(compressed)
    end = time.perf_counter()

    orig = len(input_data) 
    comp_size = len(compressed)

    compress_time = mid - start
    decomp_time = end - mid

    comp_ratio = comp_size / orig
    comp_speed = orig / compress_time
    decomp_speed = orig / decomp_time

    return {
        "algorithm": algo.name,
        "compression time": compress_time,
        "decompression time": decomp_time,
        "compression ratio": comp_ratio,
        "compression speed": comp_speed,
        "decompression speed": decomp_speed,
    }
