# Adaptive Chunk-Based Compression for Inverted Index Posting Lists

**University of Nebraska–Lincoln · individual project** · [Full paper (PDF)](./report.pdf)

Search engines store, for every term, a *posting list*: the sorted IDs of the documents that contain it. Storing the gaps between consecutive IDs and compressing those integers is standard practice, but most systems pick one codec for everything. Real posting lists mix dense runs of small gaps with sparse stretches of large ones, so no single codec fits them all.

This project implements three classic integer codecs from scratch and adds an **adaptive** scheme. It splits a list into chunks, encodes each chunk with every codec, keeps the smallest result, and records which codec it used.

## Results

### No single codec wins everywhere, but the adaptive one is always first or second

Compressed size in bytes on four synthetic gap distributions (lower is better):

| Distribution | vByte | Elias gamma | Bit packing | **Adaptive** | Winner |
|---|---:|---:|---:|---:|---|
| Uniform, small gaps | 10,000 | 6,272 | **5,009** | 5,783 | Bit packing |
| Clustered | 20,000 | 23,750 | **3,757** | 4,540 | Bit packing |
| Sparse with outliers | 10,890 | **6,858** | 21,259 | 7,091 | Elias gamma |
| Monotonic gaps | 24,541 | 33,543 | 18,759 | **12,034** | **Adaptive** |

- **Bit packing** is excellent on tight, clustered data and poor when there are outliers, because one large value forces a wide bit width on every value in the block.
- **Elias gamma** handles sparse data with outliers well and does poorly on clustered data.
- **Adaptive** comes second wherever it doesn't win, and on monotonic gaps it beats every fixed codec by mixing them chunk by chunk. Storing the codec choice costs 79 bytes, about 1–2% of the output.

### On real text, adaptive edges out vByte

The real corpora were *The Complete Works of Shakespeare* (194 documents) and a 6-group subset of 20 Newsgroups (1,928 documents). On both, vByte and Adaptive gave the best compression ratios, around 0.45 on Shakespeare (compressed size ÷ original). Adaptive was slightly smaller than vByte at every chunk size, and Elias gamma and bit packing were far behind. Example output is in [`graphics/`](./graphics).

### The trade-off is compression time

Adaptive compression encodes each chunk once per codec, so it took about 11 s on the benchmark against about 2.5 s for vByte. Decoding doesn't pay that cost: each chunk carries its codec ID, so it is decoded in one pass. This trade-off makes sense for an index that is written once and read many times.

## How it works

| Codec | Idea | File |
|---|---|---|
| **vByte** | 7 data bits per byte; the high bit marks whether more bytes follow | [`compression/vByte.py`](./compression/vByte.py) |
| **Elias gamma** | Unary length prefix plus binary suffix, packed into a continuous bitstream across byte boundaries | [`compression/elias.py`](./compression/elias.py) |
| **Bit packing** | Subtract the block minimum, then pack every value at the smallest bit width that fits the largest | [`compression/bitPacking.py`](./compression/bitPacking.py) |
| **Fixed chunking** | Split a list into equal blocks and compress each independently, which allows partial decoding | [`chunk/fixedChunk.py`](./chunk/fixedChunk.py) |
| **Adaptive** | For each chunk, try every codec, keep the smallest, and store a codec ID | [`chunk/adaptiveChunk.py`](./chunk/adaptiveChunk.py), [`compression/adaptive.py`](./compression/adaptive.py) |

The text pipeline (tokenize on whitespace → lowercase → deduplicate term/document pairs → gap-encode) is in [`utils/posting_list.py`](./utils/posting_list.py).

## Running it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CooperMalone/project_portfolio.git
cd project_portfolio/adaptive-chunk-compression
uv sync

uv run python main.py          # synthetic benchmark (no downloads needed)
uv run python -m pytest        # 98 unit tests, round-trip correctness for every codec
```

The real-text comparisons need two datasets in `data/` (not committed): `t8.shakespeare.txt` ([Project Gutenberg #100](https://www.gutenberg.org/ebooks/100)) and `6_newsgroups.zip` (a subset of [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)). Then run any of:

```bash
uv run python -m comparisons.vByte_Elias      # vByte vs Elias gamma, Shakespeare + Newsgroups
uv run python -m comparisons.vByte_bitPack    # vByte vs bit packing
uv run python -m comparisons.adaptive_comp    # adaptive vs all three codecs
uv run python -m comparisons.synthetic        # same as main.py
```

## Structure

```text
adaptive-chunk-compression/
├── compression/   # vByte, Elias gamma, bit packing, adaptive compressor
├── chunk/         # fixed and adaptive chunking
├── comparisons/   # benchmark scripts (synthetic and real text)
├── utils/         # posting-list construction, data loading, timing
├── tests/         # pytest suite
├── graphics/      # benchmark output
├── main.py        # entry point (synthetic benchmark)
└── report.pdf     # full paper
```
