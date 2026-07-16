# Data Compression Algorithm Comparison

This project compares various data compression algorithms including vByte, Elias, Bit Packing, and Adaptive compression techniques. It's designed to benchmark compression efficiency, compression ratios, and performance across different data patterns.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

## Prerequisites

- **Python 3.12+** (required by the project)
- **uv** package manager
- macOS, Linux, or Windows with a terminal/command line

### Installing Python 3.12

If you don't have Python 3.12 installed, you can:

- Download from [python.org](https://www.python.org/downloads/)
- Use a package manager like Homebrew: `brew install python@3.12`
- Use pyenv: `pyenv install 3.12.0 && pyenv shell 3.12.0`

### Installing uv

uv is a fast Python package installer. Install it with:

```bash
# macOS or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew (macOS)
brew install uv
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd csce410-team5
   ```

2. **Create a virtual environment and install dependencies:**

   Using uv (recommended):

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

## Running the Project

### Default Comparison

By default, running the project executes all synthetic dataset tests:

```bash
# Using make
make run

# Or directly with Python
python3 main.py
```

### Individual Comparisons

You can run specific comparisons by modifying `main.py` and uncommenting/commenting out the different comparison functions:

```python
if __name__ == "__main__":
    # vByteEliasComparison()              # Compares vByte vs Elias Gamma on Shakespeare
    # vByteBitPackComparison()            # Compares vByte vs Bit Packing on Shakespeare
    run_all_tests()                        # Runs all synthetic dataset tests (DEFAULT)
    # adaptiveComparison()                 # Compares adaptive compression vs other algorithms
```

**Available Comparisons:**

- **`run_all_tests()`** (from `synthetic.py`) - Runs comprehensive tests on synthetically generated datasets with various data patterns (uniform, clustered, sparse with outliers)

- **`vByteEliasComparison()`** (from `vByte_Elias.py`) - Compares vByte and Elias Gamma compression algorithms on Shakespeare text using fixed-size chunking

- **`vByteBitPackComparison()`** (from `vByte_bitPack.py`) - Compares vByte and Bit Packing compression on Shakespeare text

- **`adaptiveComparison()`** (from `adaptive_comp.py`) - Compares adaptive compression (which chooses the best algorithm per chunk) against vByte, Elias Gamma, and Bit Packing on Shakespeare

### Running Comparisons Directly

You can also run individual comparison modules directly:

```bash
# Run vByte vs Elias comparison
python3 -m comparisons.vByte_Elias

# Run vByte vs Bit Packing comparison
python3 -m comparisons.vByte_bitPack

# Run synthetic dataset tests
python3 -m comparisons.synthetic

# Run adaptive compression comparison
python3 -m comparisons.adaptive_comp
```

## Running Tests

To run the test suite:

```bash
# Using make
make test

# Or directly with pytest
python3 -m pytest
```

## Project Structure

```
csce410-team5/
├── main.py                 # Entry point for running comparisons
├── Makefile                # Build automation commands
├── pyproject.toml          # Project configuration and dependencies
├── chunk/                  # Chunking strategies
│   ├── fixedChunk.py       # Fixed-size chunking
│   └── adaptiveChunk.py    # Adaptive chunking
├── compression/            # Compression algorithms
│   ├── vByte.py            # vByte compression
│   ├── elias.py            # Elias compression
│   ├── bitPacking.py       # Bit packing compression
│   └── adaptive.py         # Adaptive compression
├── comparisons/            # Benchmarking and comparison scripts
│   ├── synthetic.py        # Synthetic dataset tests
│   ├── vByte_Elias.py      # vByte vs Elias comparison
│   ├── vByte_bitPack.py    # vByte vs Bit Packing comparison
│   └── adaptive_comp.py    # Adaptive compression comparison
├── utils/                  # Utility functions
│   ├── data_extraction.py  # Data loading utilities
│   ├── metrics.py          # Performance metrics
│   └── posting_list.py     # Posting list utilities
├── data/                   # Datasets
│   └── 6_newsgroups/       # 20 newsgroups dataset
│   └── t8.shakespeare.txt  # 20 newsgroups dataset
├── tests/                  # Unit tests
│   ├── test_vbyte.py
│   ├── test_elias.py
│   ├── test_adaptive.py
│   └── test_posting_list.py
└── graphics/              # Visualization and results
```
