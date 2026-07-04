# Semantic Plagiarism Engine

A command-line plagiarism and near-duplicate detection engine for the Data Mining course project.

The project implements and compares two document similarity detection pipelines:

1. Shingling + MinHash + LSH
2. TF-IDF Weighted SimHash

The system is designed as a reproducible CLI tool, not as a notebook-only project.

---

## Project Structure

    semantic-plagiarism-engine/
    ├── README.md
    ├── requirements.txt
    ├── pyproject.toml
    ├── data/
    │   ├── sample_corpus/
    │   ├── raw/
    │   └── processed/
    ├── docs/
    ├── notebooks/
    ├── outputs/
    ├── src/
    │   └── plagiarism_engine/
    │       ├── preprocessing.py
    │       ├── minhash.py
    │       ├── lsh.py
    │       ├── simhash.py
    │       ├── dataset.py
    │       ├── evaluation.py
    │       └── cli.py
    └── tests/

---

## Implemented Methods

### 1. Text Preprocessing

The preprocessing module performs:

- text normalization
- Persian/Arabic character normalization
- punctuation and extra whitespace removal
- URL and email removal
- word-level tokenization
- stopword removal
- word shingling

The default shingle size is 3 words.

### 2. Exact Jaccard Similarity

Each document is represented as a set of word shingles.

Jaccard similarity is computed as:

    J(A, B) = |A ∩ B| / |A ∪ B|

This method is accurate but expensive for large corpora because all document pairs require comparison.

### 3. MinHash

MinHash creates a compact signature for each document's shingle set.

The estimated similarity between two documents is computed as the fraction of equal positions in their MinHash signatures.

Default signature length:

    num_perm = 128

### 4. Locality Sensitive Hashing

LSH divides each MinHash signature into bands.

Documents that share at least one bucket in at least one band become candidate pairs.

This reduces the number of comparisons compared to all-pairs comparison.

Default setting:

    num_bands = 32

### 5. TF-IDF Weighted SimHash

SimHash creates a 64-bit fingerprint for each document.

Each token contributes to a weighted bit vector using its TF-IDF weight.

Similarity is computed from Hamming distance:

    similarity = 1 - hamming_distance / hash_bits

Default setting:

    hash_bits = 64

---

## Installation

Create a virtual environment:

    python -m venv .venv

Activate it on Windows Git Bash:

    source .venv/Scripts/activate

Install dependencies:

    pip install -r requirements.txt
    pip install -e .

---

## Running Tests

Run all tests:

    pytest tests

---

## CLI Usage

### 1. Compare Two Documents

    python -m plagiarism_engine.cli compare \
      --file-a data/sample_corpus/doc_01.txt \
      --file-b data/sample_corpus/doc_02.txt \
      --shingle-size 3 \
      --output outputs/two_file_compare.json

Output:

    outputs/two_file_compare.json

The output includes:

- exact Jaccard similarity
- MinHash similarity
- SimHash fingerprints
- SimHash Hamming distance
- SimHash similarity

---

### 2. Search Similar Documents in a Corpus

    python -m plagiarism_engine.cli corpus \
      --data data/sample_corpus \
      --threshold 0.1 \
      --shingle-size 3 \
      --output outputs/candidates.csv

Output:

    outputs/candidates.csv

The output includes candidate document pairs and LSH reduction statistics.

---

### 3. Evaluate on a Labeled Pair Dataset

Example for a Quora-like dataset:

    python -m plagiarism_engine.cli pairs \
      --pairs data/raw/quora/train.csv \
      --text-col-a question1 \
      --text-col-b question2 \
      --label-col is_duplicate \
      --limit 5000 \
      --threshold 0.25 \
      --simhash-threshold 0.75 \
      --output outputs/metrics.csv

Output:

    outputs/metrics.csv

The output includes:

- accuracy
- precision
- recall
- F1-score
- runtime

for:

- exact Jaccard
- MinHash
- SimHash

---

## Sample Corpus

The repository includes a small sample corpus:

    data/sample_corpus/doc_01.txt
    data/sample_corpus/doc_02.txt
    data/sample_corpus/doc_03.txt

These files are used for quick CLI testing.

---

## Data Policy

Large raw datasets must not be committed to GitHub.

The following folders are ignored by Git:

    data/raw/
    data/processed/

Only small sample files and final reproducible outputs should be committed.

---

## Current Status

Implemented:

- text preprocessing
- word shingling
- exact Jaccard similarity
- MinHash from scratch
- LSH candidate generation
- TF-IDF weighted SimHash
- evaluation metrics
- dataset loading utilities
- CLI commands

Remaining work:

- run experiments on a labeled dataset
- generate final metrics.csv
- write final technical report
- record short CLI demo video
