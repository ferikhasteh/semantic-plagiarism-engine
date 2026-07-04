# Semantic Plagiarism Engine

A command-line plagiarism and near-duplicate detection engine for the Data Mining course project.

## Goal

This project implements and compares two document similarity detection pipelines:

1. Shingling + MinHash + LSH
2. TF-IDF Weighted SimHash

The project is implemented as a reproducible CLI tool, not as a notebook-only project.

## Main Modules

- preprocessing.py: text cleaning, tokenization, stopword removal, and word shingling
- minhash.py: MinHash signature generation
- lsh.py: Locality Sensitive Hashing candidate generation
- simhash.py: TF-IDF weighted SimHash
- dataset.py: loading corpora and labeled pair datasets
- evaluation.py: precision, recall, F1, and runtime evaluation
- cli.py: command-line interface

## Planned Commands

Compare two documents:

python -m plagiarism_engine.cli compare --file-a data/sample_corpus/doc_01.txt --file-b data/sample_corpus/doc_02.txt --output outputs/two_file_compare.json

Find similar documents in a folder:

python -m plagiarism_engine.cli corpus --data data/sample_corpus --threshold 0.25 --shingle-size 3 --output outputs/candidates.csv

Evaluate on labeled pairs:

python -m plagiarism_engine.cli pairs --pairs data/raw/quora/train.csv --text-col-a question1 --text-col-b question2 --label-col is_duplicate --limit 5000 --output outputs/metrics.csv

## Notes

Large raw datasets must not be committed to GitHub. The data/raw and data/processed directories are ignored by Git.
