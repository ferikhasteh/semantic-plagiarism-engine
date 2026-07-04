#!/usr/bin/env bash

set -e

echo "Running Semantic Plagiarism Engine demo..."

echo
echo "1) Compare two documents"
python -m plagiarism_engine.cli compare \
  --file-a data/sample_corpus/doc_01.txt \
  --file-b data/sample_corpus/doc_02.txt \
  --shingle-size 3 \
  --output outputs/two_file_compare.json

echo
echo "2) Search similar documents in sample corpus"
python -m plagiarism_engine.cli corpus \
  --data data/sample_corpus \
  --threshold 0.1 \
  --shingle-size 3 \
  --output outputs/candidates.csv

echo
echo "3) Evaluate labeled sample pairs"
python -m plagiarism_engine.cli pairs \
  --pairs data/sample_corpus/sample_pairs.csv \
  --text-col-a text_a \
  --text-col-b text_b \
  --label-col label \
  --threshold 0.1 \
  --simhash-threshold 0.65 \
  --shingle-size 2 \
  --output outputs/metrics.csv \
  --details-output outputs/pair_predictions.csv

echo
echo "4) Build PDF report"
python scripts/build_report_pdf.py

echo
echo "Demo completed."
echo "Generated outputs:"
echo "- outputs/two_file_compare.json"
echo "- outputs/candidates.csv"
echo "- outputs/metrics.csv"
echo "- outputs/pair_predictions.csv"
echo "- docs/project_report.pdf"
