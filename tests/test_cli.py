import csv
import json
from pathlib import Path

from plagiarism_engine.cli import main


def test_cli_compare(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    output = tmp_path / "compare.json"

    file_a.write_text("lion ate zebra", encoding="utf-8")
    file_b.write_text("lion ate zebra and goat", encoding="utf-8")

    exit_code = main(
        [
            "compare",
            "--file-a",
            str(file_a),
            "--file-b",
            str(file_b),
            "--output",
            str(output),
            "--shingle-size",
            "2",
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["file_a"] == str(file_a)
    assert data["file_b"] == str(file_b)
    assert data["jaccard_similarity"] > 0


def test_cli_corpus(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    (corpus_dir / "doc_1.txt").write_text("lion ate zebra", encoding="utf-8")
    (corpus_dir / "doc_2.txt").write_text("lion ate zebra and goat", encoding="utf-8")
    (corpus_dir / "doc_3.txt").write_text("cat drinks milk", encoding="utf-8")

    output = tmp_path / "candidates.csv"

    exit_code = main(
        [
            "corpus",
            "--data",
            str(corpus_dir),
            "--threshold",
            "0.1",
            "--shingle-size",
            "2",
            "--num-bands",
            "16",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    content = output.read_text(encoding="utf-8")

    assert "doc_a" in content
    assert "__STATS__" in content


def test_cli_pairs(tmp_path: Path):
    pairs_csv = tmp_path / "pairs.csv"
    output = tmp_path / "metrics.csv"

    pairs_csv.write_text(
        "question1,question2,is_duplicate\n"
        "lion ate zebra,lion ate zebra and goat,1\n"
        "cat drinks milk,python code runs,0\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "pairs",
            "--pairs",
            str(pairs_csv),
            "--text-col-a",
            "question1",
            "--text-col-b",
            "question2",
            "--label-col",
            "is_duplicate",
            "--threshold",
            "0.1",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0

    with output.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    methods = {row["method"] for row in rows}

    assert "jaccard_exact" in methods
    assert "minhash" in methods
    assert "simhash" in methods
