from pathlib import Path

import pytest

from plagiarism_engine.dataset import (
    load_text_documents_from_folder,
    load_labeled_pairs_csv,
    labeled_pairs_to_columns,
    parse_binary_label,
)


def test_load_text_documents_from_folder(tmp_path: Path):
    doc_1 = tmp_path / "doc_1.txt"
    doc_2 = tmp_path / "doc_2.txt"
    ignored = tmp_path / "notes.md"

    doc_1.write_text("Lion ate zebra.", encoding="utf-8")
    doc_2.write_text("Cat drinks milk.", encoding="utf-8")
    ignored.write_text("This should be ignored.", encoding="utf-8")

    documents = load_text_documents_from_folder(tmp_path)

    assert len(documents) == 2
    assert documents[0].doc_id == "doc_1.txt"
    assert documents[0].text == "Lion ate zebra."
    assert documents[1].doc_id == "doc_2.txt"


def test_load_text_documents_recursive(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()

    (nested / "doc.txt").write_text("Nested document.", encoding="utf-8")

    documents = load_text_documents_from_folder(tmp_path, recursive=True)

    assert len(documents) == 1
    assert documents[0].doc_id == "nested/doc.txt"


def test_load_text_documents_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        load_text_documents_from_folder("missing_folder")


def test_parse_binary_label():
    assert parse_binary_label("1") == 1
    assert parse_binary_label("0") == 0
    assert parse_binary_label("true") == 1
    assert parse_binary_label("false") == 0
    assert parse_binary_label("duplicate") == 1
    assert parse_binary_label("different") == 0


def test_parse_binary_label_invalid():
    with pytest.raises(ValueError):
        parse_binary_label("maybe")


def test_load_labeled_pairs_csv(tmp_path: Path):
    csv_path = tmp_path / "pairs.csv"

    csv_path.write_text(
        "question1,question2,is_duplicate\n"
        "How are you?,How do you do?,1\n"
        "What is Python?,What is Java?,0\n",
        encoding="utf-8",
    )

    pairs = load_labeled_pairs_csv(
        csv_path,
        text_col_a="question1",
        text_col_b="question2",
        label_col="is_duplicate",
    )

    assert len(pairs) == 2
    assert pairs[0].text_a == "How are you?"
    assert pairs[0].text_b == "How do you do?"
    assert pairs[0].label == 1
    assert pairs[1].label == 0


def test_load_labeled_pairs_csv_with_limit(tmp_path: Path):
    csv_path = tmp_path / "pairs.csv"

    csv_path.write_text(
        "a,b,label\n"
        "text 1,text 2,1\n"
        "text 3,text 4,0\n",
        encoding="utf-8",
    )

    pairs = load_labeled_pairs_csv(
        csv_path,
        text_col_a="a",
        text_col_b="b",
        label_col="label",
        limit=1,
    )

    assert len(pairs) == 1


def test_load_labeled_pairs_missing_column(tmp_path: Path):
    csv_path = tmp_path / "pairs.csv"

    csv_path.write_text(
        "a,b\n"
        "text 1,text 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_labeled_pairs_csv(
            csv_path,
            text_col_a="a",
            text_col_b="b",
            label_col="label",
        )


def test_labeled_pairs_to_columns(tmp_path: Path):
    csv_path = tmp_path / "pairs.csv"

    csv_path.write_text(
        "a,b,label\n"
        "text a,text b,1\n"
        "text c,text d,0\n",
        encoding="utf-8",
    )

    pairs = load_labeled_pairs_csv(
        csv_path,
        text_col_a="a",
        text_col_b="b",
        label_col="label",
    )

    texts_a, texts_b, labels = labeled_pairs_to_columns(pairs)

    assert texts_a == ["text a", "text c"]
    assert texts_b == ["text b", "text d"]
    assert labels == [1, 0]
