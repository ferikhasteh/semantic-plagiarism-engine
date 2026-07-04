from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable


@dataclass(frozen=True)
class TextDocument:
    doc_id: str
    path: str
    text: str


@dataclass(frozen=True)
class LabeledPair:
    pair_id: str
    text_a: str
    text_b: str
    label: int


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    path = Path(path)
    return path.read_text(encoding=encoding, errors="ignore")


def load_text_documents_from_folder(
    folder_path: str | Path,
    extensions: tuple[str, ...] = (".txt",),
    recursive: bool = False,
    encoding: str = "utf-8",
) -> list[TextDocument]:
    """
    Load text documents from a folder.

    Each file becomes one TextDocument.
    doc_id is the relative file path, so it stays unique inside the corpus.
    """

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    normalized_extensions = tuple(ext.lower() for ext in extensions)

    if recursive:
        candidate_paths = folder.rglob("*")
    else:
        candidate_paths = folder.glob("*")

    documents: list[TextDocument] = []

    for path in sorted(candidate_paths):
        if not path.is_file():
            continue

        if path.suffix.lower() not in normalized_extensions:
            continue

        relative_path = path.relative_to(folder).as_posix()
        text = read_text_file(path, encoding=encoding)

        documents.append(
            TextDocument(
                doc_id=relative_path,
                path=str(path),
                text=text,
            )
        )

    return documents


def parse_binary_label(value: object) -> int:
    """
    Convert common binary label formats to integer 0 or 1.
    """

    text = str(value).strip().lower()

    positive_values = {"1", "true", "yes", "duplicate", "similar"}
    negative_values = {"0", "false", "no", "non_duplicate", "not_duplicate", "different"}

    if text in positive_values:
        return 1

    if text in negative_values:
        return 0

    raise ValueError(f"Cannot parse binary label: {value}")


def load_labeled_pairs_csv(
    csv_path: str | Path,
    text_col_a: str,
    text_col_b: str,
    label_col: str,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> list[LabeledPair]:
    """
    Load a labeled pair dataset from a CSV file.

    Expected format:
        text_col_a, text_col_b, label_col

    Example for Quora:
        question1, question2, is_duplicate
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None.")

    pairs: list[LabeledPair] = []

    with path.open("r", newline="", encoding=encoding, errors="ignore") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")

        required_columns = {text_col_a, text_col_b, label_col}
        available_columns = set(reader.fieldnames)
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}. "
                f"Available columns: {sorted(available_columns)}"
            )

        for row_index, row in enumerate(reader):
            if limit is not None and len(pairs) >= limit:
                break

            text_a = row.get(text_col_a, "")
            text_b = row.get(text_col_b, "")
            label_value = row.get(label_col, "")

            pair = LabeledPair(
                pair_id=str(row_index),
                text_a=text_a or "",
                text_b=text_b or "",
                label=parse_binary_label(label_value),
            )

            pairs.append(pair)

    return pairs


def labeled_pairs_to_columns(
    pairs: Iterable[LabeledPair],
) -> tuple[list[str], list[str], list[int]]:
    """
    Convert LabeledPair objects to separate columns.

    This is useful for evaluation code:
        texts_a, texts_b, labels = labeled_pairs_to_columns(pairs)
    """

    texts_a: list[str] = []
    texts_b: list[str] = []
    labels: list[int] = []

    for pair in pairs:
        texts_a.append(pair.text_a)
        texts_b.append(pair.text_b)
        labels.append(pair.label)

    return texts_a, texts_b, labels
