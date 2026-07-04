from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import re
import unicodedata
from typing import Iterable


ENGLISH_STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "from", "with", "by", "as", "at",
    "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "them", "we", "you", "i",
    "my", "your", "his", "her", "their", "our",
    "not", "no", "yes",
    "do", "does", "did", "doing",
    "have", "has", "had",
    "can", "could", "should", "would", "will", "may", "might",
    "very", "also", "just", "than", "then", "there", "here",
}


PERSIAN_STOPWORDS = {
    "از", "به", "با", "در", "را", "و", "یا", "اما", "اگر", "که",
    "این", "آن", "برای", "تا", "بر", "هم", "نیز", "چون", "پس",
    "است", "هست", "بود", "شد", "شود", "می", "های", "ها",
    "یک", "هر", "همه", "خود", "ما", "من", "تو", "او", "آنها",
    "شما", "ایشان", "کرد", "کرده", "میشود", "میکند",
}


DEFAULT_STOPWORDS = ENGLISH_STOPWORDS | PERSIAN_STOPWORDS


@dataclass(frozen=True)
class PreprocessedDocument:
    normalized_text: str
    tokens: list[str]
    shingles: set[tuple[str, ...]]
    is_valid: bool
    reason: str = ""


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    path = Path(path)
    return path.read_text(encoding=encoding, errors="ignore")


def normalize_persian_arabic_chars(text: str) -> str:
    replacements = {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ـ": "",
        "\u200c": " ",
        "\u200f": " ",
        "\u200e": " ",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    return text


def remove_urls_emails(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    return text


def strip_html_tags(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def keep_letters_numbers_spaces(text: str) -> str:
    cleaned_chars: list[str] = []

    for ch in text:
        category = unicodedata.category(ch)

        if ch.isspace():
            cleaned_chars.append(" ")
        elif category.startswith("L") or category.startswith("N"):
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")

    return "".join(cleaned_chars)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, lowercase: bool = True) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_html_tags(text)
    text = remove_urls_emails(text)
    text = normalize_persian_arabic_chars(text)

    if lowercase:
        text = text.lower()

    text = keep_letters_numbers_spaces(text)
    text = collapse_whitespace(text)

    return text


def tokenize_words(
    text: str,
    stopwords: set[str] | None = None,
    remove_stopwords: bool = True,
    min_token_length: int = 1,
) -> list[str]:
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    tokens = text.split()
    filtered_tokens: list[str] = []

    for token in tokens:
        if len(token) < min_token_length:
            continue

        if remove_stopwords and token in stopwords:
            continue

        filtered_tokens.append(token)

    return filtered_tokens


def make_word_shingles(tokens: list[str], k: int = 3) -> set[tuple[str, ...]]:
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if len(tokens) < k:
        return set()

    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0

    union = set_a | set_b

    if not union:
        return 0.0

    intersection = set_a & set_b
    return len(intersection) / len(union)


def preprocess_document(
    text: str,
    shingle_size: int = 3,
    stopwords: set[str] | None = None,
    remove_stopwords: bool = True,
    min_token_length: int = 1,
    min_tokens: int | None = None,
) -> PreprocessedDocument:
    if min_tokens is None:
        min_tokens = shingle_size

    normalized_text = normalize_text(text)

    if not normalized_text:
        return PreprocessedDocument(
            normalized_text="",
            tokens=[],
            shingles=set(),
            is_valid=False,
            reason="empty_or_invalid_text",
        )

    tokens = tokenize_words(
        normalized_text,
        stopwords=stopwords,
        remove_stopwords=remove_stopwords,
        min_token_length=min_token_length,
    )

    if len(tokens) < min_tokens:
        return PreprocessedDocument(
            normalized_text=normalized_text,
            tokens=tokens,
            shingles=set(),
            is_valid=False,
            reason="too_few_tokens",
        )

    shingles = make_word_shingles(tokens, k=shingle_size)

    if not shingles:
        return PreprocessedDocument(
            normalized_text=normalized_text,
            tokens=tokens,
            shingles=set(),
            is_valid=False,
            reason="too_few_shingles",
        )

    return PreprocessedDocument(
        normalized_text=normalized_text,
        tokens=tokens,
        shingles=shingles,
        is_valid=True,
    )


def preprocess_many(
    documents: Iterable[str],
    shingle_size: int = 3,
    remove_stopwords: bool = True,
) -> list[PreprocessedDocument]:
    return [
        preprocess_document(
            doc,
            shingle_size=shingle_size,
            remove_stopwords=remove_stopwords,
        )
        for doc in documents
    ]
