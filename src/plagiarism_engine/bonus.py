"""Optional bonus extensions on top of the required MinHash+LSH and SimHash
pipelines.

These are additive, self-contained utilities -- nothing in `cli.py`, `lsh.py`,
`minhash.py` or `simhash.py` depends on this module, and none of the three
CLI commands change behaviour because of it. Everything here is implemented
from scratch using only the standard library and the primitives already
defined in `simhash.py` (`simhash`, `build_simhash_idf`), consistent with the
project rule that MinHash/LSH/SimHash themselves may not come from a
ready-made library.

Three ideas are implemented:

1. `find_adaptive_lsh_params` -- instead of hand-picking `num_bands`, solve
   for the (bands, rows_per_band) split of a MinHash signature that best
   approximates a *target* Jaccard threshold on the LSH S-curve.
2. `persian_lemmatize` -- a tiny, transparent rule-based normalizer that
   strips common Persian plural suffixes and continuous-tense verb prefixes,
   so that inflected forms of the same word (e.g. "دانشجویان" / "دانشجو")
   collide in shingles/tokens instead of being treated as unrelated.
3. `hybrid_simhash` / `build_hybrid_simhash_idf` -- a SimHash variant that
   hashes word tokens *and* character n-grams together, so that a handful of
   typos or minor spelling changes still leave most of the fingerprint
   intact (character n-grams overlap even when whole words don't).
"""

from __future__ import annotations

from typing import Hashable, Iterable

from .simhash import DEFAULT_HASH_BITS, build_simhash_idf, simhash


# ---------------------------------------------------------------------------
# 1. Adaptive LSH band/row search
# ---------------------------------------------------------------------------


def find_adaptive_lsh_params(
    num_perm: int,
    target_threshold: float,
) -> tuple[int, int]:
    """Find (bands, rows_per_band) that split a MinHash signature of length
    `num_perm` and whose LSH S-curve threshold

        s* = (1 / bands) ** (1 / rows_per_band)

    is as close as possible to `target_threshold`.

    Only exact divisor pairs of `num_perm` are considered, since
    `split_signature_into_bands` (lsh.py) requires the signature length to
    be evenly divisible by the number of bands.
    """

    if num_perm <= 0:
        raise ValueError("num_perm must be a positive integer.")
    if not 0.0 < target_threshold < 1.0:
        raise ValueError("target_threshold must be between 0 and 1, exclusive.")

    best_bands, best_rows = 1, num_perm
    best_diff = float("inf")

    for bands in range(1, num_perm + 1):
        if num_perm % bands != 0:
            continue

        rows_per_band = num_perm // bands
        approx_threshold = (1.0 / bands) ** (1.0 / rows_per_band)
        diff = abs(approx_threshold - target_threshold)

        if diff < best_diff:
            best_diff = diff
            best_bands, best_rows = bands, rows_per_band

    return best_bands, best_rows


# ---------------------------------------------------------------------------
# 2. Light rule-based Persian lemmatizer
# ---------------------------------------------------------------------------

_PERSIAN_PLURAL_SUFFIXES = ("های", "ها", "ان")
_PERSIAN_CONTINUOUS_PREFIXES = ("نمی‌", "می‌", "نمی", "می")
_ZWNJ = "‌"


def _strip_verb_prefix(token: str) -> str:
    for prefix in _PERSIAN_CONTINUOUS_PREFIXES:
        if token.startswith(prefix) and len(token) > len(prefix) + 1:
            return token[len(prefix):]
    return token


def _strip_plural_suffix(token: str) -> str:
    for suffix in _PERSIAN_PLURAL_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def persian_lemmatize(tokens: Iterable[str]) -> tuple[str, ...]:
    """Apply a very light, transparent normalization to Persian tokens:

    - strip the continuous-tense verb prefixes "می‌"/"نمی‌"
    - strip the plural suffixes "ها"/"های"/"ان"
    - drop any leftover leading/trailing ZWNJ (half-space) characters

    This is *not* a full morphological analyzer. It is a cheap heuristic
    intended to merge obviously related inflected forms (plurals, simple
    present/continuous verb forms) before shingling, which helps catch
    paraphrased Persian text that only differs by such inflections. Words
    that do not match any rule are returned unchanged.
    """

    lemmatized: list[str] = []

    for token in tokens:
        lemma = _strip_verb_prefix(token)
        lemma = _strip_plural_suffix(lemma)
        lemma = lemma.strip(_ZWNJ)
        lemmatized.append(lemma if lemma else token)

    return tuple(lemmatized)


# ---------------------------------------------------------------------------
# 3. Hybrid word + character n-gram SimHash
# ---------------------------------------------------------------------------

_CHAR_NGRAM_PREFIX = "c{n}:"


def char_ngrams(word: str, n: int) -> list[str]:
    """Character n-grams of a single word. Short words (shorter than `n`)
    are returned as a single-element list containing the whole word, so
    they still contribute one feature instead of none."""

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if len(word) <= n:
        return [word]

    return [word[i : i + n] for i in range(len(word) - n + 1)]


def make_hybrid_tokens(
    tokens: Iterable[Hashable],
    char_ngram_sizes: tuple[int, ...] = (3,),
) -> list[Hashable]:
    """Combine word tokens with character n-gram tokens.

    Character n-grams are tagged with a small prefix (e.g. "c3:") so they
    can never collide with an actual word token, and are generated per
    n-gram size so a caller can mix e.g. 3- and 4-grams.
    """

    hybrid: list[Hashable] = list(tokens)

    for token in tokens:
        word = str(token)
        for n in char_ngram_sizes:
            for gram in char_ngrams(word, n):
                hybrid.append(_CHAR_NGRAM_PREFIX.format(n=n) + gram)

    return hybrid


def build_hybrid_simhash_idf(
    documents_tokens: list[list[Hashable]],
    char_ngram_sizes: tuple[int, ...] = (3,),
) -> dict[Hashable, float]:
    """Corpus-level IDF values for the hybrid (word + char n-gram) SimHash,
    built over the same hybrid-token representation used by
    `hybrid_simhash`."""

    hybrid_docs = [
        make_hybrid_tokens(tokens, char_ngram_sizes=char_ngram_sizes)
        for tokens in documents_tokens
    ]
    return build_simhash_idf(hybrid_docs)


def hybrid_simhash(
    tokens: Iterable[Hashable],
    idf: dict[Hashable, float] | None = None,
    hash_bits: int = DEFAULT_HASH_BITS,
    char_ngram_sizes: tuple[int, ...] = (3,),
) -> int:
    """TF-IDF weighted SimHash computed over word tokens *and* character
    n-grams of those tokens, reusing the exact same `simhash` primitive as
    the required pipeline (simhash.py) -- only the input token stream is
    different. Because a handful of character n-grams survive a small typo
    (e.g. "artificial" -> "artificail" still shares most 3-grams), the
    resulting fingerprint is more typo-resilient than plain word-level
    SimHash.
    """

    hybrid_tokens = make_hybrid_tokens(tokens, char_ngram_sizes=char_ngram_sizes)
    return simhash(hybrid_tokens, idf=idf, hash_bits=hash_bits)
