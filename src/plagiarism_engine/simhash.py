from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Iterable, Hashable


DEFAULT_HASH_BITS = 64


def stable_token_hash(token: Hashable, hash_bits: int = DEFAULT_HASH_BITS) -> int:
    """
    Create a deterministic hash for a token.

    We use hashlib instead of Python's built-in hash() because Python hash()
    is randomized between different runs.
    """

    if hash_bits <= 0 or hash_bits > 64:
        raise ValueError("hash_bits must be between 1 and 64.")

    text = str(token)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)

    if hash_bits == 64:
        return value

    mask = (1 << hash_bits) - 1
    return value & mask


def term_frequency(tokens: Iterable[Hashable]) -> dict[Hashable, float]:
    """
    Compute normalized term frequency for one document.

    tf(token) = count(token) / total_tokens
    """

    token_list = list(tokens)

    if not token_list:
        return {}

    counts = Counter(token_list)
    total = len(token_list)

    return {token: count / total for token, count in counts.items()}


def document_frequency(documents: Iterable[Iterable[Hashable]]) -> dict[Hashable, int]:
    """
    Count in how many documents each token appears.
    """

    df: Counter[Hashable] = Counter()

    for doc_tokens in documents:
        unique_tokens = set(doc_tokens)

        for token in unique_tokens:
            df[token] += 1

    return dict(df)


def inverse_document_frequency(
    documents: list[list[Hashable]],
    smooth: bool = True,
) -> dict[Hashable, float]:
    """
    Compute IDF for a corpus.

    With smoothing:
        idf(t) = log((1 + N) / (1 + df(t))) + 1

    Without smoothing:
        idf(t) = log(N / df(t))
    """

    if not documents:
        return {}

    n_documents = len(documents)
    df = document_frequency(documents)

    idf: dict[Hashable, float] = {}

    for token, freq in df.items():
        if smooth:
            idf[token] = math.log((1 + n_documents) / (1 + freq)) + 1
        else:
            idf[token] = math.log(n_documents / freq)

    return idf


def tf_idf_weights(
    tokens: Iterable[Hashable],
    idf: dict[Hashable, float] | None = None,
) -> dict[Hashable, float]:
    """
    Compute TF-IDF weights for one document.

    If idf is None, this function returns only normalized TF weights.
    """

    tf = term_frequency(tokens)

    if idf is None:
        return tf

    return {
        token: tf_value * idf.get(token, 0.0)
        for token, tf_value in tf.items()
    }


def simhash(
    tokens: Iterable[Hashable],
    idf: dict[Hashable, float] | None = None,
    hash_bits: int = DEFAULT_HASH_BITS,
) -> int:
    """
    Compute TF-IDF weighted SimHash.

    For each token:
        1. Compute token weight using TF-IDF.
        2. Hash the token into a 64-bit value.
        3. For each bit:
            - add weight if the bit is 1
            - subtract weight if the bit is 0
        4. The final bit is 1 if the accumulated value is non-negative,
           otherwise it is 0.
    """

    if hash_bits <= 0 or hash_bits > 64:
        raise ValueError("hash_bits must be between 1 and 64.")

    token_list = list(tokens)

    if not token_list:
        return 0

    weights = tf_idf_weights(token_list, idf=idf)
    vector = [0.0] * hash_bits

    for token, weight in weights.items():
        token_hash = stable_token_hash(token, hash_bits=hash_bits)

        for bit_index in range(hash_bits):
            bit_is_one = (token_hash >> bit_index) & 1

            if bit_is_one:
                vector[bit_index] += weight
            else:
                vector[bit_index] -= weight

    fingerprint = 0

    for bit_index, value in enumerate(vector):
        if value >= 0:
            fingerprint |= 1 << bit_index

    return fingerprint


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """
    Count the number of different bits between two integer hashes.
    """

    if hash_a < 0 or hash_b < 0:
        raise ValueError("hash values must be non-negative integers.")

    return (hash_a ^ hash_b).bit_count()


def simhash_similarity(
    hash_a: int,
    hash_b: int,
    hash_bits: int = DEFAULT_HASH_BITS,
) -> float:
    """
    Convert Hamming distance into a similarity score in [0, 1].

    similarity = 1 - hamming_distance / hash_bits
    """

    if hash_bits <= 0:
        raise ValueError("hash_bits must be a positive integer.")

    distance = hamming_distance(hash_a, hash_b)
    return 1.0 - (distance / hash_bits)


def build_simhash_idf(
    documents_tokens: list[list[Hashable]],
) -> dict[Hashable, float]:
    """
    Build corpus-level IDF values for weighted SimHash.
    """

    return inverse_document_frequency(documents_tokens, smooth=True)


def compute_simhashes(
    documents_tokens: dict[Hashable, list[Hashable]],
    hash_bits: int = DEFAULT_HASH_BITS,
) -> dict[Hashable, int]:
    """
    Compute TF-IDF weighted SimHash for multiple documents.

    Input:
        {
            doc_id: tokens
        }

    Output:
        {
            doc_id: simhash_integer
        }
    """

    corpus_tokens = list(documents_tokens.values())
    idf = build_simhash_idf(corpus_tokens)

    return {
        doc_id: simhash(
            tokens=tokens,
            idf=idf,
            hash_bits=hash_bits,
        )
        for doc_id, tokens in documents_tokens.items()
    }


def format_simhash(value: int, hash_bits: int = DEFAULT_HASH_BITS) -> str:
    """
    Format SimHash integer as a fixed-length hexadecimal string.
    """

    width = math.ceil(hash_bits / 4)
    return f"{value:0{width}x}"
