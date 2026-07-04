from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import hashlib
from typing import Hashable, Sequence


DocumentId = Hashable
Signature = Sequence[int]
BucketKey = tuple[int, int]


def split_signature_into_bands(
    signature: Signature,
    num_bands: int = 32,
) -> list[tuple[int, ...]]:
    """
    Split a MinHash signature into equal-sized bands.

    Example:
        signature length = 128
        num_bands = 32
        rows per band = 4
    """

    if num_bands <= 0:
        raise ValueError("num_bands must be a positive integer.")

    if len(signature) == 0:
        raise ValueError("signature must not be empty.")

    if len(signature) % num_bands != 0:
        raise ValueError(
            "Signature length must be divisible by num_bands. "
            f"Got signature length={len(signature)} and num_bands={num_bands}."
        )

    rows_per_band = len(signature) // num_bands

    return [
        tuple(signature[i : i + rows_per_band])
        for i in range(0, len(signature), rows_per_band)
    ]


def stable_band_hash(band: tuple[int, ...]) -> int:
    """
    Create a deterministic hash for one band.

    We avoid Python's built-in hash() because it is randomized between runs.
    """

    text = "|".join(map(str, band))
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def build_lsh_index(
    signatures: dict[DocumentId, Signature],
    num_bands: int = 32,
) -> dict[BucketKey, list[DocumentId]]:
    """
    Build an LSH bucket index from MinHash signatures.

    Key:
        (band_index, bucket_hash)

    Value:
        list of document ids that fell into that bucket
    """

    buckets: dict[BucketKey, list[DocumentId]] = defaultdict(list)

    for doc_id, signature in signatures.items():
        bands = split_signature_into_bands(signature, num_bands=num_bands)

        for band_index, band in enumerate(bands):
            bucket_hash = stable_band_hash(band)
            bucket_key = (band_index, bucket_hash)
            buckets[bucket_key].append(doc_id)

    return dict(buckets)


def generate_candidate_pairs(
    signatures: dict[DocumentId, Signature],
    num_bands: int = 32,
) -> set[tuple[DocumentId, DocumentId]]:
    """
    Generate candidate document pairs using LSH.

    Two documents become candidates if they share at least one bucket
    in at least one band.
    """

    buckets = build_lsh_index(signatures=signatures, num_bands=num_bands)
    candidates: set[tuple[DocumentId, DocumentId]] = set()

    for doc_ids in buckets.values():
        if len(doc_ids) < 2:
            continue

        for doc_a, doc_b in combinations(sorted(doc_ids, key=str), 2):
            candidates.add((doc_a, doc_b))

    return candidates


def count_all_pairs(num_documents: int) -> int:
    """
    Count the number of all possible pairwise comparisons.

    n choose 2 = n(n - 1) / 2
    """

    if num_documents < 0:
        raise ValueError("num_documents must be non-negative.")

    return num_documents * (num_documents - 1) // 2


def lsh_reduction_stats(
    num_documents: int,
    num_candidate_pairs: int,
) -> dict[str, float | int]:
    """
    Compute how much LSH reduces pairwise comparisons.
    """

    all_pairs = count_all_pairs(num_documents)

    if all_pairs == 0:
        reduction_ratio = 0.0
    else:
        reduction_ratio = 1.0 - (num_candidate_pairs / all_pairs)

    return {
        "num_documents": num_documents,
        "all_pairs": all_pairs,
        "candidate_pairs": num_candidate_pairs,
        "reduction_ratio": reduction_ratio,
    }
