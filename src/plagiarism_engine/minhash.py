from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from typing import Iterable, Hashable


LARGE_PRIME = 18446744073709551557
MAX_HASH = LARGE_PRIME


def stable_hash(value: Hashable) -> int:
    """
    Create a deterministic 64-bit hash for any hashable value.

    Python's built-in hash() is randomized between runs, so we use hashlib
    to make the MinHash signatures reproducible.
    """

    if isinstance(value, tuple):
        text = "\x1f".join(map(str, value))
    else:
        text = str(value)

    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


@lru_cache(maxsize=None)
def generate_hash_coefficients(
    num_perm: int = 128,
    seed: int = 42,
) -> tuple[tuple[int, int], ...]:
    """
    Generate parameters for universal hash functions:

        h_i(x) = (a_i * x + b_i) mod LARGE_PRIME

    The seed makes the output reproducible.
    """

    if num_perm <= 0:
        raise ValueError("num_perm must be a positive integer.")

    rng = random.Random(seed)
    coefficients: list[tuple[int, int]] = []

    used_a: set[int] = set()

    while len(coefficients) < num_perm:
        a = rng.randint(1, LARGE_PRIME - 1)
        b = rng.randint(0, LARGE_PRIME - 1)

        if a in used_a:
            continue

        used_a.add(a)
        coefficients.append((a, b))

    return tuple(coefficients)


def minhash_signature(
    shingles: Iterable[Hashable],
    num_perm: int = 128,
    seed: int = 42,
) -> tuple[int, ...]:
    """
    Compute a MinHash signature for a set of shingles.

    Input:
        shingles: a collection of document shingles
        num_perm: signature length, usually 128 or 256
        seed: random seed for reproducible hash functions

    Output:
        A tuple of integers representing the MinHash signature.
    """

    shingle_list = list(shingles)

    if num_perm <= 0:
        raise ValueError("num_perm must be a positive integer.")

    if not shingle_list:
        return tuple([MAX_HASH] * num_perm)

    coefficients = generate_hash_coefficients(num_perm=num_perm, seed=seed)
    signature = [MAX_HASH] * num_perm

    hashed_shingles = [stable_hash(shingle) for shingle in shingle_list]

    for shingle_hash in hashed_shingles:
        for i, (a, b) in enumerate(coefficients):
            candidate_hash = (a * shingle_hash + b) % LARGE_PRIME

            if candidate_hash < signature[i]:
                signature[i] = candidate_hash

    return tuple(signature)


def minhash_similarity(
    signature_a: tuple[int, ...] | list[int],
    signature_b: tuple[int, ...] | list[int],
) -> float:
    """
    Estimate Jaccard similarity using two MinHash signatures.

    The estimate is the fraction of equal positions in the two signatures.
    """

    if len(signature_a) != len(signature_b):
        raise ValueError("Signatures must have the same length.")

    if len(signature_a) == 0:
        raise ValueError("Signatures must not be empty.")

    matches = sum(1 for a, b in zip(signature_a, signature_b) if a == b)
    return matches / len(signature_a)


def signature_matrix(
    documents_shingles: list[Iterable[Hashable]],
    num_perm: int = 128,
    seed: int = 42,
) -> list[tuple[int, ...]]:
    """
    Compute MinHash signatures for multiple documents.
    """

    return [
        minhash_signature(
            shingles=shingles,
            num_perm=num_perm,
            seed=seed,
        )
        for shingles in documents_shingles
    ]
