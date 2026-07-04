import pytest

from plagiarism_engine.lsh import (
    split_signature_into_bands,
    stable_band_hash,
    build_lsh_index,
    generate_candidate_pairs,
    count_all_pairs,
    lsh_reduction_stats,
)


def test_split_signature_into_bands():
    signature = tuple(range(8))
    bands = split_signature_into_bands(signature, num_bands=4)

    assert bands == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]


def test_split_signature_requires_divisible_length():
    signature = tuple(range(10))

    with pytest.raises(ValueError):
        split_signature_into_bands(signature, num_bands=4)


def test_stable_band_hash_is_deterministic():
    band = (10, 20, 30, 40)

    assert stable_band_hash(band) == stable_band_hash(band)


def test_identical_signatures_become_candidates():
    signatures = {
        "doc_1": tuple(range(128)),
        "doc_2": tuple(range(128)),
        "doc_3": tuple(range(128, 256)),
    }

    candidates = generate_candidate_pairs(signatures, num_bands=32)

    assert ("doc_1", "doc_2") in candidates


def test_build_lsh_index_contains_buckets():
    signatures = {
        "doc_1": tuple(range(16)),
        "doc_2": tuple(range(16)),
    }

    buckets = build_lsh_index(signatures, num_bands=4)

    assert len(buckets) == 4
    assert all(len(doc_ids) == 2 for doc_ids in buckets.values())


def test_count_all_pairs():
    assert count_all_pairs(0) == 0
    assert count_all_pairs(1) == 0
    assert count_all_pairs(5) == 10


def test_lsh_reduction_stats():
    stats = lsh_reduction_stats(num_documents=5, num_candidate_pairs=3)

    assert stats["num_documents"] == 5
    assert stats["all_pairs"] == 10
    assert stats["candidate_pairs"] == 3
    assert stats["reduction_ratio"] == 0.7
