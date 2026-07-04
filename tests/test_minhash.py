from plagiarism_engine.minhash import (
    stable_hash,
    minhash_signature,
    minhash_similarity,
    signature_matrix,
)


def test_stable_hash_is_deterministic():
    value = ("lion", "ate", "zebra")

    assert stable_hash(value) == stable_hash(value)


def test_identical_sets_have_similarity_one():
    shingles = {
        ("lion", "ate"),
        ("ate", "zebra"),
        ("zebra", "and"),
    }

    sig_a = minhash_signature(shingles, num_perm=128, seed=42)
    sig_b = minhash_signature(shingles, num_perm=128, seed=42)

    assert minhash_similarity(sig_a, sig_b) == 1.0


def test_disjoint_sets_have_low_similarity():
    shingles_a = {
        ("lion", "ate"),
        ("ate", "zebra"),
    }

    shingles_b = {
        ("cat", "drinks"),
        ("drinks", "milk"),
    }

    sig_a = minhash_signature(shingles_a, num_perm=128, seed=42)
    sig_b = minhash_signature(shingles_b, num_perm=128, seed=42)

    assert minhash_similarity(sig_a, sig_b) < 0.2


def test_minhash_approximates_jaccard():
    set_a = {(str(i),) for i in range(100)}
    set_b = {(str(i),) for i in range(50, 150)}

    exact_jaccard = 50 / 150

    sig_a = minhash_signature(set_a, num_perm=256, seed=42)
    sig_b = minhash_signature(set_b, num_perm=256, seed=42)

    estimated = minhash_similarity(sig_a, sig_b)

    assert abs(estimated - exact_jaccard) < 0.15


def test_signature_matrix():
    docs = [
        {("a", "b"), ("b", "c")},
        {("a", "b"), ("c", "d")},
    ]

    signatures = signature_matrix(docs, num_perm=64, seed=42)

    assert len(signatures) == 2
    assert len(signatures[0]) == 64
    assert len(signatures[1]) == 64
