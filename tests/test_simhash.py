from plagiarism_engine.simhash import (
    stable_token_hash,
    term_frequency,
    document_frequency,
    inverse_document_frequency,
    simhash,
    hamming_distance,
    simhash_similarity,
    compute_simhashes,
    format_simhash,
)


def test_stable_token_hash_is_deterministic():
    assert stable_token_hash("lion") == stable_token_hash("lion")


def test_term_frequency():
    tokens = ["lion", "ate", "lion"]
    tf = term_frequency(tokens)

    assert tf["lion"] == 2 / 3
    assert tf["ate"] == 1 / 3


def test_document_frequency():
    docs = [
        ["lion", "ate", "zebra"],
        ["lion", "drinks", "water"],
        ["cat", "drinks", "milk"],
    ]

    df = document_frequency(docs)

    assert df["lion"] == 2
    assert df["drinks"] == 2
    assert df["zebra"] == 1


def test_inverse_document_frequency_common_word_is_lower():
    docs = [
        ["common", "rare_a"],
        ["common", "rare_b"],
        ["common", "rare_c"],
    ]

    idf = inverse_document_frequency(docs)

    assert idf["common"] < idf["rare_a"]


def test_identical_documents_have_same_simhash():
    tokens = ["lion", "ate", "zebra"]

    hash_a = simhash(tokens)
    hash_b = simhash(tokens)

    assert hash_a == hash_b
    assert hamming_distance(hash_a, hash_b) == 0
    assert simhash_similarity(hash_a, hash_b) == 1.0


def test_hamming_distance():
    assert hamming_distance(0b1010, 0b1001) == 2


def test_simhash_similarity_range():
    hash_a = simhash(["lion", "ate", "zebra"])
    hash_b = simhash(["cat", "drinks", "milk"])

    score = simhash_similarity(hash_a, hash_b)

    assert 0.0 <= score <= 1.0


def test_compute_simhashes():
    docs = {
        "doc_1": ["lion", "ate", "zebra"],
        "doc_2": ["lion", "ate", "zebra"],
        "doc_3": ["cat", "drinks", "milk"],
    }

    hashes = compute_simhashes(docs)

    assert len(hashes) == 3
    assert hashes["doc_1"] == hashes["doc_2"]


def test_format_simhash():
    value = 255
    formatted = format_simhash(value, hash_bits=64)

    assert len(formatted) == 16
    assert formatted.endswith("ff")
