from plagiarism_engine.preprocessing import (
    normalize_text,
    tokenize_words,
    make_word_shingles,
    jaccard_similarity,
    preprocess_document,
)


def test_normalize_text_removes_punctuation_and_lowercases():
    text = "Hello!!! This is a TEST..."
    result = normalize_text(text)
    assert result == "hello this is a test"


def test_tokenize_words_removes_stopwords():
    text = "this is a good book"
    tokens = tokenize_words(text)
    assert tokens == ["good", "book"]


def test_make_word_shingles():
    tokens = ["lion", "ate", "zebra", "and", "goat"]
    shingles = make_word_shingles(tokens, k=2)

    assert ("lion", "ate") in shingles
    assert ("ate", "zebra") in shingles
    assert len(shingles) == 4


def test_jaccard_similarity():
    a = {("lion", "ate"), ("ate", "zebra")}
    b = {("lion", "ate"), ("ate", "zebra"), ("zebra", "and")}

    score = jaccard_similarity(a, b)

    assert score == 2 / 3


def test_preprocess_document_valid():
    text = "Lion ate zebra and goat."
    doc = preprocess_document(text, shingle_size=2)

    assert doc.is_valid
    assert len(doc.tokens) > 0
    assert len(doc.shingles) > 0


def test_preprocess_document_empty():
    doc = preprocess_document("", shingle_size=3)

    assert not doc.is_valid
    assert doc.reason == "empty_or_invalid_text"


def test_preprocess_document_too_short():
    doc = preprocess_document("hello", shingle_size=3)

    assert not doc.is_valid
    assert doc.reason == "too_few_tokens"
