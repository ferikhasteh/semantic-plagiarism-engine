from plagiarism_engine.bonus import (
    build_hybrid_simhash_idf,
    char_ngrams,
    find_adaptive_lsh_params,
    hybrid_simhash,
    make_hybrid_tokens,
    persian_lemmatize,
)
from plagiarism_engine.lsh import split_signature_into_bands
from plagiarism_engine.simhash import hamming_distance, simhash, simhash_similarity


def test_find_adaptive_lsh_params_divides_num_perm_evenly():
    bands, rows_per_band = find_adaptive_lsh_params(num_perm=128, target_threshold=0.15)

    assert 128 % bands == 0
    assert bands * rows_per_band == 128


def test_find_adaptive_lsh_params_is_usable_by_split_signature_into_bands():
    bands, _ = find_adaptive_lsh_params(num_perm=128, target_threshold=0.5)
    signature = tuple(range(128))

    # Should not raise: bands must be a valid divisor of the signature length.
    result = split_signature_into_bands(signature, num_bands=bands)
    assert len(result) == bands


def test_find_adaptive_lsh_params_approximates_target_threshold():
    # 128 = 64 * 2, and (1/64)**(1/2) == 0.125 exactly, so this target is
    # exactly reachable and pins down which divisor pair should be chosen.
    bands, rows_per_band = find_adaptive_lsh_params(num_perm=128, target_threshold=0.125)
    approx = (1 / bands) ** (1 / rows_per_band)

    assert bands == 64
    assert rows_per_band == 2
    assert abs(approx - 0.125) < 1e-9


def test_find_adaptive_lsh_params_rejects_invalid_threshold():
    try:
        find_adaptive_lsh_params(num_perm=128, target_threshold=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for out-of-range threshold")


def test_persian_lemmatize_strips_plural_suffix():
    result = persian_lemmatize(["دانشجویان"])
    assert result == ("دانشجوی",)


def test_persian_lemmatize_strips_continuous_verb_prefix():
    result = persian_lemmatize(["می‌روند"])
    assert result == ("روند",)


def test_persian_lemmatize_leaves_unmatched_tokens_unchanged():
    result = persian_lemmatize(["کتاب", "را"])
    assert result == ("کتاب", "را")


def test_persian_lemmatize_preserves_token_count():
    tokens = ["دانشجویان", "کتاب‌ها", "می‌خوانند"]
    result = persian_lemmatize(tokens)
    assert len(result) == len(tokens)


def test_char_ngrams_basic():
    assert char_ngrams("fox", 3) == ["fox"]
    assert char_ngrams("foxes", 3) == ["fox", "oxe", "xes"]


def test_char_ngrams_short_word_returns_whole_word():
    assert char_ngrams("a", 3) == ["a"]


def test_make_hybrid_tokens_includes_original_words():
    tokens = ["fox"]
    hybrid = make_hybrid_tokens(tokens, char_ngram_sizes=(3,))

    assert "fox" in hybrid
    assert any(str(t).startswith("c3:") for t in hybrid)


def test_hybrid_simhash_identical_inputs_match():
    tokens = ["quick", "brown", "fox"]

    hash_a = hybrid_simhash(tokens)
    hash_b = hybrid_simhash(tokens)

    assert hash_a == hash_b
    assert simhash_similarity(hash_a, hash_b) == 1.0


def test_hybrid_tokens_share_char_ngrams_despite_word_level_typo():
    # The whole-word tokens are entirely different strings, but a couple of
    # letters swapped still leaves most 3-grams intact -- this is the actual
    # mechanism that makes the hybrid SimHash more typo-resilient.
    normal_hybrid = set(make_hybrid_tokens(["artificial"], char_ngram_sizes=(3,)))
    typo_hybrid = set(make_hybrid_tokens(["artificail"], char_ngram_sizes=(3,)))

    assert "artificial" not in typo_hybrid  # whole-word tokens don't match
    shared_ngrams = {t for t in normal_hybrid & typo_hybrid if str(t).startswith("c3:")}
    assert len(shared_ngrams) >= 3


def test_hybrid_simhash_more_typo_resilient_than_word_level_simhash():
    # A handful of shared, unaffected words keeps the SimHash sign-vote more
    # stable, so with enough overlapping context the hybrid fingerprint is
    # measurably closer for a typo'd document than the plain word-level one.
    normal = ["artificial", "intelligence", "course", "about", "machine", "learning", "systems"]
    typo = ["artificail", "intellegence", "coruse", "about", "machine", "learning", "systems"]

    idf = build_hybrid_simhash_idf([normal, typo])

    word_level_a = simhash(normal)
    word_level_b = simhash(typo)
    word_level_distance = hamming_distance(word_level_a, word_level_b)

    hybrid_a = hybrid_simhash(normal, idf=idf)
    hybrid_b = hybrid_simhash(typo, idf=idf)
    hybrid_distance = hamming_distance(hybrid_a, hybrid_b)

    assert hybrid_distance <= word_level_distance


def test_build_hybrid_simhash_idf_returns_weights_for_char_ngrams():
    docs = [["fox", "jumps"], ["fox", "sleeps"]]
    idf = build_hybrid_simhash_idf(docs, char_ngram_sizes=(3,))

    assert any(str(key).startswith("c3:") for key in idf)
