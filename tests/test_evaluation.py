from pathlib import Path

import pytest

from plagiarism_engine.evaluation import (
    classification_metrics,
    confusion_matrix_counts,
    confusion_matrix_to_dict,
    evaluate_binary_scores,
    evaluate_candidate_pairs,
    measure_runtime,
    save_metrics_csv,
    threshold_scores,
)


def test_confusion_matrix_counts():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]

    cm = confusion_matrix_counts(y_true, y_pred)

    assert cm.true_positive == 1
    assert cm.true_negative == 1
    assert cm.false_positive == 1
    assert cm.false_negative == 1


def test_classification_metrics():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_threshold_scores():
    scores = [0.9, 0.4, 0.7, 0.1]

    predictions = threshold_scores(scores, threshold=0.5)

    assert predictions == [1, 0, 1, 0]


def test_evaluate_binary_scores():
    y_true = [1, 0, 1, 0]
    scores = [0.9, 0.4, 0.7, 0.1]

    metrics = evaluate_binary_scores(y_true, scores, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["threshold"] == 0.5


def test_evaluate_candidate_pairs():
    predicted_pairs = {
        ("doc_1", "doc_2"),
        ("doc_1", "doc_3"),
    }

    true_pairs = {
        ("doc_2", "doc_1"),
        ("doc_3", "doc_4"),
    }

    metrics = evaluate_candidate_pairs(predicted_pairs, true_pairs)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_measure_runtime():
    result, elapsed = measure_runtime(lambda x: x + 1, 4)

    assert result == 5
    assert elapsed >= 0.0


def test_save_metrics_csv(tmp_path: Path):
    output_path = tmp_path / "metrics.csv"

    rows = [
        {
            "method": "minhash_lsh",
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
        },
        {
            "method": "simhash",
            "precision": 0.6,
            "recall": 0.9,
            "f1": 0.72,
        },
    ]

    save_metrics_csv(rows, output_path)

    content = output_path.read_text(encoding="utf-8")

    assert "method" in content
    assert "minhash_lsh" in content
    assert "simhash" in content


def test_confusion_matrix_to_dict():
    cm = confusion_matrix_counts([1, 0], [1, 1])
    result = confusion_matrix_to_dict(cm)

    assert result == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 0,
    }


def test_invalid_labels_raise_error():
    with pytest.raises(ValueError):
        classification_metrics([1, 2], [1, 0])
