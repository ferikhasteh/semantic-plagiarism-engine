from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import time
from typing import Any, Callable, Hashable, Iterable, Sequence


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate_binary_labels(labels: Sequence[int]) -> None:
    invalid_labels = set(labels) - {0, 1}

    if invalid_labels:
        raise ValueError(f"Labels must be binary 0/1. Invalid labels: {invalid_labels}")


def confusion_matrix_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ConfusionMatrix:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    validate_binary_labels(y_true)
    validate_binary_labels(y_pred)

    tp = fp = tn = fn = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1

    return ConfusionMatrix(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float | int]:
    cm = confusion_matrix_counts(y_true, y_pred)

    tp = cm.true_positive
    fp = cm.false_positive
    tn = cm.true_negative
    fn = cm.false_negative

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_scores(
    scores: Sequence[float],
    threshold: float,
) -> list[int]:
    return [1 if score >= threshold else 0 for score in scores]


def evaluate_binary_scores(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, float | int]:
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length.")

    y_pred = threshold_scores(scores, threshold=threshold)
    metrics = classification_metrics(y_true, y_pred)
    metrics["threshold"] = threshold

    return metrics


def normalize_pair(pair: tuple[Hashable, Hashable]) -> tuple[Hashable, Hashable]:
    doc_a, doc_b = pair

    if str(doc_a) <= str(doc_b):
        return doc_a, doc_b

    return doc_b, doc_a


def evaluate_candidate_pairs(
    predicted_pairs: Iterable[tuple[Hashable, Hashable]],
    true_pairs: Iterable[tuple[Hashable, Hashable]],
) -> dict[str, float | int]:
    predicted = {normalize_pair(pair) for pair in predicted_pairs}
    truth = {normalize_pair(pair) for pair in true_pairs}

    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def measure_runtime(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed_seconds = time.perf_counter() - start_time

    return result, elapsed_seconds


def save_metrics_csv(
    metrics_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not metrics_rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = sorted(
        {key for row in metrics_rows for key in row.keys()}
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def metrics_row(
    method: str,
    metrics: dict[str, Any],
    runtime_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method}
    row.update(metrics)

    if runtime_seconds is not None:
        row["runtime_seconds"] = runtime_seconds

    if extra is not None:
        row.update(extra)

    return row


def confusion_matrix_to_dict(confusion_matrix: ConfusionMatrix) -> dict[str, int]:
    return asdict(confusion_matrix)
