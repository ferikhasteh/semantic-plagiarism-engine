from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from plagiarism_engine.bonus import (
    build_hybrid_simhash_idf,
    find_adaptive_lsh_params,
    hybrid_simhash,
    persian_lemmatize,
)
from plagiarism_engine.dataset import (
    load_labeled_pairs_csv,
    load_text_documents_from_folder,
)
from plagiarism_engine.evaluation import (
    evaluate_binary_scores,
    measure_runtime,
    metrics_row,
    save_metrics_csv,
)
from plagiarism_engine.lsh import (
    generate_candidate_pairs,
    lsh_reduction_stats,
)
from plagiarism_engine.minhash import (
    minhash_signature,
    minhash_similarity,
)
from plagiarism_engine.preprocessing import (
    jaccard_similarity,
    preprocess_document,
    read_text_file,
)
from plagiarism_engine.simhash import (
    build_simhash_idf,
    hamming_distance,
    simhash,
    simhash_similarity,
    format_simhash,
)


def write_json(data: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def document_similarity_summary(
    text_a: str,
    text_b: str,
    shingle_size: int = 3,
    num_perm: int = 128,
    seed: int = 42,
    hash_bits: int = 64,
) -> dict[str, Any]:
    doc_a = preprocess_document(text_a, shingle_size=shingle_size)
    doc_b = preprocess_document(text_b, shingle_size=shingle_size)

    both_valid = doc_a.is_valid and doc_b.is_valid

    if both_valid:
        exact_jaccard = jaccard_similarity(doc_a.shingles, doc_b.shingles)

        sig_a = minhash_signature(
            doc_a.shingles,
            num_perm=num_perm,
            seed=seed,
        )
        sig_b = minhash_signature(
            doc_b.shingles,
            num_perm=num_perm,
            seed=seed,
        )
        minhash_score = minhash_similarity(sig_a, sig_b)
    else:
        exact_jaccard = 0.0
        minhash_score = 0.0

    if doc_a.tokens and doc_b.tokens:
        idf = build_simhash_idf([doc_a.tokens, doc_b.tokens])
        simhash_a = simhash(doc_a.tokens, idf=idf, hash_bits=hash_bits)
        simhash_b = simhash(doc_b.tokens, idf=idf, hash_bits=hash_bits)
        simhash_distance = hamming_distance(simhash_a, simhash_b)
        simhash_score = simhash_similarity(
            simhash_a,
            simhash_b,
            hash_bits=hash_bits,
        )
    else:
        simhash_a = 0
        simhash_b = 0
        simhash_distance = hash_bits
        simhash_score = 0.0

    return {
        "doc_a_valid": doc_a.is_valid,
        "doc_b_valid": doc_b.is_valid,
        "doc_a_reason": doc_a.reason,
        "doc_b_reason": doc_b.reason,
        "doc_a_num_tokens": len(doc_a.tokens),
        "doc_b_num_tokens": len(doc_b.tokens),
        "doc_a_num_shingles": len(doc_a.shingles),
        "doc_b_num_shingles": len(doc_b.shingles),
        "jaccard_similarity": exact_jaccard,
        "minhash_similarity": minhash_score,
        "simhash_a": format_simhash(simhash_a, hash_bits=hash_bits),
        "simhash_b": format_simhash(simhash_b, hash_bits=hash_bits),
        "simhash_hamming_distance": simhash_distance,
        "simhash_similarity": simhash_score,
    }


def command_compare(args: argparse.Namespace) -> int:
    text_a = read_text_file(args.file_a)
    text_b = read_text_file(args.file_b)

    result = document_similarity_summary(
        text_a=text_a,
        text_b=text_b,
        shingle_size=args.shingle_size,
        num_perm=args.num_perm,
        seed=args.seed,
        hash_bits=args.hash_bits,
    )

    result.update(
        {
            "file_a": str(args.file_a),
            "file_b": str(args.file_b),
            "shingle_size": args.shingle_size,
            "num_perm": args.num_perm,
            "hash_bits": args.hash_bits,
        }
    )

    write_json(result, args.output)
    print(f"Saved comparison result to {args.output}")

    return 0


def command_corpus(args: argparse.Namespace) -> int:
    documents = load_text_documents_from_folder(
        folder_path=args.data,
        extensions=(".txt",),
        recursive=args.recursive,
    )

    preprocessed = {
        doc.doc_id: preprocess_document(
            doc.text,
            shingle_size=args.shingle_size,
        )
        for doc in documents
    }

    valid_docs = {
        doc_id: doc
        for doc_id, doc in preprocessed.items()
        if doc.is_valid
    }

    signatures = {
        doc_id: minhash_signature(
            doc.shingles,
            num_perm=args.num_perm,
            seed=args.seed,
        )
        for doc_id, doc in valid_docs.items()
    }

    candidate_pairs, runtime_seconds = measure_runtime(
        generate_candidate_pairs,
        signatures,
        args.num_bands,
    )

    rows: list[dict[str, Any]] = []

    for doc_a, doc_b in sorted(candidate_pairs, key=lambda pair: (str(pair[0]), str(pair[1]))):
        shingles_a = valid_docs[doc_a].shingles
        shingles_b = valid_docs[doc_b].shingles

        jaccard_score = jaccard_similarity(shingles_a, shingles_b)
        minhash_score = minhash_similarity(signatures[doc_a], signatures[doc_b])

        if jaccard_score >= args.threshold:
            rows.append(
                {
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "jaccard_similarity": jaccard_score,
                    "minhash_similarity": minhash_score,
                }
            )

    stats = lsh_reduction_stats(
        num_documents=len(valid_docs),
        num_candidate_pairs=len(candidate_pairs),
    )

    rows.append(
        {
            "doc_a": "__STATS__",
            "doc_b": "",
            "jaccard_similarity": "",
            "minhash_similarity": "",
            "num_documents": stats["num_documents"],
            "all_pairs": stats["all_pairs"],
            "candidate_pairs": stats["candidate_pairs"],
            "reduction_ratio": stats["reduction_ratio"],
            "runtime_seconds": runtime_seconds,
        }
    )

    write_csv(rows, args.output)
    print(f"Saved corpus candidates to {args.output}")

    return 0


def command_pairs(args: argparse.Namespace) -> int:
    pairs = load_labeled_pairs_csv(
        csv_path=args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        limit=args.limit,
    )

    y_true: list[int] = []
    jaccard_scores: list[float] = []
    minhash_scores: list[float] = []
    simhash_scores: list[float] = []
    detail_rows: list[dict[str, Any]] = []

    def compute_scores() -> None:
        for pair in pairs:
            summary = document_similarity_summary(
                text_a=pair.text_a,
                text_b=pair.text_b,
                shingle_size=args.shingle_size,
                num_perm=args.num_perm,
                seed=args.seed,
                hash_bits=args.hash_bits,
            )

            jaccard_score = float(summary["jaccard_similarity"])
            minhash_score = float(summary["minhash_similarity"])
            simhash_score = float(summary["simhash_similarity"])

            jaccard_pred = 1 if jaccard_score >= args.threshold else 0
            minhash_pred = 1 if minhash_score >= args.threshold else 0
            simhash_pred = 1 if simhash_score >= args.simhash_threshold else 0

            y_true.append(pair.label)
            jaccard_scores.append(jaccard_score)
            minhash_scores.append(minhash_score)
            simhash_scores.append(simhash_score)

            detail_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "text_a": pair.text_a,
                    "text_b": pair.text_b,
                    "label": pair.label,
                    "jaccard_similarity": jaccard_score,
                    "minhash_similarity": minhash_score,
                    "simhash_similarity": simhash_score,
                    "jaccard_prediction": jaccard_pred,
                    "minhash_prediction": minhash_pred,
                    "simhash_prediction": simhash_pred,
                    "jaccard_error": int(jaccard_pred != pair.label),
                    "minhash_error": int(minhash_pred != pair.label),
                    "simhash_error": int(simhash_pred != pair.label),
                }
            )

    _, runtime_seconds = measure_runtime(compute_scores)

    rows = [
        metrics_row(
            method="jaccard_exact",
            metrics=evaluate_binary_scores(
                y_true,
                jaccard_scores,
                threshold=args.threshold,
            ),
            runtime_seconds=runtime_seconds,
            extra={"num_pairs": len(pairs)},
        ),
        metrics_row(
            method="minhash",
            metrics=evaluate_binary_scores(
                y_true,
                minhash_scores,
                threshold=args.threshold,
            ),
            runtime_seconds=runtime_seconds,
            extra={"num_pairs": len(pairs)},
        ),
        metrics_row(
            method="simhash",
            metrics=evaluate_binary_scores(
                y_true,
                simhash_scores,
                threshold=args.simhash_threshold,
            ),
            runtime_seconds=runtime_seconds,
            extra={"num_pairs": len(pairs)},
        ),
    ]

    save_metrics_csv(rows, args.output)

    if args.details_output is not None:
        write_csv(detail_rows, args.details_output)
        print(f"Saved pair prediction details to {args.details_output}")

    print(f"Saved evaluation metrics to {args.output}")

    return 0


def command_bonus_eval(args: argparse.Namespace) -> int:
    """Optional bonus command: evaluate a labeled pair CSV with the extra
    (non-required) techniques from bonus.py, side by side with the
    standard word-level SimHash from the required pipeline:

    - hybrid SimHash (word tokens + character n-grams), for typo/paraphrase
      resilience
    - optional light Persian lemmatization of tokens before hashing
    - the adaptively chosen LSH (bands, rows_per_band) for --num-perm that
      best approximates --threshold, reported for reference (this command
      does not itself run LSH candidate generation -- see the `corpus`
      command for that)

    This command is additive: it does not change the behaviour of
    `compare`, `corpus`, or `pairs`.
    """

    pairs = load_labeled_pairs_csv(
        csv_path=args.pairs,
        text_col_a=args.text_col_a,
        text_col_b=args.text_col_b,
        label_col=args.label_col,
        limit=args.limit,
    )

    y_true: list[int] = []
    standard_scores: list[float] = []
    hybrid_scores: list[float] = []

    def compute_scores() -> None:
        for pair in pairs:
            doc_a = preprocess_document(pair.text_a, shingle_size=args.shingle_size)
            doc_b = preprocess_document(pair.text_b, shingle_size=args.shingle_size)

            tokens_a = list(doc_a.tokens)
            tokens_b = list(doc_b.tokens)

            if args.persian_lemmatize:
                tokens_a = list(persian_lemmatize(tokens_a))
                tokens_b = list(persian_lemmatize(tokens_b))

            if not tokens_a or not tokens_b:
                standard_scores.append(0.0)
                hybrid_scores.append(0.0)
                y_true.append(pair.label)
                continue

            standard_idf = build_simhash_idf([tokens_a, tokens_b])
            standard_a = simhash(tokens_a, idf=standard_idf, hash_bits=args.hash_bits)
            standard_b = simhash(tokens_b, idf=standard_idf, hash_bits=args.hash_bits)
            standard_scores.append(
                simhash_similarity(standard_a, standard_b, hash_bits=args.hash_bits)
            )

            hybrid_idf = build_hybrid_simhash_idf(
                [tokens_a, tokens_b],
                char_ngram_sizes=tuple(args.char_ngram_sizes),
            )
            hybrid_a = hybrid_simhash(
                tokens_a,
                idf=hybrid_idf,
                hash_bits=args.hash_bits,
                char_ngram_sizes=tuple(args.char_ngram_sizes),
            )
            hybrid_b = hybrid_simhash(
                tokens_b,
                idf=hybrid_idf,
                hash_bits=args.hash_bits,
                char_ngram_sizes=tuple(args.char_ngram_sizes),
            )
            hybrid_scores.append(
                simhash_similarity(hybrid_a, hybrid_b, hash_bits=args.hash_bits)
            )

            y_true.append(pair.label)

    _, runtime_seconds = measure_runtime(compute_scores)

    adaptive_bands, adaptive_rows = find_adaptive_lsh_params(
        num_perm=args.num_perm,
        target_threshold=args.threshold,
    )

    rows = [
        metrics_row(
            method="simhash_standard",
            metrics=evaluate_binary_scores(
                y_true, standard_scores, threshold=args.simhash_threshold
            ),
            runtime_seconds=runtime_seconds,
            extra={
                "num_pairs": len(pairs),
                "persian_lemmatize": args.persian_lemmatize,
            },
        ),
        metrics_row(
            method="simhash_hybrid_bonus",
            metrics=evaluate_binary_scores(
                y_true, hybrid_scores, threshold=args.simhash_threshold
            ),
            runtime_seconds=runtime_seconds,
            extra={
                "num_pairs": len(pairs),
                "char_ngram_sizes": ",".join(str(n) for n in args.char_ngram_sizes),
                "persian_lemmatize": args.persian_lemmatize,
            },
        ),
    ]

    save_metrics_csv(rows, args.output)

    print(
        "Adaptive LSH suggestion for num_perm="
        f"{args.num_perm}, target_threshold={args.threshold}: "
        f"bands={adaptive_bands}, rows_per_band={adaptive_rows} "
        f"(approx_threshold={(1 / adaptive_bands) ** (1 / adaptive_rows):.4f})"
    )
    print(f"Saved bonus evaluation metrics to {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plagiarism-engine",
        description="Semantic duplicate and near-plagiarism detection CLI.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two text files.",
    )
    compare_parser.add_argument("--file-a", required=True, type=Path)
    compare_parser.add_argument("--file-b", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    compare_parser.add_argument("--shingle-size", type=int, default=3)
    compare_parser.add_argument("--num-perm", type=int, default=128)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--hash-bits", type=int, default=64)
    compare_parser.set_defaults(func=command_compare)

    corpus_parser = subparsers.add_parser(
        "corpus",
        help="Find similar documents in a folder.",
    )
    corpus_parser.add_argument("--data", required=True, type=Path)
    corpus_parser.add_argument("--threshold", type=float, default=0.25)
    corpus_parser.add_argument("--shingle-size", type=int, default=3)
    corpus_parser.add_argument("--num-perm", type=int, default=128)
    corpus_parser.add_argument("--num-bands", type=int, default=32)
    corpus_parser.add_argument("--seed", type=int, default=42)
    corpus_parser.add_argument("--recursive", action="store_true")
    corpus_parser.add_argument("--output", required=True, type=Path)
    corpus_parser.set_defaults(func=command_corpus)

    pairs_parser = subparsers.add_parser(
        "pairs",
        help="Evaluate on a labeled pair CSV dataset.",
    )
    pairs_parser.add_argument("--pairs", required=True, type=Path)
    pairs_parser.add_argument("--text-col-a", required=True)
    pairs_parser.add_argument("--text-col-b", required=True)
    pairs_parser.add_argument("--label-col", required=True)
    pairs_parser.add_argument("--limit", type=int, default=None)
    pairs_parser.add_argument("--threshold", type=float, default=0.25)
    pairs_parser.add_argument("--simhash-threshold", type=float, default=0.75)
    pairs_parser.add_argument("--shingle-size", type=int, default=3)
    pairs_parser.add_argument("--num-perm", type=int, default=128)
    pairs_parser.add_argument("--seed", type=int, default=42)
    pairs_parser.add_argument("--hash-bits", type=int, default=64)
    pairs_parser.add_argument("--output", required=True, type=Path)
    pairs_parser.add_argument("--details-output", type=Path, default=None)
    pairs_parser.set_defaults(func=command_pairs)

    bonus_parser = subparsers.add_parser(
        "bonus-eval",
        help=(
            "Optional: evaluate a labeled pair CSV with the bonus hybrid "
            "SimHash (word + character n-grams) against the standard "
            "word-level SimHash. Does not affect compare/corpus/pairs."
        ),
    )
    bonus_parser.add_argument("--pairs", required=True, type=Path)
    bonus_parser.add_argument("--text-col-a", required=True)
    bonus_parser.add_argument("--text-col-b", required=True)
    bonus_parser.add_argument("--label-col", required=True)
    bonus_parser.add_argument("--limit", type=int, default=None)
    bonus_parser.add_argument("--threshold", type=float, default=0.25)
    bonus_parser.add_argument("--simhash-threshold", type=float, default=0.75)
    bonus_parser.add_argument("--shingle-size", type=int, default=3)
    bonus_parser.add_argument("--num-perm", type=int, default=128)
    bonus_parser.add_argument("--hash-bits", type=int, default=64)
    bonus_parser.add_argument(
        "--char-ngram-sizes",
        type=int,
        nargs="+",
        default=[3],
        help="Character n-gram sizes for the hybrid SimHash (default: 3).",
    )
    bonus_parser.add_argument(
        "--persian-lemmatize",
        action="store_true",
        help="Apply the light rule-based Persian lemmatizer before hashing.",
    )
    bonus_parser.add_argument("--output", required=True, type=Path)
    bonus_parser.set_defaults(func=command_bonus_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
