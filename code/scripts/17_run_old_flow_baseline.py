#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.evaluation import RankingAccumulator, top_k_from_scores
from avito_recsys.old_flow import (
    OldFlowStatistics,
    count_strict_ordered_category_pairs,
    leaf_scores_from_category_flow,
)
from avito_recsys.snapshot_identity import (
    SNAPSHOT_ID_CONTRACT,
    duckdb_pseudonymous_snapshot_id_sql,
)

METRICS = ("p_target_given_source", "lift", "mu", "potential")


def parquet_list(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{path.resolve()}'" for path in paths) + "]"


def history_sequence(
    row, history_seconds: int, leaf_to_category: np.ndarray
) -> tuple[list[int], list[int], list[int]]:
    cutoff = int(pd.Timestamp(row.snapshot_ts).timestamp())
    threshold = cutoff - history_seconds
    selected = [
        (int(leaf_to_category[int(leaf)]), int(leaf), int(ts))
        for leaf, ts in zip(row.leaf, row.ts)
        if threshold <= int(ts) < cutoff
    ]
    if not selected:
        return [], [], []
    selected.sort(key=lambda item: item[2])
    return (
        [item[0] for item in selected],
        [item[1] for item in selected],
        [item[2] for item in selected],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--train-cap", type=int, default=300000)
    parser.add_argument("--eval-cap", type=int, default=100000)
    parser.add_argument("--history-days", type=int, default=7)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="Legacy flow score variants to evaluate.",
    )
    parser.add_argument(
        "--eval-ids",
        type=Path,
        help="Optional predictions parquet whose snapshot_id values define the exact evaluation audience.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--allow-final-holdout", action="store_true")
    args = parser.parse_args()
    if args.history_days <= 0:
        raise ValueError("--history-days must be positive")
    history_seconds = args.history_days * 24 * 60 * 60

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fold = {int(item["fold"]): item for item in config["folds"]}[args.fold]
    if fold["role"] == "final_holdout" and args.eval_split == "test" and not args.allow_final_holdout:
        raise RuntimeError("Final holdout is locked until the model and score variant are frozen.")

    processed = ROOT / config.get("processed_dir", "data/processed")
    snapshots = args.snapshots_dir or (processed / "snapshots")
    train_paths = [snapshots / f"snapshot_{date}.parquet" for date in fold["train_snapshots"]]
    eval_date = fold[f"{args.eval_split}_snapshot"]
    eval_path = snapshots / f"snapshot_{eval_date}.parquet"
    vocab_path = args.vocab or (processed / config.get("vocab_file", "vocab_leaf_v90.json"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    n_leaf = len(vocab["leaf"])
    leaf_to_category = np.zeros(n_leaf + 1, dtype=np.int64)
    for leaf_id, category_id in vocab["leaf2cat"].items():
        leaf_to_category[int(leaf_id)] = int(category_id)
    n_category = int(leaf_to_category.max())

    con = duckdb.connect()
    con.execute("PRAGMA threads=4; SET memory_limit='12GB';")
    train_limit = "" if args.train_cap <= 0 else f"LIMIT {args.train_cap}"
    train_cursor = con.execute(
        f"""
        SELECT snapshot_ts, leaf, ts
        FROM read_parquet({parquet_list(train_paths)})
        ORDER BY hash(snapshot_id)
        {train_limit}
        """
    )
    pair_count = np.zeros((n_category + 1, n_category + 1), dtype=np.float64)
    leaf_popularity = np.zeros(n_leaf + 1, dtype=np.float64)
    while True:
        train_chunk = train_cursor.fetch_df_chunk(vectors_per_chunk=16)
        if train_chunk.empty:
            break
        history_train = [
            history_sequence(row, history_seconds, leaf_to_category)
            for row in train_chunk.itertuples(index=False)
        ]
        sequences = ((cat, ts) for cat, _leaf, ts in history_train if len(cat) >= 2)
        pair_count += count_strict_ordered_category_pairs(
            sequences, n_category
        ).pair_count
        for _cat, leaf, _ts in history_train:
            if leaf:
                np.add.at(leaf_popularity, np.asarray(leaf, dtype=np.int64), 1)
    pair_count[0, :] = 0
    pair_count[:, 0] = 0
    stats = OldFlowStatistics(
        pair_count=pair_count,
        source_pair_count=pair_count.sum(axis=1),
        target_pair_count=pair_count.sum(axis=0),
        total_pairs=float(pair_count.sum()),
    )

    eval_limit = "" if args.eval_cap <= 0 else f"LIMIT {args.eval_cap}"
    safe_snapshot_id = duckdb_pseudonymous_snapshot_id_sql("snapshot_id")
    if args.eval_ids:
        eval_ids_path = args.eval_ids.resolve()
        evaluation = con.execute(
            f"""
            WITH requested AS (
                SELECT DISTINCT lower(CAST(snapshot_id AS VARCHAR)) AS snapshot_id
                FROM read_parquet('{eval_ids_path}')
            ), eligible AS (
                SELECT {safe_snapshot_id} AS snapshot_id,
                       snapshot_ts, leaf, ts, label_leaf, masked_leaf
                FROM read_parquet('{eval_path.resolve()}')
                WHERE n_labels > 0
            )
            SELECT eligible.*
            FROM eligible
            INNER JOIN requested USING (snapshot_id)
            ORDER BY snapshot_id
            {eval_limit}
            """
        ).fetchdf()
        requested_count = con.execute(
            f"""
            SELECT count(DISTINCT lower(CAST(snapshot_id AS VARCHAR)))
            FROM read_parquet('{eval_ids_path}')
            """
        ).fetchone()[0]
        if len(evaluation) != min(requested_count, args.eval_cap if args.eval_cap > 0 else requested_count):
            raise RuntimeError(
                f"Exact audience join mismatch: requested={requested_count}, matched={len(evaluation)}"
            )
    else:
        evaluation = con.execute(
            f"""
            SELECT {safe_snapshot_id} AS snapshot_id,
                   snapshot_ts, leaf, ts, label_leaf, masked_leaf
            FROM read_parquet('{eval_path.resolve()}')
            WHERE n_labels > 0
            ORDER BY hash(snapshot_id)
            {eval_limit}
            """
        ).fetchdf()
    con.close()

    matrices = {metric: stats.metric_matrix(metric) for metric in args.metrics}
    accumulators = {
        f"old_flow_{metric}": RankingAccumulator() for metric in args.metrics
    }
    predictions: list[dict] = []
    started = time.time()
    no_history = 0
    for row in evaluation.itertuples(index=False):
        category, _leaf, _ts = history_sequence(
            row, history_seconds, leaf_to_category
        )
        source_category = category[-1] if category else None
        no_history += int(source_category is None)
        for metric, matrix in matrices.items():
            method = f"old_flow_{metric}"
            scores = leaf_scores_from_category_flow(
                source_category, matrix, leaf_to_category, leaf_popularity
            )
            ranking = top_k_from_scores(scores, row.masked_leaf, 20)
            accumulators[method].add(ranking, row.label_leaf)
            predictions.append(
                {
                    "snapshot_id": row.snapshot_id,
                    "method": method,
                    "source_category_id": source_category,
                    "ranking": ranking,
                    "labels": [int(value) for value in row.label_leaf],
                }
            )

    category_names: dict[int, str] = {}
    for leaf_key, leaf_id in vocab["leaf"].items():
        category_id = int(vocab["leaf2cat"][str(leaf_id)])
        category_names.setdefault(category_id, leaf_key.split("|", 2)[1])
    source_id, target_id = np.nonzero(stats.pair_count)
    matrix_frame = pd.DataFrame(
        {
            "source_category_id": source_id,
            "source_category": [category_names.get(int(value), "") for value in source_id],
            "target_category_id": target_id,
            "target_category": [category_names.get(int(value), "") for value in target_id],
            "pair_count": stats.pair_count[source_id, target_id],
            "source_pair_count": stats.source_pair_count[source_id],
            "target_pair_count": stats.target_pair_count[target_id],
            "all_pairs": stats.total_pairs,
        }
    )
    for metric, matrix in matrices.items():
        matrix_frame[metric] = matrix[source_id, target_id]

    result_rows = [
        {
            "experiment": f"old_flow_baseline_{config.get('data_version', 'v90')}",
            "method": method,
            "fold": args.fold,
            "eval_split": args.eval_split,
            "eval_snapshot": eval_date,
            "train_snapshots": fold["train_snapshots"],
            "history_days": args.history_days,
            "source_experience": "buyer",
            "target_experience": "buyer",
            **accumulator.result(n_leaf),
        }
        for method, accumulator in accumulators.items()
    ]
    payload = {
        "config": str(args.config),
        "fold": fold,
        "train_cap": args.train_cap,
        "eval_cap": args.eval_cap,
        "n_pair_occurrences": stats.total_pairs,
        "n_nonzero_category_pairs": int((stats.pair_count > 0).sum()),
        f"n_eval_without_last_{args.history_days}d_history": no_history,
        "eval_ids": str(args.eval_ids.resolve()) if args.eval_ids else None,
        "score_variants": list(args.metrics),
        "snapshot_id_contract": SNAPSHOT_ID_CONTRACT,
        "prediction_artifact_contains_raw_user_id": False,
        "evaluation_runtime_seconds": time.time() - started,
        "formula_contract": {
            "p_target_given_source": "pair_count / source_pair_count",
            "lift": "p_target_given_source / (target_pair_count / all_pairs)",
            "p_target_without_source": "(target_pair_count - pair_count) / (all_pairs - source_pair_count)",
            "mu": "p_target_given_source / p_target_without_source",
            "potential": "(source_pair_count - pair_count) * p_target_given_source * 0.08",
        },
        "results": result_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    matrix_frame.to_parquet(args.output / "flow_matrix.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(args.output / "predictions.parquet", index=False)
    print(pd.DataFrame(result_rows).to_string(index=False))


if __name__ == "__main__":
    main()
