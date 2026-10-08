#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.evaluation import RankingAccumulator, top_k_from_scores
from avito_recsys.snapshot_identity import (
    SNAPSHOT_ID_CONTRACT,
    duckdb_pseudonymous_snapshot_id_sql,
)


def parquet_list(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{path.resolve()}'" for path in paths) + "]"


def build_statistics(con: duckdb.DuckDBPyConnection, train_paths: list[Path], n_leaf: int):
    source = parquet_list(train_paths)
    popularity_df = con.execute(
        f"""
        SELECT target.leaf_id, count(*) n
        FROM read_parquet({source}) t, unnest(t.label_leaf) AS target(leaf_id)
        GROUP BY 1
        """
    ).fetchdf()
    popularity = np.zeros(n_leaf + 1, dtype=np.float64)
    popularity[popularity_df["leaf_id"].to_numpy(dtype=int)] = popularity_df["n"].to_numpy(dtype=float)

    cooc_df = con.execute(
        f"""
        SELECT history.leaf_id history_leaf, target.leaf_id target_leaf, count(*) n
        FROM read_parquet({source}) t,
             unnest(list_distinct(t.leaf)) AS history(leaf_id),
             unnest(t.label_leaf) AS target(leaf_id)
        WHERE t.n_labels > 0
        GROUP BY 1, 2
        """
    ).fetchdf()
    cooc = np.zeros((n_leaf + 1, n_leaf + 1), dtype=np.float32)
    h = cooc_df["history_leaf"].to_numpy(dtype=int)
    target = cooc_df["target_leaf"].to_numpy(dtype=int)
    count = cooc_df["n"].to_numpy(dtype=float)
    target_norm = np.sqrt(np.maximum(popularity[target], 1.0))
    cooc[h, target] = (count / target_norm).astype(np.float32)
    row_norm = np.linalg.norm(cooc, axis=1, keepdims=True)
    cooc = np.divide(cooc, row_norm, out=np.zeros_like(cooc), where=row_norm > 0)
    pop_fallback = np.log1p(popularity)
    if pop_fallback.max() > 0:
        pop_fallback /= pop_fallback.max()
    return popularity, cooc, pop_fallback


def random_scores(user_key: object, n_leaf: int, seed: int) -> np.ndarray:
    ids = np.arange(n_leaf + 1, dtype=np.uint64)
    if isinstance(user_key, (int, np.integer)):
        base_key = int(user_key) & 0xFFFFFFFFFFFFFFFF
    else:
        digest = hashlib.sha256(str(user_key).encode("utf-8")).digest()
        base_key = int.from_bytes(digest[:8], "little")
    key = np.uint64(base_key) ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        values = ids * np.uint64(6364136223846793005) + key * np.uint64(1442695040888963407)
    values ^= values >> np.uint64(33)
    return values.astype(np.float64)


def cooc_scores(history: list[int], cooc: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    scores = np.zeros(cooc.shape[1], dtype=np.float64)
    used: set[int] = set()
    rank = 0
    for item in reversed(history):
        item = int(item)
        if item <= 0 or item in used:
            continue
        used.add(item)
        rank += 1
        scores += cooc[item] / math.sqrt(rank)
    return scores + 0.05 * fallback


def build_puresvd(
    con: duckdb.DuckDBPyConnection,
    train_paths: list[Path],
    n_leaf: int,
    train_cap: int,
    seed: int,
    n_components: int = 64,
) -> np.ndarray:
    source = parquet_list(train_paths)
    frame = con.execute(
        f"""
        SELECT leaf, action FROM read_parquet({source})
        ORDER BY hash(snapshot_id)
        LIMIT {train_cap}
        """
    ).fetchdf()
    total = int(sum(len(values) for values in frame["leaf"]))
    rows = np.empty(total, dtype=np.int32)
    cols = np.empty(total, dtype=np.int32)
    values = np.empty(total, dtype=np.float32)
    action_weight = np.array([0.0, 1.0, 3.0, 5.0, 5.0], dtype=np.float32)
    offset = 0
    for row_id, (leaf, action) in enumerate(zip(frame["leaf"], frame["action"])):
        length = len(leaf)
        rows[offset : offset + length] = row_id
        cols[offset : offset + length] = np.asarray(leaf, dtype=np.int32)
        values[offset : offset + length] = action_weight[np.asarray(action, dtype=np.int32)]
        offset += length
    matrix = sp.csr_matrix(
        (values, (rows, cols)), shape=(len(frame), n_leaf + 1), dtype=np.float32
    )[:, 1:]
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    components = min(n_components, n_leaf - 1)
    svd = TruncatedSVD(n_components=components, n_iter=7, random_state=seed)
    svd.fit(matrix)
    return svd.components_.astype(np.float32)


def puresvd_scores(history: list[int], action: list[int], components: np.ndarray, n_leaf: int) -> np.ndarray:
    profile = np.zeros(n_leaf, dtype=np.float32)
    action_weight = np.array([0.0, 1.0, 3.0, 5.0, 5.0], dtype=np.float32)
    leaf = np.asarray(history, dtype=np.int64) - 1
    weights = action_weight[np.asarray(action, dtype=np.int64)]
    np.add.at(profile, leaf, weights)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile /= norm
    reconstructed = (profile @ components.T) @ components
    scores = np.zeros(n_leaf + 1, dtype=np.float64)
    scores[1:] = reconstructed
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--eval-cap", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svd-train-cap", type=int, default=300000)
    parser.add_argument("--skip-svd", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path)
    parser.add_argument("--train-snapshots-dir", type=Path)
    parser.add_argument("--eval-snapshots-dir", type=Path)
    parser.add_argument(
        "--eval-ids",
        type=Path,
        help="Optional parquet with snapshot_id values defining the exact evaluation audience.",
    )
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--allow-final-holdout", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    folds = {int(item["fold"]): item for item in config["folds"]}
    fold = folds[args.fold]
    if fold["role"] == "final_holdout" and args.eval_split == "test" and not args.allow_final_holdout:
        raise RuntimeError("Final holdout is locked. Use --allow-final-holdout only after freezing the model.")

    processed = ROOT / config.get("processed_dir", "data/processed")
    snapshots = args.snapshots_dir or (processed / "snapshots")
    train_snapshots = args.train_snapshots_dir or snapshots
    eval_snapshots = args.eval_snapshots_dir or snapshots
    train_paths = [train_snapshots / f"snapshot_{date}.parquet" for date in fold["train_snapshots"]]
    eval_date = fold[f"{args.eval_split}_snapshot"]
    eval_path = eval_snapshots / f"snapshot_{eval_date}.parquet"
    vocab_path = args.vocab or (processed / config.get("vocab_file", "vocab_leaf_v90.json"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    n_leaf = len(vocab["leaf"])

    con = duckdb.connect()
    con.execute("PRAGMA threads=4; SET memory_limit='12GB';")
    started = time.time()
    popularity, cooc, fallback = build_statistics(con, train_paths, n_leaf)
    svd_components = None
    if not args.skip_svd:
        svd_components = build_puresvd(
            con, train_paths, n_leaf, args.svd_train_cap, args.seed
        )
    cap_clause = "" if args.eval_cap <= 0 else f"LIMIT {args.eval_cap}"
    safe_snapshot_id = duckdb_pseudonymous_snapshot_id_sql("s.snapshot_id")
    if args.eval_ids:
        audience_id = duckdb_pseudonymous_snapshot_id_sql("i.snapshot_id")
        audience_source = f"read_parquet('{args.eval_ids.resolve()}') AS i"
        legacy_double_id = "sha256(CAST(s.snapshot_id AS VARCHAR))"
        audience_filter = (
            f"AND (({safe_snapshot_id}) IN (SELECT {audience_id} FROM {audience_source}) "
            f"OR ({legacy_double_id}) IN (SELECT {audience_id} FROM {audience_source}))"
        )
        cap_clause = ""
    else:
        audience_filter = ""
    eval_columns = {
        str(row[0])
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{eval_path.resolve()}')"
        ).fetchall()
    }
    if "user_id" in eval_columns:
        user_key_expr = "s.user_id"
    elif "hashed_user_key" in eval_columns:
        user_key_expr = "s.hashed_user_key"
    else:
        user_key_expr = safe_snapshot_id
    evaluation = con.execute(
        f"""
        SELECT {safe_snapshot_id} AS snapshot_id,
               {user_key_expr} AS user_key,
               s.leaf, s.action, s.label_leaf, s.masked_leaf
        FROM read_parquet('{eval_path.resolve()}') AS s
        WHERE s.n_labels > 0
        {audience_filter}
        ORDER BY hash(s.snapshot_id)
        {cap_clause}
        """
    ).fetchdf()
    con.close()

    methods = {
        "random": RankingAccumulator(),
        "most_popular": RankingAccumulator(),
        "weighted_cooccurrence": RankingAccumulator(),
    }
    if svd_components is not None:
        methods["puresvd_fold_in"] = RankingAccumulator()
    predictions: list[dict] = []
    for row in evaluation.itertuples(index=False):
        score_map = {
            "random": random_scores(row.user_key, n_leaf, args.seed),
            "most_popular": popularity,
            "weighted_cooccurrence": cooc_scores(row.leaf, cooc, fallback),
        }
        if svd_components is not None:
            score_map["puresvd_fold_in"] = puresvd_scores(
                row.leaf, row.action, svd_components, n_leaf
            )
        for method, scores in score_map.items():
            ranking = top_k_from_scores(scores, row.masked_leaf, 20)
            methods[method].add(ranking, row.label_leaf)
            predictions.append(
                {
                    "snapshot_id": row.snapshot_id,
                    "method": method,
                    "ranking": ranking,
                    "labels": [int(x) for x in row.label_leaf],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, accumulator in methods.items():
        rows.append(
            {
                "experiment": f"baseline_{config.get('data_version', 'v90')}",
                "method": method,
                "fold": args.fold,
                "eval_split": args.eval_split,
                "eval_snapshot": eval_date,
                "train_snapshots": fold["train_snapshots"],
                "seed": args.seed,
                **accumulator.result(n_leaf),
            }
        )
    metrics = {
        "config": str(args.config),
        "fold": fold,
        "eval_cap": args.eval_cap,
        "eval_ids": (str(args.eval_ids) if args.eval_ids else None),
        "svd_train_cap": (None if args.skip_svd else args.svd_train_cap),
        "train_snapshots_dir": str(train_snapshots),
        "eval_snapshots_dir": str(eval_snapshots),
        "snapshot_id_contract": SNAPSHOT_ID_CONTRACT,
        "prediction_artifact_contains_raw_user_id": False,
        "runtime_seconds": time.time() - started,
        "results": rows,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(predictions).to_parquet(args.output / "predictions.parquet", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
