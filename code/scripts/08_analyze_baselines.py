#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.metrics import binary_ndcg_at_k, recall_at_k
from avito_recsys.snapshot_identity import (
    duckdb_pseudonymous_snapshot_id_sql,
    pseudonymize_snapshot_ids,
)


def row_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = pseudonymize_snapshot_ids(frame, copy=False)
    frame["recall@5"] = [recall_at_k(r, y, 5) for r, y in zip(frame.ranking, frame.labels)]
    frame["ndcg@5"] = [binary_ndcg_at_k(r, y, 5) for r, y in zip(frame.ranking, frame.labels)]
    return frame


def paired_bootstrap(
    reference: np.ndarray, challenger: np.ndarray, repeats: int, seed: int
) -> dict[str, float]:
    difference = reference - challenger
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sample = rng.integers(0, len(difference), size=len(difference))
        means[repeat] = difference[sample].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_bootstrap_le_zero": float((means <= 0).mean()),
    }


def load_history_lengths(snapshot: Path, snapshot_ids: pd.Series) -> pd.DataFrame:
    ids = pd.DataFrame({"snapshot_id": snapshot_ids.drop_duplicates().astype(str)})
    ids = pseudonymize_snapshot_ids(ids, copy=False)
    con = duckdb.connect()
    con.register("selected_ids", ids)
    primary_id = duckdb_pseudonymous_snapshot_id_sql("s.snapshot_id")
    legacy_double_id = "sha256(CAST(s.snapshot_id AS VARCHAR))"
    result = con.execute(
        f"""
        WITH source_rows AS (
            SELECT {primary_id} AS primary_id,
                   {legacy_double_id} AS legacy_double_id,
                   len(s.leaf) AS history_length
            FROM read_parquet(?) AS s
        ), source_candidates AS (
            SELECT primary_id AS join_id, 0 AS compatibility_rank, history_length
            FROM source_rows
            UNION ALL
            SELECT legacy_double_id AS join_id, 1 AS compatibility_rank, history_length
            FROM source_rows
            WHERE legacy_double_id <> primary_id
        ), matched AS (
            SELECT i.snapshot_id, c.history_length,
                   row_number() OVER (
                       PARTITION BY i.snapshot_id ORDER BY c.compatibility_rank
                   ) AS match_rank
            FROM selected_ids AS i
            JOIN source_candidates AS c ON c.join_id = i.snapshot_id
        )
        SELECT snapshot_id, history_length
        FROM matched
        WHERE match_rank = 1
        """,
        [str(snapshot.resolve())],
    ).fetchdf()
    con.close()
    return result


def target_frequency(train_paths: list[Path]) -> pd.DataFrame:
    source = "[" + ",".join(f"'{path.resolve()}'" for path in train_paths) + "]"
    con = duckdb.connect()
    result = con.execute(
        f"""
        SELECT target_leaf, count(*) AS target_frequency
        FROM read_parquet({source}), UNNEST(label_leaf) AS t(target_leaf)
        WHERE n_labels > 0
        GROUP BY target_leaf
        ORDER BY target_frequency DESC, target_leaf
        """
    ).fetchdf()
    con.close()
    n = len(result)
    result["popularity_segment"] = np.select(
        [np.arange(n) < max(1, int(np.ceil(0.20 * n))),
         np.arange(n) < max(2, int(np.ceil(0.50 * n)))],
        ["head_top_20pct_leaf", "medium_next_30pct_leaf"],
        default="tail_bottom_50pct_leaf",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data_v90.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/baselines")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/baselines/diagnostics")
    parser.add_argument("--bootstrap-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--snapshots-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    snapshots = args.snapshots_dir or ROOT / config.get("processed_dir", "data/processed") / "snapshots"
    args.output.mkdir(parents=True, exist_ok=True)

    uncertainty_rows = []
    history_rows = []
    popularity_rows = []
    for fold in config["folds"]:
        if fold["role"] != "cv":
            continue
        fold_id = int(fold["fold"])
        eval_snapshot = fold[f"{args.eval_split}_snapshot"]
        prediction_path = args.artifacts / f"fold_{fold_id}_{args.eval_split}/predictions.parquet"
        if not prediction_path.exists():
            continue
        predictions = row_metrics(pd.read_parquet(prediction_path))

        pivot = predictions.pivot(index="snapshot_id", columns="method", values="recall@5")
        reference = "puresvd_fold_in"
        for challenger in ["weighted_cooccurrence", "most_popular", "random"]:
            paired = pivot[[reference, challenger]].dropna()
            stats = paired_bootstrap(
                paired[reference].to_numpy(), paired[challenger].to_numpy(),
                args.bootstrap_repeats, args.seed + fold_id,
            )
            uncertainty_rows.append(
                {"fold": fold_id, "reference": reference, "challenger": challenger,
                 "metric": "recall@5", "n_users": len(paired), **stats}
            )

        lengths = load_history_lengths(
            snapshots / f"snapshot_{eval_snapshot}.parquet", predictions.snapshot_id
        )
        with_lengths = predictions.merge(lengths, on="snapshot_id", how="left", validate="many_to_one")
        with_lengths["history_segment"] = pd.cut(
            with_lengths.history_length,
            bins=[0, 20, 50, 128, np.inf],
            labels=["short_5_20", "medium_21_50", "long_51_128", "over_128"],
        )
        grouped = (
            with_lengths.groupby(["method", "history_segment"], observed=True)
            .agg(n_users=("snapshot_id", "size"), recall_at_5=("recall@5", "mean"),
                 ndcg_at_5=("ndcg@5", "mean"))
            .reset_index()
        )
        grouped.insert(0, "fold", fold_id)
        history_rows.append(grouped)

        frequencies = target_frequency(
            [snapshots / f"snapshot_{date}.parquet" for date in fold["train_snapshots"]]
        )
        exploded = predictions[["method", "ranking", "labels"]].explode("labels")
        exploded = exploded.rename(columns={"labels": "target_leaf"})
        exploded["hit_at_5"] = [
            int(int(target) in list(ranking)[:5])
            for target, ranking in zip(exploded.target_leaf, exploded.ranking)
        ]
        exploded = exploded.merge(frequencies, on="target_leaf", how="left")
        pop_grouped = (
            exploded.groupby(["method", "popularity_segment"], observed=True)
            .agg(n_target_occurrences=("target_leaf", "size"), target_hit_rate_at_5=("hit_at_5", "mean"))
            .reset_index()
        )
        pop_grouped.insert(0, "fold", fold_id)
        popularity_rows.append(pop_grouped)

    if not uncertainty_rows:
        raise FileNotFoundError(
            f"No completed {args.eval_split} prediction files found under {args.artifacts}"
        )

    uncertainty = pd.DataFrame(uncertainty_rows)
    history = pd.concat(history_rows, ignore_index=True)
    popularity = pd.concat(popularity_rows, ignore_index=True)
    uncertainty.to_csv(args.output / "paired_bootstrap_recall5.csv", index=False)
    history.to_csv(args.output / "history_segments.csv", index=False)
    popularity.to_csv(args.output / "target_popularity_segments.csv", index=False)

    summary = (
        "# Baseline diagnostics\n\n"
        "All comparisons use the same new-only candidate mask. Confidence intervals are paired "
        f"user-level bootstrap intervals on completed {args.eval_split} folds. "
        "The final August holdout is untouched.\n\n"
        "## PureSVD minus alternatives, Recall@5\n\n"
        + uncertainty.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Metrics by history length\n\n"
        + history.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Target-level hit rate by training popularity\n\n"
        + popularity.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    (args.output / "README.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
