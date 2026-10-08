#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.snapshot_identity import (
    duckdb_pseudonymous_snapshot_id_sql,
    pseudonymize_snapshot_ids,
)


def recall_at_k(ranking, labels, k: int = 5) -> float:
    positives = set(int(value) for value in labels)
    return len(set(int(value) for value in ranking[:k]) & positives) / len(positives)


def ndcg_at_k(ranking, labels, k: int = 5) -> float:
    positives = set(int(value) for value in labels)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(ranking[:k])
        if int(item) in positives
    )
    ideal_hits = min(k, len(positives))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def add_row_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["snapshot_id", "ranking", "labels"]].copy()
    result = pseudonymize_snapshot_ids(result, copy=False)
    result["recall@5"] = [
        recall_at_k(ranking, labels) for ranking, labels in zip(result.ranking, result.labels)
    ]
    result["ndcg@5"] = [
        ndcg_at_k(ranking, labels) for ranking, labels in zip(result.ranking, result.labels)
    ]
    return result


def paired_bootstrap(delta: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sample = rng.integers(0, len(delta), len(delta))
        means[repeat] = delta[sample].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_bootstrap_le_zero": float((means <= 0).mean()),
    }


def compare_pair(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    reference_name: str,
    challenger_name: str,
    repeats: int,
    seed: int,
) -> list[dict]:
    left = add_row_metrics(reference)
    right = add_row_metrics(challenger)
    paired = left.merge(
        right,
        on="snapshot_id",
        how="inner",
        suffixes=("_reference", "_challenger"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError(
            f"Audience mismatch for {reference_name} vs {challenger_name}: "
            f"reference={len(left)}, challenger={len(right)}, paired={len(paired)}"
        )
    rows = []
    for metric in ("recall@5", "ndcg@5"):
        reference_values = paired[f"{metric}_reference"].to_numpy(dtype=float)
        challenger_values = paired[f"{metric}_challenger"].to_numpy(dtype=float)
        rows.append(
            {
                "comparison": f"{reference_name}_vs_{challenger_name}",
                "metric": metric,
                "reference": reference_name,
                "challenger": challenger_name,
                "n_users": len(paired),
                "reference_mean": float(reference_values.mean()),
                "challenger_mean": float(challenger_values.mean()),
                **paired_bootstrap(reference_values - challenger_values, repeats, seed),
            }
        )
    return rows


def parquet_source(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{path.resolve()}'" for path in paths) + "]"


def load_validation_metadata(snapshot_path: Path, audience_path: Path) -> pd.DataFrame:
    audience_id = duckdb_pseudonymous_snapshot_id_sql("a.snapshot_id")
    source_id = duckdb_pseudonymous_snapshot_id_sql("s.snapshot_id")
    legacy_source_id = "sha256(CAST(s.snapshot_id AS VARCHAR))"
    con = duckdb.connect()
    result = con.execute(
        f"""
        WITH audience AS (
            SELECT DISTINCT {audience_id} AS snapshot_id
            FROM read_parquet('{audience_path.resolve()}') AS a
        ), source_rows AS (
            SELECT {source_id} AS primary_id,
                   {legacy_source_id} AS legacy_double_id,
                   array_length(s.leaf) AS history_length,
                   s.n_labels
            FROM read_parquet('{snapshot_path.resolve()}') AS s
            WHERE s.n_labels > 0
        ), source_candidates AS (
            SELECT primary_id AS join_id, 0 AS compatibility_rank,
                   history_length, n_labels
            FROM source_rows
            UNION ALL
            SELECT legacy_double_id AS join_id, 1 AS compatibility_rank,
                   history_length, n_labels
            FROM source_rows
            WHERE legacy_double_id <> primary_id
        ), matched AS (
            SELECT a.snapshot_id, c.history_length, c.n_labels,
                   row_number() OVER (
                       PARTITION BY a.snapshot_id ORDER BY c.compatibility_rank
                   ) AS match_rank
            FROM audience AS a
            JOIN source_candidates AS c ON c.join_id = a.snapshot_id
        )
        SELECT snapshot_id, history_length, n_labels
        FROM matched
        WHERE match_rank = 1
        ORDER BY hash(snapshot_id)
        """
    ).fetchdf()
    con.close()
    result["history_segment"] = pd.cut(
        result["history_length"],
        bins=[0, 20, 50, np.inf],
        labels=["short_5_20", "medium_21_50", "long_51_128"],
    ).astype(str)
    result["label_segment"] = np.where(result["n_labels"] == 1, "single_label", "multi_label")
    return result


def build_leaf_segments(train_paths: list[Path], n_leaf: int) -> dict[int, str]:
    con = duckdb.connect()
    counts = con.execute(
        f"""
        SELECT target.leaf_id, count(*) AS n
        FROM read_parquet({parquet_source(train_paths)}) t,
             unnest(t.label_leaf) AS target(leaf_id)
        GROUP BY 1
        ORDER BY n DESC, target.leaf_id
        """
    ).fetchdf()
    con.close()
    count_map = dict(zip(counts.leaf_id.astype(int), counts.n.astype(int)))
    ordered = sorted(range(1, n_leaf + 1), key=lambda leaf: (-count_map.get(leaf, 0), leaf))
    head_end = int(math.ceil(0.20 * n_leaf))
    middle_end = head_end + int(math.ceil(0.30 * n_leaf))
    segments = {}
    for rank, leaf in enumerate(ordered):
        if rank < head_end:
            segments[leaf] = "head_top_20pct_leaf"
        elif rank < middle_end:
            segments[leaf] = "medium_next_30pct_leaf"
        else:
            segments[leaf] = "tail_bottom_50pct_leaf"
    return segments


def segment_metrics(
    predictions: dict[str, pd.DataFrame], metadata: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for method, frame in predictions.items():
        enriched = add_row_metrics(frame).merge(metadata, on="snapshot_id", validate="one_to_one")
        for dimension in ("history_segment", "label_segment"):
            for segment, group in enriched.groupby(dimension, observed=True):
                rows.append(
                    {
                        "method": method,
                        "dimension": dimension,
                        "segment": segment,
                        "n_users": len(group),
                        "recall@5": float(group["recall@5"].mean()),
                        "ndcg@5": float(group["ndcg@5"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def popularity_hit_rates(
    predictions: dict[str, pd.DataFrame], leaf_segments: dict[int, str]
) -> pd.DataFrame:
    rows = []
    for method, frame in predictions.items():
        counters: dict[str, list[int]] = {}
        for ranking, labels in zip(frame.ranking, frame.labels):
            top5 = set(int(value) for value in ranking[:5])
            for label in labels:
                segment = leaf_segments[int(label)]
                counters.setdefault(segment, [0, 0])
                counters[segment][0] += int(int(label) in top5)
                counters[segment][1] += 1
        for segment, (hits, total) in counters.items():
            rows.append(
                {
                    "method": method,
                    "popularity_segment": segment,
                    "n_target_occurrences": total,
                    "target_hit_rate@5": hits / total,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/grade10_evidence_2026_08",
    )
    args = parser.parse_args()

    final_dir = ROOT / "artifacts/final_model_fold4_august"
    baseline_dir = ROOT / "artifacts/baselines_refresh_2026_08/fold_4_validation_final_exact"
    k1_path = ROOT / "artifacts/data_scaling_refresh_2026_08/neural_k1/fold_3_fixed_eval/train400k/validation_predictions.parquet"
    k2_path = ROOT / "artifacts/data_scaling_refresh_2026_08/neural_k2/fold_3_fixed_eval/train400k/validation_predictions.parquet"

    final_predictions = pd.read_parquet(final_dir / "validation_predictions.parquet")
    baselines = pd.read_parquet(baseline_dir / "predictions.parquet")
    puresvd = baselines.loc[baselines.method == "puresvd_fold_in", ["snapshot_id", "ranking", "labels"]]
    k1 = pd.read_parquet(k1_path)
    k2 = pd.read_parquet(k2_path)

    comparisons = []
    comparisons.extend(
        compare_pair(
            final_predictions,
            puresvd,
            "final_d128_k2",
            "puresvd_full_train",
            args.repeats,
            args.seed,
        )
    )
    comparisons.extend(
        compare_pair(k2, k1, "k2_train400k", "k1_train400k", args.repeats, args.seed)
    )

    snapshot_dir = ROOT / "data/refresh_2026_08/processed/snapshots"
    metadata = load_validation_metadata(
        snapshot_dir / "snapshot_2026-07-01.parquet",
        final_dir / "validation_predictions.parquet",
    )
    predictions = {"final_d128_k2": final_predictions}
    for method in ("most_popular", "weighted_cooccurrence", "puresvd_fold_in"):
        predictions[method] = baselines.loc[
            baselines.method == method, ["snapshot_id", "ranking", "labels"]
        ]
    segments = segment_metrics(predictions, metadata)

    train_dates = ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01"]
    train_paths = [snapshot_dir / f"snapshot_{date}.parquet" for date in train_dates]
    leaf_segments = build_leaf_segments(train_paths, 351)
    popularity = popularity_hit_rates(predictions, leaf_segments)

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(args.output / "paired_bootstrap.csv", index=False)
    segments.to_csv(args.output / "segment_metrics.csv", index=False)
    popularity.to_csv(args.output / "target_popularity_hit_rate.csv", index=False)

    summary = {
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
        "validation_snapshot": "2026-07-01",
        "validation_users": len(final_predictions),
        "comparisons": comparison_frame.to_dict(orient="records"),
        "notes": [
            "Final d=128 and PureSVD use the exact same July validation users.",
            "PureSVD is fitted on up to 768708 February-June train snapshots.",
            "K=2 vs K=1 uses the exact same June validation users, 400000 train examples, seed and eight epochs.",
            "Popularity segments are defined by train-only target frequency over 351 leaf classes.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nPAIRED COMPARISONS")
    print(comparison_frame.to_string(index=False))
    print("\nSEGMENTS")
    print(segments.to_string(index=False))
    print("\nTARGET POPULARITY")
    print(popularity.to_string(index=False))


if __name__ == "__main__":
    main()
