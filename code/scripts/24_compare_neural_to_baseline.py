#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.metrics import binary_ndcg_at_k, recall_at_k
from avito_recsys.snapshot_identity import pseudonymize_snapshot_ids


def row_scores(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    result = frame[["snapshot_id", "ranking", "labels"]].copy()
    # Legacy predictions contain raw IDs; current artifacts contain v1 SHA-256
    # IDs.  Normalize both before the paired one-to-one join.
    result = pseudonymize_snapshot_ids(result, copy=False)
    result["method"] = method
    result["recall@5"] = [
        recall_at_k(ranking, labels, 5)
        for ranking, labels in zip(result.ranking, result.labels)
    ]
    result["ndcg@5"] = [
        binary_ndcg_at_k(ranking, labels, 5)
        for ranking, labels in zip(result.ranking, result.labels)
    ]
    return result[["snapshot_id", "method", "recall@5", "ndcg@5"]]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--baseline-method", default="puresvd_fold_in")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    neural = row_scores(pd.read_parquet(args.neural), "multi_interest")
    baselines = pd.read_parquet(args.baselines)
    baseline = baselines.loc[baselines.method == args.baseline_method]
    baseline = row_scores(baseline, args.baseline_method)
    paired = neural.merge(
        baseline,
        on="snapshot_id",
        how="inner",
        suffixes=("_neural", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(neural):
        raise RuntimeError(
            f"Audience mismatch: neural={len(neural)}, paired={len(paired)}"
        )

    rows = []
    for metric in ["recall@5", "ndcg@5"]:
        delta = paired[f"{metric}_neural"].to_numpy() - paired[f"{metric}_baseline"].to_numpy()
        rows.append({
            "metric": metric,
            "reference": "multi_interest",
            "challenger": args.baseline_method,
            "n_users": len(paired),
            "reference_mean": float(paired[f"{metric}_neural"].mean()),
            "challenger_mean": float(paired[f"{metric}_baseline"].mean()),
            **paired_bootstrap(delta, args.repeats, args.seed),
        })

    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "paired_bootstrap.csv", index=False)
    metadata = {
        "neural_predictions": str(args.neural),
        "baseline_predictions": str(args.baselines),
        "baseline_method": args.baseline_method,
        "repeats": args.repeats,
        "seed": args.seed,
        "note": (
            "This paired comparison uses the locally persisted fold-3 validation checkpoint. "
            "It is supporting robustness evidence and must not be relabeled as the later GPU d=128 run."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
