#!/usr/bin/env python3
"""Paired comparison of two neural scaling checkpoints on one fixed audience."""

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


def score_rows(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["snapshot_id", "ranking", "labels"]].copy()
    frame = pseudonymize_snapshot_ids(frame, copy=False)
    frame[f"recall@5_{suffix}"] = [
        recall_at_k(ranking, labels, 5)
        for ranking, labels in zip(frame["ranking"], frame["labels"])
    ]
    frame[f"ndcg@5_{suffix}"] = [
        binary_ndcg_at_k(ranking, labels, 5)
        for ranking, labels in zip(frame["ranking"], frame["labels"])
    ]
    return frame[["snapshot_id", f"recall@5_{suffix}", f"ndcg@5_{suffix}"]]


def bootstrap(delta: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for idx in range(repeats):
        sample = rng.integers(0, len(delta), size=len(delta))
        means[idx] = delta[sample].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_bootstrap_le_zero": float((means <= 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--reference-name", default="full")
    parser.add_argument("--challenger-name", default="1200k")
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = score_rows(args.reference, "reference")
    challenger = score_rows(args.challenger, "challenger")
    paired = reference.merge(challenger, on="snapshot_id", validate="one_to_one")
    if len(paired) != len(reference) or len(paired) != len(challenger):
        raise RuntimeError(
            f"Audience mismatch: reference={len(reference)}, "
            f"challenger={len(challenger)}, paired={len(paired)}"
        )

    rows: list[dict[str, object]] = []
    for metric in ("recall@5", "ndcg@5"):
        delta = (
            paired[f"{metric}_reference"].to_numpy()
            - paired[f"{metric}_challenger"].to_numpy()
        )
        rows.append(
            {
                "metric": metric,
                "reference": args.reference_name,
                "challenger": args.challenger_name,
                "n_users": len(paired),
                "reference_mean": float(paired[f"{metric}_reference"].mean()),
                "challenger_mean": float(paired[f"{metric}_challenger"].mean()),
                **bootstrap(delta, args.repeats, args.seed),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "paired_bootstrap.csv", index=False)
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "reference_predictions": str(args.reference),
                "challenger_predictions": str(args.challenger),
                "repeats": args.repeats,
                "seed": args.seed,
                "audience_is_paired": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
