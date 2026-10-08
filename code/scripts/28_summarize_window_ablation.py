#!/usr/bin/env python3
"""Summarize window runs and compute paired uncertainty on the common audience."""

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


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PREDICTIONS_PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def row_metrics(path: Path, name: str, method: str | None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if method is not None:
        if "method" not in frame:
            raise ValueError(f"{path} has no method column")
        frame = frame.loc[frame["method"] == method].copy()
    frame = frame[["snapshot_id", "ranking", "labels"]].copy()
    frame = pseudonymize_snapshot_ids(frame, copy=False)
    frame[f"recall@5_{name}"] = [
        recall_at_k(ranking, labels, 5)
        for ranking, labels in zip(frame["ranking"], frame["labels"])
    ]
    frame[f"ndcg@5_{name}"] = [
        binary_ndcg_at_k(ranking, labels, 5)
        for ranking, labels in zip(frame["ranking"], frame["labels"])
    ]
    return frame[["snapshot_id", f"recall@5_{name}", f"ndcg@5_{name}"]]


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
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--reference", default="h90")
    parser.add_argument("--method")
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = dict(args.run)
    if args.reference not in runs:
        raise ValueError(f"Reference {args.reference!r} absent from runs")
    scored = {
        name: row_metrics(path, name, args.method)
        for name, path in runs.items()
    }
    merged: pd.DataFrame | None = None
    for frame in scored.values():
        merged = frame if merged is None else merged.merge(
            frame, on="snapshot_id", validate="one_to_one"
        )
    assert merged is not None
    expected = {len(frame) for frame in scored.values()}
    if len(expected) != 1 or len(merged) != next(iter(expected)):
        raise RuntimeError(
            f"Audience mismatch: {[len(frame) for frame in scored.values()]}, "
            f"intersection={len(merged)}"
        )

    summary = pd.DataFrame(
        [
            {
                "window": name,
                "n_users": len(merged),
                "recall@5": float(merged[f"recall@5_{name}"].mean()),
                "ndcg@5": float(merged[f"ndcg@5_{name}"].mean()),
            }
            for name in runs
        ]
    ).sort_values("window")
    comparisons: list[dict[str, object]] = []
    for name in runs:
        if name == args.reference:
            continue
        for metric in ("recall@5", "ndcg@5"):
            delta = (
                merged[f"{metric}_{name}"].to_numpy()
                - merged[f"{metric}_{args.reference}"].to_numpy()
            )
            comparisons.append(
                {
                    "metric": metric,
                    "challenger": name,
                    "reference": args.reference,
                    "n_users": len(merged),
                    **bootstrap(delta, args.repeats, args.seed),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "paired_bootstrap_vs_reference.csv", index=False
    )
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "runs": {name: str(path) for name, path in runs.items()},
                "method": args.method,
                "reference": args.reference,
                "repeats": args.repeats,
                "seed": args.seed,
                "selection_metric": "recall@5",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
