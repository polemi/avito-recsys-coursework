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

from avito_recsys.metrics import recall_at_k


def paired_bootstrap(delta: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sample = rng.integers(0, len(delta), size=len(delta))
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
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-k", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = []
    summary_rows = []
    for run_dir in sorted(args.artifacts.glob("k*"), key=lambda p: int(p.name[1:])):
        k = int(run_dir.name[1:])
        predictions = pd.read_parquet(run_dir / "validation_predictions.parquet")
        predictions["recall@5"] = [
            recall_at_k(ranking, labels, 5)
            for ranking, labels in zip(predictions.ranking, predictions.labels)
        ]
        frames.append(predictions[["snapshot_id", "recall@5"]].assign(num_interests=k))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        summary_rows.append(
            {key: metrics[key] for key in [
                "num_interests", "selected_epoch", "recall@5", "ndcg@5",
                "recall@10", "recall@20", "coverage", "interest_cosine",
                "runtime_seconds", "n_parameters",
            ]}
        )

    scores = pd.concat(frames, ignore_index=True).pivot(
        index="snapshot_id", columns="num_interests", values="recall@5"
    )
    if args.reference_k not in scores.columns:
        raise ValueError(f"Reference K={args.reference_k} is absent")

    comparisons = []
    for challenger_k in scores.columns:
        if challenger_k == args.reference_k:
            continue
        paired = scores[[args.reference_k, challenger_k]].dropna()
        delta = paired[args.reference_k].to_numpy() - paired[challenger_k].to_numpy()
        comparisons.append({
            "reference_k": args.reference_k,
            "challenger_k": int(challenger_k),
            "n_users": len(paired),
            **paired_bootstrap(delta, args.repeats, args.seed + int(challenger_k)),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values("num_interests")
    uncertainty = pd.DataFrame(comparisons).sort_values("challenger_k")
    summary.to_csv(args.output / "k_ablation_8epoch_summary.csv", index=False)
    uncertainty.to_csv(args.output / "paired_bootstrap_recall5.csv", index=False)
    note = (
        "# K-ablation on refreshed data\n\n"
        "All K variants use the same fold, train cap, validation users, seed and eight-epoch budget. "
        "The final holdout is untouched.\n\n"
        "## Validation metrics\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Paired bootstrap: reference K minus challenger\n\n"
        + uncertainty.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )
    (args.output / "README.md").write_text(note, encoding="utf-8")
    print(note)


if __name__ == "__main__":
    main()
