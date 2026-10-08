#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/data_v90.json")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_reduced.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--interests", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--train-cap", type=int, default=10000)
    parser.add_argument("--eval-cap", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/k_ablation_reduced")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in args.interests:
        run_dir = args.output / f"k{k}"
        command = [
            sys.executable,
            str(ROOT / "scripts/06_train_multi_interest.py"),
            "--data-config", str(args.data_config),
            "--model-config", str(args.model_config),
            "--fold", str(args.fold),
            "--num-interests", str(k),
            "--train-cap", str(args.train_cap),
            "--eval-cap", str(args.eval_cap),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed),
            "--output", str(run_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        rows.append(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))

    results = pd.DataFrame(rows).sort_values("num_interests")
    results.to_csv(args.output / "k_ablation_summary.csv", index=False)
    columns = [
        "num_interests", "selected_epoch", "recall@5", "recall@10", "recall@20",
        "ndcg@5", "coverage", "interest_cosine", "runtime_seconds", "n_parameters",
    ]
    table = results[columns].to_markdown(index=False, floatfmt=".4f")
    note = (
        "# Reduced K ablation\n\n"
        f"Fold: {args.fold}; train cap: {args.train_cap}; validation cap: {args.eval_cap}; "
        f"epochs: {args.epochs}; seed: {args.seed}.\n\n"
        "This is a model-selection run, not the final holdout evaluation.\n\n"
        + table
        + "\n"
    )
    (args.output / "README.md").write_text(note, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
