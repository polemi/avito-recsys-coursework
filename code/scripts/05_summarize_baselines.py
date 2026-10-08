#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "artifacts" / "baselines"
    output = root / "reports" / "baselines"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    pattern = re.compile(r"fold_[0-3]_(validation|test)$")
    for path in sorted(source.glob("fold_*_*/metrics.json")):
        if not pattern.match(path.parent.name):
            continue
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_by_fold.csv", index=False)
    test = frame[frame["eval_split"] == "test"]
    validation = frame[frame["eval_split"] == "validation"]
    metrics = ["recall@5", "ndcg@5", "recall@10", "ndcg@10", "coverage"]
    test_summary = test.groupby("method")[metrics].agg(["mean", "std"])
    validation_summary = validation.groupby("method")[metrics].agg(["mean", "std"])
    test_summary.to_csv(output / "test_summary.csv")
    validation_summary.to_csv(output / "validation_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, group in frame.groupby("method"):
        group = group.sort_values(["eval_snapshot", "eval_split"])
        ax.plot(
            range(len(group)), group["recall@5"], marker="o", label=method
        )
    ordered = frame.sort_values(["eval_snapshot", "eval_split"])[["eval_snapshot", "eval_split"]].drop_duplicates()
    ax.set_xticks(
        range(len(ordered)),
        [f"{date}\n{split}" for date, split in ordered.itertuples(index=False)],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Recall@5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "baseline_recall5_over_time.png", dpi=160)
    plt.close(fig)

    display = test.groupby("method").agg(
        recall5_mean=("recall@5", "mean"),
        recall5_std=("recall@5", "std"),
        ndcg5_mean=("ndcg@5", "mean"),
        recall10_mean=("recall@10", "mean"),
        coverage_mean=("coverage", "mean"),
    ).reset_index()
    lines = [
        "# Бейзлайны v90 — промежуточный отчёт",
        "",
        "Оценка выполнена на 100 000 positive snapshots каждого доступного test-фолда. Final holdout 2026-06-08 не открыт.",
        "",
        "| Метод | Recall@5 mean±std | NDCG@5 mean | Recall@10 mean | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.recall5_mean:.4f} ± {row.recall5_std:.4f} | "
            f"{row.ndcg5_mean:.4f} | {row.recall10_mean:.4f} | {row.coverage_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "PureSVD использует 64 компоненты и фиксированный train-cap 300 000 snapshots. Это сильный локальный baseline, но перед финальным сравнением нужна sensitivity-кривая по train-cap.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
