#!/usr/bin/env python
"""Calculate descriptive segments for the final full-pool d128 predictions."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "artifacts/runpod/d128_full2m_2026_10/extracted/output/test_predictions.parquet",
    )
    parser.add_argument(
        "--evaluation-snapshot",
        type=Path,
        default=(
            ROOT
            / "reports/kaggle/expanded_user_scaling_2026_10/scaling_data/"
            "snapshots/snapshot_2026-07-01.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/runpod_full2m_exact100k_comparison_2026_10_08",
    )
    args = parser.parse_args()

    predictions = pseudonymize_snapshot_ids(pd.read_parquet(args.predictions), copy=False)
    source = pd.read_parquet(
        args.evaluation_snapshot,
        columns=["snapshot_id", "leaf", "label_leaf", "n_labels"],
    )
    source = source.loc[source.n_labels > 0].copy()
    source = pseudonymize_snapshot_ids(source, copy=False)
    frame = predictions.merge(
        source,
        on="snapshot_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )
    if frame.leaf.isna().any():
        raise RuntimeError("Some prediction identifiers are absent from the evaluation snapshot")
    if any(
        list(map(int, left)) != list(map(int, right))
        for left, right in zip(frame.labels, frame.label_leaf)
    ):
        raise RuntimeError("Prediction labels do not match the evaluation snapshot")

    frame["history_length"] = frame.leaf.map(len)
    frame["target_count"] = frame.labels.map(len)
    frame["recall@5"] = [
        recall_at_k(ranking, labels, 5)
        for ranking, labels in zip(frame.ranking, frame.labels)
    ]
    frame["ndcg@5"] = [
        binary_ndcg_at_k(ranking, labels, 5)
        for ranking, labels in zip(frame.ranking, frame.labels)
    ]
    frame["history_group"] = pd.cut(
        frame.history_length,
        bins=[4, 20, 50, 128],
        labels=["5-20 events", "21-50 events", "51-128 events"],
    )
    frame["target_group"] = np.where(frame.target_count == 1, "one target", "multiple targets")

    rows: list[dict[str, object]] = []
    for dimension, column in (
        ("history_length", "history_group"),
        ("target_count", "target_group"),
    ):
        grouped = frame.groupby(column, observed=True, sort=False)
        for group, part in grouped:
            rows.append(
                {
                    "dimension": dimension,
                    "segment": str(group),
                    "n_users": int(len(part)),
                    "recall@5": float(part["recall@5"].mean()),
                    "ndcg@5": float(part["ndcg@5"].mean()),
                }
            )
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "segment_metrics.csv", index=False)
    (args.output / "segment_metrics.json").write_text(
        json.dumps(result.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
