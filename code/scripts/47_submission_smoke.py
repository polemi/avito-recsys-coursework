#!/usr/bin/env python3
"""Run a tiny, data-free end-to-end smoke check for the public submission.

The script creates synthetic parquet rows in a temporary directory, exercises
privacy-safe distinct-user sampling, collates a batch and runs one forward pass.
It never reads project raw data or a production identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import duckdb
import torch

from avito_recsys.metrics import binary_ndcg_at_k, recall_at_k
from avito_recsys.model import MultiInterestTransformer
from avito_recsys.scaling_sampling import load_fixed_evaluation_frame, load_positive_frame
from avito_recsys.training_data import SnapshotDataset, make_collate


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _anonymous_key(index: int) -> str:
    return hashlib.sha256(f"synthetic-user-{index}".encode()).hexdigest()


def _write_synthetic_snapshot(path: Path) -> None:
    con = duckdb.connect()
    try:
        rows = ",".join(
            "(" + ",".join(
                [
                    str(index),
                    f"'{_anonymous_key(index)}'",
                    f"'synthetic-row-{index}'",
                    str(1 + index % 5),
                    str(2 + index % 5),
                ]
            ) + ")"
            for index in range(1, 9)
        )
        con.execute(
            f"""
            COPY (
                SELECT hashed_user_key,
                       TIMESTAMP '2026-01-01' AS snapshot_ts,
                       snapshot_id,
                       [history_leaf, target_leaf]::INTEGER[] AS leaf,
                       [1, 3]::INTEGER[] AS action,
                       [epoch(TIMESTAMP '2025-12-20')::BIGINT,
                        epoch(TIMESTAMP '2025-12-29')::BIGINT]::BIGINT[] AS ts,
                       [target_leaf]::INTEGER[] AS label_leaf,
                       []::INTEGER[] AS masked_leaf,
                       1::INTEGER AS n_labels
                FROM (VALUES {rows}) AS source(
                    synthetic_index, hashed_user_key, snapshot_id,
                    history_leaf, target_leaf
                )
            ) TO '{str(path).replace("'", "''")}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def run_smoke(workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    snapshot = workdir / "synthetic_snapshot.parquet"
    _write_synthetic_snapshot(snapshot)

    train, stats = load_positive_frame(
        [snapshot], cap=4, sampling_unit="users", subset_seed=42
    )
    evaluation = load_fixed_evaluation_frame([snapshot], cap=5)
    assert stats.selected_users == 4
    assert len(train) == 4 and len(evaluation) == 5
    assert "hashed_user_key" not in train.columns and "user_id" not in train.columns
    assert all(HEX64.fullmatch(value) for value in train.snapshot_id.astype(str))
    assert all(HEX64.fullmatch(value) for value in evaluation.snapshot_id.astype(str))

    dataset = SnapshotDataset(train)
    batch = make_collate(n_leaf=8, max_len=4, n_time_buckets=16)(
        [dataset[index] for index in range(len(dataset))]
    )
    torch.manual_seed(42)
    model = MultiInterestTransformer(
        n_leaf=8,
        num_interests=2,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        max_len=4,
        n_time_buckets=16,
    ).eval()
    with torch.no_grad():
        scores, interests, attention = model(
            batch.leaf, batch.action, batch.time_bucket, batch.padding_mask
        )
    assert scores.shape == (4, 9)
    assert interests.shape == (4, 2, 16)
    assert attention.shape == (4, 2, 4)
    assert torch.isfinite(scores).all()

    ranking = [2, 4, 1, 3, 5]
    labels = [2, 3]
    recall = recall_at_k(ranking, labels, 5)
    ndcg = binary_ndcg_at_k(ranking, labels, 5)
    assert recall == 1.0 and 0.0 < ndcg <= 1.0
    return {
        "status": "pass",
        "synthetic_only": True,
        "selected_users": stats.selected_users,
        "selected_rows": stats.selected_rows,
        "evaluation_rows": len(evaluation),
        "snapshot_ids_are_sha256": True,
        "model_output_shape": list(scores.shape),
        "recall_at_5": recall,
        "ndcg_at_5": ndcg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workdir:
        result = run_smoke(args.workdir)
    else:
        with tempfile.TemporaryDirectory(prefix="avito_submission_smoke_") as tmp:
            result = run_smoke(Path(tmp))
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
