#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.model import MultiInterestTransformer
from avito_recsys.training_data import SnapshotDataset, make_collate


def load_snapshot(path: Path, cap: int, seed: int) -> pd.DataFrame:
    limit = "" if cap <= 0 else f"LIMIT {cap}"
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT snapshot_id, user_id, epoch(snapshot_ts)::BIGINT snapshot_epoch,
               leaf, action, ts, label_leaf, masked_leaf
        FROM read_parquet(?)
        ORDER BY hash(snapshot_id || '{seed}')
        {limit}
        """,
        [str(path.resolve())],
    ).fetchdf()
    con.close()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_reduced.json")
    parser.add_argument("--num-interests", type=int, required=True)
    parser.add_argument("--leaf-ids", type=int, nargs="+", required=True)
    parser.add_argument("--top-n", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/processed/vocab_leaf_v90.json")
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    reverse_vocab = {int(value): key for key, value in vocab["leaf"].items()}
    n_leaf = len(vocab["leaf"])
    unknown = sorted(set(args.leaf_ids) - set(reverse_vocab))
    if unknown:
        raise ValueError(f"Unknown leaf ids: {unknown}")
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    frame = load_snapshot(args.snapshot, args.cap, args.seed)
    collate = make_collate(n_leaf, model_config["max_len"], model_config["n_time_buckets"])
    loader = DataLoader(
        SnapshotDataset(frame), batch_size=model_config["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    model = MultiInterestTransformer(
        n_leaf=n_leaf,
        num_interests=args.num_interests,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        ff_dim=model_config["ff_dim"],
        dropout=model_config["dropout"],
        max_len=model_config["max_len"],
        n_time_buckets=model_config["n_time_buckets"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    score_parts = {leaf_id: [] for leaf_id in args.leaf_ids}
    label_parts = {leaf_id: [] for leaf_id in args.leaf_ids}
    eligible_parts = {leaf_id: [] for leaf_id in args.leaf_ids}
    snapshot_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            scores, _, _ = model(batch.leaf, batch.action, batch.time_bucket, batch.padding_mask)
            snapshot_ids.extend(batch.snapshot_ids)
            for leaf_id in args.leaf_ids:
                score_parts[leaf_id].append(scores[:, leaf_id].numpy())
                label_parts[leaf_id].append(batch.labels[:, leaf_id].numpy().astype(bool))
                eligible_parts[leaf_id].append(batch.candidate_mask[:, leaf_id].numpy())

    args.output.mkdir(parents=True, exist_ok=True)
    user_lookup = frame.set_index("snapshot_id").user_id.astype(str).to_dict()
    metric_rows = []
    audience_rows = []
    for leaf_id in args.leaf_ids:
        scores = np.concatenate(score_parts[leaf_id])
        labels = np.concatenate(label_parts[leaf_id])
        eligible = np.concatenate(eligible_parts[leaf_id])
        eligible_index = np.flatnonzero(eligible)
        eligible_scores = scores[eligible]
        eligible_labels = labels[eligible]
        order = eligible_index[np.argsort(eligible_scores)[::-1]]
        positives = int(eligible_labels.sum())
        base_rate = positives / max(1, len(eligible_index))
        for requested_n in args.top_n:
            n = min(requested_n, len(order))
            selected = order[:n]
            hits = int(labels[selected].sum())
            precision = hits / max(1, n)
            metric_rows.append({
                "leaf_id": leaf_id,
                "leaf_path": reverse_vocab[leaf_id],
                "requested_n": requested_n,
                "actual_n": n,
                "n_eligible_users": len(eligible_index),
                "n_positive_users": positives,
                "base_rate": base_rate,
                "precision_at_n": precision,
                "recall_at_n": hits / max(1, positives),
                "lift_at_n": precision / base_rate if base_rate > 0 else np.nan,
            })
        max_n = min(max(args.top_n), len(order))
        for rank, row_index in enumerate(order[:max_n], start=1):
            snapshot_id = snapshot_ids[row_index]
            audience_rows.append({
                "leaf_id": leaf_id,
                "leaf_path": reverse_vocab[leaf_id],
                "rank": rank,
                "snapshot_id": snapshot_id,
                "user_id": user_lookup[snapshot_id],
                "score": float(scores[row_index]),
                "label": int(labels[row_index]),
            })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "reverse_ranking_metrics.csv", index=False)
    pd.DataFrame(audience_rows).to_parquet(args.output / "ranked_audiences.parquet", index=False)
    metadata = {
        "snapshot": str(args.snapshot),
        "checkpoint": str(args.checkpoint),
        "num_interests": args.num_interests,
        "leaf_ids": args.leaf_ids,
        "n_sampled_users": len(frame),
        "cap": args.cap,
        "warning": "If cap > 0, results are a pipeline smoke test, not CRM evidence.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
