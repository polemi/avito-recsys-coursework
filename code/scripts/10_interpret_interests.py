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

from avito_recsys.model import MultiInterestTransformer, mean_off_diagonal_cosine
from avito_recsys.training_data import SnapshotDataset, make_collate


def load_rows(snapshot: Path, cap: int, seed: int) -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT snapshot_id, epoch(snapshot_ts)::BIGINT snapshot_epoch,
               leaf, action, ts, label_leaf, masked_leaf
        FROM read_parquet(?)
        WHERE n_labels > 0
        ORDER BY hash(snapshot_id || '{seed}')
        LIMIT {cap}
        """,
        [str(snapshot.resolve())],
    ).fetchdf()
    con.close()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model_reduced.json")
    parser.add_argument("--num-interests", type=int, required=True)
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/processed/vocab_leaf_v90.json")
    args = parser.parse_args()

    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    reverse_vocab = {int(value): key for key, value in vocab["leaf"].items()}
    n_leaf = len(reverse_vocab)
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    frame = load_rows(args.snapshot, args.cap, args.seed)
    collate = make_collate(n_leaf, config["max_len"], config["n_time_buckets"])
    loader = DataLoader(SnapshotDataset(frame), batch_size=config["batch_size"], collate_fn=collate)
    model = MultiInterestTransformer(
        n_leaf=n_leaf, num_interests=args.num_interests, d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"], ff_dim=config["ff_dim"],
        dropout=config["dropout"], max_len=config["max_len"],
        n_time_buckets=config["n_time_buckets"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    examples = []
    winner_counts = np.zeros(args.num_interests, dtype=np.int64)
    cosines = []
    with torch.no_grad():
        for batch in loader:
            interests, weights = model.encode(
                batch.leaf, batch.action, batch.time_bucket, batch.padding_mask
            )
            per_interest_scores = torch.einsum(
                "bkd,nd->bkn", interests, model.leaf_embedding.weight
            ) / np.sqrt(model.d_model)
            candidate_mask = batch.candidate_mask.unsqueeze(1)
            per_interest_scores = per_interest_scores.masked_fill(~candidate_mask, -torch.inf)
            aggregate_scores, winners = per_interest_scores.max(dim=1)
            top_recommendations = aggregate_scores.topk(5, dim=1).indices
            for row in range(len(batch.snapshot_ids)):
                for leaf_id in top_recommendations[row]:
                    winner_counts[int(winners[row, leaf_id])] += 1
                valid_positions = np.flatnonzero(~batch.padding_mask[row].numpy())
                history_leaf = batch.leaf[row].numpy()
                interest_details = []
                for interest_id in range(args.num_interests):
                    top_leaf = per_interest_scores[row, interest_id].topk(5).indices.tolist()
                    attention_order = valid_positions[
                        np.argsort(weights[row, interest_id, valid_positions].numpy())[::-1][:5]
                    ]
                    interest_details.append({
                        "interest_id": interest_id,
                        "top_scored_leaf": [reverse_vocab[int(x)] for x in top_leaf],
                        "top_attended_history_leaf": [
                            reverse_vocab[int(history_leaf[position])] for position in attention_order
                        ],
                        "top_attention_weight": [
                            float(weights[row, interest_id, position]) for position in attention_order
                        ],
                    })
                labels = np.flatnonzero(batch.labels[row].numpy()).tolist()
                examples.append({
                    "snapshot_id": batch.snapshot_ids[row],
                    "labels": [reverse_vocab[int(x)] for x in labels],
                    "recommendations": [reverse_vocab[int(x)] for x in top_recommendations[row]],
                    "interests": interest_details,
                })
            cosines.append(mean_off_diagonal_cosine(interests))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "interest_examples.jsonl").open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    usage = pd.DataFrame({
        "interest_id": np.arange(args.num_interests),
        "winning_recommendations": winner_counts,
        "usage_share": winner_counts / max(1, winner_counts.sum()),
    })
    usage.to_csv(args.output / "interest_usage.csv", index=False)
    report = {
        "n_examples": len(examples),
        "num_interests": args.num_interests,
        "mean_interest_cosine": float(np.mean(cosines)),
        "collapsed_at_cosine_0_95": bool(float(np.mean(cosines)) >= 0.95),
        "warning": "Small-checkpoint outputs validate interpretation code only; use the selected full model for coursework conclusions.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(usage.to_string(index=False))


if __name__ == "__main__":
    main()
