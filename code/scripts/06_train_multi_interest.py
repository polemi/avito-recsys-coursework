#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.evaluation import RankingAccumulator
from avito_recsys.model import (
    MultiInterestTransformer,
    attention_diversity_penalty,
    diversity_penalty,
    mean_off_diagonal_cosine,
)
from avito_recsys.training_data import SnapshotDataset, make_collate
from avito_recsys.scaling_sampling import (
    load_fixed_evaluation_frame,
    load_positive_frame,
    validate_temporal_split,
)
from avito_recsys.snapshot_identity import (
    SNAPSHOT_ID_CONTRACT,
    pseudonymize_snapshot_id,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, n_leaf: int):
    model.eval()
    accumulator = RankingAccumulator()
    cosine = []
    predictions = []
    for batch in loader:
        leaf = batch.leaf.to(device)
        action = batch.action.to(device)
        time_bucket = batch.time_bucket.to(device)
        padding_mask = batch.padding_mask.to(device)
        scores, interests, _ = model(leaf, action, time_bucket, padding_mask)
        scores = scores.masked_fill(~batch.candidate_mask.to(device), -torch.inf)
        top20 = scores.topk(20, dim=1).indices.cpu().numpy()
        cosine.append(mean_off_diagonal_cosine(interests))
        labels = batch.labels.numpy()
        for row_id, ranking in enumerate(top20):
            positive = np.flatnonzero(labels[row_id]).astype(int).tolist()
            ranked = ranking.astype(int).tolist()
            accumulator.add(ranked, positive)
            predictions.append(
                {
                    "snapshot_id": pseudonymize_snapshot_id(batch.snapshot_ids[row_id]),
                    "ranking": ranked,
                    "labels": positive,
                }
            )
    return accumulator.result(n_leaf), float(np.mean(cosine)), predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-interests", type=int, required=True)
    parser.add_argument("--train-cap", type=int, default=5000)
    parser.add_argument("--eval-cap", type=int, default=2000)
    parser.add_argument(
        "--test-cap", type=int,
        help="Reported test audience cap; defaults to --eval-cap when omitted.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--min-epochs", type=int, default=1,
        help="Minimum number of epochs before early stopping may end training.",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=0,
        help=(
            "Stop after this many consecutive validation epochs without a new best "
            "Recall@5. Zero disables early stopping."
        ),
    )
    parser.add_argument(
        "--train-sampling-unit", choices=["rows", "users"], default="rows",
        help=(
            "Unit constrained by --train-cap. 'users' selects a deterministic prefix "
            "of distinct positive users and keeps every positive snapshot row for them."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subset-seed", type=int,
        help="Seed for deterministic train membership; defaults to --seed for compatibility.",
    )
    parser.add_argument(
        "--model-seed", type=int,
        help="Seed for initialization and minibatch order; defaults to --seed for compatibility.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="CPU worker processes used to prepare training batches.",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--unlock-final-holdout", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path)
    parser.add_argument("--train-snapshots-dir", type=Path)
    parser.add_argument("--eval-snapshots-dir", type=Path)
    parser.add_argument("--vocab", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 1 <= args.min_epochs <= args.epochs:
        parser.error("--min-epochs must be between 1 and --epochs")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")
    subset_seed = args.seed if args.subset_seed is None else args.subset_seed
    model_seed = args.seed if args.model_seed is None else args.model_seed
    test_cap = args.eval_cap if args.test_cap is None else args.test_cap
    set_seed(model_seed)

    data_config = json.loads(args.data_config.read_text(encoding="utf-8"))
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    fold = next(item for item in data_config["folds"] if int(item["fold"]) == args.fold)
    validate_temporal_split(
        fold["train_snapshots"],
        fold["validation_snapshot"],
        int(data_config["target_days"]),
    )
    processed = ROOT / data_config.get("processed_dir", "data/processed")
    snapshots = args.snapshots_dir or (processed / "snapshots")
    train_snapshots = args.train_snapshots_dir or snapshots
    eval_snapshots = args.eval_snapshots_dir or snapshots
    train_paths = [
        train_snapshots / f"snapshot_{date}.parquet"
        for date in fold["train_snapshots"]
    ]
    validation_path = eval_snapshots / f"snapshot_{fold['validation_snapshot']}.parquet"
    vocab_path = args.vocab or (processed / data_config.get("vocab_file", "vocab_leaf_v90.json"))
    n_leaf = len(json.loads(vocab_path.read_text())["leaf"])

    train_frame, train_sampling = load_positive_frame(
        train_paths,
        cap=args.train_cap,
        sampling_unit=args.train_sampling_unit,
        subset_seed=subset_seed,
    )
    # Evaluation membership must not change across training seeds or model variants.
    validation_frame = load_fixed_evaluation_frame([validation_path], args.eval_cap)
    collate = make_collate(n_leaf, model_config["max_len"], model_config["n_time_buckets"])
    train_loader_options = {
        "batch_size": model_config["batch_size"],
        "shuffle": True,
        "collate_fn": collate,
        "num_workers": args.num_workers,
    }
    if args.num_workers > 0:
        train_loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(SnapshotDataset(train_frame), **train_loader_options)
    validation_loader = DataLoader(
        SnapshotDataset(validation_frame), batch_size=model_config["batch_size"], shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
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
        use_action_embedding=model_config.get("use_action_embedding", True),
        use_time_embedding=model_config.get("use_time_embedding", True),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"]
    )
    positive_weight = torch.tensor(model_config["positive_weight"], device=device)
    learning_curve = []
    best_state = None
    best_epoch = None
    best_recall_at_5 = -float("inf")
    epochs_without_improvement = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            leaf = batch.leaf.to(device)
            action = batch.action.to(device)
            time_bucket = batch.time_bucket.to(device)
            padding_mask = batch.padding_mask.to(device)
            labels = batch.labels.to(device)
            candidates = batch.candidate_mask.to(device)
            scores, interests, attention = model(leaf, action, time_bucket, padding_mask)
            raw = F.binary_cross_entropy_with_logits(
                scores, labels, pos_weight=positive_weight, reduction="none"
            )
            ranking_loss = raw[candidates].mean()
            loss = (
                ranking_loss
                + model_config["attention_diversity_weight"]
                * attention_diversity_penalty(attention)
                + model_config["interest_diversity_weight"]
                * diversity_penalty(interests)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics, cosine, _ = evaluate(model, validation_loader, device, n_leaf)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "interest_cosine": cosine, **metrics}
        learning_curve.append(row)
        if metrics["recall@5"] > best_recall_at_5:
            best_recall_at_5 = metrics["recall@5"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(json.dumps(row), flush=True)
        if (
            args.early_stopping_patience > 0
            and epoch + 1 >= args.min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epochs_completed": epoch + 1,
                        "selected_epoch": best_epoch,
                        "patience": args.early_stopping_patience,
                    }
                ),
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was produced")
    model.load_state_dict(best_state)
    metrics, cosine, predictions = evaluate(model, validation_loader, device, n_leaf)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(learning_curve).to_csv(args.output / "learning_curve.csv", index=False)
    pd.DataFrame(predictions).to_parquet(args.output / "validation_predictions.parquet", index=False)
    torch.save(model.state_dict(), args.output / "checkpoint.pt")
    report = {
        "experiment": "multi_interest_smoke",
        "fold": args.fold,
        "validation_snapshot": fold["validation_snapshot"],
        "train_snapshots": fold["train_snapshots"],
        "train_source_snapshots": fold.get("source_train_snapshots", fold["train_snapshots"]),
        "num_interests": args.num_interests,
        # Kept for compatibility with old result readers: this is actual train rows.
        "train_cap": len(train_frame),
        "train_sampling_unit": train_sampling.sampling_unit,
        "train_cap_requested": train_sampling.requested_cap,
        "train_selected_rows": train_sampling.selected_rows,
        "train_selected_distinct_users": train_sampling.selected_users,
        "eval_cap": len(validation_frame),
        "epochs": args.epochs,
        "epochs_completed": len(learning_curve),
        "min_epochs": args.min_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_triggered": len(learning_curve) < args.epochs,
        "selected_epoch": best_epoch,
        "checkpoint_selection_metric": "validation_recall_at_5",
        "seed": args.seed,
        "subset_seed": subset_seed,
        "model_seed": model_seed,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "interest_cosine": cosine,
        "snapshot_id_contract": SNAPSHOT_ID_CONTRACT,
        **metrics,
    }
    if args.evaluate_test:
        if fold["role"] == "final_holdout" and not args.unlock_final_holdout:
            raise RuntimeError(
                "Final holdout is locked. Freeze the model/config first, then pass "
                "--unlock-final-holdout for the single final run."
            )
        validate_temporal_split(
            [fold["validation_snapshot"]],
            fold["test_snapshot"],
            int(data_config["target_days"]),
        )
        test_path = eval_snapshots / f"snapshot_{fold['test_snapshot']}.parquet"
        test_frame = load_fixed_evaluation_frame([test_path], test_cap)
        test_loader = DataLoader(
            SnapshotDataset(test_frame), batch_size=model_config["batch_size"], shuffle=False,
            collate_fn=collate, num_workers=0,
        )
        test_metrics, test_cosine, test_predictions = evaluate(model, test_loader, device, n_leaf)
        pd.DataFrame(test_predictions).to_parquet(
            args.output / "test_predictions.parquet", index=False
        )
        report["test_snapshot"] = fold["test_snapshot"]
        report["test_cap"] = len(test_frame)
        report["test_interest_cosine"] = test_cosine
        report["test_metrics"] = test_metrics
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(
        json.dumps({"data": data_config, "model": model_config, "arguments": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
