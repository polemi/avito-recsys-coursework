#!/usr/bin/env python3
"""Resume-safe orchestrator for the Nov--May data-scaling curve.

The script intentionally launches each point in a separate process.  This keeps
peak RAM bounded and guarantees that model/data state is not reused between
points.  A completed point is skipped only when its saved configuration matches
the requested cap, seed, epochs, fold, and evaluation size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_POINTS = (
    ("200k_users", 200_000),
    ("400k_users", 400_000),
    ("600k_users", 600_000),
    ("800k_users", 800_000),
    ("1200k_users", 1_200_000),
    ("1600k_users", 1_600_000),
)


def parse_point(value: str) -> tuple[str, int]:
    try:
        label, raw_cap = value.split(":", 1)
        cap = int(raw_cap)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected LABEL:CAP, for example 1600k:1600000") from exc
    if not label or cap < 0:
        raise argparse.ArgumentTypeError("LABEL must be non-empty and CAP must be >= 0")
    return label, cap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_root: Path, verify_hashes: bool) -> dict:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("train", "validation", "test", "vocab", "config"):
        item = manifest["files"][key]
        path = data_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["size_bytes"]):
            raise RuntimeError(f"Size mismatch for {path}")
        if verify_hashes and sha256(path) != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
    return manifest


def completed_metrics(
    path: Path,
    *,
    cap: int,
    subset_seed: int,
    model_seed: int,
    epochs: int,
    min_epochs: int,
    early_stopping_patience: int,
    fold: int,
    eval_cap: int,
    test_cap: int,
) -> dict | None:
    if not path.is_file():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if (
        metrics.get("train_sampling_unit") != "users"
        or int(metrics.get("train_cap_requested", -1)) != cap
        or int(metrics.get("subset_seed", -1)) != subset_seed
        or int(metrics.get("model_seed", -1)) != model_seed
        or int(metrics.get("epochs", -1)) != epochs
        or int(metrics.get("min_epochs", -1)) != min_epochs
        or int(metrics.get("early_stopping_patience", -1)) != early_stopping_patience
        or int(metrics.get("fold", -1)) != fold
        or int(metrics.get("eval_cap", -1)) != eval_cap
        or int(metrics.get("test_cap", -1)) != test_cap
        or "test_metrics" not in metrics
    ):
        return None
    if cap > 0 and int(metrics.get("train_selected_distinct_users", -1)) != cap:
        return None
    return metrics


def write_summary(rows: list[dict], output_root: Path) -> None:
    ordered = [
        "label", "requested_user_cap", "train_selected_distinct_users", "train_selected_rows",
        "subset_seed", "model_seed", "epochs", "epochs_completed", "selected_epoch",
        "validation_recall@5", "validation_ndcg@5", "validation_coverage",
        "test_recall@5", "test_ndcg@5", "test_coverage", "runtime_seconds", "device",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "scaling_summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "scaling_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--point", action="append", type=parse_point, dest="points")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-interests", type=int, default=2)
    parser.add_argument("--eval-cap", type=int, default=20_000)
    parser.add_argument("--test-cap", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cuda")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    manifest = load_manifest(data_root, args.verify_hashes)
    privacy = manifest.get("pseudonymization", {})
    if (
        int(privacy.get("version", -1)) != 2
        or privacy.get("key_in_package") is not False
        or privacy.get("raw_user_id_in_package") is not False
        or privacy.get("raw_snapshot_id_in_package") is not False
    ):
        raise RuntimeError("Refusing to launch: package pseudonymization audit is missing or unsafe")
    train_positive_rows = int(manifest.get("train_positive_rows", 0))
    train_distinct_users = int(manifest.get("train_distinct_hashed_users", 0))
    if train_positive_rows <= 0 or train_distinct_users <= 0:
        raise RuntimeError("Manifest has an empty or missing train-pool count")
    if int(manifest["validation_selected_rows"]) != args.eval_cap:
        raise RuntimeError("Evaluation cap differs from the fixed validation audience")
    if int(manifest["test_selected_rows"]) != args.test_cap:
        raise RuntimeError("Test cap differs from the fixed July audience")

    data_config = data_root / "config.json"
    model_config = project / "configs/model_reduced.json"
    training_script = project / "scripts/06_train_multi_interest.py"
    for path in (data_config, model_config, training_script):
        if not path.is_file():
            raise FileNotFoundError(path)

    points = args.points or [
        *[(label, cap) for label, cap in DEFAULT_POINTS if cap < train_distinct_users],
        ("full", 0),
    ]
    if len({label for label, _ in points}) != len(points):
        raise ValueError("Point labels must be unique")
    oversized = [(label, cap) for label, cap in points if cap > train_distinct_users]
    if oversized:
        raise ValueError(
            f"User caps exceed the available {train_distinct_users:,} distinct users: {oversized}"
        )
    summary: list[dict] = []
    for label, cap in points:
        point_output = output_root / f"modelseed{args.model_seed}" / f"train{label}"
        metrics_path = point_output / "metrics.json"
        metrics = completed_metrics(
            metrics_path,
            cap=cap,
            subset_seed=args.subset_seed,
            model_seed=args.model_seed,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            early_stopping_patience=args.early_stopping_patience,
            fold=args.fold,
            eval_cap=args.eval_cap,
            test_cap=args.test_cap,
        )
        if metrics is None:
            command = [
                sys.executable,
                str(training_script),
                "--data-config", str(data_config),
                "--model-config", str(model_config),
                "--fold", str(args.fold),
                "--num-interests", str(args.num_interests),
                "--train-cap", str(cap),
                "--train-sampling-unit", "users",
                "--eval-cap", str(args.eval_cap),
                "--test-cap", str(args.test_cap),
                "--epochs", str(args.epochs),
                "--min-epochs", str(args.min_epochs),
                "--early-stopping-patience", str(args.early_stopping_patience),
                "--seed", str(args.model_seed),
                "--subset-seed", str(args.subset_seed),
                "--model-seed", str(args.model_seed),
                "--device", args.device,
                "--evaluate-test",
                "--snapshots-dir", str(data_root / "snapshots"),
                "--vocab", str(data_root / "vocab.json"),
                "--output", str(point_output),
            ]
            print(json.dumps({"point": label, "command": command}, ensure_ascii=False), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=project, check=True)
            metrics = completed_metrics(
                metrics_path,
                cap=cap,
                subset_seed=args.subset_seed,
                model_seed=args.model_seed,
                epochs=args.epochs,
                min_epochs=args.min_epochs,
                early_stopping_patience=args.early_stopping_patience,
                fold=args.fold,
                eval_cap=args.eval_cap,
                test_cap=args.test_cap,
            )
            if metrics is None:
                raise RuntimeError(f"Point {label} finished without matching metrics")
        else:
            print(f"Skipping completed point {label}: {metrics_path}", flush=True)
        test_metrics = metrics["test_metrics"]
        summary.append(
            {
                "label": label,
                "requested_user_cap": cap,
                "train_selected_distinct_users": metrics["train_selected_distinct_users"],
                "train_selected_rows": metrics["train_selected_rows"],
                "subset_seed": metrics["subset_seed"],
                "model_seed": metrics["model_seed"],
                "epochs": metrics["epochs"],
                "epochs_completed": metrics["epochs_completed"],
                "selected_epoch": metrics["selected_epoch"],
                "validation_recall@5": metrics["recall@5"],
                "validation_ndcg@5": metrics["ndcg@5"],
                "validation_coverage": metrics["coverage"],
                "test_recall@5": test_metrics["recall@5"],
                "test_ndcg@5": test_metrics["ndcg@5"],
                "test_coverage": test_metrics["coverage"],
                "runtime_seconds": metrics["runtime_seconds"],
                "device": metrics["device"],
            }
        )
        write_summary(summary, output_root)

    if args.dry_run:
        print("Dry run complete; no training was launched.")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
