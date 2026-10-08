#!/usr/bin/env python3
"""Audit the 1.2M-versus-full scaling replication across model seeds.

The script verifies the downloaded seed-123 recovery, checks that all three
model seeds use the same ordered July audience and labels, recomputes July
Recall@5/NDCG@5 from predictions, and writes a compact canonical report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "data_scaling_expanded_2026_10"
OUT = ROOT / "reports" / "data_scaling_replication_2026_10"

SEED_PATHS = {
    42: ARTIFACT_ROOT / "seed42_kaggle" / "results" / "modelseed42",
    7: (
        ARTIFACT_ROOT
        / "endpoint_replication_v2"
        / "extracted"
        / "endpoint_seed7"
        / "modelseed7"
    ),
    123: ARTIFACT_ROOT / "endpoint_replication_seed123_v3" / "extracted" / "modelseed123",
}

BOOTSTRAP_PATHS = {
    42: (
        ARTIFACT_ROOT
        / "seed42_kaggle"
        / "results"
        / "audit_2026_10_05"
        / "paired_bootstrap_required.csv"
    ),
    7: ARTIFACT_ROOT / "endpoint_replication_v2" / "seed7_paired_bootstrap" / "paired_bootstrap.csv",
    123: (
        ARTIFACT_ROOT
        / "endpoint_replication_seed123_v3"
        / "extracted"
        / "endpoint_paired_bootstrap.csv"
    ),
}

SEED123_ZIP = (
    ARTIFACT_ROOT
    / "endpoint_replication_seed123_v3"
    / "download"
    / "avito_endpoint_scaling_recovery_seed123.zip"
)
EXPECTED_SEED123_ZIP_SHA256 = "372ddcbff77398d074917a8bd9c20853418bf13625803173d0577d516c437fea"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recall_at_5(ranking: list[int], labels: list[int]) -> float:
    positives = set(labels)
    if not positives:
        raise AssertionError("positive-only evaluation contains an empty label set")
    return len(set(ranking[:5]) & positives) / len(positives)


def ndcg_at_5(ranking: list[int], labels: list[int]) -> float:
    positives = set(labels)
    if not positives:
        raise AssertionError("positive-only evaluation contains an empty label set")
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(ranking[:5])
        if item in positives
    )
    ideal_hits = min(5, len(positives))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def load_predictions(path: Path) -> dict[str, object]:
    table = pq.read_table(path, columns=["snapshot_id", "ranking", "labels"])
    assert table.num_rows == 100_000, (path, table.num_rows)
    ids = table.column("snapshot_id").to_pylist()
    rankings = table.column("ranking").to_pylist()
    labels = table.column("labels").to_pylist()
    assert len(ids) == len(set(ids)), f"duplicate snapshot_id in {path}"
    assert all(HEX64.fullmatch(value or "") for value in ids), f"non-pseudonymous id in {path}"
    recall = sum(recall_at_5(r, y) for r, y in zip(rankings, labels)) / len(ids)
    ndcg = sum(ndcg_at_5(r, y) for r, y in zip(rankings, labels)) / len(ids)
    audience_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    label_hash = hashlib.sha256(
        "\n".join(",".join(map(str, row)) for row in labels).encode()
    ).hexdigest()
    return {
        "ids": ids,
        "labels": labels,
        "recall@5": recall,
        "ndcg@5": ndcg,
        "audience_hash": audience_hash,
        "label_hash": label_hash,
    }


def load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, object], split: str, name: str) -> float:
    if split == "test":
        nested_test = metrics.get("test_metrics")
        if isinstance(nested_test, dict) and name in nested_test:
            return float(nested_test[name])
    if split == "validation" and name in metrics:
        return float(metrics[name])
    nested = metrics.get(split)
    if isinstance(nested, dict) and name in nested:
        return float(nested[name])
    flat = metrics.get(f"{split}_{name}")
    if flat is not None:
        return float(flat)
    raise KeyError((split, name, sorted(metrics)))


def read_bootstrap(seed: int) -> list[dict[str, object]]:
    with BOOTSTRAP_PATHS[seed].open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows: list[dict[str, object]] = []
    for metric in ("recall@5", "ndcg@5"):
        candidates = [row for row in source if row["metric"] == metric]
        if seed == 42:
            candidates = [
                row
                for row in candidates
                if row["reference"] == "full" and row["challenger"] == "1200k_users"
            ]
        assert len(candidates) == 1, (seed, metric, candidates)
        row = candidates[0]
        rows.append(
            {
                "model_seed": seed,
                "metric": metric,
                "n_users": int(row["n_users"]),
                "full_mean": float(row["reference_mean"]),
                "train_1_2m_mean": float(row["challenger_mean"]),
                "delta_full_minus_1_2m": float(row["mean_difference"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
                "bootstrap_fraction_delta_le_zero": float(row["p_bootstrap_le_zero"]),
            }
        )
    return rows


def main() -> None:
    assert sha256_file(SEED123_ZIP) == EXPECTED_SEED123_ZIP_SHA256
    recovery = json.loads(
        (
            ARTIFACT_ROOT
            / "endpoint_replication_seed123_v3"
            / "extracted"
            / "recovery_status.json"
        ).read_text(encoding="utf-8")
    )
    assert recovery["overall_status"] == "complete"
    assert recovery["data_hashes_verified"] is True
    assert recovery["paired_bootstrap"]["complete"] is True
    assert set(recovery["completed_labels"]) == {"1200k_users", "full"}
    assert all(not run["gpu_oom"] and not run["host_oom"] for run in recovery["point_runs"])

    metric_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    reference_ids: list[str] | None = None
    reference_labels: list[list[int]] | None = None
    audience_hashes: set[str] = set()
    label_hashes: set[str] = set()

    for seed, base in SEED_PATHS.items():
        per_seed: dict[str, dict[str, object]] = {}
        for label in ("train1200k_users", "trainfull"):
            run = base / label
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            arguments = config["arguments"]
            metrics = load_metrics(run / "metrics.json")
            predictions = load_predictions(run / "test_predictions.parquet")
            assert int(arguments["subset_seed"]) == 42
            assert int(arguments["model_seed"]) == seed
            assert arguments["train_sampling_unit"] == "users"
            assert int(arguments["test_cap"]) == 100_000
            assert abs(predictions["recall@5"] - metric_value(metrics, "test", "recall@5")) < 1e-12
            assert abs(predictions["ndcg@5"] - metric_value(metrics, "test", "ndcg@5")) < 1e-12
            endpoint_rows.append(
                {
                    "model_seed": seed,
                    "endpoint": "1.2M" if label == "train1200k_users" else "2M",
                    "exact_distinct_train_users": int(metrics["train_selected_distinct_users"]),
                    "train_rows": int(metrics["train_selected_rows"]),
                    "epochs_completed": int(metrics["epochs_completed"]),
                    "selected_epoch": int(metrics["selected_epoch"]),
                    "june_validation_recall@5": metric_value(metrics, "validation", "recall@5"),
                    "june_validation_ndcg@5": metric_value(metrics, "validation", "ndcg@5"),
                    "july_test_recall@5": predictions["recall@5"],
                    "july_test_ndcg@5": predictions["ndcg@5"],
                }
            )
            audience_hashes.add(str(predictions["audience_hash"]))
            label_hashes.add(str(predictions["label_hash"]))
            if reference_ids is None:
                reference_ids = predictions["ids"]  # type: ignore[assignment]
                reference_labels = predictions["labels"]  # type: ignore[assignment]
            else:
                assert predictions["ids"] == reference_ids
                assert predictions["labels"] == reference_labels
            per_seed[label] = predictions

        for row in read_bootstrap(seed):
            metric = str(row["metric"])
            assert abs(
                float(row["full_mean"]) - float(per_seed["trainfull"][metric])
            ) < 1e-12
            assert abs(
                float(row["train_1_2m_mean"])
                - float(per_seed["train1200k_users"][metric])
            ) < 1e-12
            assert float(row["ci95_low"]) > 0.0
            metric_rows.append(row)

    assert len(audience_hashes) == 1
    assert len(label_hashes) == 1

    descriptive: list[dict[str, object]] = []
    for metric in ("recall@5", "ndcg@5"):
        values = [
            float(row["delta_full_minus_1_2m"])
            for row in metric_rows
            if row["metric"] == metric
        ]
        descriptive.append(
            {
                "metric": metric,
                "model_seeds": "42,7,123",
                "n_model_seeds": len(values),
                "positive_seeds": sum(value > 0 for value in values),
                "mean_delta": statistics.mean(values),
                "sample_sd_delta": statistics.stdev(values),
                "min_delta": min(values),
                "max_delta": max(values),
                "interpretation": "descriptive only; n=3 is not used for an across-seed confidence interval",
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "paired_bootstrap_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (OUT / "endpoint_metrics_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    with (OUT / "descriptive_across_seeds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(descriptive[0]))
        writer.writeheader()
        writer.writerows(descriptive)

    summary = {
        "status": "pass",
        "estimand": "July test metric at full (~2M) minus 1.2M distinct positive train users",
        "configuration": {
            "model": "reduced Transformer, d_model=64, K=2",
            "subset_seed": 42,
            "model_seeds": [42, 7, 123],
            "checkpoint_selection": "fixed June positive-only audience, n=20,000",
            "final_evaluation": "fixed July positive-only audience, n=100,000",
            "full_exact_users": 1_985_972,
            "human_readable_label": "2 млн пользователей",
        },
        "audience": {
            "ordered_snapshot_id_sha256": next(iter(audience_hashes)),
            "ordered_labels_sha256": next(iter(label_hashes)),
            "same_ordered_ids_and_labels_all_runs": True,
            "raw_ids_present": False,
        },
        "results": metric_rows,
        "endpoint_metrics": endpoint_rows,
        "descriptive_across_model_seeds": descriptive,
        "conclusion": (
            "The full (~2M) endpoint exceeds 1.2M for Recall@5 and NDCG@5 "
            "in all three model initializations; every within-seed paired 95% interval excludes zero. "
            "The across-seed mean and SD are descriptive because only three model seeds were run."
        ),
        "seed123_recovery": {
            "version_id": 355414766,
            "zip_sha256": EXPECTED_SEED123_ZIP_SHA256,
            "status": "complete",
            "oom": False,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    provenance = {
        "seed42_bootstrap": str(BOOTSTRAP_PATHS[42].relative_to(ROOT)),
        "seed7_bootstrap": str(BOOTSTRAP_PATHS[7].relative_to(ROOT)),
        "seed123_bootstrap": str(BOOTSTRAP_PATHS[123].relative_to(ROOT)),
        "seed123_download": str(SEED123_ZIP.relative_to(ROOT)),
        "seed123_download_sha256": EXPECTED_SEED123_ZIP_SHA256,
        "audit_script": str(Path(__file__).resolve().relative_to(ROOT)),
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
