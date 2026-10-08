#!/usr/bin/env python
"""Reconcile the completed full-pool d128 run with exact-audience baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.metrics import binary_ndcg_at_k, recall_at_k
from avito_recsys.snapshot_identity import pseudonymize_snapshot_ids


DISPLAY_NAMES = {
    "random": "Случайное ранжирование",
    "most_popular": "Популярность",
    "weighted_cooccurrence": "Совместная встречаемость",
    "old_flow_monthly_adaptation": "Адаптация модели переходов (30 дней)",
    "puresvd_fold_in": "PureSVD",
    "transformer_d128_k2_full2m": "Transformer d=128, K=2, полный пул",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Store repository-relative provenance paths instead of workstation paths."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def row_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["snapshot_id", "ranking", "labels"]].copy()
    result = pseudonymize_snapshot_ids(result, copy=False)
    result["recall@5"] = [
        recall_at_k(ranking, labels, 5)
        for ranking, labels in zip(result.ranking, result.labels)
    ]
    result["ndcg@5"] = [
        binary_ndcg_at_k(ranking, labels, 5)
        for ranking, labels in zip(result.ranking, result.labels)
    ]
    return result.sort_values("snapshot_id").reset_index(drop=True)


def paired_bootstrap(delta: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    """User-level paired bootstrap using a multinomial count representation."""

    values, counts = np.unique(delta.astype(np.float64), return_counts=True)
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    chunk = 1_000
    for start in range(0, repeats, chunk):
        stop = min(start + chunk, repeats)
        sampled_counts = rng.multinomial(len(delta), probabilities, size=stop - start)
        means[start:stop] = sampled_counts @ values / len(delta)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_bootstrap_le_zero": float((means <= 0).mean()),
        "statistically_significant_at_alpha_0_05": bool(low > 0 or high < 0),
    }


def assert_same_audience(reference: pd.DataFrame, candidate: pd.DataFrame, method: str) -> None:
    if len(candidate) != len(reference):
        raise RuntimeError(f"{method}: row count {len(candidate)} != {len(reference)}")
    if candidate.snapshot_id.duplicated().any():
        raise RuntimeError(f"{method}: duplicate snapshot_id")
    if not np.array_equal(reference.snapshot_id.to_numpy(), candidate.snapshot_id.to_numpy()):
        raise RuntimeError(f"{method}: audience mismatch")
    for left, right in zip(reference.labels, candidate.labels):
        if list(map(int, left)) != list(map(int, right)):
            raise RuntimeError(f"{method}: label mismatch")


def validate_rankings(frame: pd.DataFrame, method: str, minimum_length: int = 5) -> None:
    """Validate the persisted ranking contract before calculating metrics."""

    for row in frame.itertuples(index=False):
        ranking = [int(value) for value in row.ranking]
        if len(ranking) < minimum_length:
            raise RuntimeError(
                f"{method}: ranking shorter than {minimum_length} for {row.snapshot_id}"
            )
        if len(ranking) != len(set(ranking)):
            raise RuntimeError(f"{method}: duplicate category in ranking")
        if any(value <= 0 for value in ranking):
            raise RuntimeError(f"{method}: padding or invalid category in ranking")


def validate_against_evaluation_snapshot(
    reference: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    d64: pd.DataFrame,
    evaluation_snapshot: Path,
) -> dict[str, int | bool | str]:
    """Verify labels and candidate masks from the source evaluation snapshot.

    Prediction files intentionally contain no raw user identifiers or candidate
    masks.  This check joins them back to the privacy-safe prepared snapshot and
    confirms that every reported ranking excludes categories with a recent
    contact.  It turns the protocol statement into a machine-checked invariant.
    """

    source = pd.read_parquet(
        evaluation_snapshot,
        columns=["snapshot_id", "label_leaf", "masked_leaf", "n_labels"],
    )
    source = source.loc[source["n_labels"] > 0].rename(columns={"label_leaf": "labels"})
    source = pseudonymize_snapshot_ids(source, copy=False)
    if source.snapshot_id.duplicated().any():
        raise RuntimeError("Evaluation snapshot contains duplicate snapshot_id")
    source = source.set_index("snapshot_id")
    missing = set(reference.snapshot_id) - set(source.index)
    if missing:
        raise RuntimeError(
            f"Evaluation snapshot is missing {len(missing)} reported audience identifiers"
        )
    source = source.loc[reference.snapshot_id].reset_index()
    for left, right in zip(reference.labels, source.labels):
        if list(map(int, left)) != list(map(int, right)):
            raise RuntimeError("Neural labels do not match the evaluation snapshot")

    mask_map = source.set_index("snapshot_id")["masked_leaf"]
    violations: dict[str, int] = {}
    all_frames = {**frames, "transformer_d64_k2_full2m": d64}
    for method, frame in all_frames.items():
        validate_rankings(frame, method)
        count = 0
        for row in frame.itertuples(index=False):
            blocked = set(map(int, mask_map.loc[row.snapshot_id]))
            if blocked.intersection(map(int, row.ranking)):
                count += 1
        violations[method] = count
        if count:
            raise RuntimeError(f"{method}: {count} masked-category ranking violations")
    return {
        "evaluation_snapshot": portable_path(evaluation_snapshot),
        "source_rows": int(len(source)),
        "labels_verified": True,
        "candidate_masks_verified": True,
        "masked_category_ranking_violations_total": int(sum(violations.values())),
    }


def normalize_legacy_double_hash(
    frame: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Map legacy sha256(sha256-id) artifacts back to the current id contract."""

    reverse = {
        hashlib.sha256(value.encode("utf-8")).hexdigest(): value
        for value in reference.snapshot_id.astype(str)
    }
    result = frame.copy()
    mapped = result.snapshot_id.astype(str).map(reverse)
    if mapped.notna().all():
        result["snapshot_id"] = mapped
    return result.sort_values("snapshot_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neural",
        type=Path,
        default=ROOT / "artifacts/runpod/d128_full2m_2026_10/extracted/output/test_predictions.parquet",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=ROOT / "artifacts/baselines_full2m_exact_runpod100k_2026_10_08/predictions.parquet",
    )
    parser.add_argument(
        "--old-flow",
        type=Path,
        default=ROOT / "artifacts/old_flow_model/monthly_exact_runpod100k_2026_10_08/predictions.parquet",
    )
    parser.add_argument(
        "--d64",
        type=Path,
        default=(
            ROOT
            / "artifacts/data_scaling_expanded_2026_10/seed42_kaggle/results/"
            "modelseed42/trainfull/test_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/runpod_full2m_exact100k_comparison_2026_10_08",
    )
    parser.add_argument(
        "--evaluation-snapshot",
        type=Path,
        default=(
            ROOT
            / "reports/kaggle/expanded_user_scaling_2026_10/scaling_data/"
            "snapshots/snapshot_2026-07-01.parquet"
        ),
        help="Prepared July snapshot used to verify labels and candidate masks.",
    )
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    neural_raw = pd.read_parquet(args.neural)
    neural = row_metrics(neural_raw)
    if neural.snapshot_id.duplicated().any():
        raise RuntimeError("Duplicate neural snapshot_id")
    validate_rankings(neural, "transformer_d128_k2_full2m")

    baseline_raw = pd.read_parquet(args.baselines)
    frames: dict[str, pd.DataFrame] = {}
    for method in ("random", "most_popular", "weighted_cooccurrence", "puresvd_fold_in"):
        frame = row_metrics(baseline_raw.loc[baseline_raw.method == method])
        assert_same_audience(neural, frame, method)
        frames[method] = frame

    old_flow = row_metrics(pd.read_parquet(args.old_flow))
    assert_same_audience(neural, old_flow, "old_flow_monthly_adaptation")
    frames["old_flow_monthly_adaptation"] = old_flow
    frames["transformer_d128_k2_full2m"] = neural

    d64 = normalize_legacy_double_hash(row_metrics(pd.read_parquet(args.d64)), neural)
    assert_same_audience(neural, d64, "transformer_d64_k2_full2m")

    evaluation_validation = validate_against_evaluation_snapshot(
        neural,
        frames,
        d64,
        args.evaluation_snapshot,
    )

    comparison_rows = []
    for method, frame in frames.items():
        comparison_rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                "n_users": len(frame),
                "recall@5": float(frame["recall@5"].mean()),
                "ndcg@5": float(frame["ndcg@5"].mean()),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("recall@5")

    bootstrap_rows = []
    for method, frame in frames.items():
        if method == "transformer_d128_k2_full2m":
            continue
        for metric in ("recall@5", "ndcg@5"):
            delta = neural[metric].to_numpy() - frame[metric].to_numpy()
            bootstrap_rows.append(
                {
                    "reference": "transformer_d128_k2_full2m",
                    "comparison": method,
                    "metric": metric,
                    "n_users": len(delta),
                    "alpha": 0.05,
                    "repeats": args.repeats,
                    "seed": args.seed,
                    **paired_bootstrap(delta, args.repeats, args.seed),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)

    configuration_rows = []
    for metric in ("recall@5", "ndcg@5"):
        delta = neural[metric].to_numpy() - d64[metric].to_numpy()
        configuration_rows.append(
            {
                "reference": "transformer_d128_k2_full2m",
                "comparison": "transformer_d64_k2_full2m",
                "metric": metric,
                "n_users": len(delta),
                "alpha": 0.05,
                "repeats": args.repeats,
                "seed": args.seed,
                "d128_metric": float(neural[metric].mean()),
                "d64_metric": float(d64[metric].mean()),
                **paired_bootstrap(delta, args.repeats, args.seed),
            }
        )
    configuration_comparison = pd.DataFrame(configuration_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output / "comparison.csv"
    bootstrap_path = args.output / "paired_bootstrap.csv"
    configuration_path = args.output / "d128_vs_d64_full2m.csv"
    summary_path = args.output / "summary.json"
    comparison.to_csv(comparison_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    configuration_comparison.to_csv(configuration_path, index=False)
    summary = {
        "protocol": {
            "evaluation_users": len(neural),
            "evaluation_period": "2026-07-01",
            "same_snapshot_ids_labels_and_candidate_masks": True,
            "evaluation_snapshot_validation": evaluation_validation,
            "transformer": {
                "configuration": "d=128, K=2",
                "train_distinct_users": 1_985_972,
                "train_rows": 4_162_636,
                "completed_epochs": 12,
                "selected_epoch": 12,
                "seed": 42,
            },
            "puresvd_train_cap_rows": 768_708,
            "old_flow": (
                "Reproducible 30-day offline adaptation of the historical transition logic; "
                "not a metric of the original production business model."
            ),
            "alpha": 0.05,
            "bootstrap_repeats": args.repeats,
        },
        "comparison": comparison.to_dict(orient="records"),
        "paired_bootstrap": bootstrap.to_dict(orient="records"),
        "d128_vs_d64_full2m": configuration_comparison.to_dict(orient="records"),
        "input_sha256": {
            "neural": sha256_file(args.neural),
            "baselines": sha256_file(args.baselines),
            "old_flow": sha256_file(args.old_flow),
            "d64": sha256_file(args.d64),
            "evaluation_snapshot": sha256_file(args.evaluation_snapshot),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print()
    print(bootstrap.to_string(index=False))
    print()
    print(configuration_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
