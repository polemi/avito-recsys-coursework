#!/usr/bin/env python3
"""Build privacy-safe, reproducible product-demo evidence for the coursework.

The script deliberately separates three different objects that are easy to mix up:

1. user -> leaf: the final fold-4 August holdout predictions;
2. leaf -> users (within-user view): how often a leaf appears in users' Top-1/Top-5;
3. leaf -> users (cross-user view): the existing full fold-3 reverse-ranking run,
   where one leaf score is compared across all eligible users.

No raw ``user_id`` or ``snapshot_id`` is written to the output directory.  The
bundle is a supplementary descriptive co-interest analysis over full leaf paths;
it is not presented as a causal result or as the primary recommender output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.snapshot_identity import (
    duckdb_snapshot_id_match_sql,
    pseudonymize_snapshot_ids,
)

DEFAULT_OUTPUT = ROOT / "reports" / "practical_demo_2026_09"
SEED_PATH = "services|Services.Appliances|Ремонт и обслуживание техники"
DEMO_USER_TOP1_PATH = "services|Services.CargoTransportation|Перевозки и доставка"
SEED_MIN_BUNDLE_SUPPORT = 100
N_BUNDLE_NEIGHBOURS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "artifacts/final_model_fold4_august/test_predictions.parquet",
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=ROOT / "artifacts/final_model_fold4_august/metrics.json",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=ROOT / "data/refresh_2026_08/processed/vocab_leaf_refresh_2026_08.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT / "data/refresh_2026_08/raw/taxonomy.parquet",
    )
    parser.add_argument(
        "--bundle-snapshot",
        type=Path,
        default=ROOT / "data/refresh_2026_08/processed/snapshots/snapshot_2026-08-01.parquet",
    )
    parser.add_argument(
        "--reverse-metrics",
        type=Path,
        default=(
            ROOT
            / "artifacts/reverse_rank_refresh_2026_08/fold_3_local_full/reverse_ranking_metrics.csv"
        ),
    )
    parser.add_argument(
        "--reverse-audiences",
        type=Path,
        default=(
            ROOT
            / "artifacts/reverse_rank_refresh_2026_08/fold_3_local_full/ranked_audiences.parquet"
        ),
    )
    parser.add_argument(
        "--reverse-metadata",
        type=Path,
        default=(
            ROOT
            / "artifacts/reverse_rank_refresh_2026_08/fold_3_local_full/metadata.json"
        ),
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_path(leaf_path: str) -> tuple[str, str, str]:
    parts = str(leaf_path).split("|", 2)
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(f"Invalid full leaf path: {leaf_path!r}")
    return parts[0], parts[1], parts[2]


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_leaf_summary(
    predictions: pd.DataFrame,
    reverse_vocab: dict[int, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    n_leaf = max(reverse_vocab)
    predicted_top1 = np.zeros(n_leaf + 1, dtype=np.int64)
    predicted_top5 = np.zeros(n_leaf + 1, dtype=np.int64)
    positive_users = np.zeros(n_leaf + 1, dtype=np.int64)
    hits_top1 = np.zeros(n_leaf + 1, dtype=np.int64)
    hits_top5 = np.zeros(n_leaf + 1, dtype=np.int64)
    users_any_hit_top1 = 0
    users_any_hit_top5 = 0
    pair_hits_top1 = 0
    pair_hits_top5 = 0
    total_positive_pairs = 0

    for ranking_raw, labels_raw in zip(predictions["ranking"], predictions["labels"]):
        ranking = [int(value) for value in ranking_raw]
        labels = {int(value) for value in labels_raw}
        assert len(ranking) >= 5, "Every final prediction must contain at least Top-5"
        assert len(set(ranking[:5])) == 5, "Top-5 recommendations must be unique"
        assert labels, "The fixed August evaluation audience is positive-only"
        assert set(ranking[:5]).issubset(reverse_vocab), "Unknown leaf id in ranking"
        assert labels.issubset(reverse_vocab), "Unknown leaf id in target"

        top1 = ranking[0]
        top5 = ranking[:5]
        predicted_top1[top1] += 1
        for leaf_id in top5:
            predicted_top5[leaf_id] += 1
        for leaf_id in labels:
            positive_users[leaf_id] += 1
        top1_hits = labels.intersection([top1])
        top5_hits = labels.intersection(top5)
        for leaf_id in top1_hits:
            hits_top1[leaf_id] += 1
        for leaf_id in top5_hits:
            hits_top5[leaf_id] += 1
        users_any_hit_top1 += int(bool(top1_hits))
        users_any_hit_top5 += int(bool(top5_hits))
        pair_hits_top1 += len(top1_hits)
        pair_hits_top5 += len(top5_hits)
        total_positive_pairs += len(labels)

    rows = []
    for leaf_id, leaf_path in sorted(reverse_vocab.items()):
        vertical, category, param1 = split_path(leaf_path)
        rows.append(
            {
                "leaf_id": leaf_id,
                "leaf_path": leaf_path,
                "vertical": vertical,
                "category": category,
                "param1": param1,
                "positive_users": int(positive_users[leaf_id]),
                "predicted_top1_users": int(predicted_top1[leaf_id]),
                "predicted_top5_users": int(predicted_top5[leaf_id]),
                "target_hits_top1": int(hits_top1[leaf_id]),
                "target_hits_top5": int(hits_top5[leaf_id]),
                "hit_rate_among_top1": safe_ratio(hits_top1[leaf_id], predicted_top1[leaf_id]),
                "hit_rate_among_top5": safe_ratio(hits_top5[leaf_id], predicted_top5[leaf_id]),
                "target_recall_top1": safe_ratio(hits_top1[leaf_id], positive_users[leaf_id]),
                "target_recall_top5": safe_ratio(hits_top5[leaf_id], positive_users[leaf_id]),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["predicted_top5_users", "leaf_id"], ascending=[False, True]
    )
    totals = {
        "n_positive_eval_users": int(len(predictions)),
        "total_top1_assignments": int(predicted_top1.sum()),
        "total_top5_assignments": int(predicted_top5.sum()),
        "total_positive_user_leaf_pairs": int(total_positive_pairs),
        "users_with_any_top1_hit": int(users_any_hit_top1),
        "users_with_any_top5_hit": int(users_any_hit_top5),
        "correct_user_leaf_pairs_top1": int(pair_hits_top1),
        "correct_user_leaf_pairs_top5": int(pair_hits_top5),
    }
    assert totals["total_top1_assignments"] == len(predictions)
    assert totals["total_top5_assignments"] == 5 * len(predictions)
    return summary, totals


def build_bundle(
    snapshot_path: Path,
    vocab: dict[str, int],
    reverse_vocab: dict[int, str],
    seed_id: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    con = duckdb.connect()
    con.execute("PRAGMA threads=4; SET memory_limit='8GB';")
    n_users = int(
        con.execute("SELECT count(*) FROM read_parquet(?)", [str(snapshot_path.resolve())]).fetchone()[
            0
        ]
    )
    n_seed_users = int(
        con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE list_contains(leaf, ?)",
            [str(snapshot_path.resolve()), seed_id],
        ).fetchone()[0]
    )
    counts = con.execute(
        """
        WITH base AS (
            SELECT list_distinct(leaf) AS leaves
            FROM read_parquet(?)
        ),
        marginal AS (
            SELECT item.leaf_id AS leaf_id, count(*) AS n_users
            FROM base, unnest(leaves) AS item(leaf_id)
            GROUP BY 1
        ),
        seed_users AS (
            SELECT leaves
            FROM base
            WHERE list_contains(leaves, ?)
        ),
        joint AS (
            SELECT item.leaf_id AS leaf_id, count(*) AS n_pair
            FROM seed_users, unnest(leaves) AS item(leaf_id)
            WHERE item.leaf_id <> ?
            GROUP BY 1
        )
        SELECT joint.leaf_id, joint.n_pair, marginal.n_users
        FROM joint
        JOIN marginal USING (leaf_id)
        """,
        [str(snapshot_path.resolve()), seed_id, seed_id],
    ).fetchdf()
    con.close()

    rows: list[dict[str, Any]] = []
    for row in counts.itertuples(index=False):
        leaf_id = int(row.leaf_id)
        leaf_path = reverse_vocab[leaf_id]
        vertical, category, param1 = split_path(leaf_path)
        n_pair = int(row.n_pair)
        n_candidate_users = int(row.n_users)
        pair_probability = n_pair / n_users
        seed_probability = n_seed_users / n_users
        candidate_probability = n_candidate_users / n_users
        lift = pair_probability / (seed_probability * candidate_probability)
        pmi = math.log(lift)
        npmi = pmi / -math.log(pair_probability)
        rows.append(
            {
                "leaf_id": leaf_id,
                "leaf_path": leaf_path,
                "vertical": vertical,
                "category": category,
                "param1": param1,
                "joint_user_support": n_pair,
                "candidate_users": n_candidate_users,
                "p_candidate_given_seed": n_pair / n_seed_users,
                "lift": lift,
                "npmi": npmi,
            }
        )

    candidates = pd.DataFrame(rows)
    candidates = candidates[
        (candidates["vertical"] == "services")
        & (candidates["category"] != "Services.Other")
        & (candidates["param1"] != "Другое")
        & (candidates["joint_user_support"] >= SEED_MIN_BUNDLE_SUPPORT)
    ].copy()
    candidates = candidates.sort_values(
        ["npmi", "joint_user_support", "leaf_id"], ascending=[False, False, True]
    ).head(N_BUNDLE_NEIGHBOURS)
    candidates.insert(0, "bundle_rank", np.arange(1, len(candidates) + 1, dtype=int))
    candidates.insert(1, "seed_leaf_id", seed_id)
    candidates.insert(2, "seed_leaf_path", reverse_vocab[seed_id])

    assert len(candidates) == N_BUNDLE_NEIGHBOURS
    assert candidates["leaf_id"].nunique() == N_BUNDLE_NEIGHBOURS
    assert not (candidates["leaf_id"] == seed_id).any()
    assert (candidates["vertical"] == "services").all()
    assert len(vocab) == len(reverse_vocab)
    return candidates, {"n_snapshot_users": n_users, "n_seed_history_users": n_seed_users}


def build_user_example(
    predictions: pd.DataFrame,
    reverse_vocab: dict[int, str],
    demo_leaf_id: int,
    snapshot_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    candidates: list[tuple[int, list[int], set[int]]] = []
    for source_row, row in enumerate(predictions.itertuples(index=False)):
        ranking = [int(value) for value in row.ranking[:5]]
        labels = {int(value) for value in row.labels}
        if ranking[0] != demo_leaf_id or labels != {demo_leaf_id}:
            continue
        candidates.append((source_row, ranking, labels))
    if not candidates:
        raise RuntimeError("No privacy-safe U-001 example satisfies the deterministic rule")
    # The first immutable source row is a deterministic, non-score-based tie-break.
    source_row, ranking, labels = candidates[0]
    rows = []
    for rank, leaf_id in enumerate(ranking, start=1):
        leaf_path = reverse_vocab[leaf_id]
        vertical, category, param1 = split_path(leaf_path)
        rows.append(
            {
                "anonymous_user": "U-001",
                "rank": rank,
                "leaf_id": leaf_id,
                "leaf_path": leaf_path,
                "vertical": vertical,
                "category": category,
                "param1": param1,
                "future_target_hit": int(leaf_id in labels),
            }
        )
    snapshot_id = str(predictions.iloc[source_row]["snapshot_id"])
    identity_match = duckdb_snapshot_id_match_sql(
        "s.snapshot_id", "requested.snapshot_id"
    )
    con = duckdb.connect()
    history = con.execute(
        f"""
        WITH requested AS (SELECT CAST(? AS VARCHAR) AS snapshot_id)
        SELECT s.leaf, s.action, s.masked_leaf, s.label_leaf
        FROM read_parquet(?) AS s
        CROSS JOIN requested
        WHERE {identity_match}
        """,
        [snapshot_id, str(snapshot_path.resolve())],
    ).fetchone()
    con.close()
    if history is None:
        raise RuntimeError("U-001 source row is missing from the fixed August snapshot")
    history_leaf, history_action, masked_leaf, label_leaf = history
    top1_view_events = sum(
        int(leaf_id == demo_leaf_id and action_id == 1)
        for leaf_id, action_id in zip(history_leaf, history_action)
    )
    top1_contact_events = sum(
        int(leaf_id == demo_leaf_id and action_id in {3, 4})
        for leaf_id, action_id in zip(history_leaf, history_action)
    )
    top1_in_prior_contact_mask = demo_leaf_id in set(masked_leaf)
    assert rows[0]["leaf_id"] == demo_leaf_id and rows[0]["future_target_hit"] == 1
    assert demo_leaf_id in set(label_leaf)
    assert top1_view_events == 2
    assert top1_contact_events == 0
    assert not top1_in_prior_contact_mask
    return pd.DataFrame(rows), {
        "source_row_zero_based": source_row,
        "top1_pre_t_view_events": top1_view_events,
        "top1_pre_t_contact_events": top1_contact_events,
        "top1_in_prior_contact_mask": top1_in_prior_contact_mask,
    }


def build_category_examples(
    predictions: pd.DataFrame,
    reverse_vocab: dict[int, str],
    seed_id: int,
    limit: int = 12,
) -> pd.DataFrame:
    selected: list[tuple[int, int, int]] = []
    for source_row, row in enumerate(predictions.itertuples(index=False)):
        ranking = [int(value) for value in row.ranking[:5]]
        if seed_id not in ranking:
            continue
        within_user_rank = ranking.index(seed_id) + 1
        future_hit = int(seed_id in {int(value) for value in row.labels})
        selected.append((within_user_rank, source_row, future_hit))
    selected.sort(key=lambda item: (item[0], item[1]))
    rows = []
    for alias_index, (within_user_rank, _source_row, future_hit) in enumerate(
        selected[:limit], start=1
    ):
        leaf_path = reverse_vocab[seed_id]
        vertical, category, param1 = split_path(leaf_path)
        rows.append(
            {
                "anonymous_user": f"A-{alias_index:03d}",
                "leaf_id": seed_id,
                "leaf_path": leaf_path,
                "vertical": vertical,
                "category": category,
                "param1": param1,
                "within_user_rank": within_user_rank,
                "future_target_hit": future_hit,
                "ordering_note": "not cross-user ordered; seed position inside each user's Top-5",
            }
        )
    assert rows and all(1 <= row["within_user_rank"] <= 5 for row in rows)
    return pd.DataFrame(rows)


def sanitize_reverse_ranking(
    metrics: pd.DataFrame,
    audiences: pd.DataFrame,
    limit: int = 10,
) -> tuple[pd.DataFrame, int]:
    cutoff100 = metrics[metrics["requested_n"] == 100].copy()
    chosen = cutoff100.sort_values(
        ["precision_at_n", "lift_at_n", "leaf_id"], ascending=[False, False, True]
    ).iloc[0]
    leaf_id = int(chosen["leaf_id"])
    source = audiences[audiences["leaf_id"] == leaf_id].sort_values("rank").head(limit)
    safe = source[["leaf_id", "leaf_path", "rank", "score", "label"]].copy()
    safe.insert(0, "anonymous_user", [f"RR-{index:03d}" for index in range(1, len(safe) + 1)])
    safe = safe.rename(
        columns={
            "rank": "cross_user_rank",
            "score": "model_score",
            "label": "future_target_hit",
        }
    )
    safe["model_score"] = safe["model_score"].round(6)
    safe["ordering_note"] = "cross-user ordering by one fixed leaf score"
    assert "user_id" not in safe.columns and "snapshot_id" not in safe.columns
    return safe, leaf_id


def build_same_name_audit(vocab: dict[str, int]) -> pd.DataFrame:
    rows = []
    by_name: dict[str, list[tuple[int, str, str, str]]] = {}
    for leaf_path, leaf_id in vocab.items():
        vertical, category, param1 = split_path(leaf_path)
        by_name.setdefault(param1, []).append((int(leaf_id), leaf_path, vertical, category))
    for param1, values in sorted(by_name.items()):
        if len(values) < 2:
            continue
        for leaf_id, leaf_path, vertical, category in sorted(values):
            rows.append(
                {
                    "param1": param1,
                    "distinct_full_paths": len(values),
                    "leaf_id": leaf_id,
                    "leaf_path": leaf_path,
                    "vertical": vertical,
                    "category": category,
                }
            )
    audit = pd.DataFrame(rows)
    repair = audit[audit["param1"] == "Ремонт и отделка"]
    assert repair["leaf_id"].nunique() >= 2, "Same-name full paths must remain distinct"
    return audit


def assert_no_raw_ids(output: Path, sensitive_user_ids: set[str]) -> dict[str, Any]:
    checked_files = []
    suspicious_matches: dict[str, list[str]] = {}
    for path in sorted(output.iterdir()):
        if path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        numeric_tokens = set(re.findall(r"(?<!\d)\d{8,}(?!\d)", text))
        exposed = sorted(numeric_tokens.intersection(sensitive_user_ids))
        if exposed:
            suspicious_matches[path.name] = exposed[:5]
        checked_files.append(path.name)
    assert not suspicious_matches, f"Raw user IDs found in outputs: {suspicious_matches}"
    return {"checked_files": checked_files, "raw_user_ids_found": 0}


def main() -> None:
    args = parse_args()
    input_paths = [
        args.predictions,
        args.model_metrics,
        args.vocab,
        args.taxonomy,
        args.bundle_snapshot,
        args.reverse_metrics,
        args.reverse_audiences,
        args.reverse_metadata,
    ]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    vocab_payload = json.loads(args.vocab.read_text(encoding="utf-8"))
    vocab = {str(path): int(leaf_id) for path, leaf_id in vocab_payload["leaf"].items()}
    reverse_vocab = {leaf_id: path for path, leaf_id in vocab.items()}
    assert len(vocab) == len(reverse_vocab), "Leaf ids must be one-to-one with full paths"
    assert SEED_PATH in vocab, f"Missing seed path: {SEED_PATH}"
    assert DEMO_USER_TOP1_PATH in vocab, f"Missing demo path: {DEMO_USER_TOP1_PATH}"
    seed_id = vocab[SEED_PATH]
    demo_leaf_id = vocab[DEMO_USER_TOP1_PATH]

    taxonomy = pd.read_parquet(args.taxonomy)
    taxonomy_paths = {
        "|".join(map(str, row))
        for row in taxonomy[["vertical", "category", "param1"]].itertuples(index=False, name=None)
    }
    assert taxonomy_paths == set(vocab), "Taxonomy and full-path vocab must match exactly"

    predictions = pd.read_parquet(args.predictions)
    required_prediction_columns = {"snapshot_id", "ranking", "labels"}
    assert required_prediction_columns.issubset(predictions.columns)
    assert len(predictions) == 20_000, "Expected fixed 20k positive August holdout audience"
    assert predictions["snapshot_id"].nunique() == len(predictions)
    # Keep legacy raw IDs only in memory for the privacy scan, then use the
    # pseudonymous contract for all joins and downstream processing.
    sensitive_ids = {
        str(value).split("_", 1)[0]
        for value in predictions["snapshot_id"].astype(str)
    }
    predictions = pseudonymize_snapshot_ids(predictions, copy=False)

    leaf_summary, global_totals = build_leaf_summary(predictions, reverse_vocab)
    leaf_summary.to_csv(args.output / "august_eval_leaf_topk_summary.csv", index=False)

    bundle, bundle_totals = build_bundle(
        args.bundle_snapshot, vocab, reverse_vocab, seed_id
    )
    bundle.to_csv(args.output / "bundle_seed_repair_equipment.csv", index=False)

    user_example, user_example_meta = build_user_example(
        predictions, reverse_vocab, demo_leaf_id, args.bundle_snapshot
    )
    user_example.to_csv(args.output / "user_to_leaf_U-001.csv", index=False)

    category_examples = build_category_examples(predictions, reverse_vocab, seed_id)
    category_examples.to_csv(args.output / "leaf_to_users_within_top5_examples.csv", index=False)

    seed_summary = leaf_summary[leaf_summary["leaf_id"] == seed_id].iloc[0].to_dict()
    for key, value in list(seed_summary.items()):
        if isinstance(value, np.generic):
            seed_summary[key] = value.item()
    dump_json(args.output / "seed_leaf_audience_summary.json", seed_summary)
    demo_leaf_summary = leaf_summary[leaf_summary["leaf_id"] == demo_leaf_id].iloc[0].to_dict()
    for key, value in list(demo_leaf_summary.items()):
        if isinstance(value, np.generic):
            demo_leaf_summary[key] = value.item()

    reverse_metrics = pd.read_csv(args.reverse_metrics)
    reverse_audiences = pd.read_parquet(args.reverse_audiences)
    reverse_metadata = json.loads(args.reverse_metadata.read_text(encoding="utf-8"))
    safe_reverse, chosen_reverse_leaf = sanitize_reverse_ranking(
        reverse_metrics, reverse_audiences
    )
    reverse_metrics.to_csv(args.output / "reverse_ranking_metrics_fold3_full.csv", index=False)
    safe_reverse.to_csv(args.output / "reverse_ranking_example_fold3_full.csv", index=False)

    same_name_audit = build_same_name_audit(vocab)
    same_name_audit.to_csv(args.output / "same_name_full_path_audit.csv", index=False)

    model_metrics = json.loads(args.model_metrics.read_text(encoding="utf-8"))
    provenance = {
        "schema_version": 1,
        "purpose": "privacy-safe practical CRM demo evidence",
        "seed": {"leaf_id": seed_id, "leaf_path": SEED_PATH},
        "u_001_top1": {"leaf_id": demo_leaf_id, "leaf_path": DEMO_USER_TOP1_PATH},
        "inputs": {
            relative(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in input_paths
        },
        "final_august_user_to_leaf": {
            "fold": int(model_metrics["fold"]),
            "test_snapshot": model_metrics["test_snapshot"],
            "selected_epoch": int(model_metrics["selected_epoch"]),
            "test_recall_at_5": float(model_metrics["test_metrics"]["recall@5"]),
            "test_ndcg_at_5": float(model_metrics["test_metrics"]["ndcg@5"]),
            **global_totals,
        },
        "u_001_selection_rule": (
            "Take the first immutable source row where DEMO_USER_TOP1_PATH is rank 1 and the "
            "sole future label. The example is illustrative and is not used as a performance estimate."
        ),
        "u_001_selection_metadata": user_example_meta,
        "u_001_leaf_eval_summary": demo_leaf_summary,
        "within_user_leaf_to_users_note": (
            "Counts and examples use the seed's position inside each user's final August Top-5. "
            "They are not a cross-user ordering because the saved final file has rankings, not per-leaf scores."
        ),
        "cross_user_reverse_ranking": {
            "provenance": reverse_metadata,
            "selected_example_leaf_id": chosen_reverse_leaf,
            "distinction": (
                "Existing fold-3 evidence: one fixed leaf score was compared across the full eligible "
                "June audience. It is supporting operational evidence, not the fold-4 August result."
            ),
        },
        "bundle": {
            "source": relative(args.bundle_snapshot),
            "definition": (
                "Distinct pre-T full leaf paths per user; Services candidates only; exclude Services.Other "
                "and param1='Другое'; require joint support >= 100; rank by normalized PMI."
            ),
            "interpretation": (
                "Supplementary descriptive co-interest bundle. It is neither the primary recommender "
                "output nor a causal effect."
            ),
            **bundle_totals,
        },
        "privacy": {
            "raw_user_ids_written": False,
            "raw_snapshot_ids_written": False,
            "aliases": ["U-001", "A-001...", "RR-001..."],
        },
    }
    dump_json(args.output / "provenance.json", provenance)

    bundle_lines = [
        f"{int(row.bundle_rank)}. `{row.leaf_path}` — support {int(row.joint_user_support):,}, "
        f"lift {row.lift:.2f}, NPMI {row.npmi:.3f}"
        for row in bundle.itertuples(index=False)
    ]
    u_lines = [
        f"{int(row.rank)}. `{row.leaf_path}`" + (" — future hit" if row.future_target_hit else "")
        for row in user_example.itertuples(index=False)
    ]
    readme = f"""# Практический CRM-demo: воспроизводимые артефакты

Эта папка показывает, как одна модель используется в двух направлениях, и не
раскрывает исходные `user_id`/`snapshot_id`.

## 1. Пользователь → новые категории

Для анонимного примера `U-001` финальная fold-4 модель на August holdout выдала:

{chr(10).join(u_lines)}

`U-001` выбран детерминированным правилом, записанным в `provenance.json`; это
иллюстрация интерфейса, а не отдельная оценка качества. Общие holdout-метрики
остаются Recall@5 = {model_metrics['test_metrics']['recall@5']:.4f} и
NDCG@5 = {model_metrics['test_metrics']['ndcg@5']:.4f} на 20 000 positive users.
До момента T у top-1 категории было два просмотра, но ни одного контакта, поэтому
она не попала в prior-contact mask и оставалась допустимым new-only кандидатом.

## 2. Категория → пользователи

Seed: `{SEED_PATH}` (`leaf_id={seed_id}`). На фиксированной August-выборке модель
поставила seed на первое место у **{int(seed_summary['predicted_top1_users']):,}**
пользователей и в Top-5 у **{int(seed_summary['predicted_top5_users']):,}**.
Из них будущий контакт с seed наблюдался соответственно у
**{int(seed_summary['target_hits_top1']):,}** и **{int(seed_summary['target_hits_top5']):,}**.
Это честный offline backtest на positive-only аудитории, а не размер всей CRM-базы.

`leaf_to_users_within_top5_examples.csv` показывает позиции seed внутри Top-5
отдельных анонимных пользователей. Это ещё не полноценная сортировка пользователей
между собой: в сохранённом final August prediction-файле нет per-leaf scores.

Полная cross-user проверка уже была сделана отдельным fold-3 прогоном на
**{int(reverse_metadata['n_sampled_users']):,}** пользователях. Безопасная копия метрик
лежит в `reverse_ranking_metrics_fold3_full.csv`, а 10 обезличенных строк — в
`reverse_ranking_example_fold3_full.csv`. Эти результаты нельзя выдавать за
fold-4 August rerun; provenance явно сохраняет snapshot и checkpoint.

## 3. Дополнительный bundle категорий

Bundle не является основной целью модели. Это описательная co-interest функция:
в August snapshot берутся различные full-leaf пути из pre-T истории пользователя,
после чего связи с seed ранжируются по NPMI при support >=
{SEED_MIN_BUNDLE_SUPPORT}. `Services.Other` и `param1=Другое` исключены как
неинтерпретируемые CRM-направления.

{chr(10).join(bundle_lines)}

Путь целиком (`vertical|category|param1`) остаётся ключом. Одинаковые названия
`param1` в разных ветках не объединяются; это проверяет
`same_name_full_path_audit.csv`.

## Файлы

- `user_to_leaf_U-001.csv` — анонимный Top-5 пример;
- `august_eval_leaf_topk_summary.csv` — Top-1/Top-5 audience и hit counts для всех leaf;
- `seed_leaf_audience_summary.json` — компактные числа для seed;
- `leaf_to_users_within_top5_examples.csv` — seed внутри пользовательских Top-5;
- `reverse_ranking_metrics_fold3_full.csv` — full-audience cross-user evidence;
- `reverse_ranking_example_fold3_full.csv` — обезличенный cross-user пример;
- `bundle_seed_repair_equipment.csv` — full-path bundle;
- `same_name_full_path_audit.csv` — контроль несхлопывания одинаковых названий;
- `provenance.json` — источники, SHA-256 и границы интерпретации;
- `validation_report.json` — машинно проверенные инварианты и privacy scan.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")

    sensitive_ids.update(reverse_audiences["user_id"].astype(str))
    privacy_scan = assert_no_raw_ids(args.output, sensitive_ids)
    validation = {
        "all_checks_passed": True,
        "checks": {
            "taxonomy_equals_full_path_vocab": True,
            "vocab_is_one_to_one": True,
            "same_named_param1_paths_remain_distinct": True,
            "august_eval_has_20000_unique_positive_users": True,
            "every_top5_is_unique_and_known": True,
            "seed_exists_exactly_as_full_path": True,
            "u_001_top1_exists_exactly_as_full_path": True,
            "bundle_has_five_unique_services_full_paths": True,
            "reverse_example_has_no_identifier_columns": True,
            "privacy_scan": privacy_scan,
        },
        "key_counts": {
            **global_totals,
            **bundle_totals,
            "seed_leaf_id": seed_id,
            "seed_predicted_top1_users": int(seed_summary["predicted_top1_users"]),
            "seed_predicted_top5_users": int(seed_summary["predicted_top5_users"]),
            "seed_target_hits_top1": int(seed_summary["target_hits_top1"]),
            "seed_target_hits_top5": int(seed_summary["target_hits_top5"]),
            "u_001_leaf_id": demo_leaf_id,
            "u_001_leaf_predicted_top1_users": int(
                demo_leaf_summary["predicted_top1_users"]
            ),
            "u_001_leaf_target_hits_top1": int(demo_leaf_summary["target_hits_top1"]),
        },
    }
    dump_json(args.output / "validation_report.json", validation)
    # Include the validation file itself in a final scan after it is written.
    assert_no_raw_ids(args.output, sensitive_ids)

    print(json.dumps(validation["key_counts"], ensure_ascii=False, indent=2))
    print(f"Wrote privacy-safe demo to {args.output}")


if __name__ == "__main__":
    main()
