#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    n_leaf = len(vocab["leaf"])
    history_days = int(config["history_days"])
    dataset = str(args.dataset.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    summary = con.execute(
        f"""
        SELECT count(*) n_rows,
               count(DISTINCT snapshot_id) n_unique_snapshot,
               count(DISTINCT user_id) n_users,
               min(snapshot_ts) min_snapshot,
               max(snapshot_ts) max_snapshot,
               avg(history_events) mean_history,
               approx_quantile(history_events, 0.5) median_history,
               max(history_events) max_history,
               avg(n_labels) mean_labels_all,
               avg(n_labels) FILTER (WHERE n_labels > 0) mean_labels_positive,
               sum(n_labels > 0) n_positive,
               sum(n_labels > 1) n_multilabel,
               avg(len(masked_leaf)) mean_masked,
               sum(history_events_before_truncation > history_events) n_truncated
        FROM read_parquet('{dataset}')
        """
    ).fetchone()
    keys = [
        "n_rows", "n_unique_snapshot", "n_users", "min_snapshot", "max_snapshot",
        "mean_history", "median_history", "max_history", "mean_labels_all",
        "mean_labels_positive", "n_positive", "n_multilabel", "mean_masked", "n_truncated",
    ]
    stats = dict(zip(keys, summary))

    checks = {}
    checks["unique_snapshot_id"] = stats["n_rows"] == stats["n_unique_snapshot"]
    checks["history_length_bounds"] = con.execute(
        f"SELECT count(*) = 0 FROM read_parquet('{dataset}') "
        f"WHERE history_events < {int(config['min_history_events'])} "
        f"OR history_events > {int(config['max_history_events'])}"
    ).fetchone()[0]
    checks["sequence_arrays_aligned"] = con.execute(
        f"""
        SELECT count(*) = 0 FROM read_parquet('{dataset}')
        WHERE len(action) != len(vert) OR len(action) != len(cat)
           OR len(action) != len(param1) OR len(action) != len(leaf)
           OR len(action) != len(count_views) OR len(action) != len(ts)
           OR len(action) != history_events
        """
    ).fetchone()[0]
    checks["history_strictly_before_T"] = con.execute(
        f"""
        SELECT count(*) = 0 FROM read_parquet('{dataset}')
        WHERE list_max(ts) >= epoch(snapshot_ts)
           OR list_min(ts) < epoch(snapshot_ts - INTERVAL '{history_days} days')
        """
    ).fetchone()[0]
    checks["labels_not_masked"] = con.execute(
        f"""
        SELECT count(*) = 0 FROM read_parquet('{dataset}') t,
             unnest(t.label_leaf) AS label(leaf_id)
        WHERE list_contains(t.masked_leaf, label.leaf_id)
        """
    ).fetchone()[0]
    checks["label_grade_aligned"] = con.execute(
        f"SELECT count(*) = 0 FROM read_parquet('{dataset}') "
        "WHERE len(label_leaf) != len(label_grade) OR len(label_leaf) != n_labels"
    ).fetchone()[0]
    checks["candidate_count_consistent"] = con.execute(
        f"SELECT count(*) = 0 FROM read_parquet('{dataset}') "
        f"WHERE n_candidates != {n_leaf} - len(masked_leaf)"
    ).fetchone()[0]
    checks["leaf_ids_in_vocab"] = con.execute(
        f"""
        SELECT count(*) = 0 FROM (
            SELECT unnest(leaf) leaf_id FROM read_parquet('{dataset}')
            UNION ALL SELECT unnest(label_leaf) FROM read_parquet('{dataset}')
            UNION ALL SELECT unnest(masked_leaf) FROM read_parquet('{dataset}')
        ) WHERE leaf_id < 1 OR leaf_id > {n_leaf}
        """
    ).fetchone()[0]
    checks["multilabel_preserved"] = stats["n_multilabel"] > 0
    checks["positive_examples_exist"] = stats["n_positive"] > 0

    samples = con.execute(
        f"""
        SELECT user_id, snapshot_ts, history_events,
               list_slice(leaf, greatest(len(leaf)-9, 1), len(leaf)) AS last_leaf_ids,
               label_leaf, masked_leaf
        FROM read_parquet('{dataset}')
        WHERE n_labels > 0
        ORDER BY hash(snapshot_id)
        LIMIT 20
        """
    ).fetchdf().to_dict(orient="records")
    con.close()

    status = "pass" if all(checks.values()) else "fail"
    report = {
        "status": status,
        "dataset": dataset,
        "n_leaf": n_leaf,
        "stats": stats,
        "checks": checks,
        "manual_samples": samples,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": status, "output": str(args.output)}, ensure_ascii=False))
    if status != "pass":
        failed = [name for name, passed in checks.items() if not passed]
        print(f"Failed checks: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

