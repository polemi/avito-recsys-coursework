#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash", action="store_true", dest="with_hash")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("events_c*.parquet")) + [
        raw / "contacts.parquet",
        raw / "taxonomy.parquet",
        raw / "socdem.parquet",
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    con = duckdb.connect()
    events_glob = str(raw / "events_c*.parquet")
    contacts = str(raw / "contacts.parquet")
    taxonomy = str(raw / "taxonomy.parquet")
    event_stats = con.execute(
        f"""
        SELECT count(*) n_rows, approx_count_distinct(user_id) n_users,
               min(ts) min_ts, max(ts) max_ts,
               sum(user_id IS NULL) null_user, sum(ts IS NULL) null_ts,
               sum(vertical IS NULL OR category IS NULL OR param1 IS NULL) null_taxonomy
        FROM read_parquet('{events_glob}')
        """
    ).fetchone()
    contact_stats = con.execute(
        f"""
        SELECT count(*) n_rows, approx_count_distinct(user_id) n_users,
               min(ts) min_ts, max(ts) max_ts,
               sum(user_id IS NULL) null_user, sum(ts IS NULL) null_ts,
               sum(vertical IS NULL OR category IS NULL OR param1 IS NULL) null_taxonomy
        FROM read_parquet('{contacts}')
        """
    ).fetchone()
    taxonomy_stats = con.execute(
        f"""
        SELECT count(*) n_rows,
               count(*) FILTER (
                   WHERE vertical IS NOT NULL AND category IS NOT NULL AND param1 IS NOT NULL
               ) n_rows_valid,
               sum(vertical IS NULL OR category IS NULL OR param1 IS NULL) n_rows_incomplete,
               count(DISTINCT vertical) n_vertical,
               count(DISTINCT category) n_category, count(DISTINCT param1) n_param1,
               count(DISTINCT (vertical, category, param1)) FILTER (
                   WHERE vertical IS NOT NULL AND category IS NOT NULL AND param1 IS NOT NULL
               ) n_leaf_valid
        FROM read_parquet('{taxonomy}')
        """
    ).fetchone()
    duplicate_param_names = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT param1 FROM read_parquet('{taxonomy}')
            WHERE vertical IS NOT NULL AND category IS NOT NULL AND param1 IS NOT NULL
            GROUP BY param1 HAVING count(DISTINCT (vertical, category)) > 1
        )
        """
    ).fetchone()[0]
    con.close()

    event_min, event_max = event_stats[2], event_stats[3]
    contact_min, contact_max = contact_stats[2], contact_stats[3]
    snapshots = []
    all_feasible = True
    for value in config["snapshot_dates"]:
        snapshot = datetime.fromisoformat(value)
        history_start = snapshot - timedelta(days=config["history_days"])
        novelty_start = snapshot - timedelta(days=config["novelty_days"])
        target_end = snapshot + timedelta(days=config["target_days"])
        feasible = (
            event_min <= history_start
            and event_max >= snapshot - timedelta(seconds=1)
            and contact_min <= novelty_start
            and contact_max >= target_end - timedelta(seconds=1)
        )
        all_feasible = all_feasible and feasible
        snapshots.append(
            {
                "snapshot": value,
                "history_start": history_start.isoformat(),
                "target_start": snapshot.isoformat(),
                "target_end_exclusive": target_end.isoformat(),
                "feasible": feasible,
            }
        )

    manifest_files = []
    for path in files:
        resolved = path.resolve()
        item = {
            "logical_path": str(path.relative_to(root)),
            "source_path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "mtime": datetime.fromtimestamp(resolved.stat().st_mtime).isoformat(),
        }
        if args.with_hash:
            item["sha256"] = sha256(resolved)
        manifest_files.append(item)

    report = {
        "status": "pass" if all_feasible and not missing else "fail",
        "config": config,
        "events": dict(zip(
            ["n_rows", "n_users_approx", "min_ts", "max_ts", "null_user", "null_ts", "null_taxonomy"],
            event_stats,
        )),
        "contacts": dict(zip(
            ["n_rows", "n_users_approx", "min_ts", "max_ts", "null_user", "null_ts", "null_taxonomy"],
            contact_stats,
        )),
        "taxonomy": dict(zip(
            ["n_rows", "n_rows_valid", "n_rows_incomplete", "n_vertical", "n_category",
             "n_param1", "n_leaf_valid"], taxonomy_stats
        )),
        "duplicate_param1_names_across_paths": duplicate_param_names,
        "snapshots": snapshots,
        "files": manifest_files,
    }
    report_path = args.output / "input_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    with (args.output / "fold_calendar.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["fold", "role", "train_snapshots", "validation_snapshot", "test_snapshot"],
        )
        writer.writeheader()
        for fold in config["folds"]:
            writer.writerow({**fold, "train_snapshots": ";".join(fold["train_snapshots"])})
    print(json.dumps({"status": report["status"], "report": str(report_path)}, ensure_ascii=False))
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
