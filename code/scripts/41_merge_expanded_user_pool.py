#!/usr/bin/env python3
"""Merge two disjoint modulo-30 samples into one leakage-safe scaling pool.

The original refreshed data use hash bucket 7 modulo 30.  The expansion uses
bucket 8 modulo 30, so raw users must be disjoint.  This script verifies that
contract for every train snapshot, keeps the original June/July evaluation
files unchanged, and records exact row/user counts without exporting user ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb


TRAIN_DATES = (
    "2025-11-01",
    "2025-12-01",
    "2026-01-01",
    "2026-02-01",
    "2026-03-01",
    "2026-04-01",
    "2026-05-01",
)
VALIDATION_DATE = "2026-06-01"
TEST_DATE = "2026-07-01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-snapshots",
        type=Path,
        default=Path("data/refresh_2026_08/processed/snapshots"),
    )
    parser.add_argument(
        "--extension-snapshots",
        type=Path,
        default=Path("data/scaling_user_expansion_2026_10/bucket8/processed/snapshots"),
    )
    parser.add_argument(
        "--base-vocab",
        type=Path,
        default=Path("data/refresh_2026_08/processed/vocab_leaf_refresh_2026_08.json"),
    )
    parser.add_argument(
        "--extension-vocab",
        type=Path,
        default=Path("data/scaling_user_expansion_2026_10/bucket8/processed/vocab_leaf_bucket8.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scaling_user_expansion_2026_10/combined_2of30"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    resolve = lambda value: value if value.is_absolute() else project / value
    base = resolve(args.base_snapshots)
    extension = resolve(args.extension_snapshots)
    base_vocab = resolve(args.base_vocab)
    extension_vocab = resolve(args.extension_vocab)
    output = resolve(args.output_dir)
    snapshots = output / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    for path in [base_vocab, extension_vocab]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if json.loads(base_vocab.read_text()) != json.loads(extension_vocab.read_text()):
        raise RuntimeError("Base and extension vocabularies differ")
    shutil.copy2(base_vocab, output / "vocab.json")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4; SET memory_limit='16GB'; SET preserve_insertion_order=false;")
    base_train_paths = [base / f"snapshot_{date}.parquet" for date in TRAIN_DATES]
    extension_train_paths = [extension / f"snapshot_{date}.parquet" for date in TRAIN_DATES]
    for path in [*base_train_paths, *extension_train_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    base_source = ",".join(f"'{path.resolve()}'" for path in base_train_paths)
    extension_source = ",".join(f"'{path.resolve()}'" for path in extension_train_paths)
    pool_user_overlap = con.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT DISTINCT user_id
            FROM read_parquet([{base_source}], union_by_name=true)
        ) a
        JOIN (
            SELECT DISTINCT user_id
            FROM read_parquet([{extension_source}], union_by_name=true)
        ) b USING (user_id)
        """
    ).fetchone()[0]
    if pool_user_overlap:
        raise RuntimeError(f"Base and extension train pools overlap: {pool_user_overlap:,} users")
    per_snapshot: list[dict] = []
    for date in TRAIN_DATES:
        left = base / f"snapshot_{date}.parquet"
        right = extension / f"snapshot_{date}.parquet"
        target = snapshots / f"snapshot_{date}.parquet"
        for path in (left, right):
            if not path.is_file():
                raise FileNotFoundError(path)
        overlap = con.execute(
            f"""
            SELECT count(*)
            FROM (SELECT DISTINCT user_id FROM read_parquet('{left.resolve()}')) a
            JOIN (SELECT DISTINCT user_id FROM read_parquet('{right.resolve()}')) b USING (user_id)
            """
        ).fetchone()[0]
        if overlap:
            raise RuntimeError(f"Samples overlap on {date}: {overlap:,} users")
        if target.exists() and args.overwrite:
            target.unlink()
        if not target.exists():
            con.execute(
                f"""
                COPY (
                    SELECT * REPLACE (user_bucket_index::INTEGER AS user_bucket_index),
                           7::INTEGER AS source_sample_bucket
                    FROM read_parquet('{left.resolve()}')
                    UNION ALL
                    SELECT * REPLACE ((user_bucket_index + 20)::INTEGER AS user_bucket_index),
                           8::INTEGER AS source_sample_bucket
                    FROM read_parquet('{right.resolve()}')
                ) TO '{target.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        stats = con.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT user_id) AS users,
                   count_if(n_labels > 0) AS positive_rows,
                   count(DISTINCT user_id) FILTER (WHERE n_labels > 0) AS positive_users,
                   count(DISTINCT snapshot_id) AS snapshot_ids
            FROM read_parquet('{target.resolve()}')
            """
        ).fetchone()
        if stats[0] != stats[4]:
            raise RuntimeError(f"Duplicate snapshot_id values after merge on {date}")
        per_snapshot.append(
            {
                "date": date,
                "rows": int(stats[0]),
                "users": int(stats[1]),
                "positive_rows": int(stats[2]),
                "positive_users": int(stats[3]),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )

    fixed_evaluation: dict[str, dict] = {}
    for date in (VALIDATION_DATE, TEST_DATE):
        source = base / f"snapshot_{date}.parquet"
        target = snapshots / f"snapshot_{date}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        link_or_copy(source, target)
        source_digest = sha256(source)
        target_digest = sha256(target)
        if source_digest != target_digest:
            raise RuntimeError(f"Fixed evaluation snapshot changed during copy: {date}")
        fixed_evaluation[date] = {
            "source": str(source),
            "bytes": target.stat().st_size,
            "sha256": target_digest,
            "byte_identical_to_original": True,
        }

    sources = ",".join(
        f"'{(snapshots / f'snapshot_{date}.parquet').resolve()}'" for date in TRAIN_DATES
    )
    pool = con.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT user_id) AS users,
               count_if(n_labels > 0) AS positive_rows,
               count(DISTINCT user_id) FILTER (WHERE n_labels > 0) AS positive_users,
               count(DISTINCT snapshot_id) AS snapshot_ids
        FROM read_parquet([{sources}], union_by_name=true)
        """
    ).fetchone()
    con.close()
    if pool[0] != pool[4]:
        raise RuntimeError("Duplicate train snapshot_id values in combined pool")
    config = {
        "data_version": "scaling_user_expansion_2of30_2026_10",
        "processed_dir": str(output.relative_to(project)) if output.is_relative_to(project) else str(output),
        "vocab_file": "vocab.json",
        "history_days": 90,
        "novelty_days": 90,
        "target_days": 14,
        "folds": [
            {
                "fold": 0,
                "train_snapshots": list(TRAIN_DATES),
                "validation_snapshot": VALIDATION_DATE,
                "test_snapshot": TEST_DATE,
                "role": "scaling_extension",
            }
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_samples": ["hash(user_id) mod 30 = 7", "hash(user_id) mod 30 = 8"],
        "train_dates": list(TRAIN_DATES),
        "validation_date": VALIDATION_DATE,
        "test_date": TEST_DATE,
        "validation_and_test_source": "unchanged original bucket-7 snapshots",
        "fixed_evaluation": fixed_evaluation,
        "per_snapshot": per_snapshot,
        "train_pool": {
            "rows": int(pool[0]),
            "distinct_users": int(pool[1]),
            "positive_user_snapshot_rows": int(pool[2]),
            "distinct_positive_users": int(pool[3]),
            "distinct_snapshot_ids": int(pool[4]),
        },
        "disjoint_samples_verified": True,
        "cross_bucket_train_user_overlap": int(pool_user_overlap),
        "duplicate_snapshot_ids": 0,
        "temporal_contract": {
            "last_train_snapshot": "2026-05-01",
            "last_train_target_end_exclusive": "2026-05-15",
            "validation_snapshot": VALIDATION_DATE,
            "test_snapshot": TEST_DATE,
            "passed": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
