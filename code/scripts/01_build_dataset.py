#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.dataset_builder import BuildConfig, SnapshotBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", action="append", help="YYYY-MM-DD; repeat to build several")
    parser.add_argument("--bucket-limit", type=int, default=None, help="Smoke-test only")
    parser.add_argument("--n-buckets", type=int, default=None, help="Override config for performance tests")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--shards-per-bucket", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = BuildConfig.from_json(args.config)
    if args.n_buckets is not None:
        if args.n_buckets < 1:
            raise ValueError("--n-buckets must be positive")
        config = replace(config, n_buckets=args.n_buckets)
    snapshots = tuple(args.snapshot or config.snapshot_dates)
    unknown = set(snapshots) - set(config.snapshot_dates)
    if unknown:
        raise ValueError(f"Snapshots absent from config: {sorted(unknown)}")

    builder = SnapshotBuilder(
        ROOT,
        config,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    try:
        for snapshot in snapshots:
            output = builder.build_snapshot(
                snapshot,
                bucket_limit=args.bucket_limit,
                overwrite=args.overwrite,
                shards_per_bucket=args.shards_per_bucket,
            )
            print(output)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
