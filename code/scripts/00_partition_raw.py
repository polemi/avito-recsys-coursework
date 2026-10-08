#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb


def partition(con: duckdb.DuckDBPyConnection, source: str, output: Path, n_buckets: int) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    con.execute(
        f"""
        COPY (
            SELECT *, (hash(user_id) % {n_buckets})::INTEGER AS bucket
            FROM read_parquet('{source}')
        ) TO '{output}' (
            FORMAT PARQUET,
            PARTITION_BY (bucket),
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """
    )
    (output / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-buckets", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_buckets < 1:
        raise ValueError("--n-buckets must be positive")

    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    output = root / "data" / "processed" / "raw_partitioned"
    events_out = output / "events"
    contacts_out = output / "contacts"
    complete = (events_out / "_SUCCESS").exists() and (contacts_out / "_SUCCESS").exists()
    if complete and not args.overwrite:
        print(f"Already complete: {output}")
        return

    tmp = root / "data" / "processed" / "_partition_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "PRAGMA threads=4; SET memory_limit='12GB'; "
        f"SET temp_directory='{tmp}'; SET preserve_insertion_order=false;"
    )
    try:
        partition(con, str(raw / "events_c*.parquet"), events_out, args.n_buckets)
        partition(con, str(raw / "contacts.parquet"), contacts_out, args.n_buckets)
    finally:
        con.close()
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()
    print(output)


if __name__ == "__main__":
    main()
