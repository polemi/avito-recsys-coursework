from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class BuildConfig:
    history_days: int
    novelty_days: int
    target_days: int
    min_history_events: int
    max_history_events: int
    max_views_per_leaf: int
    n_buckets: int
    snapshot_dates: tuple[str, ...]
    data_version: str = "v90"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    events_pattern: str = "events_c*.parquet"
    contacts_pattern: str = "contacts.parquet"
    taxonomy_file: str = "taxonomy.parquet"
    partition_column: str = "bucket"
    event_count_column: str | None = None
    vocab_file: str = "vocab_leaf_v90.json"

    @classmethod
    def from_json(cls, path: Path) -> "BuildConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            history_days=int(raw["history_days"]),
            novelty_days=int(raw["novelty_days"]),
            target_days=int(raw["target_days"]),
            min_history_events=int(raw["min_history_events"]),
            max_history_events=int(raw["max_history_events"]),
            max_views_per_leaf=int(raw["max_views_per_leaf"]),
            n_buckets=int(raw["n_buckets"]),
            snapshot_dates=tuple(raw["snapshot_dates"]),
            data_version=str(raw.get("data_version", "v90")),
            raw_dir=str(raw.get("raw_dir", "data/raw")),
            processed_dir=str(raw.get("processed_dir", "data/processed")),
            events_pattern=str(raw.get("events_pattern", "events_c*.parquet")),
            contacts_pattern=str(raw.get("contacts_pattern", "contacts.parquet")),
            taxonomy_file=str(raw.get("taxonomy_file", "taxonomy.parquet")),
            partition_column=str(raw.get("partition_column", "bucket")),
            event_count_column=raw.get("event_count_column"),
            vocab_file=str(raw.get("vocab_file", "vocab_leaf_v90.json")),
        )


class SnapshotBuilder:
    def __init__(
        self,
        project_root: Path,
        config: BuildConfig,
        threads: int = 4,
        memory_limit: str = "16GB",
    ):
        self.root = project_root.resolve()
        self.config = config
        self.raw = self.root / config.raw_dir
        self.output = self.root / config.processed_dir
        self.snapshot_dir = self.output / "snapshots"
        self.tmp = self.output / "_duck_tmp"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.events_glob = str(self.raw / config.events_pattern)
        self.contacts = str(self.raw / config.contacts_pattern)
        self.taxonomy = str(self.raw / config.taxonomy_file)
        self.partitioned = self.output / "raw_partitioned"
        self.use_partitioned = (
            (self.partitioned / "events" / "_SUCCESS").exists()
            and (self.partitioned / "contacts" / "_SUCCESS").exists()
        )
        self.con = duckdb.connect()
        self.con.execute(
            f"PRAGMA threads={int(threads)}; "
            f"SET memory_limit='{memory_limit}'; "
            f"SET temp_directory='{self.tmp}'; "
            "SET preserve_insertion_order=false;"
        )
        self.n_leaf = self._prepare_taxonomy()

    def _prepare_taxonomy(self) -> int:
        self.con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE v_leaf AS
            SELECT vertical, category, param1,
                   row_number() OVER (ORDER BY vertical, category, param1)::INTEGER AS leaf_id,
                   dense_rank() OVER (ORDER BY vertical)::INTEGER AS vert_id,
                   dense_rank() OVER (ORDER BY category)::INTEGER AS cat_id,
                   dense_rank() OVER (ORDER BY param1)::INTEGER AS param1_id
            FROM (
                SELECT DISTINCT vertical, category, param1
                FROM read_parquet('{self.taxonomy}')
                WHERE vertical IS NOT NULL AND category IS NOT NULL AND param1 IS NOT NULL
            )
            """
        )
        rows = self.con.execute(
            "SELECT vertical, category, param1, leaf_id, vert_id, cat_id, param1_id "
            "FROM v_leaf ORDER BY leaf_id"
        ).fetchall()
        vocab = {
            "leaf": {f"{v}|{c}|{p}": int(leaf) for v, c, p, leaf, _, _, _ in rows},
            "leaf2vert": {str(leaf): int(vert) for _, _, _, leaf, vert, _, _ in rows},
            "leaf2cat": {str(leaf): int(cat) for _, _, _, leaf, _, cat, _ in rows},
            "leaf2param1": {str(leaf): int(param1) for _, _, _, leaf, _, _, param1 in rows},
            "action": {"<PAD>": 0, "view": 1, "favorite": 2, "contact_dab": 3, "contact_dtb": 4},
        }
        (self.output / self.config.vocab_file).write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return len(rows)

    def _query(
        self,
        snapshot: datetime,
        bucket: int,
        n_buckets: int,
        shard: int = 0,
        shards_per_bucket: int = 1,
    ) -> str:
        cfg = self.config
        history_start = snapshot - timedelta(days=cfg.history_days)
        novelty_start = snapshot - timedelta(days=cfg.novelty_days)
        target_end = snapshot + timedelta(days=cfg.target_days)
        if self.use_partitioned:
            partition = self.config.partition_column
            event_source = str(self.partitioned / "events" / f"{partition}={bucket}" / "*.parquet")
            contact_source = str(self.partitioned / "contacts" / f"{partition}={bucket}" / "*.parquet")
            bucket_filter = f"hash(user_id) % {shards_per_bucket} = {shard}"
        else:
            event_source = self.events_glob
            contact_source = self.contacts
            bucket_filter = (
                f"hash(user_id) % {n_buckets} = {bucket} "
                f"AND hash(user_id) % {shards_per_bucket} = {shard}"
            )
        raw_view_count = self.config.event_count_column or "1"
        return f"""
        WITH history_raw AS (
            SELECT user_id, ts,
                   CASE action_type WHEN 'view' THEN 1 ELSE 2 END AS action,
                   vertical, category, param1,
                   action_type = 'view' AS is_view,
                   CASE WHEN action_type = 'view' THEN {raw_view_count} ELSE 0 END::INTEGER AS raw_view_count,
                   {bucket}::INTEGER AS user_bucket_index
            FROM read_parquet('{event_source}', union_by_name=true)
            WHERE {bucket_filter}
              AND ts >= TIMESTAMP '{history_start:%Y-%m-%d %H:%M:%S}'
              AND ts <  TIMESTAMP '{snapshot:%Y-%m-%d %H:%M:%S}'
            UNION ALL
            SELECT user_id, ts,
                   CASE action_type WHEN 'contact_dtb' THEN 4 ELSE 3 END AS action,
                   vertical, category, param1,
                   false AS is_view,
                   0::INTEGER AS raw_view_count,
                   {bucket}::INTEGER AS user_bucket_index
            FROM read_parquet('{contact_source}', union_by_name=true)
            WHERE {bucket_filter}
              AND ts >= TIMESTAMP '{history_start:%Y-%m-%d %H:%M:%S}'
              AND ts <  TIMESTAMP '{snapshot:%Y-%m-%d %H:%M:%S}'
        ),
        encoded AS (
            SELECT h.user_id, h.ts, h.action, h.is_view, h.raw_view_count, h.user_bucket_index,
                   l.vert_id, l.cat_id, l.param1_id, l.leaf_id
            FROM history_raw h
            JOIN v_leaf l USING (vertical, category, param1)
        ),
        with_view_stats AS (
            SELECT *,
                   sum(raw_view_count) FILTER (WHERE is_view)
                       OVER (PARTITION BY user_id, leaf_id) AS total_leaf_views,
                   row_number() OVER (
                       PARTITION BY user_id, leaf_id, is_view
                       ORDER BY ts DESC, action DESC
                   ) AS view_rank
            FROM encoded
        ),
        compressed AS (
            SELECT user_id, ts, action, vert_id, cat_id, param1_id, leaf_id, user_bucket_index,
                   CASE WHEN is_view THEN total_leaf_views ELSE 0 END::INTEGER AS count_views
            FROM with_view_stats
            WHERE NOT is_view OR view_rank <= {cfg.max_views_per_leaf}
        ),
        ranked_history AS (
            SELECT *,
                   count(*) OVER (PARTITION BY user_id) AS history_events_before_truncation,
                   row_number() OVER (
                       PARTITION BY user_id
                       ORDER BY ts DESC, action DESC, leaf_id DESC
                   ) AS reverse_rank
            FROM compressed
        ),
        truncated AS (
            SELECT * FROM ranked_history
            WHERE reverse_rank <= {cfg.max_history_events}
        ),
        sequences AS (
            SELECT user_id,
                   list(action ORDER BY ts, action, leaf_id) AS action,
                   list(vert_id ORDER BY ts, action, leaf_id) AS vert,
                   list(cat_id ORDER BY ts, action, leaf_id) AS cat,
                   list(param1_id ORDER BY ts, action, leaf_id) AS param1,
                   list(leaf_id ORDER BY ts, action, leaf_id) AS leaf,
                   list(count_views ORDER BY ts, action, leaf_id) AS count_views,
                   list(epoch(ts)::BIGINT ORDER BY ts, action, leaf_id) AS ts,
                   min(user_bucket_index)::INTEGER AS user_bucket_index,
                   max(history_events_before_truncation)::INTEGER AS history_events_before_truncation,
                   count(*)::INTEGER AS history_events
            FROM truncated
            GROUP BY user_id
            HAVING count(*) >= {cfg.min_history_events}
        ),
        masked AS (
            SELECT c.user_id, l.leaf_id
            FROM read_parquet('{contact_source}', union_by_name=true) c
            JOIN v_leaf l USING (vertical, category, param1)
            WHERE {bucket_filter}
              AND c.ts >= TIMESTAMP '{novelty_start:%Y-%m-%d %H:%M:%S}'
              AND c.ts <  TIMESTAMP '{snapshot:%Y-%m-%d %H:%M:%S}'
            GROUP BY c.user_id, l.leaf_id
        ),
        masked_lists AS (
            SELECT user_id, list(leaf_id ORDER BY leaf_id) AS masked_leaf
            FROM masked GROUP BY user_id
        ),
        future AS (
            SELECT c.user_id, l.leaf_id,
                   max(CASE WHEN c.action_type = 'contact_dtb' THEN 2 ELSE 1 END)::INTEGER AS grade
            FROM read_parquet('{contact_source}', union_by_name=true) c
            JOIN v_leaf l USING (vertical, category, param1)
            WHERE {bucket_filter}
              AND c.ts >= TIMESTAMP '{snapshot:%Y-%m-%d %H:%M:%S}'
              AND c.ts <  TIMESTAMP '{target_end:%Y-%m-%d %H:%M:%S}'
            GROUP BY c.user_id, l.leaf_id
        ),
        novel_labels AS (
            SELECT f.user_id, f.leaf_id, f.grade
            FROM future f
            LEFT JOIN masked m USING (user_id, leaf_id)
            WHERE m.leaf_id IS NULL
        ),
        label_lists AS (
            SELECT user_id,
                   list(leaf_id ORDER BY leaf_id) AS label_leaf,
                   list(grade ORDER BY leaf_id) AS label_grade
            FROM novel_labels GROUP BY user_id
        )
        SELECT s.user_id,
               TIMESTAMP '{snapshot:%Y-%m-%d %H:%M:%S}' AS snapshot_ts,
               concat(s.user_id::VARCHAR, '_', epoch(TIMESTAMP '{snapshot:%Y-%m-%d %H:%M:%S}')::BIGINT::VARCHAR) AS snapshot_id,
               s.action, s.vert, s.cat, s.param1, s.leaf, s.count_views, s.ts,
               s.history_events_before_truncation, s.history_events, s.user_bucket_index,
               COALESCE(l.label_leaf, []::INTEGER[]) AS label_leaf,
               COALESCE(l.label_grade, []::INTEGER[]) AS label_grade,
               COALESCE(m.masked_leaf, []::INTEGER[]) AS masked_leaf,
               len(COALESCE(l.label_leaf, []::INTEGER[]))::INTEGER AS n_labels,
               ({self.n_leaf} - len(COALESCE(m.masked_leaf, []::INTEGER[])))::INTEGER AS n_candidates
        FROM sequences s
        LEFT JOIN label_lists l USING (user_id)
        LEFT JOIN masked_lists m USING (user_id)
        """

    def build_snapshot(
        self,
        date: str,
        bucket_limit: int | None = None,
        overwrite: bool = False,
        shards_per_bucket: int = 1,
    ) -> Path:
        snapshot = datetime.fromisoformat(date)
        output_path = self.snapshot_dir / f"snapshot_{date}.parquet"
        if output_path.exists() and not overwrite:
            return output_path
        n_buckets = self.config.n_buckets
        selected_buckets = n_buckets if bucket_limit is None else min(bucket_limit, n_buckets)
        if shards_per_bucket < 1:
            raise ValueError("shards_per_bucket must be positive")
        parts: list[Path] = []
        for bucket in range(selected_buckets):
            for shard in range(shards_per_bucket):
                part = self.tmp / (
                    f"snapshot_{date}_bucket_{bucket:02d}_"
                    f"shard_{shard:02d}-of-{shards_per_bucket:02d}.parquet"
                )
                if not part.exists():
                    query = self._query(
                        snapshot,
                        bucket,
                        n_buckets,
                        shard=shard,
                        shards_per_bucket=shards_per_bucket,
                    )
                    self.con.execute(
                        f"COPY ({query}) TO '{part}' (FORMAT PARQUET, COMPRESSION ZSTD)"
                    )
                n_rows = self.con.execute(
                    f"SELECT count(*) FROM read_parquet('{part}')"
                ).fetchone()[0]
                print(
                    f"[{date}] bucket {bucket + 1}/{selected_buckets}, "
                    f"shard {shard + 1}/{shards_per_bucket}: {n_rows:,} snapshots",
                    flush=True,
                )
                parts.append(part)
        quoted = ", ".join(f"'{part}'" for part in parts)
        if output_path.exists():
            output_path.unlink()
        self.con.execute(
            f"COPY (SELECT * FROM read_parquet([{quoted}])) "
            f"TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        for part in parts:
            part.unlink()
        return output_path

    def close(self) -> None:
        self.con.close()
        if self.tmp.exists() and not any(self.tmp.iterdir()):
            shutil.rmtree(self.tmp)
