from datetime import date
from pathlib import Path

import duckdb
import pytest

from avito_recsys.scaling_sampling import (
    audit_nested_user_prefixes,
    audit_temporal_parquet_boundaries,
    load_fixed_evaluation_frame,
    load_positive_frame,
    validate_temporal_split,
)
from avito_recsys.snapshot_identity import (
    is_pseudonymous_snapshot_id,
    pseudonymize_snapshot_id,
)


def _write_snapshot(
    path: Path,
    snapshot_date: date,
    rows: list[tuple[int, int]],
    user_column: str = "user_id",
) -> None:
    values = ",".join(f"({user_id}, {n_labels})" for user_id, n_labels in rows)
    timestamp = snapshot_date.isoformat()
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
                SELECT user_id AS {user_column},
                       TIMESTAMP '{timestamp}' AS snapshot_ts,
                       CAST(user_id AS VARCHAR) || '_{timestamp}' AS snapshot_id,
                       [1]::INTEGER[] AS leaf,
                       [1]::INTEGER[] AS action,
                       [epoch(TIMESTAMP '{timestamp}')::BIGINT]::BIGINT[] AS ts,
                       CASE WHEN n_labels > 0 THEN [2]::INTEGER[] ELSE []::INTEGER[] END AS label_leaf,
                       []::INTEGER[] AS masked_leaf,
                       n_labels::INTEGER AS n_labels
                FROM (VALUES {values}) AS source(user_id, n_labels)
            ) TO '{path}' (FORMAT PARQUET)
            """
        )
    finally:
        con.close()


def test_user_sampling_keeps_all_positive_snapshots_and_is_nested(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_snapshot(first, date(2026, 1, 1), [(1, 1), (2, 1), (3, 0), (4, 1)])
    _write_snapshot(second, date(2026, 2, 1), [(1, 1), (2, 0), (3, 1), (4, 1)])
    paths = [first, second]

    one_user, one_stats = load_positive_frame(paths, 1, "users", subset_seed=17)
    two_users, two_stats = load_positive_frame(paths, 2, "users", subset_seed=17)
    assert one_stats.selected_users == 1
    assert two_stats.selected_users == 2
    assert set(one_user.snapshot_id).issubset(set(two_users.snapshot_id))

    audit = audit_nested_user_prefixes(paths, [1, 2, 3], subset_seed=17)
    assert audit["nested_prefixes_verified"] is True
    assert [point["user_cap"] for point in audit["points"]] == [1, 2, 3]
    # A selected user contributes all of their positive snapshots, not one row.
    assert audit["points"][0]["selected_positive_user_snapshot_rows"] >= 1


def test_subset_seed_is_separate_from_model_seed_contract(tmp_path):
    snapshot = tmp_path / "snapshot.parquet"
    _write_snapshot(snapshot, date(2026, 1, 1), [(user_id, 1) for user_id in range(1, 21)])
    first, _ = load_positive_frame([snapshot], 5, "users", subset_seed=11)
    repeated, _ = load_positive_frame([snapshot], 5, "users", subset_seed=11)
    other, _ = load_positive_frame([snapshot], 5, "users", subset_seed=12)
    assert first.snapshot_id.tolist() == repeated.snapshot_id.tolist()
    assert set(first.snapshot_id) != set(other.snapshot_id)


def test_pseudonymous_user_key_schema_is_supported_and_source_ids_are_not_returned(tmp_path):
    snapshot = tmp_path / "snapshot.parquet"
    _write_snapshot(
        snapshot,
        date(2026, 1, 1),
        [(1, 1), (2, 1), (3, 1)],
        user_column="user_key",
    )
    frame, stats = load_positive_frame([snapshot], 2, "users", subset_seed=42)
    assert stats.selected_users == 2
    assert "user_key" not in frame.columns
    assert "user_id" not in frame.columns
    # Source snapshot IDs contain user identifiers in this fixture; outputs do not.
    assert all("_2026-01-01" not in value for value in frame.snapshot_id)


def test_scaling_loader_keeps_already_pseudonymous_snapshot_ids_stable(tmp_path):
    source = tmp_path / "source.parquet"
    prepared = tmp_path / "prepared.parquet"
    _write_snapshot(source, date(2026, 1, 1), [(1, 1), (2, 1)])
    con = duckdb.connect()
    try:
        con.execute(
            "COPY (SELECT * REPLACE (sha256(CAST(snapshot_id AS VARCHAR)) AS snapshot_id) "
            f"FROM read_parquet('{source}')) TO '{prepared}' (FORMAT PARQUET)"
        )
    finally:
        con.close()

    raw = "1_2026-01-01"
    expected = pseudonymize_snapshot_id(raw)
    frame = load_fixed_evaluation_frame([prepared], cap=0)
    assert expected in set(frame.snapshot_id)
    assert frame.snapshot_id.map(is_pseudonymous_snapshot_id).all()


def test_temporal_guard_accounts_for_target_window():
    validate_temporal_split(["2026-04-01", "2026-05-01"], "2026-06-01", 14)
    with pytest.raises(ValueError, match="Temporal leakage"):
        validate_temporal_split(["2026-05-25"], "2026-06-01", 14)
    with pytest.raises(ValueError, match="also present"):
        validate_temporal_split(["2026-06-01"], "2026-06-01", 14)


def test_parquet_temporal_audit_checks_actual_snapshot_timestamps(tmp_path):
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    _write_snapshot(train, date(2026, 5, 1), [(1, 1)])
    _write_snapshot(validation, date(2026, 6, 1), [(1, 1)])
    audit = audit_temporal_parquet_boundaries([train], validation, target_days=14)
    assert audit["passed"] is True
    assert audit["train_snapshot_ts_max"].startswith("2026-05-01")
