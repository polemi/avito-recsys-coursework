import hashlib

import duckdb
import pandas as pd
import pytest

from avito_recsys.snapshot_identity import (
    SNAPSHOT_ID_CONTRACT,
    duckdb_pseudonymous_snapshot_id_sql,
    duckdb_snapshot_id_match_sql,
    is_pseudonymous_snapshot_id,
    pseudonymize_snapshot_id,
    pseudonymize_snapshot_ids,
)


def test_snapshot_id_contract_is_private_deterministic_and_idempotent():
    raw = "315980500007_1782864000"
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    assert SNAPSHOT_ID_CONTRACT == "sha256-lower-hex-v1"
    assert pseudonymize_snapshot_id(raw) == expected
    assert pseudonymize_snapshot_id(expected) == expected
    assert pseudonymize_snapshot_id(expected.upper()) == expected
    assert is_pseudonymous_snapshot_id(expected)
    assert raw not in expected
    with pytest.raises(ValueError, match="non-null"):
        pseudonymize_snapshot_id(None)


def test_legacy_raw_and_current_hashed_artifacts_pair_one_to_one():
    raw_ids = ["101_1782864000", "202_1782864000"]
    current_ids = [pseudonymize_snapshot_id(value) for value in raw_ids]
    legacy = pd.DataFrame({"snapshot_id": raw_ids, "legacy_metric": [0.0, 1.0]})
    current = pd.DataFrame({"snapshot_id": current_ids, "current_metric": [1.0, 0.0]})

    legacy = pseudonymize_snapshot_ids(legacy)
    current = pseudonymize_snapshot_ids(current)
    paired = legacy.merge(current, on="snapshot_id", validate="one_to_one")

    assert len(paired) == 2
    assert paired["snapshot_id"].map(is_pseudonymous_snapshot_id).all()


def test_duckdb_contract_matches_python_and_supports_legacy_double_hash():
    raw = "315980500007_1782864000"
    once = pseudonymize_snapshot_id(raw)
    twice = hashlib.sha256(once.encode("utf-8")).hexdigest()
    con = duckdb.connect()
    try:
        expression = duckdb_pseudonymous_snapshot_id_sql("value")
        assert con.execute(
            f"SELECT {expression} FROM (VALUES (?)) AS ids(value)", [raw]
        ).fetchone()[0] == once
        assert con.execute(
            f"SELECT {expression} FROM (VALUES (?)) AS ids(value)", [once]
        ).fetchone()[0] == once

        match = duckdb_snapshot_id_match_sql("left_id", "right_id")
        raw_to_current = con.execute(
            f"SELECT {match} FROM (VALUES (?, ?)) AS ids(left_id, right_id)",
            [raw, once],
        ).fetchone()[0]
        prepared_to_legacy_double = con.execute(
            f"SELECT {match} FROM (VALUES (?, ?)) AS ids(left_id, right_id)",
            [once, twice],
        ).fetchone()[0]
        unrelated = con.execute(
            f"SELECT {match} FROM (VALUES (?, ?)) AS ids(left_id, right_id)",
            [raw, pseudonymize_snapshot_id("different")],
        ).fetchone()[0]
    finally:
        con.close()

    assert raw_to_current is True
    assert prepared_to_legacy_double is True
    assert unrelated is False
