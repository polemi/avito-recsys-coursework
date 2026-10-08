"""Privacy-safe identity contract for user-snapshot rows.

The persisted identifier is the lowercase SHA-256 digest of the source
``snapshot_id``.  The conversion is deliberately idempotent: identifiers that
already satisfy the 64-hex contract are kept (and lowercased).  This lets new
code read both legacy artifacts containing raw identifiers and current
pseudonymous artifacts without rewriting either one.

SHA-256 here is a pseudonymisation boundary, not a claim of anonymisation.  Raw
identifiers must still remain inside the controlled snapshot store.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Hashable

import pandas as pd


SNAPSHOT_ID_CONTRACT = "sha256-lower-hex-v1"
_PSEUDONYMOUS_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DUCKDB_PSEUDONYMOUS_RE = "[0-9a-fA-F]{64}"


def is_pseudonymous_snapshot_id(value: object) -> bool:
    """Return whether *value* already satisfies the persisted ID contract."""
    return value is not None and bool(_PSEUDONYMOUS_RE.fullmatch(str(value)))


def pseudonymize_snapshot_id(value: object) -> str:
    """Return an idempotent, privacy-safe snapshot identifier.

    Nulls are rejected because they cannot participate in an auditable paired
    comparison.  Existing 64-hex identifiers are not hashed for a second time.
    """
    if value is None or pd.isna(value):
        raise ValueError("snapshot_id must be non-null")
    text = str(value)
    if _PSEUDONYMOUS_RE.fullmatch(text):
        return text.lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pseudonymize_snapshot_ids(
    frame: pd.DataFrame,
    column: Hashable = "snapshot_id",
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Normalize a dataframe ID column under the v1 contract.

    The function intentionally does not require uniqueness: baseline artifacts
    contain one row per ``(snapshot_id, method)``.  Callers that require a
    one-to-one audience should keep using pandas' ``validate=`` merge contract.
    """
    if column not in frame.columns:
        raise KeyError(f"Missing snapshot identity column: {column!r}")
    result = frame.copy() if copy else frame
    result[column] = result[column].map(pseudonymize_snapshot_id)
    return result


def duckdb_pseudonymous_snapshot_id_sql(expression: str) -> str:
    """Build DuckDB SQL implementing the same idempotent v1 contract."""
    text = f"CAST({expression} AS VARCHAR)"
    return (
        "CASE "
        f"WHEN regexp_full_match({text}, '{_DUCKDB_PSEUDONYMOUS_RE}') "
        f"THEN lower({text}) "
        f"ELSE sha256({text}) END"
    )


def duckdb_snapshot_id_match_sql(left: str, right: str) -> str:
    """Build a backward-compatible DuckDB equality predicate.

    Primary comparison uses the idempotent contract.  The two compatibility
    branches also resolve the brief legacy state in which an already hashed
    source identifier was hashed a second time by the scaling loader.
    """
    left_primary = duckdb_pseudonymous_snapshot_id_sql(left)
    right_primary = duckdb_pseudonymous_snapshot_id_sql(right)
    left_legacy = f"sha256(CAST({left} AS VARCHAR))"
    right_legacy = f"sha256(CAST({right} AS VARCHAR))"
    return (
        f"(({left_primary}) = ({right_primary}) "
        f"OR ({left_legacy}) = ({right_primary}) "
        f"OR ({left_primary}) = ({right_legacy}))"
    )
