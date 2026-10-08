#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_recsys.scaling_sampling import (  # noqa: E402
    audit_nested_user_prefixes,
    audit_temporal_parquet_boundaries,
    validate_temporal_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit nested distinct-user train prefixes without training a model."
    )
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--caps", type=int, nargs="+", required=True)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.data_config.read_text(encoding="utf-8"))
    fold = next(item for item in config["folds"] if int(item["fold"]) == args.fold)
    validate_temporal_split(
        fold["train_snapshots"],
        fold["validation_snapshot"],
        int(config["target_days"]),
    )
    processed = ROOT / config.get("processed_dir", "data/processed")
    train_paths = [
        processed / "snapshots" / f"snapshot_{snapshot}.parquet"
        for snapshot in fold["train_snapshots"]
    ]
    validation_path = processed / "snapshots" / f"snapshot_{fold['validation_snapshot']}.parquet"
    audit = audit_nested_user_prefixes(train_paths, args.caps, args.subset_seed)
    parquet_boundaries = audit_temporal_parquet_boundaries(
        train_paths, validation_path, int(config["target_days"])
    )
    audit.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_config": str(args.data_config),
            "fold": args.fold,
            "train_snapshots": fold["train_snapshots"],
            "validation_snapshot": fold["validation_snapshot"],
            "target_days": int(config["target_days"]),
            "temporal_leakage_check": "passed",
            "parquet_temporal_boundaries": parquet_boundaries,
            "validation_membership": "fixed by ORDER BY hash(snapshot_id), snapshot_id",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
