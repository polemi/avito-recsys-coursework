#!/usr/bin/env python3
"""Create a compact, privacy-safe Kaggle package for the extended scaling run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import nbformat


TRAIN_DATES = (
    "2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01",
    "2026-03-01", "2026-04-01", "2026-05-01",
)
VALIDATION_DATE = "2026-06-01"
TEST_DATE = "2026-07-01"
PAYLOAD_COLUMNS = "snapshot_ts, leaf, action, ts, label_leaf, masked_leaf, n_labels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_list(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{path.resolve()}'" for path in paths) + "]"


def add_tree(archive: zipfile.ZipFile, root: Path, paths: list[Path]) -> None:
    for path in paths:
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            archive.write(path, path.relative_to(root))


def build_code_archive(project: Path, output: Path) -> dict:
    required = [
        project / "configs/model_reduced.json",
        project / "configs/distinct_user_scaling_2026_10.json",
        project / "requirements.txt",
        project / "pyproject.toml",
        project / "scripts/06_train_multi_interest.py",
        project / "scripts/26_compare_scaling_points.py",
        project / "scripts/39_run_extended_scaling.py",
    ]
    required.extend(sorted((project / "src/avito_recsys").glob("*.py")))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        add_tree(archive, project, required)
    return {"path": output.name, "size_bytes": output.stat().st_size, "sha256": sha256(output)}


def build_notebook(output: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Avito RecSys — distinct-user scaling curve\n\n"
            "One common Nov–May pool is sampled by nested prefixes of distinct pseudonymous "
            "users: 200k, 400k, 600k, 800k, and full. June (20k) is used only for epoch "
            "selection; reported comparisons use an untouched fixed July audience (100k). "
            "The notebook runs a small smoke test before the full seed-42 curve."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, shutil, subprocess, sys, zipfile\n"
            "import pandas as pd\n"
            "import torch\n"
            "print('Python:', sys.version)\n"
            "print('PyTorch:', torch.__version__)\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "if not torch.cuda.is_available(): raise RuntimeError('Enable an NVIDIA GPU accelerator')\n"
            "print('GPU:', torch.cuda.get_device_name(0))"
        ),
        nbformat.v4.new_code_cell(
            "INPUT = Path('/kaggle/input')\n"
            "WORK = Path('/kaggle/working')\n"
            "PROJECT = WORK / 'avito_recsys_project'\n"
            "DATA = WORK / 'scaling_data'\n"
            "for path in (PROJECT, DATA):\n"
            "    if path.exists(): shutil.rmtree(path)\n"
            "    path.mkdir(parents=True)\n"
            "code_zips = list(INPUT.rglob('avito_recsys_scaling_code_2026_10_v2.zip'))\n"
            "if code_zips:\n"
            "    with zipfile.ZipFile(code_zips[0]) as z: z.extractall(PROJECT)\n"
            "else:\n"
            "    code_runner = next(INPUT.rglob('39_run_extended_scaling.py'))\n"
            "    shutil.copytree(code_runner.parents[1], PROJECT, dirs_exist_ok=True)\n"
            "data_zips = list(INPUT.rglob('avito_recsys_scaling_data_2026_10_v2.zip'))\n"
            "if data_zips:\n"
            "    with zipfile.ZipFile(data_zips[0]) as z: z.extractall(DATA)\n"
            "else:\n"
            "    manifests = [p for p in INPUT.rglob('manifest.json') if 'train_positive_rows' in p.read_text()]\n"
            "    if len(manifests) != 1: raise RuntimeError(f'Expected one scaling manifest, found {manifests}')\n"
            "    shutil.copytree(manifests[0].parent, DATA, dirs_exist_ok=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'duckdb==1.5.4'], check=True)\n"
            "print('Project:', PROJECT)\n"
            "print('Data:', DATA)"
        ),
        nbformat.v4.new_code_cell(
            "# Hash verification catches incomplete/corrupted GUI uploads before GPU time is spent.\n"
            "runner = PROJECT / 'scripts/39_run_extended_scaling.py'\n"
            "dry = [sys.executable, str(runner), '--project', str(PROJECT), '--data-root', str(DATA),\n"
            "       '--output-root', str(WORK/'scaling_runs'), '--device', 'cuda', '--verify-hashes', '--dry-run']\n"
            "subprocess.run(dry, check=True)"
        ),
        nbformat.v4.new_code_cell(
            "# Short end-to-end smoke: data loading, CUDA forward/backward, evaluation, and artifact write.\n"
            "smoke = [sys.executable, str(PROJECT/'scripts/06_train_multi_interest.py'),\n"
            "  '--data-config', str(DATA/'config.json'), '--model-config', str(PROJECT/'configs/model_reduced.json'),\n"
            "  '--fold', '0', '--num-interests', '2', '--train-cap', '2000',\n"
            "  '--train-sampling-unit', 'users', '--eval-cap', '1000', '--test-cap', '2000',\n"
            "  '--epochs', '1', '--seed', '42', '--subset-seed', '42', '--model-seed', '42',\n"
            "  '--device', 'cuda', '--evaluate-test',\n"
            "  '--snapshots-dir', str(DATA/'snapshots'), '--vocab', str(DATA/'vocab.json'),\n"
            "  '--output', str(WORK/'scaling_smoke')]\n"
            "subprocess.run(smoke, cwd=PROJECT, check=True)\n"
            "print((WORK/'scaling_smoke/metrics.json').read_text())"
        ),
        nbformat.v4.new_code_cell(
            "# Main curve. Resume-safe: matching completed points are skipped.\n"
            "run = [sys.executable, str(runner), '--project', str(PROJECT), '--data-root', str(DATA),\n"
            "  '--output-root', str(WORK/'scaling_runs'), '--epochs', '8',\n"
            "  '--subset-seed', '42', '--model-seed', '42', '--eval-cap', '20000',\n"
            "  '--test-cap', '100000', '--device', 'cuda', '--verify-hashes']\n"
            "subprocess.run(run, cwd=PROJECT, check=True)\n"
            "display(pd.read_csv(WORK/'scaling_runs/scaling_summary.csv'))"
        ),
        nbformat.v4.new_code_cell(
            "# Primary paired bootstrap on exactly the same untouched July audience.\n"
            "pairs = [('400k_users','200k_users'), ('600k_users','400k_users'),\n"
            "         ('800k_users','600k_users'), ('full','800k_users')]\n"
            "bootstrap_rows = []\n"
            "for larger, smaller in pairs:\n"
            "    out = WORK/'scaling_runs/bootstrap'/f'{larger}_vs_{smaller}'\n"
            "    cmd = [sys.executable, str(PROJECT/'scripts/26_compare_scaling_points.py'),\n"
            "      '--reference', str(WORK/f'scaling_runs/modelseed42/train{larger}/test_predictions.parquet'),\n"
            "      '--challenger', str(WORK/f'scaling_runs/modelseed42/train{smaller}/test_predictions.parquet'),\n"
            "      '--reference-name', larger, '--challenger-name', smaller,\n"
            "      '--repeats', '20000', '--seed', '42', '--output', str(out)]\n"
            "    subprocess.run(cmd, cwd=PROJECT, check=True)\n"
            "    bootstrap_rows.append(pd.read_csv(out/'paired_bootstrap.csv'))\n"
            "bootstrap = pd.concat(bootstrap_rows, ignore_index=True)\n"
            "bootstrap.to_csv(WORK/'scaling_runs/adjacent_paired_bootstrap.csv', index=False)\n"
            "display(bootstrap)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation guard\n\n"
            "This first curve uses one model seed with frozen user membership. A confidence "
            "interval crossing zero means the direction is unresolved, not that the points "
            "are equivalent. If a key contrast is consequential, keep subset seed 42 fixed "
            "and rerun that pair with model seeds 7 and 123."
        ),
        nbformat.v4.new_code_cell(
            "archive = shutil.make_archive(str(WORK/'avito_distinct_user_scaling_results_seed42'), 'zip', WORK/'scaling_runs')\n"
            "print('Download or save as notebook output:', archive)"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
    }
    nbformat.write(nb, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("reports/kaggle/extended_scaling_2026_10_v2"))
    parser.add_argument(
        "--pseudonym-key-file",
        type=Path,
        required=True,
        help="Local file containing a 64-character random hex key. It is never copied to the package.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    key_text = args.pseudonym_key_file.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", key_text):
        raise ValueError("Pseudonym key must be exactly 32 random bytes encoded as 64 hex characters")
    pseudonym_key = bytes.fromhex(key_text)
    key_id = hashlib.sha256(pseudonym_key).hexdigest()[:16]
    data_dir = output_dir / "scaling_data"
    if data_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{data_dir} exists; pass --overwrite to rebuild")
        shutil.rmtree(data_dir)
    snapshots_dir = data_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)

    source_dir = project / "data/refresh_2026_08/processed/snapshots"
    train_sources = [source_dir / f"snapshot_{date}.parquet" for date in TRAIN_DATES]
    validation_source = source_dir / f"snapshot_{VALIDATION_DATE}.parquet"
    test_source = source_dir / f"snapshot_{TEST_DATE}.parquet"
    vocab_source = project / "data/refresh_2026_08/processed/vocab_leaf_refresh_2026_08.json"
    for path in [*train_sources, validation_source, test_source, vocab_source]:
        if not path.is_file():
            raise FileNotFoundError(path)

    con = duckdb.connect()
    if duckdb.__version__ != "1.5.4":
        raise RuntimeError(f"DuckDB 1.5.4 required for frozen hash order, found {duckdb.__version__}")
    con.execute("PRAGMA threads=4; SET memory_limit='12GB';")
    # The compact files use real ISO snapshot names so the training script can
    # enforce temporal ordering, even though the train file merges Nov--May.
    train_target = snapshots_dir / "snapshot_2026-05-01.parquet"
    validation_target = snapshots_dir / "snapshot_2026-06-01.parquet"
    test_target = snapshots_dir / "snapshot_2026-07-01.parquet"
    # The keyed prefix is deliberately absent from every generated artifact.  It
    # prevents practical dictionary recovery of raw numeric user ids while the
    # stable hashed_user_key still permits user-level sampling/aggregation.
    user_key_sql = f"sha256('{key_text}:user:' || CAST(user_id AS VARCHAR))"
    snapshot_key_sql = (
        f"sha256('{key_text}:snapshot:' || CAST(user_id AS VARCHAR) || ':' || "
        "CAST(epoch(snapshot_ts) AS VARCHAR))"
    )
    con.execute(
        f"COPY (SELECT {snapshot_key_sql} AS snapshot_id, {user_key_sql} AS hashed_user_key, "
        f"{PAYLOAD_COLUMNS} FROM read_parquet({parquet_list(train_sources)}) "
        f"WHERE n_labels > 0) TO '{train_target.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT {snapshot_key_sql} AS snapshot_id, {user_key_sql} AS hashed_user_key, "
        f"{PAYLOAD_COLUMNS} FROM read_parquet('{validation_source.resolve()}') "
        f"WHERE n_labels > 0 ORDER BY hash(snapshot_id), snapshot_id LIMIT 20000) "
        f"TO '{validation_target.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.execute(
        f"COPY (SELECT {snapshot_key_sql} AS snapshot_id, {user_key_sql} AS hashed_user_key, "
        f"{PAYLOAD_COLUMNS} FROM read_parquet('{test_source.resolve()}') "
        f"WHERE n_labels > 0 ORDER BY hash(snapshot_id), snapshot_id LIMIT 100000) "
        f"TO '{test_target.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    train_rows = con.execute(f"SELECT count(*) FROM read_parquet('{train_target.resolve()}')").fetchone()[0]
    validation_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{validation_target.resolve()}')"
    ).fetchone()[0]
    test_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{test_target.resolve()}')"
    ).fetchone()[0]
    forbidden_columns = {"user_id", "vert", "cat", "param1", "count_views"}
    actual_columns = {
        row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{train_target.resolve()}')"
        ).fetchall()
    }
    con.close()
    if train_rows != 2_093_298 or validation_rows != 20_000 or test_rows != 100_000:
        raise RuntimeError(
            f"Unexpected row counts: train={train_rows}, validation={validation_rows}, test={test_rows}"
        )
    if forbidden_columns.intersection(actual_columns):
        raise RuntimeError("Compact package contains forbidden/unused source columns")
    if actual_columns != {
        "snapshot_id", "hashed_user_key", "snapshot_ts", "leaf", "action", "ts",
        "label_leaf", "masked_leaf", "n_labels",
    }:
        raise RuntimeError(f"Unexpected compact schema: {sorted(actual_columns)}")
    con = duckdb.connect()
    invalid_snapshot_ids, invalid_user_keys, duplicate_snapshots = con.execute(
        f"SELECT count_if(NOT regexp_full_match(snapshot_id, '[0-9a-f]{{64}}')), "
        f"count_if(NOT regexp_full_match(hashed_user_key, '[0-9a-f]{{64}}')), "
        f"count(*) - count(DISTINCT snapshot_id) FROM read_parquet('{train_target.resolve()}')"
    ).fetchone()
    validation_invalid_snapshot_ids, validation_invalid_user_keys = con.execute(
        f"SELECT count_if(NOT regexp_full_match(snapshot_id, '[0-9a-f]{{64}}')), "
        f"count_if(NOT regexp_full_match(hashed_user_key, '[0-9a-f]{{64}}')) "
        f"FROM read_parquet('{validation_target.resolve()}')"
    ).fetchone()
    test_invalid_snapshot_ids, test_invalid_user_keys = con.execute(
        f"SELECT count_if(NOT regexp_full_match(snapshot_id, '[0-9a-f]{{64}}')), "
        f"count_if(NOT regexp_full_match(hashed_user_key, '[0-9a-f]{{64}}')) "
        f"FROM read_parquet('{test_target.resolve()}')"
    ).fetchone()
    if any((invalid_snapshot_ids, invalid_user_keys, duplicate_snapshots,
            validation_invalid_snapshot_ids, validation_invalid_user_keys,
            test_invalid_snapshot_ids, test_invalid_user_keys)):
        raise RuntimeError("Pseudonym or snapshot uniqueness audit failed")
    cap_audit = []
    for label, cap in (("200k_users", 200_000), ("400k_users", 400_000),
                       ("600k_users", 600_000), ("800k_users", 800_000), ("full", 0)):
        user_filter = "" if cap == 0 else f"WHERE user_rank <= {cap}"
        selected_rows, selected_users = con.execute(
            f"WITH positive AS (SELECT * FROM read_parquet('{train_target.resolve()}')), "
            "ranked_users AS (SELECT hashed_user_key, row_number() OVER ("
            "ORDER BY hash(CAST(hashed_user_key AS VARCHAR) || '42'), "
            "CAST(hashed_user_key AS VARCHAR)) AS user_rank "
            "FROM (SELECT DISTINCT hashed_user_key FROM positive)), "
            f"selected_users AS (SELECT hashed_user_key FROM ranked_users {user_filter}) "
            "SELECT count(*), count(DISTINCT p.hashed_user_key) FROM positive p "
            "JOIN selected_users s USING (hashed_user_key)"
        ).fetchone()
        cap_audit.append({"label": label, "user_cap": cap, "selected_positive_rows": selected_rows,
                          "selected_distinct_hashed_users": selected_users})
    ordered_validation_ids = [
        row[0] for row in con.execute(
            f"SELECT snapshot_id FROM read_parquet('{validation_target.resolve()}') "
            "ORDER BY hash(snapshot_id)"
        ).fetchall()
    ]
    ordered_test_ids = [
        row[0] for row in con.execute(
            f"SELECT snapshot_id FROM read_parquet('{test_target.resolve()}') "
            "ORDER BY hash(snapshot_id), snapshot_id"
        ).fetchall()
    ]
    con.close()
    validation_id_sha256 = hashlib.sha256("\n".join(ordered_validation_ids).encode("ascii")).hexdigest()
    test_id_sha256 = hashlib.sha256("\n".join(ordered_test_ids).encode("ascii")).hexdigest()

    shutil.copy2(vocab_source, data_dir / "vocab.json")
    config = {
        "data_version": "scaling_extended_2026_10_compact_pseudonymous_v2",
        "processed_dir": ".",
        "vocab_file": "vocab.json",
        "history_days": 90,
        "novelty_days": 90,
        "target_days": 14,
        "folds": [{
            "fold": 0,
            "train_snapshots": ["2026-05-01"],
            "source_train_snapshots": list(TRAIN_DATES),
            "validation_snapshot": "2026-06-01",
            "test_snapshot": "2026-07-01",
            "role": "scaling_extension",
        }],
    }
    (data_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    files = {
        "train": train_target,
        "validation": validation_target,
        "test": test_target,
        "vocab": data_dir / "vocab.json",
        "config": data_dir / "config.json",
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "pseudonymization": {
            "version": 2,
            "method": "SHA-256 with a random 256-bit keyed prefix generated and retained outside the package",
            "key_id": key_id,
            "key_in_package": False,
            "raw_user_id_in_package": False,
            "raw_snapshot_id_in_package": False,
            "snapshot_id_format": "64 lowercase hex characters",
            "hashed_user_key_format": "64 lowercase hex characters",
        },
        "train_dates": list(TRAIN_DATES),
        "validation_date": VALIDATION_DATE,
        "test_date": TEST_DATE,
        "train_positive_rows": train_rows,
        "train_distinct_hashed_users": cap_audit[-1]["selected_distinct_hashed_users"],
        "validation_selected_rows": validation_rows,
        "test_selected_rows": test_rows,
        "validation_ordered_pseudonymous_snapshot_id_sha256": validation_id_sha256,
        "test_ordered_pseudonymous_snapshot_id_sha256": test_id_sha256,
        "train_order_for_caps": "distinct hashed_user_key ordered by hash(hashed_user_key || '42')",
        "validation_order": "hash(snapshot_id), snapshot_id",
        "test_order": "hash(snapshot_id), snapshot_id",
        "contains_raw_user_id": False,
        "columns": sorted(actual_columns),
        "planned_curve": cap_audit,
        "identifier_audit": {
            "invalid_train_snapshot_ids": invalid_snapshot_ids,
            "invalid_train_hashed_user_keys": invalid_user_keys,
            "duplicate_train_snapshot_ids": duplicate_snapshots,
            "invalid_validation_snapshot_ids": validation_invalid_snapshot_ids,
            "invalid_validation_hashed_user_keys": validation_invalid_user_keys,
            "invalid_test_snapshot_ids": test_invalid_snapshot_ids,
            "invalid_test_hashed_user_keys": test_invalid_user_keys,
        },
        "files": {
            key: {"path": str(path.relative_to(data_dir)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in files.items()
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    data_zip = output_dir / "avito_recsys_scaling_data_2026_10_v2.zip"
    with zipfile.ZipFile(data_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(data_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data_dir))
    code_zip = output_dir / "avito_recsys_scaling_code_2026_10_v2.zip"
    code_info = build_code_archive(project, code_zip)
    notebook = output_dir / "04_extended_data_scaling_gpu_v2.ipynb"
    build_notebook(notebook)
    upload_manifest = {
        "data_archive": {"path": data_zip.name, "size_bytes": data_zip.stat().st_size, "sha256": sha256(data_zip)},
        "code_archive": code_info,
        "notebook": {"path": notebook.name, "size_bytes": notebook.stat().st_size, "sha256": sha256(notebook)},
        "inner_manifest": manifest,
    }
    (output_dir / "upload_manifest.json").write_text(
        json.dumps(upload_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(upload_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
