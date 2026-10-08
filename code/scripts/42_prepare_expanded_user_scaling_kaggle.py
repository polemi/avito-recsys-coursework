#!/usr/bin/env python3
"""Build the browser-upload package for the >1.2M distinct-user curve.

Train is rebuilt from the verified local 2/30 combined snapshots.  June and
July are copied byte-for-byte from the already audited 906 MB package, so the
evaluation audiences do not change.  Raw identifiers and the pseudonym key are
never included in the output archive.
"""

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
PAYLOAD_COLUMNS = "snapshot_ts, leaf, action, ts, label_leaf, masked_leaf, n_labels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote(paths: list[Path]) -> str:
    return "[" + ",".join("'" + str(path.resolve()).replace("'", "''") + "'" for path in paths) + "]"


def add_code_archive(project: Path, output: Path) -> dict:
    required = [
        project / "configs/model_reduced.json",
        project / "requirements.txt",
        project / "pyproject.toml",
        project / "scripts/06_train_multi_interest.py",
        project / "scripts/26_compare_scaling_points.py",
        project / "scripts/39_run_extended_scaling.py",
    ] + sorted((project / "src/avito_recsys").glob("*.py"))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in required:
            archive.write(path, path.relative_to(project))
    return {"path": output.name, "size_bytes": output.stat().st_size, "sha256": sha256(output)}


def build_notebook(output: Path, data_zip_name: str, code_zip_name: str) -> None:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Avito RecSys — dual-GPU scaling v3\n\n"
            "Nested train prefixes contain 200k, 400k, 600k, 800k, 1.2M, 1.6M and all "
            "distinct positive users. June is used only to select the epoch; "
            "the fixed July audience is used for the reported comparison. This version is "
            "designed for Kaggle Save & Run All and assigns one independent process to each T4."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, os, shutil, subprocess, sys, time, zipfile\n"
            "import pandas as pd\nimport torch\n"
            "gpu_count=torch.cuda.device_count()\n"
            "print('CUDA:',torch.cuda.is_available(),'GPU count:',gpu_count)\n"
            "if gpu_count<2: raise RuntimeError('Enable Kaggle T4 x2 before Save & Run All')\n"
            "print([torch.cuda.get_device_name(i) for i in range(gpu_count)])"
        ),
        nbformat.v4.new_code_cell(
            "INPUT=Path('/kaggle/input'); WORK=Path('/kaggle/working')\n"
            "PROJECT=WORK/'avito_recsys_project'; DATA_FALLBACK=WORK/'expanded_scaling_data'\n"
            "for p in (PROJECT,DATA_FALLBACK):\n"
            "    if p.exists(): shutil.rmtree(p)\n"
            "PROJECT.mkdir(parents=True)\n\n"
            "def unique_or_none(paths,label):\n"
            "    paths=sorted({p.resolve() for p in paths})\n"
            "    if len(paths)>1: raise RuntimeError(f'Multiple {label} candidates: {paths}')\n"
            "    return paths[0] if paths else None\n\n"
            "def valid_code_root(root):\n"
            "    return all((root/p).is_file() for p in (\n"
            "        'scripts/06_train_multi_interest.py',\n"
            "        'scripts/26_compare_scaling_points.py',\n"
            "        'scripts/39_run_extended_scaling.py',\n"
            "        'configs/model_reduced.json')) and (root/'src/avito_recsys').is_dir()\n\n"
            "code_roots=[]\n"
            "for runner in INPUT.rglob('39_run_extended_scaling.py'):\n"
            "    if runner.is_file() and runner.parent.name=='scripts':\n"
            "        root=runner.parent.parent\n"
            "        if valid_code_root(root): code_roots.append(root)\n"
            "code_root=unique_or_none(code_roots,'extracted code root')\n"
            "if code_root is not None:\n"
            "    for name in ('configs','scripts','src'):\n"
            "        shutil.copytree(code_root/name,PROJECT/name)\n"
            "    for name in ('requirements.txt','pyproject.toml'):\n"
            "        source=code_root/name\n"
            "        if source.is_file(): shutil.copy2(source,PROJECT/name)\n"
            "    code_mode='extracted directory'\n"
            "else:\n"
            f"    code_zip=unique_or_none(INPUT.rglob('{code_zip_name}'),'code ZIP')\n"
            f"    if code_zip is None: raise FileNotFoundError('Neither extracted code nor {code_zip_name} was found under /kaggle/input')\n"
            "    with zipfile.ZipFile(code_zip) as z: z.extractall(PROJECT)\n"
            "    if not valid_code_root(PROJECT): raise RuntimeError('Code ZIP has an unexpected layout')\n"
            "    code_mode='ZIP fallback'\n\n"
            "def valid_data_root(root):\n"
            "    manifest=root/'manifest.json'\n"
            "    if not all((root/p).is_file() for p in (\n"
            "        'config.json','vocab.json',\n"
            "        'snapshots/snapshot_2026-05-01.parquet',\n"
            "        'snapshots/snapshot_2026-06-01.parquet',\n"
            "        'snapshots/snapshot_2026-07-01.parquet')) or not manifest.is_file():\n"
            "        return False\n"
            "    try: payload=json.loads(manifest.read_text())\n"
            "    except (OSError,json.JSONDecodeError): return False\n"
            "    return 'train_distinct_hashed_users' in payload and 'planned_curve' in payload\n\n"
            "data_roots=[]\n"
            "for manifest in INPUT.rglob('manifest.json'):\n"
            "    if valid_data_root(manifest.parent): data_roots.append(manifest.parent)\n"
            "DATA=unique_or_none(data_roots,'extracted data root')\n"
            "if DATA is not None:\n"
            "    data_mode='extracted directory (read-only)'\n"
            "else:\n"
            f"    data_zip=unique_or_none(INPUT.rglob('{data_zip_name}'),'data ZIP')\n"
            f"    if data_zip is None: raise FileNotFoundError('Neither extracted data nor {data_zip_name} was found under /kaggle/input')\n"
            "    DATA_FALLBACK.mkdir(parents=True)\n"
            "    with zipfile.ZipFile(data_zip) as z: z.extractall(DATA_FALLBACK)\n"
            "    fallback_roots=[p.parent for p in DATA_FALLBACK.rglob('manifest.json') if valid_data_root(p.parent)]\n"
            "    DATA=unique_or_none(fallback_roots,'data root extracted from ZIP')\n"
            "    if DATA is None: raise RuntimeError('Data ZIP has an unexpected layout')\n"
            "    data_mode='ZIP fallback'\n\n"
            "subprocess.run([sys.executable,'-m','pip','install','-q','duckdb==1.5.4'],check=True)\n"
            "manifest=json.loads((DATA/'manifest.json').read_text())\n"
            "print({'code_mode':code_mode,'project':str(PROJECT),'data_mode':data_mode,"
            "'data_root':str(DATA),'train_distinct_hashed_users':manifest['train_distinct_hashed_users']})"
        ),
        nbformat.v4.new_code_cell(
            "smoke=WORK/'gpu_smoke'\n"
            "smoke_cmd=[sys.executable,str(PROJECT/'scripts/06_train_multi_interest.py'),"
            "'--data-config',str(DATA/'config.json'),'--model-config',str(PROJECT/'configs/model_reduced.json'),"
            "'--fold','0','--num-interests','2','--train-cap','2000','--train-sampling-unit','users',"
            "'--eval-cap','1000','--test-cap','1000','--epochs','1','--min-epochs','1',"
            "'--early-stopping-patience','0','--seed','42','--subset-seed','42','--model-seed','42',"
            "'--device','cuda','--evaluate-test','--snapshots-dir',str(DATA/'snapshots'),"
            "'--vocab',str(DATA/'vocab.json'),'--output',str(smoke)]\n"
            "subprocess.run(smoke_cmd,cwd=PROJECT,check=True)\n"
            "smoke_metrics=json.loads((smoke/'metrics.json').read_text())\n"
            "print({'gpu_smoke':'PASS','train_users':smoke_metrics['train_selected_distinct_users'],"
            "'validation_recall@5':smoke_metrics['recall@5'],"
            "'test_recall@5':smoke_metrics['test_metrics']['recall@5']})"
        ),
        nbformat.v4.new_code_cell(
            "runner=PROJECT/'scripts/39_run_extended_scaling.py'\n"
            "FINAL_ROOT=WORK/'scaling_runs'; FINAL_MODEL_ROOT=FINAL_ROOT/'modelseed42'\n"
            "TEMP_ROOTS={'gpu0':WORK/'scaling_runs_gpu0','gpu1':WORK/'scaling_runs_gpu1'}\n"
            "ASSIGNMENTS={'gpu0':[('200k_users',200000),('400k_users',400000),"
            "('600k_users',600000),('800k_users',800000),('1200k_users',1200000)],"
            "'gpu1':[('1600k_users',1600000),('full',0)]}\n"
            "ORDER=[label for gpu in ('gpu0','gpu1') for label,_ in ASSIGNMENTS[gpu]]\n"
            "def reusable_v3_point(path,cap):\n"
            "    metrics_path=path/'metrics.json'\n"
            "    if not metrics_path.is_file() or not (path/'test_predictions.parquet').is_file(): return False\n"
            "    try: metrics=json.loads(metrics_path.read_text())\n"
            "    except (OSError,json.JSONDecodeError): return False\n"
            "    return (metrics.get('train_sampling_unit')=='users' and metrics.get('train_cap_requested')==cap "
            "and metrics.get('subset_seed')==42 and metrics.get('model_seed')==42 and metrics.get('epochs')==12 "
            "and metrics.get('min_epochs')==8 and metrics.get('early_stopping_patience')==3 "
            "and metrics.get('eval_cap')==20000 and metrics.get('test_cap')==100000 and 'test_metrics' in metrics)\n"
            "for gpu,points in ASSIGNMENTS.items():\n"
            "    for label,cap in points:\n"
            "        previous=FINAL_MODEL_ROOT/f'train{label}'\n"
            "        resume=TEMP_ROOTS[gpu]/'modelseed42'/f'train{label}'\n"
            "        if reusable_v3_point(previous,cap) and not reusable_v3_point(resume,cap):\n"
            "            if resume.exists(): shutil.rmtree(resume)\n"
            "            resume.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(previous),str(resume))\n"
            "common=[sys.executable,str(runner),'--project',str(PROJECT),'--data-root',str(DATA),"
            "'--epochs','12','--min-epochs','8',"
            "'--early-stopping-patience','3','--subset-seed','42',"
            "'--model-seed','42','--eval-cap','20000','--test-cap','100000','--device','cuda',"
            "'--verify-hashes']\n"
            "LOG_ROOT=WORK/'scaling_parallel_logs'; LOG_ROOT.mkdir(parents=True,exist_ok=True)\n"
            "processes=[]\n"
            "for gpu_index,gpu in enumerate(('gpu0','gpu1')):\n"
            "    points=ASSIGNMENTS[gpu]\n"
            "    cmd=common+['--output-root',str(TEMP_ROOTS[gpu])]+sum((['--point',f'{label}:{cap}'] for label,cap in points),[])\n"
            "    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu_index); env['PYTHONUNBUFFERED']='1'; env['PYTHONDONTWRITEBYTECODE']='1'\n"
            "    log_path=LOG_ROOT/f'{gpu}.log'; log_handle=log_path.open('w',buffering=1)\n"
            "    process=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log_handle,stderr=subprocess.STDOUT,text=True)\n"
            "    processes.append({'gpu':gpu,'process':process,'log_path':log_path,'log_handle':log_handle,'cmd':cmd})\n"
            "    print({'started':gpu,'physical_gpu':gpu_index,'pid':process.pid,'points':[p[0] for p in points],'log':str(log_path)},flush=True)\n"
            "while any(item['process'].poll() is None for item in processes):\n"
            "    print({item['gpu']:item['process'].poll() for item in processes},flush=True); time.sleep(60)\n"
            "for item in processes: item['log_handle'].close()\n"
            "failures=[item for item in processes if item['process'].returncode!=0]\n"
            "if failures:\n"
            "    tails={}\n"
            "    for item in failures:\n"
            "        lines=item['log_path'].read_text(errors='replace').splitlines(); tails[item['gpu']]='\\n'.join(lines[-80:])\n"
            "    raise RuntimeError(f'Parallel scaling failed: {tails}')\n"
            "FINAL_MODEL_ROOT.mkdir(parents=True,exist_ok=True)\n"
            "for gpu,points in ASSIGNMENTS.items():\n"
            "    for label,_ in points:\n"
            "        source=TEMP_ROOTS[gpu]/'modelseed42'/f'train{label}'; target=FINAL_MODEL_ROOT/f'train{label}'\n"
            "        if not (source/'metrics.json').is_file(): raise RuntimeError(f'Missing completed point: {source}')\n"
            "        if target.exists(): shutil.rmtree(target)\n"
            "        shutil.move(str(source),str(target))\n"
            "frames=[pd.read_csv(TEMP_ROOTS[gpu]/'scaling_summary.csv') for gpu in ('gpu0','gpu1')]\n"
            "summary=pd.concat(frames,ignore_index=True)\n"
            "if summary['label'].duplicated().any() or set(summary['label'])!=set(ORDER): raise RuntimeError(f'Unexpected summary labels: {summary.label.tolist()}')\n"
            "summary=summary.set_index('label').loc[ORDER].reset_index()\n"
            "FINAL_ROOT.mkdir(parents=True,exist_ok=True); summary.to_csv(FINAL_ROOT/'scaling_summary.csv',index=False)\n"
            "records=json.loads(summary.to_json(orient='records'))\n"
            "(FINAL_ROOT/'scaling_summary.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))\n"
            "(FINAL_ROOT/'parallel_run.json').write_text(json.dumps({'epochs':12,'min_epochs':8,'patience':3,'assignments':{gpu:[label for label,_ in points] for gpu,points in ASSIGNMENTS.items()},'returncodes':{item['gpu']:item['process'].returncode for item in processes}},indent=2))\n"
            "shutil.copytree(LOG_ROOT,FINAL_ROOT/'logs',dirs_exist_ok=True)\n"
            "display(summary)"
        ),
        nbformat.v4.new_code_cell(
            "pairs=[('400k_users','200k_users'),('600k_users','400k_users'),"
            "('800k_users','600k_users'),('1200k_users','800k_users'),"
            "('1600k_users','1200k_users'),('full','1600k_users')]\nrows=[]\n"
            "for larger,smaller in pairs:\n"
            " out=WORK/'scaling_runs/bootstrap'/f'{larger}_vs_{smaller}'\n"
            " cmd=[sys.executable,str(PROJECT/'scripts/26_compare_scaling_points.py'),"
            "'--reference',str(WORK/f'scaling_runs/modelseed42/train{larger}/test_predictions.parquet'),"
            "'--challenger',str(WORK/f'scaling_runs/modelseed42/train{smaller}/test_predictions.parquet'),"
            "'--reference-name',larger,'--challenger-name',smaller,'--repeats','20000','--seed','42','--output',str(out)]\n"
            " subprocess.run(cmd,cwd=PROJECT,check=True); rows.append(pd.read_csv(out/'paired_bootstrap.csv'))\n"
            "bootstrap=pd.concat(rows,ignore_index=True); display(bootstrap)\n"
            "bootstrap.to_csv(WORK/'scaling_runs/adjacent_paired_bootstrap.csv',index=False)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Required replication\n\nThe first pass estimates the complete learning curve. "
            "Before claiming a plateau, rerun the 800k, 1.2M, 1.6M and full points "
            "with model seeds 7 and 123 and compare the same fixed July users. The v3 primary "
            "run uses at most 12 epochs, a minimum of 8 epochs and patience 3 for every point."
        ),
        nbformat.v4.new_code_cell(
            "print(shutil.make_archive(str(WORK/'avito_expanded_user_scaling_results_seed42'),"
            "'zip',WORK/'scaling_runs'))"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
    }
    nbformat.write(nb, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--combined-dir", type=Path,
        default=Path("data/scaling_user_expansion_2026_10/combined_2of30"),
    )
    parser.add_argument(
        "--fixed-eval-dir", type=Path,
        default=Path("reports/kaggle/extended_scaling_2026_10_v2/scaling_data"),
    )
    parser.add_argument("--pseudonym-key-file", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/kaggle/expanded_user_scaling_2026_10"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    resolve = lambda value: value if value.is_absolute() else project / value
    combined = resolve(args.combined_dir)
    fixed_eval = resolve(args.fixed_eval_dir)
    output = resolve(args.output_dir)
    data_dir = output / "scaling_data"
    if data_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{data_dir} exists; use --overwrite")
        shutil.rmtree(data_dir)
    (data_dir / "snapshots").mkdir(parents=True)

    combined_manifest = json.loads((combined / "manifest.json").read_text())
    if not combined_manifest.get("disjoint_samples_verified"):
        raise RuntimeError("Combined-pool disjointness audit is missing")
    full_users = int(combined_manifest["train_pool"]["distinct_positive_users"])
    if full_users <= 1_600_000:
        raise RuntimeError(f"Expanded pool is too small for the planned curve: {full_users:,}")
    key_text = args.pseudonym_key_file.read_text(encoding="ascii").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", key_text):
        raise ValueError("Pseudonym key must be 64 lowercase hex characters")

    train_sources = [combined / "snapshots" / f"snapshot_{date}.parquet" for date in TRAIN_DATES]
    for path in train_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    target_train = data_dir / "snapshots/snapshot_2026-05-01.parquet"
    con = duckdb.connect()
    con.execute("PRAGMA threads=4; SET memory_limit='16GB'; SET preserve_insertion_order=false;")
    user_key = f"sha256('{key_text}:user:' || CAST(user_id AS VARCHAR))"
    snapshot_key = (
        f"sha256('{key_text}:snapshot:' || CAST(user_id AS VARCHAR) || ':' || "
        "CAST(epoch(snapshot_ts) AS VARCHAR))"
    )
    con.execute(
        f"COPY (SELECT {snapshot_key} AS snapshot_id, {user_key} AS hashed_user_key, "
        f"{PAYLOAD_COLUMNS} FROM read_parquet({quote(train_sources)}, union_by_name=true) "
        f"WHERE n_labels > 0) TO '{target_train.resolve()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    train_rows, train_users, duplicate_ids, invalid_snapshot_ids, invalid_user_keys = con.execute(
        f"SELECT count(*), count(DISTINCT hashed_user_key), count(*)-count(DISTINCT snapshot_id), "
        "count_if(NOT regexp_full_match(snapshot_id, '[0-9a-f]{64}')), "
        "count_if(NOT regexp_full_match(hashed_user_key, '[0-9a-f]{64}')) "
        f"FROM read_parquet('{target_train.resolve()}')"
    ).fetchone()
    actual_columns = {
        str(row[0])
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{target_train.resolve()}')"
        ).fetchall()
    }
    con.close()
    expected_columns = {
        "snapshot_id", "hashed_user_key", "snapshot_ts", "leaf", "action", "ts",
        "label_leaf", "masked_leaf", "n_labels",
    }
    if actual_columns != expected_columns:
        raise RuntimeError(f"Unsafe or unexpected compact schema: {sorted(actual_columns)}")
    if train_users != full_users or duplicate_ids or invalid_snapshot_ids or invalid_user_keys:
        raise RuntimeError(
            "Train identity audit failed: "
            f"users={train_users:,}, expected={full_users:,}, duplicates={duplicate_ids}, "
            f"invalid_snapshot_ids={invalid_snapshot_ids}, invalid_user_keys={invalid_user_keys}"
        )

    base_manifest = json.loads((fixed_eval / "manifest.json").read_text())
    if base_manifest.get("pseudonymization", {}).get("raw_user_id_in_package") is not False:
        raise RuntimeError("Fixed evaluation package failed its privacy contract")
    for key, date in (("validation", "2026-06-01"), ("test", "2026-07-01")):
        source = fixed_eval / "snapshots" / f"snapshot_{date}.parquet"
        expected = base_manifest["files"][key]
        if source.stat().st_size != int(expected["size_bytes"]) or sha256(source) != expected["sha256"]:
            raise RuntimeError(f"The fixed {key} file differs from its audited base manifest")
        shutil.copy2(source, data_dir / "snapshots" / f"snapshot_{date}.parquet")
    shutil.copy2(fixed_eval / "vocab.json", data_dir / "vocab.json")
    config = {
        "data_version": "expanded_distinct_user_scaling_2of30_2026_10",
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
        "train": target_train,
        "validation": data_dir / "snapshots/snapshot_2026-06-01.parquet",
        "test": data_dir / "snapshots/snapshot_2026-07-01.parquet",
        "vocab": data_dir / "vocab.json",
        "config": data_dir / "config.json",
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duckdb_version": duckdb.__version__,
        "pseudonymization": {
            "version": 2,
            "method": "SHA-256 with a random 256-bit keyed prefix retained outside the package",
            "key_id": hashlib.sha256(bytes.fromhex(key_text)).hexdigest()[:16],
            "key_in_package": False,
            "raw_user_id_in_package": False,
            "raw_snapshot_id_in_package": False,
        },
        "train_dates": list(TRAIN_DATES),
        "validation_date": "2026-06-01",
        "test_date": "2026-07-01",
        "train_positive_rows": int(train_rows),
        "train_distinct_hashed_users": int(train_users),
        "contains_raw_user_id": False,
        "columns": sorted(actual_columns),
        "identifier_audit": {
            "duplicate_train_snapshot_ids": int(duplicate_ids),
            "invalid_train_snapshot_ids": int(invalid_snapshot_ids),
            "invalid_train_hashed_user_keys": int(invalid_user_keys),
        },
        "validation_selected_rows": int(base_manifest["validation_selected_rows"]),
        "test_selected_rows": int(base_manifest["test_selected_rows"]),
        "fixed_evaluation_provenance": {
            "source_manifest_sha256": sha256(fixed_eval / "manifest.json"),
            "validation_file_sha256": sha256(files["validation"]),
            "test_file_sha256": sha256(files["test"]),
        },
        "planned_curve": [
            {"label": label, "user_cap": cap}
            for label, cap in (
                ("200k_users", 200_000), ("400k_users", 400_000),
                ("600k_users", 600_000), ("800k_users", 800_000),
                ("1200k_users", 1_200_000), ("1600k_users", 1_600_000),
                ("full", 0),
            )
        ],
        "files": {
            key: {"path": str(path.relative_to(data_dir)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for key, path in files.items()
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    data_zip = output / "avito_recsys_expanded_user_scaling_data_2026_10.zip"
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(data_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(data_dir))
    code_zip = output / "avito_recsys_expanded_user_scaling_code_2026_10.zip"
    code_info = add_code_archive(project, code_zip)
    notebook = output / "05_expanded_distinct_user_scaling_gpu.ipynb"
    build_notebook(notebook, data_zip.name, code_zip.name)
    upload = {
        "data_archive": {"path": data_zip.name, "size_bytes": data_zip.stat().st_size, "sha256": sha256(data_zip)},
        "code_archive": code_info,
        "notebook": {"path": notebook.name, "size_bytes": notebook.stat().st_size, "sha256": sha256(notebook)},
        "execution_profile": {
            "version": 3,
            "mode": "Kaggle Save & Run All",
            "accelerator": "T4 x2",
            "max_epochs": 12,
            "min_epochs": 8,
            "early_stopping_patience": 3,
            "gpu_assignments": {
                "0": ["200k_users", "400k_users", "600k_users", "800k_users", "1200k_users"],
                "1": ["1600k_users", "full"],
            },
        },
        "inner_manifest": manifest,
        "upload_method": "Kaggle browser UI only; Save & Run All",
    }
    (output / "upload_manifest.json").write_text(json.dumps(upload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(upload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
