#!/usr/bin/env python3
"""Build a single-GPU, failure-tolerant recovery notebook for model seed 123.

The dual-GPU endpoint replication can put both workers' in-memory copies of the
1.8 GB train parquet on the same Kaggle host.  This recovery notebook avoids
that shared host-RAM pressure: it runs the 1.2M and full endpoints sequentially
on one GPU, in separate child processes.  Completed point artifacts are never
deleted, the second point is attempted even when the first returns non-zero,
and a diagnostic ZIP is written for complete as well as partial runs.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT
    / "reports/kaggle/expanded_user_scaling_2026_10/06_endpoint_seed_replication_gpu.ipynb"
)
OUTPUT = (
    PROJECT
    / "reports/kaggle/expanded_user_scaling_2026_10/07_endpoint_seed123_recovery_gpu.ipynb"
)


def source(text: str) -> list[str]:
    """Return canonical notebook source lines."""
    return text.splitlines(keepends=True)


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def main() -> None:
    source_nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(source_nb["cells"]) < 3:
        raise RuntimeError("Unexpected source notebook layout")

    model_config_path = PROJECT / "configs/model_reduced.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    expected_model = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "ff_dim": 128,
        "dropout": 0.15,
        "max_len": 128,
        "n_time_buckets": 32,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "positive_weight": 50.0,
        "attention_diversity_weight": 0.1,
        "interest_diversity_weight": 0.5,
    }
    if model_config != expected_model:
        raise RuntimeError(
            f"model_reduced.json drifted: expected {expected_model}, got {model_config}"
        )

    intro = markdown_cell(
        "# Avito RecSys — single-GPU recovery for model seed 123\n\n"
        "This notebook repeats only the 1.2M → all 1,985,972 distinct-positive-user "
        "contrast for model seed 123. It uses subset seed 42, June 20k only for "
        "checkpoint selection, and the fixed July 100k audience only for final "
        "reporting. The endpoints run sequentially in isolated child processes to "
        "avoid the shared host-RAM pressure of the dual-GPU attempt.\n\n"
        "The run is deliberately failure-tolerant: each endpoint is attempted "
        "independently, completed artifacts are retained, and a status file plus ZIP "
        "are produced even when one endpoint or the paired bootstrap fails."
    )

    imports = code_cell(
        "from pathlib import Path\n"
        "from datetime import datetime, timezone\n"
        "import gc, importlib.util, json, os, shutil, subprocess, sys, time, traceback, zipfile\n"
        "import pandas as pd\n"
        "import torch\n"
        "gpu_count=torch.cuda.device_count()\n"
        "print('CUDA:',torch.cuda.is_available(),'GPU count:',gpu_count)\n"
        "if not torch.cuda.is_available() or gpu_count<1:\n"
        "    raise RuntimeError('Enable a Kaggle GPU before Save & Run All')\n"
        "print('Recovery GPU:',torch.cuda.get_device_name(0))"
    )

    # Reuse the already exercised private-input discovery and extraction logic from
    # Version #2.  It supports both Kaggle's extracted dataset layout and the ZIP
    # fallback without embedding any private path or identifier in this notebook.
    setup = code_cell("".join(source_nb["cells"][2]["source"]))

    recovery = code_cell(
        "RUNNER=PROJECT/'scripts/39_run_extended_scaling.py'\n"
        "COMPARE=PROJECT/'scripts/26_compare_scaling_points.py'\n"
        "RUN_ROOT=WORK/'endpoint_seed123_recovery'\n"
        "MODEL_ROOT=RUN_ROOT/'modelseed123'\n"
        "LOG_ROOT=WORK/'endpoint_seed123_recovery_logs'\n"
        "ZIP_BASE=WORK/'avito_endpoint_scaling_recovery_seed123'\n"
        "POINTS=[('1200k_users',1200000),('full',0)]\n"
        "MODEL_SEED=123; SUBSET_SEED=42; MAX_EPOCHS=12; MIN_EPOCHS=8; PATIENCE=3\n"
        "EVAL_CAP=20000; TEST_CAP=100000; BOOTSTRAP_REPEATS=20000; BOOTSTRAP_SEED=1123\n"
        "RUN_ROOT.mkdir(parents=True,exist_ok=True); LOG_ROOT.mkdir(parents=True,exist_ok=True)\n"
        "started_at=datetime.now(timezone.utc)\n"
        "\n"
        "expected_model={'d_model':64,'n_heads':4,'n_layers':2,'ff_dim':128,'dropout':0.15,'max_len':128,'n_time_buckets':32,'batch_size':128,'learning_rate':0.001,'weight_decay':0.00001,'positive_weight':50.0,'attention_diversity_weight':0.1,'interest_diversity_weight':0.5}\n"
        "actual_model=json.loads((PROJECT/'configs/model_reduced.json').read_text())\n"
        "if actual_model!=expected_model:\n"
        "    raise RuntimeError(f'Reduced-model config drift: {actual_model}')\n"
        "manifest=json.loads((DATA/'manifest.json').read_text())\n"
        "if int(manifest.get('train_distinct_hashed_users',0))!=1985972:\n"
        "    raise RuntimeError(f\"Unexpected full audience: {manifest.get('train_distinct_hashed_users')}\")\n"
        "if int(manifest.get('validation_selected_rows',0))!=EVAL_CAP or int(manifest.get('test_selected_rows',0))!=TEST_CAP:\n"
        "    raise RuntimeError('Fixed June/July evaluation audiences changed')\n"
        "# Verify every packaged file once. Individual endpoint runners still check sizes.\n"
        "runner_spec=importlib.util.spec_from_file_location('seed123_scaling_runner',RUNNER)\n"
        "runner_module=importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner_module)\n"
        "runner_module.load_manifest(DATA,verify_hashes=True)\n"
        "print({'preflight':'PASS','data_hashes_verified':True,'model':actual_model,'full_users':1985972})\n"
        "\n"
        "def load_valid_point(label,cap):\n"
        "    point_dir=MODEL_ROOT/f'train{label}'\n"
        "    metrics_path=point_dir/'metrics.json'; predictions_path=point_dir/'test_predictions.parquet'\n"
        "    if not metrics_path.is_file() or not predictions_path.is_file(): return None\n"
        "    try: metrics=json.loads(metrics_path.read_text())\n"
        "    except (OSError,json.JSONDecodeError): return None\n"
        "    expected=(metrics.get('train_sampling_unit')=='users' and int(metrics.get('train_cap_requested',-1))==cap and int(metrics.get('subset_seed',-1))==SUBSET_SEED and int(metrics.get('model_seed',-1))==MODEL_SEED and int(metrics.get('epochs',-1))==MAX_EPOCHS and int(metrics.get('min_epochs',-1))==MIN_EPOCHS and int(metrics.get('early_stopping_patience',-1))==PATIENCE and int(metrics.get('fold',-1))==0 and int(metrics.get('eval_cap',-1))==EVAL_CAP and int(metrics.get('test_cap',-1))==TEST_CAP and 'test_metrics' in metrics)\n"
        "    if not expected: return None\n"
        "    selected=int(metrics.get('train_selected_distinct_users',-1))\n"
        "    if (cap>0 and selected!=cap) or (cap==0 and selected!=1985972): return None\n"
        "    ids=pd.read_parquet(predictions_path,columns=['snapshot_id'])['snapshot_id']\n"
        "    if len(ids)!=TEST_CAP or ids.nunique()!=TEST_CAP: return None\n"
        "    return metrics\n"
        "\n"
        "def summary_row(label,cap,metrics):\n"
        "    test=metrics['test_metrics']\n"
        "    return {'label':label,'requested_user_cap':cap,'train_selected_distinct_users':metrics['train_selected_distinct_users'],'train_selected_rows':metrics['train_selected_rows'],'subset_seed':metrics['subset_seed'],'model_seed':metrics['model_seed'],'epochs':metrics['epochs'],'epochs_completed':metrics['epochs_completed'],'selected_epoch':metrics['selected_epoch'],'validation_recall@5':metrics['recall@5'],'validation_ndcg@5':metrics['ndcg@5'],'validation_coverage':metrics['coverage'],'test_recall@5':test['recall@5'],'test_ndcg@5':test['ndcg@5'],'test_coverage':test['coverage'],'runtime_seconds':metrics['runtime_seconds'],'device':metrics['device']}\n"
        "\n"
        "def failure_flags(returncode,tail):\n"
        "    lowered=tail.lower()\n"
        "    gpu_oom=('cuda out of memory' in lowered or 'cublas_status_alloc_failed' in lowered)\n"
        "    host_oom=(returncode in (-9,137) or 'memoryerror' in lowered or 'std::bad_alloc' in lowered or ('killed' in lowered and not gpu_oom))\n"
        "    return {'gpu_oom':gpu_oom,'host_oom':host_oom}\n"
        "\n"
        "attempt_id=started_at.strftime('%Y%m%dT%H%M%SZ')\n"
        "point_runs=[]\n"
        "for label,cap in POINTS:\n"
        "    point_started=datetime.now(timezone.utc); log_path=LOG_ROOT/f'{label}_{attempt_id}.log'\n"
        "    point_dir=MODEL_ROOT/f'train{label}'; reusable_before_launch=load_valid_point(label,cap) is not None\n"
        "    quarantined=None\n"
        "    if point_dir.exists() and not reusable_before_launch:\n"
        "        quarantine=RUN_ROOT/'quarantined_incomplete'/f'{label}_{attempt_id}'\n"
        "        quarantine.parent.mkdir(parents=True,exist_ok=True)\n"
        "        if quarantine.exists(): shutil.rmtree(quarantine)\n"
        "        shutil.move(str(point_dir),str(quarantine)); quarantined=str(quarantine.relative_to(RUN_ROOT))\n"
        "    cmd=[sys.executable,str(RUNNER),'--project',str(PROJECT),'--data-root',str(DATA),'--output-root',str(RUN_ROOT),'--point',f'{label}:{cap}','--epochs',str(MAX_EPOCHS),'--min-epochs',str(MIN_EPOCHS),'--early-stopping-patience',str(PATIENCE),'--subset-seed',str(SUBSET_SEED),'--model-seed',str(MODEL_SEED),'--eval-cap',str(EVAL_CAP),'--test-cap',str(TEST_CAP),'--device','cuda']\n"
        "    env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','MALLOC_ARENA_MAX':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_MAX_THREADS':'2','PYARROW_NUM_THREADS':'2'})\n"
        "    returncode=None; launch_error=None\n"
        "    try:\n"
        "        with log_path.open('a',buffering=1) as log_handle:\n"
        "            log_handle.write('\\n=== '+point_started.isoformat()+' ===\\n'+json.dumps(cmd)+'\\n')\n"
        "            process=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log_handle,stderr=subprocess.STDOUT,text=True)\n"
        "            print({'started':label,'pid':process.pid,'gpu':0,'log':str(log_path)},flush=True)\n"
        "            while process.poll() is None:\n"
        "                elapsed=(datetime.now(timezone.utc)-point_started).total_seconds()\n"
        "                print({'running':label,'elapsed_minutes':round(elapsed/60,1)},flush=True); time.sleep(60)\n"
        "            returncode=process.returncode\n"
        "    except BaseException as exc:\n"
        "        launch_error=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))\n"
        "        with log_path.open('a') as log_handle: log_handle.write('\\nPARENT_EXCEPTION\\n'+launch_error)\n"
        "    lines=log_path.read_text(errors='replace').splitlines() if log_path.is_file() else []\n"
        "    tail='\\n'.join(lines[-120:]); metrics=load_valid_point(label,cap)\n"
        "    flags=failure_flags(returncode,tail)\n"
        "    point_runs.append({'label':label,'requested_user_cap':cap,'returncode':returncode,'complete':metrics is not None,'reused_matching_endpoint':reusable_before_launch,'quarantined_incomplete':quarantined,'started_at_utc':point_started.isoformat(),'finished_at_utc':datetime.now(timezone.utc).isoformat(),'log_file':log_path.name,'launch_error':launch_error,'log_tail':tail if metrics is None else None,**flags})\n"
        "    print({'finished':label,'returncode':returncode,'complete':metrics is not None,**flags},flush=True)\n"
        "    gc.collect(); torch.cuda.empty_cache(); time.sleep(5)\n"
        "\n"
        "rows=[]\n"
        "for label,cap in POINTS:\n"
        "    metrics=load_valid_point(label,cap)\n"
        "    if metrics is not None: rows.append(summary_row(label,cap,metrics))\n"
        "summary=pd.DataFrame(rows)\n"
        "if not summary.empty:\n"
        "    order={label:index for index,(label,_) in enumerate(POINTS)}\n"
        "    summary=summary.sort_values('label',key=lambda s:s.map(order)).reset_index(drop=True)\n"
        "summary.to_csv(RUN_ROOT/'replication_summary.csv',index=False)\n"
        "(RUN_ROOT/'replication_summary.json').write_text(json.dumps(json.loads(summary.to_json(orient='records')),indent=2))\n"
        "\n"
        "bootstrap_status={'attempted':False,'complete':False,'returncode':None,'error':None}\n"
        "bootstrap_log=LOG_ROOT/f'paired_bootstrap_{attempt_id}.log'\n"
        "if set(summary.get('label',pd.Series(dtype=str)))=={'1200k_users','full'}:\n"
        "    bootstrap_status['attempted']=True\n"
        "    out=RUN_ROOT/'bootstrap'/'seed123_full_vs_1200k'\n"
        "    cmd=[sys.executable,str(COMPARE),'--reference',str(MODEL_ROOT/'trainfull/test_predictions.parquet'),'--challenger',str(MODEL_ROOT/'train1200k_users/test_predictions.parquet'),'--reference-name','full_seed123','--challenger-name','1200k_seed123','--repeats',str(BOOTSTRAP_REPEATS),'--seed',str(BOOTSTRAP_SEED),'--output',str(out)]\n"
        "    try:\n"
        "        with bootstrap_log.open('w',buffering=1) as log_handle:\n"
        "            completed=subprocess.run(cmd,cwd=PROJECT,stdout=log_handle,stderr=subprocess.STDOUT,text=True,check=False)\n"
        "        bootstrap_status['returncode']=completed.returncode\n"
        "        source=out/'paired_bootstrap.csv'\n"
        "        if completed.returncode==0 and source.is_file():\n"
        "            bootstrap=pd.read_csv(source); bootstrap.insert(0,'model_seed',MODEL_SEED)\n"
        "            bootstrap.to_csv(RUN_ROOT/'endpoint_paired_bootstrap.csv',index=False)\n"
        "            bootstrap_status['complete']=True\n"
        "        else: bootstrap_status['error']=bootstrap_log.read_text(errors='replace')[-12000:]\n"
        "    except BaseException as exc:\n"
        "        bootstrap_status['error']=''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))\n"
        "    part=summary.set_index('label')\n"
        "    delta=pd.DataFrame([{'model_seed':MODEL_SEED,'recall_delta_full_minus_1200k':part.loc['full','test_recall@5']-part.loc['1200k_users','test_recall@5'],'ndcg_delta_full_minus_1200k':part.loc['full','test_ndcg@5']-part.loc['1200k_users','test_ndcg@5']}])\n"
        "    delta.to_csv(RUN_ROOT/'endpoint_seed_delta.csv',index=False)\n"
        "\n"
        "complete_labels=summary['label'].tolist() if 'label' in summary else []\n"
        "overall_status='complete' if set(complete_labels)=={'1200k_users','full'} and bootstrap_status['complete'] else ('partial' if complete_labels else 'failed')\n"
        "status={'overall_status':overall_status,'model_seed':MODEL_SEED,'subset_seed':SUBSET_SEED,'points':[{'label':label,'cap':cap} for label,cap in POINTS],'training_policy':{**expected_model,'max_epochs':MAX_EPOCHS,'min_epochs':MIN_EPOCHS,'early_stopping_patience':PATIENCE},'evaluation':{'checkpoint_selection':'June fixed 20k','final_report':'July fixed 100k'},'resource_strategy':{'gpu_processes_at_once':1,'endpoint_order':[label for label,_ in POINTS],'child_process_per_endpoint':True,'thread_cap':2},'data_hashes_verified':True,'completed_labels':complete_labels,'point_runs':point_runs,'paired_bootstrap':bootstrap_status,'started_at_utc':started_at.isoformat(),'finished_at_utc':datetime.now(timezone.utc).isoformat()}\n"
        "(RUN_ROOT/'recovery_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))\n"
        "shutil.copytree(LOG_ROOT,RUN_ROOT/'logs',dirs_exist_ok=True)\n"
        "archive_path=shutil.make_archive(str(ZIP_BASE),'zip',RUN_ROOT)\n"
        "print({'overall_status':overall_status,'completed_labels':complete_labels,'bootstrap_complete':bootstrap_status['complete'],'archive':archive_path})\n"
        "display(summary)"
    )

    interpretation = markdown_cell(
        "## Reading the result\n\n"
        "A successful run must contain both endpoint rows, July metrics for 100,000 "
        "users, and `endpoint_paired_bootstrap.csv`. The bootstrap estimates uncertainty "
        "over the fixed July audience conditional on this trained seed; it does not replace "
        "the across-seed comparison. If `overall_status` is `partial` or `failed`, inspect "
        "`recovery_status.json` and the included logs rather than treating the Kaggle version "
        "status itself as evidence of a completed experiment."
    )

    final_check = code_cell(
        "status=json.loads((RUN_ROOT/'recovery_status.json').read_text())\n"
        "print(json.dumps({key:status[key] for key in ('overall_status','model_seed','subset_seed','completed_labels','paired_bootstrap')},indent=2))\n"
        "print('ZIP:',str(ZIP_BASE)+'.zip')"
    )

    notebook = {
        "cells": [intro, imports, setup, recovery, interpretation, final_check],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")
    if notebook["nbformat"] != 4 or not notebook["cells"]:
        raise RuntimeError("Invalid notebook envelope")

    rendered = json.dumps(notebook, ensure_ascii=False, indent=1)
    required_markers = (
        "MODEL_SEED=123",
        "SUBSET_SEED=42",
        "MAX_EPOCHS=12",
        "MIN_EPOCHS=8",
        "PATIENCE=3",
        "EVAL_CAP=20000",
        "TEST_CAP=100000",
        "MALLOC_ARENA_MAX",
        "endpoint_paired_bootstrap.csv",
        "recovery_status.json",
        "avito_endpoint_scaling_recovery_seed123",
    )
    missing = [marker for marker in required_markers if marker not in rendered]
    if missing:
        raise RuntimeError(f"Notebook is missing required markers: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
