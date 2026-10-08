#!/usr/bin/env python3
"""Build a Kaggle notebook that replicates the 1.2M -> full scaling contrast.

The completed seed-42 curve answers the exploratory scaling question, but its
paired bootstrap is conditional on one trained model per size.  This notebook
adds two independent model initialisations (seeds 7 and 123).  Each T4 trains
the 1.2M and full-data endpoints for one seed, using exactly the same nested
user subsets, June checkpoint-selection audience, July test audience and
training policy as the primary curve.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT
    / "reports/kaggle/expanded_user_scaling_2026_10/05_expanded_distinct_user_scaling_gpu.ipynb"
)
OUTPUT = (
    PROJECT
    / "reports/kaggle/expanded_user_scaling_2026_10/06_endpoint_seed_replication_gpu.ipynb"
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
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    base_cells = notebook["cells"]
    if len(base_cells) < 4:
        raise RuntimeError("Unexpected source notebook layout")

    intro = markdown_cell(
        "# Avito RecSys — replication of the high-data scaling contrast\n\n"
        "The primary seed-42 curve is complete. This notebook repeats the meaningful "
        "1.2M → all 1,985,972 distinct-user comparison with model seeds 7 and 123. "
        "Each GPU trains one seed. June selects the epoch; the same fixed 100k July "
        "users are used only for reporting."
    )

    run_parallel = code_cell(
        "runner=PROJECT/'scripts/39_run_extended_scaling.py'\n"
        "RUN_ROOT=WORK/'endpoint_replication'\n"
        "SPECS=[\n"
        " {'name':'seed7','gpu':0,'model_seed':7,'temp':WORK/'endpoint_seed7'},\n"
        " {'name':'seed123','gpu':1,'model_seed':123,'temp':WORK/'endpoint_seed123'},\n"
        "]\n"
        "POINTS=[('1200k_users',1200000),('full',0)]\n"
        "LOG_ROOT=WORK/'endpoint_replication_logs'; LOG_ROOT.mkdir(parents=True,exist_ok=True)\n"
        "processes=[]\n"
        "for spec in SPECS:\n"
        " cmd=[sys.executable,str(runner),'--project',str(PROJECT),'--data-root',str(DATA),"
        "'--output-root',str(spec['temp']),'--epochs','12','--min-epochs','8',"
        "'--early-stopping-patience','3','--subset-seed','42','--model-seed',str(spec['model_seed']),"
        "'--eval-cap','20000','--test-cap','100000','--device','cuda','--verify-hashes']"
        "+sum((['--point',f'{label}:{cap}'] for label,cap in POINTS),[])\n"
        " env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(spec['gpu']); env['PYTHONUNBUFFERED']='1'; env['PYTHONDONTWRITEBYTECODE']='1'\n"
        " log_path=LOG_ROOT/f\"{spec['name']}.log\"; log_handle=log_path.open('w',buffering=1)\n"
        " process=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=log_handle,stderr=subprocess.STDOUT,text=True)\n"
        " processes.append({**spec,'process':process,'log_path':log_path,'log_handle':log_handle,'cmd':cmd})\n"
        " print({'started':spec['name'],'physical_gpu':spec['gpu'],'pid':process.pid,'points':[p[0] for p in POINTS]},flush=True)\n"
        "while any(item['process'].poll() is None for item in processes):\n"
        " print({item['name']:item['process'].poll() for item in processes},flush=True); time.sleep(60)\n"
        "for item in processes: item['log_handle'].close()\n"
        "failures=[item for item in processes if item['process'].returncode!=0]\n"
        "if failures:\n"
        " tails={}\n"
        " for item in failures:\n"
        "  lines=item['log_path'].read_text(errors='replace').splitlines(); tails[item['name']]='\\n'.join(lines[-100:])\n"
        " raise RuntimeError(f'Endpoint replication failed: {tails}')\n"
        "if RUN_ROOT.exists(): shutil.rmtree(RUN_ROOT)\n"
        "RUN_ROOT.mkdir(parents=True)\n"
        "frames=[]\n"
        "for spec in SPECS:\n"
        " model_dir=spec['temp']/f\"modelseed{spec['model_seed']}\"\n"
        " if not all((model_dir/f'train{label}'/'metrics.json').is_file() for label,_ in POINTS): raise RuntimeError(f'Missing endpoint for {spec}')\n"
        " shutil.copytree(model_dir,RUN_ROOT/f\"modelseed{spec['model_seed']}\")\n"
        " frame=pd.read_csv(spec['temp']/'scaling_summary.csv'); frames.append(frame)\n"
        "summary=pd.concat(frames,ignore_index=True).sort_values(['model_seed','requested_user_cap'],ascending=[True,False])\n"
        "summary.to_csv(RUN_ROOT/'replication_summary.csv',index=False)\n"
        "(RUN_ROOT/'replication_summary.json').write_text(json.dumps(json.loads(summary.to_json(orient='records')),indent=2))\n"
        "(RUN_ROOT/'parallel_run.json').write_text(json.dumps({'points':[p[0] for p in POINTS],'subset_seed':42,'model_seeds':[7,123],'epochs':12,'min_epochs':8,'patience':3,'returncodes':{i['name']:i['process'].returncode for i in processes}},indent=2))\n"
        "shutil.copytree(LOG_ROOT,RUN_ROOT/'logs')\n"
        "display(summary)"
    )

    bootstrap = code_cell(
        "bootstrap_frames=[]\n"
        "for model_seed in (7,123):\n"
        " out=RUN_ROOT/'bootstrap'/f'seed{model_seed}_full_vs_1200k'\n"
        " cmd=[sys.executable,str(PROJECT/'scripts/26_compare_scaling_points.py'),"
        "'--reference',str(RUN_ROOT/f'modelseed{model_seed}/trainfull/test_predictions.parquet'),"
        "'--challenger',str(RUN_ROOT/f'modelseed{model_seed}/train1200k_users/test_predictions.parquet'),"
        "'--reference-name',f'full_seed{model_seed}','--challenger-name',f'1200k_seed{model_seed}',"
        "'--repeats','20000','--seed',str(1000+model_seed),'--output',str(out)]\n"
        " subprocess.run(cmd,cwd=PROJECT,check=True)\n"
        " frame=pd.read_csv(out/'paired_bootstrap.csv'); frame.insert(0,'model_seed',model_seed); bootstrap_frames.append(frame)\n"
        "bootstrap=pd.concat(bootstrap_frames,ignore_index=True)\n"
        "bootstrap.to_csv(RUN_ROOT/'endpoint_paired_bootstrap.csv',index=False)\n"
        "display(bootstrap)\n"
        "delta_rows=[]\n"
        "for model_seed in (7,123):\n"
        " part=summary[summary.model_seed==model_seed].set_index('label')\n"
        " delta_rows.append({'model_seed':model_seed,"
        "'recall_delta_full_minus_1200k':part.loc['full','test_recall@5']-part.loc['1200k_users','test_recall@5'],"
        "'ndcg_delta_full_minus_1200k':part.loc['full','test_ndcg@5']-part.loc['1200k_users','test_ndcg@5']})\n"
        "deltas=pd.DataFrame(delta_rows); deltas.to_csv(RUN_ROOT/'endpoint_seed_deltas.csv',index=False); display(deltas)"
    )

    interpretation = markdown_cell(
        "## Interpretation rule\n\n"
        "The July-user bootstrap quantifies evaluation-sample uncertainty conditional on each "
        "trained model. Agreement of the sign across seeds 42, 7 and 123 is the check against "
        "initialisation noise. These three seeds support a directional scaling conclusion but "
        "are not an equivalence test and must not be used to claim a universal plateau."
    )

    archive = code_cell(
        "archive_path=shutil.make_archive(str(WORK/'avito_endpoint_scaling_replication_seeds7_123'),'zip',RUN_ROOT)\n"
        "print(archive_path)"
    )

    notebook["cells"] = [intro, base_cells[1], base_cells[2], base_cells[3], run_parallel, bootstrap, interpretation, archive]
    notebook.setdefault("metadata", {})["accelerator"] = "GPU"
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell_{index}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
