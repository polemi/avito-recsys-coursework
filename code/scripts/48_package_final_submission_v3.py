#!/usr/bin/env python3
"""Build and validate the final October 2026 privacy-safe academic package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
PACKAGE_NAME = "avito_recsys_academic_final_2026_10_08"
OUT = DELIVERABLES / PACKAGE_NAME
ARCHIVE = DELIVERABLES / f"{PACKAGE_NAME}.zip"


DIRECT_FILES = {
    "docs/coursework_grade10_academic_final_v9.docx": "materials/coursework.docx",
    "docs/coursework_grade10_academic_final_v9.pdf": "materials/coursework.pdf",
    "artifacts/presentation/final/defense_recsys_academic_final_v17.pptx": "materials/defense.pptx",
    "artifacts/presentation/final/defense_recsys_academic_final_v17.pdf": "materials/defense.pdf",
    "docs/defense_recsys_academic_final_speaker_script.docx": "materials/speaker_script.docx",
    "docs/defense_recsys_academic_final_speaker_script.pdf": "materials/speaker_script.pdf",
    "docs/coursework_full_story_revised_2026_10_07.docx": "materials/full_story.docx",
    "docs/defense_slide_plan_detailed_2026_10_07.docx": "materials/slide_plan.docx",
    "deliverables/final_submission_grade10_2026_09/AB_RESULTS_AND_PROVENANCE.md": "evidence/AB_RESULTS_AND_PROVENANCE.md",
    "artifacts/runpod/d128_full2m_2026_10/extracted/output/checkpoint.pt": "model/checkpoint.pt",
    "artifacts/runpod/d128_full2m_2026_10/extracted/output/learning_curve.csv": "model/learning_curve.csv",
    "artifacts/runpod/d128_full2m_2026_10/extracted/input/model_d128_full2m.json": "model/model_d128_full2m.json",
    "reports/kaggle/expanded_user_scaling_2026_10/scaling_data/vocab.json": "model/vocab_leaf_refresh_2026_08.json",
    "reports/kaggle/expanded_user_scaling_2026_10/scaling_data/config.json": "code/reports/kaggle/expanded_user_scaling_2026_10/scaling_data/config.json",
    "reports/kaggle/expanded_user_scaling_2026_10/scaling_data/manifest.json": "code/reports/kaggle/expanded_user_scaling_2026_10/scaling_data/manifest.json",
    "reports/practical_demo_2026_09/reverse_ranking_metrics_fold3_full.csv": "evidence/reverse_ranking/metrics.csv",
    "reports/practical_demo_2026_09/reverse_ranking_example_fold3_full.csv": "evidence/reverse_ranking/example.csv",
    "reports/practical_demo_2026_09/same_name_full_path_audit.csv": "evidence/reverse_ranking/taxonomy_name_audit.csv",
    "pyproject.toml": "code/pyproject.toml",
    "requirements.txt": "code/requirements.txt",
}


TREE_FILES = {
    "reports/runpod_full2m_exact100k_comparison_2026_10_08": "evidence/final_model_exact100k",
    "reports/data_scaling_replication_2026_10": "evidence/data_scaling_replication",
    "src/avito_recsys": "code/src/avito_recsys",
    "tests": "code/tests",
}


CONFIGS = [
    "configs/data_v90.json",
    "configs/data_refresh_2026_08.json",
    "configs/data_scaling_extended_2026_10.json",
    "configs/distinct_user_scaling_2026_10.json",
    "configs/model_reduced.json",
    "configs/model_final_d128.json",
    "configs/window_history_180.json",
]


SCRIPTS = [
    "scripts/00_audit_inputs.py",
    "scripts/00_partition_raw.py",
    "scripts/01_build_dataset.py",
    "scripts/02_validate_dataset.py",
    "scripts/03_run_eda.py",
    "scripts/04_run_baselines.py",
    "scripts/05_summarize_baselines.py",
    "scripts/06_train_multi_interest.py",
    "scripts/07_run_k_ablation.py",
    "scripts/08_analyze_baselines.py",
    "scripts/09_reverse_rank.py",
    "scripts/10_interpret_interests.py",
    "scripts/17_run_old_flow_baseline.py",
    "scripts/20_analyze_k_ablation.py",
    "scripts/24_compare_neural_to_baseline.py",
    "scripts/26_compare_scaling_points.py",
    "scripts/28_summarize_window_ablation.py",
    "scripts/30_build_grade10_evidence.py",
    "scripts/35_build_practical_demo.py",
    "scripts/39_audit_distinct_user_scaling.py",
    "scripts/39_run_extended_scaling.py",
    "scripts/40_prepare_extended_scaling_kaggle.py",
    "scripts/41_merge_expanded_user_pool.py",
    "scripts/42_prepare_expanded_user_scaling_kaggle.py",
    "scripts/44_build_scaling_endpoint_replication_notebook.py",
    "scripts/45_build_scaling_seed123_recovery_notebook.py",
    "scripts/46_audit_endpoint_replication.py",
    "scripts/47_submission_smoke.py",
    "scripts/48_package_final_submission_v3.py",
    "scripts/56_compare_runpod_full2m_exact100k.py",
    "scripts/58_analyze_full2m_segments.py",
]


NOTEBOOKS: list[str] = []


README = """# Рекомендательная система новых категорий Авито

Финальный privacy-safe комплект курсовой: материалы защиты, агрегированные
результаты, checkpoint, конфигурации и исполняемый код. Исходные DWH-выгрузки,
построчные прогнозы, сформированные аудитории и реальные идентификаторы
пользователей в архив не входят.

## Материалы

- `materials/coursework.docx` и `.pdf` — текст курсовой;
- `materials/defense.pptx` и `.pdf` — презентация;
- `materials/speaker_script.docx` и `.pdf` — текст выступления;
- `materials/full_story.docx` — полный связный рассказ о работе;
- `materials/slide_plan.docx` — подробный план основной и резервной части;
- `evidence/` — агрегированные результаты и обезличенные примеры;
- `model/` — checkpoint полного d=128 запуска, конфигурация, vocabulary и кривая обучения;
- `code/` — Python-пакет, тесты, скрипты и конфигурации.

## Итоговый результат

- Transformer `d=128`, `K=2`: 331 904 параметра.
- Обучение: 1 985 972 уникальных пользователя и 4 162 636 примеров
  «пользователь — дата прогноза».
- Выполнены и выбраны все 12 эпох. Полный запуск занял 2 часа 9 минут на
  NVIDIA H100 80GB.
- Оценка: одна фиксированная аудитория из 100 000 пользователей, 351 категория.
- Transformer: Recall@5 `0.404634`, NDCG@5 `0.313043`.
- PureSVD: Recall@5 `0.307383`, NDCG@5 `0.233241`.
- Разность Transformer − PureSVD: Recall@5 `+0.097251`, парный 95%-й ДИ
  `[0.094848; 0.099644]`; NDCG@5 `+0.079801`, ДИ `[0.077968; 0.081640]`.
  Обе разности статистически значимы при `alpha=0.05`.
- 30-дневная офлайн-адаптация исторической модели переходов на тех же
  пользователях получила Recall@5 `0.168512`. Это воспроизводимая адаптация
  идеи старого решения, а не ретроспективная метрика промышленной модели.

Главный отчёт: `evidence/final_model_exact100k/summary.json`. Интервалы относятся
к фиксированным прогнозам и не учитывают разброс повторного обучения. PureSVD
обучался на 768 708 примерах, поэтому основная таблица сравнивает законченные
методы, а не изолированный эффект архитектуры.

Вспомогательная d=64 модель использована для контролируемых экспериментов.
Контраст 1,2 млн → 2 млн пользователей положителен при инициализациях 42, 7 и
123; средний прирост Recall@5 равен `+0.002936`. Это описательное среднее трёх
запусков, а не интервал неопределённости повторного обучения финальной d=128.

Офлайн-метрики не доказывают причинный эффект CRM-коммуникации. История трёх
кампаний и границы online-вывода зафиксированы в
`evidence/AB_RESULTS_AND_PROVENANCE.md`.

## Проверка без закрытых данных

Из каталога `code/`:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/47_submission_smoke.py
```

Smoke создаёт синтетический parquet во временной папке, проверяет
псевдонимизацию `snapshot_id`, выборку уникальных пользователей, collate и один
forward pass. Для полного обучения нужны закрытые обезличенные snapshots.

Основные команды на разрешённых snapshots:

```bash
python scripts/02_validate_dataset.py --help
python scripts/04_run_baselines.py --help
python scripts/06_train_multi_interest.py --help
python scripts/09_reverse_rank.py --help
python scripts/56_compare_runpod_full2m_exact100k.py --help
```

## Границы интерпретации

- метрики рассчитаны среди пользователей, у которых в следующие 14 дней был
  хотя бы один контакт в новой категории;
- основной полный d=128 запуск выполнен с одной инициализацией;
- сравнение 90 и 180 дней не является чистой абляцией только длины истории;
- сценарий «категория → пользователи» проверен на шести категориях и d=64;
- CRM-кампании различались категорией, временем и сообщением и не являются
  прямым A/B-сравнением алгоритмов.

## Целостность и приватность

- `MANIFEST.json` содержит размер и SHA-256 каждого файла;
- `SHA256SUMS.txt` проверяет копию комплекта;
- `qa/QA_REPORT.json` фиксирует тесты, synthetic smoke и privacy scan;
- в архив не включены notebooks со старыми промежуточными результатами;
- в архиве нет parquet, raw DWH, значений `user_id` или абсолютных локальных путей.
"""


EVIDENCE_GUIDE = """# Карта доказательств и границы выводов

## Основное сравнение

Источник: `final_model_exact100k/summary.json`, `comparison.csv` и
`paired_bootstrap.csv`. Все методы оценены на одной аудитории из 100 000
пользователей с одинаковыми правильными ответами и масками допустимых
категорий. Это главный результат работы.

Полная d=128 модель обучена на 1 985 972 уникальных пользователях и 4 162 636
примерах. PureSVD обучался на 768 708 примерах, поэтому сравнение отвечает на
вопрос о качестве законченных методов, но не является чистой абляцией
архитектуры при полностью одинаковом объёме обучения.

Парный bootstrap перевыбирает пользователей и оценивает неопределённость
разницы фиксированных прогнозов. Он не учитывает вариацию полного повторного
обучения. Финальный d=128 запуск выполнен с seed 42.

## Историческая модель переходов

Число `0.168512` — результат 30-дневной офлайн-адаптации идеи старой модели на
той же аудитории, целях и масках. Это не сохранённая метрика исходного
промышленного расчёта. Бизнес-результаты старого решения сопоставляются отдельно
по CRM-кампаниям.

## Дополнительные эксперименты

- `data_scaling_replication/`: d=64, полная кривая для seed 42 и повторение
  сравнения 1,2 млн → 2 млн при seeds 42, 7 и 123;
- `practical_demo/`: обезличенные примеры выдачи и сценария
  «категория → пользователи»; reverse ranking проверен на d=64 и шести
  категориях;
- сравнение окна 90/180 дней в записке не трактуется как чистая абляция:
  составы обучающих подвыборок совпали примерно на 96,5%, а статистически
  подтверждённого преимущества 180 дней не получено.

## CRM

`AB_RESULTS_AND_PROVENANCE.md` содержит агрегированные внутренние результаты.
Локальная копия не содержит исходной таблицы размеров и событий групп, поэтому
проценты не пересчитываются из первичных строк. Кампании различались категорией,
временем и сообщением и не являются прямым A/B-сравнением алгоритмов.
"""


REVERSE_RANKING_GUIDE = """# Сценарий «категория → пользователи»

Для заранее выбранной категории вспомогательная модель `d=64`, `K=2`
рассчитала оценку для каждого допустимого пользователя, после чего пользователи
были отсортированы по этой оценке. Проверка выполнена на полном июньском срезе
из 1 541 603 пользователей и шести категориях.

- `metrics.csv` — Precision, Recall и lift при нескольких размерах аудитории;
- `example.csv` — десять обезличенных строк одного ранжирования;
- `taxonomy_name_audit.csv` — контроль того, что одинаковые названия в разных
  ветках каталога не склеиваются.

Это поддерживающий операционный эксперимент. Его нельзя переносить на все 351
категорию или выдавать за результат финальной d=128 конфигурации.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source_relative: str, destination_relative: str) -> None:
    source = ROOT / source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = OUT / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
        copy_office_sanitized(source, destination)
    else:
        shutil.copy2(source, destination)


def copy_office_sanitized(source: Path, destination: Path) -> None:
    """Copy OOXML while removing machine-local path prefixes from XML parts."""
    with zipfile.ZipFile(source) as source_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as destination_archive:
        for info in source_archive.infolist():
            payload = source_archive.read(info.filename)
            if info.filename.endswith((".xml", ".rels")):
                text = payload.decode("utf-8")
                # Keep the referenced filename, but remove a local Downloads or
                # Desktop prefix. Any other absolute home path becomes a marker.
                text = re.sub(
                    r"/(?:Users|home)/[^/<>]+/(?:Downloads|Desktop)/",
                    "",
                    text,
                )
                text = re.sub(
                    r"/(?:Users|home)/[^<\r\n\"]+",
                    "[local path removed]",
                    text,
                )
                text = re.sub(
                    "file" + r"://[^<\r\n\"]+",
                    "[local link removed]",
                    text,
                    flags=re.IGNORECASE,
                )
                payload = text.encode("utf-8")
            destination_archive.writestr(info, payload)


def copy_tree(source_relative: str, destination_relative: str) -> None:
    source = ROOT / source_relative
    if not source.is_dir():
        raise FileNotFoundError(source)
    for item in sorted(source.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        relative = item.relative_to(source)
        destination = OUT / destination_relative / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def copy_notebook_sanitized(source_relative: str) -> None:
    source = ROOT / source_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    metadata = notebook.get("metadata", {})
    notebook["metadata"] = {
        key: metadata[key]
        for key in ("kernelspec", "language_info")
        if key in metadata
    }
    destination = OUT / "code" / source_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def run_checked(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}"
        )
    clean_output = completed.stdout.replace(str(ROOT), "<PROJECT_ROOT>")
    clean_output = re.sub(
        r"/(?:Users|home)/[^\s:]+", "<LOCAL_PATH>", clean_output
    )
    clean_output = re.sub(
        r"/(?:opt|private|tmp|var)/[^\s:]+", "<RUNTIME_PATH>", clean_output
    )
    display_command = ["python", *command[1:]]
    display_command = [item.replace(str(ROOT), "<PROJECT_ROOT>") for item in display_command]
    display_command = [
        re.sub(r"/(?:opt|private|tmp|var)/[^\s:]+", "<RUNTIME_PATH>", item)
        for item in display_command
    ]
    return {
        "command": display_command,
        "returncode": completed.returncode,
        "output": clean_output.strip(),
    }


def verify_packaged_code() -> dict:
    """Install the copied package into a disposable venv and test that copy."""
    with tempfile.TemporaryDirectory(prefix="avito_final_package_venv_") as temporary:
        temporary_path = Path(temporary)
        venv = temporary_path / "venv"
        create = run_checked(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            ROOT,
        )
        python = venv / "bin" / "python"
        install = run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                str(OUT / "code"),
            ],
            OUT / "code",
        )
        tests = run_checked(
            [str(python), "-m", "pytest", "-q"],
            OUT / "code",
        )
        smoke = run_checked(
            [str(python), "scripts/47_submission_smoke.py"],
            OUT / "code",
        )
    return {
        "venv_creation": create,
        "package_install": install,
        "pytest": tests,
        "synthetic_smoke": smoke,
    }


def office_xml_text(path: Path) -> str:
    if path.suffix.lower() not in {".docx", ".pptx", ".xlsx"}:
        return ""
    with zipfile.ZipFile(path) as archive:
        return "\n".join(
            archive.read(name).decode("utf-8", errors="ignore")
            for name in archive.namelist()
            if name.endswith((".xml", ".rels"))
        )


def privacy_and_stale_scan() -> dict:
    local_path_patterns = [
        re.compile(r"/(?:Users|home)/[^\s<>'\"]+"),
        re.compile(r"[A-Za-z]:\\\\[^\s<>'\"]+"),
        re.compile("file" + r"://", re.IGNORECASE),
        re.compile(r"\.codex(?:/|\\\\)"),
        re.compile(r'/(?:private/)?' + r'var/folders/[^\s<>\'\"]+'),
        re.compile('/opt' + r'/anaconda[^\s<>\'\"]+'),
    ]
    stale_patterns = [
        re.compile(r"seed\s*123.{0,80}(?:pending|ожида|не включ)", re.IGNORECASE | re.DOTALL),
        re.compile(r"третья.{0,60}инициализац.{0,60}не включ", re.IGNORECASE | re.DOTALL),
        re.compile(r"две\s+инициализац", re.IGNORECASE),
        re.compile(r"1[,.]215\s*(?:m|млн)", re.IGNORECASE),
        re.compile(r"0[,.]403752.{0,120}0[,.]310124", re.DOTALL),
        re.compile(r"0[,.]402978.{0,120}0[,.]315051", re.DOTALL),
        re.compile(r"\+0[,.]093628"),
    ]
    path_hits: list[dict] = []
    stale_hits: list[dict] = []
    notebook_outputs = 0
    forbidden_data_files: list[str] = []
    for file in sorted(path for path in OUT.rglob("*") if path.is_file()):
        relative = str(file.relative_to(OUT))
        if file.suffix.lower() in {".parquet", ".feather", ".arrow", ".pkl", ".pickle"}:
            forbidden_data_files.append(relative)
        text = ""
        if file.suffix.lower() in {".md", ".txt", ".csv", ".json", ".py", ".toml", ".ipynb"}:
            text = file.read_text(encoding="utf-8", errors="ignore")
        elif file.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
            text = office_xml_text(file)
        raw = file.read_bytes()
        for pattern in local_path_patterns:
            match = pattern.search(text)
            if match:
                path_hits.append({"file": relative, "match": match.group(0)[:180]})
            byte_match = pattern.search(raw.decode("latin-1", errors="ignore"))
            if byte_match and not match:
                path_hits.append({"file": relative, "match": byte_match.group(0)[:180]})
        if relative.startswith("materials/") or relative.startswith("evidence/"):
            for pattern in stale_patterns:
                match = pattern.search(text)
                if match:
                    stale_hits.append({"file": relative, "match": match.group(0)[:240]})
        if file.suffix.lower() == ".ipynb":
            notebook = json.loads(file.read_text(encoding="utf-8"))
            notebook_outputs += sum(
                len(cell.get("outputs", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            )
    if path_hits or stale_hits or notebook_outputs or forbidden_data_files:
        raise RuntimeError(
            json.dumps(
                {
                    "absolute_path_hits": path_hits,
                    "stale_claim_hits": stale_hits,
                    "notebook_outputs": notebook_outputs,
                    "forbidden_data_files": forbidden_data_files,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return {
        "status": "pass",
        "absolute_local_path_hits": 0,
        "stale_claim_hits": 0,
        "notebook_outputs": 0,
        "forbidden_row_level_data_files": 0,
        "raw_data_included": False,
        "direct_identifier_values_included": False,
        "note": "Schema names such as user_id remain in source code; no source values are packaged.",
    }


def build_manifest() -> dict:
    files = []
    for file in sorted(path for path in OUT.rglob("*") if path.is_file()):
        if file.name in {"MANIFEST.json", "SHA256SUMS.txt"}:
            continue
        files.append(
            {
                "path": str(file.relative_to(OUT)),
                "size_bytes": file.stat().st_size,
                "sha256": sha256(file),
            }
        )
    return {
        "manifest_version": 1,
        "package": PACKAGE_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_class": "no raw data; no direct identifiers; aggregate or pseudonymous evidence only",
        "full_train_users_exact": 1_985_972,
        "full_train_users_display": "2 млн",
        "files": files,
    }


def write_checksums() -> None:
    lines = []
    for file in sorted(path for path in OUT.rglob("*") if path.is_file()):
        if file.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256(file)}  {file.relative_to(OUT)}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksums() -> int:
    lines = (OUT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        actual = sha256(OUT / relative)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
    return len(lines)


def create_archive() -> None:
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(path for path in OUT.rglob("*") if path.is_file()):
            archive.write(file, arcname=str(Path(PACKAGE_NAME) / file.relative_to(OUT)))
    with zipfile.ZipFile(ARCHIVE) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace only the fixed October package path",
    )
    args = parser.parse_args()
    if OUT.exists() or ARCHIVE.exists():
        if not args.replace:
            raise FileExistsError("Package already exists; rerun with --replace")
        if OUT.exists():
            shutil.rmtree(OUT)
        if ARCHIVE.exists():
            ARCHIVE.unlink()
    OUT.mkdir(parents=True)

    for source, destination in DIRECT_FILES.items():
        copy_file(source, destination)
    for source, destination in TREE_FILES.items():
        copy_tree(source, destination)
    for source in CONFIGS:
        copy_file(source, str(Path("code") / source))
    for source in SCRIPTS:
        copy_file(source, str(Path("code") / source))
    for source in NOTEBOOKS:
        copy_notebook_sanitized(source)

    (OUT / "README.md").write_text(README, encoding="utf-8")
    evidence_guide_path = OUT / "evidence/METHODOLOGY_AND_LIMITATIONS.md"
    evidence_guide_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_guide_path.write_text(EVIDENCE_GUIDE, encoding="utf-8")
    reverse_guide_path = OUT / "evidence/reverse_ranking/README.md"
    reverse_guide_path.parent.mkdir(parents=True, exist_ok=True)
    reverse_guide_path.write_text(REVERSE_RANKING_GUIDE, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    pytest_result = run_checked([sys.executable, "-m", "pytest", "-q"], ROOT, env)
    smoke_result = run_checked(
        [sys.executable, "scripts/47_submission_smoke.py"], ROOT, env
    )
    compile_result = run_checked(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            str(OUT / "code/src"),
            str(OUT / "code/scripts"),
        ],
        OUT / "code",
    )
    for cache_dir in sorted(OUT.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir)
    packaged_verification = verify_packaged_code()
    for generated in list(OUT.rglob("__pycache__")) + list(OUT.rglob(".pytest_cache")):
        if generated.is_dir():
            shutil.rmtree(generated)
    for generated in list((OUT / "code").glob("build")) + list((OUT / "code/src").glob("*.egg-info")):
        if generated.is_dir():
            shutil.rmtree(generated)
    privacy_result = privacy_and_stale_scan()
    qa_report = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "pytest": pytest_result,
        "synthetic_smoke": smoke_result,
        "compileall": compile_result,
        "isolated_packaged_copy": packaged_verification,
        "privacy_and_stale_scan": privacy_result,
        "zip_integrity": "checked after report generation",
        "scope_note": "Full training needs restricted snapshots and GPU; the public package validates code with synthetic data.",
    }
    qa_path = OUT / "qa/QA_REPORT.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    privacy_and_stale_scan()

    manifest = build_manifest()
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_checksums()
    checksum_count = verify_checksums()
    create_archive()
    print(
        json.dumps(
            {
                "status": "pass",
                "folder": str(OUT),
                "archive": str(ARCHIVE),
                "archive_sha256": sha256(ARCHIVE),
                "files": len([path for path in OUT.rglob("*") if path.is_file()]),
                "checksums_verified": checksum_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
