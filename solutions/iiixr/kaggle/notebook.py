"""Concise Kaggle notebook helper for iiixr_box_sanity_t4.ipynb and iiixr_box_full_t4.ipynb."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

WORKING = Path("/kaggle/working")
CACHE = Path("/kaggle/temp/iiixr")
ERROR_LOG = WORKING / "iiixr_launch_error.txt"
SUMMARY_PATH = WORKING / "iiixr_summary.json"


def check_kaggle_t4x2() -> list[str]:
    missing = [c for c in ("git", "nvidia-smi") if shutil.which(c) is None]
    if missing:
        raise RuntimeError(f"Missing required system tools: {missing}. Enable GPU T4 x2 in Kaggle Settings.")
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if len(gpus) < 2 or any("T4" not in g for g in gpus[:2]):
        raise RuntimeError(f"Select 'GPU T4 x2' in Kaggle Notebook Settings. Detected GPUs: {gpus}")
    return gpus


def _auto_find_caw_tar() -> str:
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        matches = sorted(p for p in input_root.rglob("quantiphy_full.tar") if p.is_file())
        if len(matches) == 1:
            return str(matches[0])
    return ""


def launch(source: Path, mode: str, config: dict[str, Any]) -> dict[str, Any]:
    WORKING.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.unlink(missing_ok=True)
    try:
        if mode in ("sanity", "full"):
            check_kaggle_t4x2()

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "solutions.iiixr.kaggle.workflow",
            mode,
            "--cache",
            str(CACHE),
            "--output",
            str(WORKING),
            "--dtype-profile",
            str(config.get("DTYPE_PROFILE", "t4_fp16")),
            "--output-budget-gib",
            str(config.get("OUTPUT_BUDGET_GIB", 18.0)),
        ]
        if config.get("SCOPE"):
            cmd += ["--scope", str(config["SCOPE"])]
        if config.get("DATASET_DIR"):
            cmd += ["--dataset-dir", str(config["DATASET_DIR"])]
        if config.get("MODELS_DIR"):
            cmd += ["--models-dir", str(config["MODELS_DIR"])]

        caw_tar = config.get("CAW_FULL_TAR") or _auto_find_caw_tar()
        if caw_tar:
            cmd += ["--caw-tar", str(caw_tar)]

        if config.get("RULE_PARSE_ONLY"):
            cmd.append("--rule-parse-only")
        if int(config.get("PARTITION_COUNT", 1)) > 1:
            cmd += [
                "--partition-count",
                str(config["PARTITION_COUNT"]),
                "--partition-index",
                str(config.get("PARTITION_INDEX", 0)),
            ]

        proc = subprocess.run(cmd, cwd=str(source), check=False)
        if SUMMARY_PATH.is_file():
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            show_summary(summary)
            if summary["status"] in ("ERROR", "FAIL") or proc.returncode != 0:
                raise RuntimeError(f"IIIXR {mode} finished with status={summary['status']} (rc={proc.returncode}).")
            return summary
        raise RuntimeError(f"IIIXR {mode} exited with code {proc.returncode} before writing {SUMMARY_PATH}.")
    except Exception:
        ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise


def show_summary(summary: dict[str, Any]) -> None:
    scope = str(summary.get("scope", ""))
    is_static = scope in ("static2d_sanity", "static2d_41", "static2d_survey158", "all2d_69")
    label = "IIIXR 2D Static" if is_static else "IIIXR g_box"
    print(f"=== {label} · {summary['mode'].upper()} · {summary['status']} ===")
    print(f"Scope                  : {scope}")
    print(f"Resolved Source Commit : {summary.get('source_commit')}")
    print(f"Pinned Upstream Commit : {summary.get('upstream_commit')}")
    print(f"Run Directory          : {summary.get('run_dir')}")
    for arch in summary.get("archives", []):
        print(f"Archive                : {arch['path']} ({arch['bytes'] / (1024**2):.2f} MiB, sha256={arch['sha256'][:16]}...)")
    print("Offline Viewer         : Extract archive and open run/index.html in any browser.")

    if summary.get("stage_errors"):
        first = summary["stage_errors"][0]
        print("\n[FIRST WORKER ERROR]")
        print(json.dumps(first, indent=2, ensure_ascii=False))

    ev = summary.get("evaluation") or {}
    subset_desc = "2D static subset" if is_static else "2D motion subset"
    print(f"\nMean per-question MRA ({subset_desc}):")
    print(json.dumps(ev.get("mean_per_question_mra", {}), indent=2))

    pq = Path(summary.get("per_question_csv", ""))
    if pq.is_file():
        with pq.open("r", encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("in_scope_2d_motion") == "True"]
        print(f"\nPer-question summary ({len(rows)} rows):")
        hdr = f"{'ID':>5}  {'Video':<25}  {'Kind':<9}  {'CachedBox':>10}  {'FreshBox':>10}  {'CAW':>8}  {'GT':>8}  {'Status':<8}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(
                f"{r['id']:>5}  {r['video_id']:<25}  {r['kind']:<9}  "
                f"{str(r['cached_box'])[:10]:>10}  {str(r['fresh_box'])[:10]:>10}  "
                f"{str(r['caw'])[:8]:>8}  {str(r['ground_truth'])[:8]:>8}  {r['status']:<8}"
            )
