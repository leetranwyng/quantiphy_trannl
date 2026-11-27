"""Main CLI orchestrator for IIIXR g_box (replay, sanity on 2xT4, full 18-question on 2xT4)."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from solutions.iiixr.kaggle.kinematics import (
    evaluate_xground_and_fuse,
    measure_video_instrumented,
    parse_questions_v1_v2,
    run_scene_ruler_on_video,
)
from solutions.iiixr.kaggle.schema import (
    ALL_2D_12_VIDEOS,
    EXPECTED_VIDEO_BY_ID,
    REPO_ROOT,
    SANITY_4_IDS,
    STATIC_2D_11_VIDEOS,
    STATIC_2D_41_IDS,
    STATIC_2D_SCOPES,
    TARGET_6_VIDEOS,
    TARGET_18_IDS,
    UPSTREAM_DIR,
    VALID_SCOPES,
    FreshAntiCacheGuard,
    TraceStore,
    ensure_upstream_tree,
    extract_iiixr_file_fps_val,
    extract_owl_threshold,
    fingerprint,
    load_cached_mot_val,
    load_cached_val_geo_raw,
    load_validation_inference_inputs,
    probe_and_compare_video,
    read_caw_predictions_from_tar,
    read_json,
    read_snapshots,
    resolve_video_files,
    sha256_file,
    stage_assets_from_caw_tar,
    utc_now_iso,
    write_json_atomic,
)
from solutions.iiixr.kaggle.viewer import build_viewer_payload_for_question, render_failure_analysis_html

GIB = 1024**3


def _git_head_sha(repo: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _query_gpu_devices(require_two_t4: bool = False) -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        if require_two_t4:
            raise RuntimeError("nvidia-smi not found. Select Kaggle Accelerator 'GPU T4 x2'.")
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        if require_two_t4:
            raise RuntimeError(f"nvidia-smi query failed: {exc}") from exc
        return []
    keys = ["index", "uuid", "name", "driver", "compute_capability", "total_mib", "free_mib"]
    devices = [dict(zip(keys, row)) for row in csv.reader(out.splitlines(), skipinitialspace=True) if row]
    if require_two_t4:
        if len(devices) < 2 or not all("T4" in d["name"] for d in devices[:2]):
            raise RuntimeError(f"Expected Kaggle GPU T4 x2 accelerator, got: {devices}")
    return devices


def _ensure_hf_assets(
    cache_dir: Path,
    dataset_dir_arg: Path | None,
    models_dir_arg: Path | None,
    caw_tar_arg: Path | None,
    need_models: bool,
    video_ids: list[str],
) -> dict[str, Any]:
    """Resolve or download pinned validation dataset + 4 g_box models."""
    snaps = read_snapshots()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset resolution (offline mount, CAW tar fallback for local replay, or pinned HF snapshot_download)
    csv_path: Path | None = None
    video_root: Path | None = None

    if dataset_dir_arg and dataset_dir_arg.exists():
        for cand_csv in (
            dataset_dir_arg / "validation_dataset.csv",
            dataset_dir_arg / "quantiphy_validation.csv",
            dataset_dir_arg / "run" / "configs" / "quantiphy_validation.csv",
        ):
            if cand_csv.is_file():
                csv_path = cand_csv
                break
        video_root = dataset_dir_arg
    elif caw_tar_arg and caw_tar_arg.is_file() and not need_models:
        staged = stage_assets_from_caw_tar(caw_tar_arg, cache_dir / "staged_from_caw_tar", tuple(video_ids))
        csv_path = staged["csv_path"]
        video_root = staged["video_dir"]
    else:
        from huggingface_hub import snapshot_download

        ds_cfg = snaps["dataset"]
        ds_dest = cache_dir / "dataset"
        allow = ["validation_dataset.csv"] + [f"*{vid}.mp4" for vid in video_ids]
        snapshot_download(
            repo_id=ds_cfg["repo_id"],
            repo_type="dataset",
            revision=ds_cfg["iiixr_revision"],
            local_dir=str(ds_dest),
            allow_patterns=allow,
            max_workers=2,
        )
        csv_path = ds_dest / "validation_dataset.csv"
        video_root = ds_dest

    if csv_path is None or not csv_path.is_file() or video_root is None:
        raise FileNotFoundError(
            "Could not locate validation_dataset.csv / quantiphy_validation.csv. "
            "Pass --dataset-dir, --caw-tar (for replay), or enable internet for pinned HF download."
        )

    video_paths = resolve_video_files(video_root, video_ids)

    # 2. Model directories (only when running GPU fresh/sanity)
    model_dirs: dict[str, Path] = {}
    model_revs: dict[str, str] = {}
    if need_models:
        for m_key, m_cfg in snaps["models"].items():
            model_revs[m_key] = m_cfg["revision"]
            local_name = m_cfg["repo_id"].split("/")[-1]
            if models_dir_arg and (models_dir_arg / local_name).is_dir():
                model_dirs[m_key] = models_dir_arg / local_name
            elif models_dir_arg and (models_dir_arg / m_key).is_dir():
                model_dirs[m_key] = models_dir_arg / m_key
            else:
                from huggingface_hub import snapshot_download

                dest = cache_dir / "models" / local_name
                snapshot_download(
                    repo_id=m_cfg["repo_id"],
                    repo_type="model",
                    revision=m_cfg["revision"],
                    local_dir=str(dest),
                    allow_patterns=m_cfg["allow_patterns"],
                    max_workers=2,
                )
                model_dirs[m_key] = dest

    return {
        "csv_path": csv_path,
        "video_paths": video_paths,
        "model_dirs": model_dirs,
        "model_revs": model_revs,
    }


def _copy_videos_into_run(run_dir: Path, video_paths: dict[str, Path]) -> None:
    vid_out = run_dir / "videos"
    vid_out.mkdir(parents=True, exist_ok=True)
    for vid, src in video_paths.items():
        dst = vid_out / f"{vid}.mp4"
        if not dst.is_file():
            shutil.copy2(src, dst)


def _snapshot_minimal_source(run_dir: Path) -> None:
    """Bundle verifier + pure-CPU kinematics/upstream geo scripts so run/verify.py works completely standalone."""
    src_out = run_dir / "source"
    files_to_copy = [
        Path("solutions/__init__.py"),
        Path("solutions/iiixr/__init__.py"),
        Path("solutions/iiixr/kaggle/__init__.py"),
        Path("solutions/iiixr/kaggle/snapshots.json"),
        Path("solutions/iiixr/kaggle/schema.py"),
        Path("solutions/iiixr/kaggle/kinematics.py"),
        Path("solutions/iiixr/kaggle/evaluate.py"),
        Path("solutions/iiixr/kaggle/verify.py"),
        Path("solutions/iiixr/upstream/geo/rules.py"),
        Path("solutions/iiixr/upstream/geo/rules_v2.py"),
        Path("solutions/iiixr/upstream/geo/parse.py"),
        Path("solutions/iiixr/upstream/geo/parse_v2.py"),
        Path("solutions/iiixr/upstream/geo/track_common.py"),
        Path("solutions/iiixr/upstream/geo/measure.py"),
        Path("solutions/iiixr/upstream/geo/measure_v3.py"),
        Path("solutions/iiixr/upstream/geo/scene_ruler.py"),
        Path("solutions/iiixr/upstream/tools/qp_common.py"),
    ]
    for rel in files_to_copy:
        full = REPO_ROOT / rel
        if full.is_file():
            dst = src_out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full, dst)
    shutil.copy2(REPO_ROOT / "solutions/iiixr/kaggle/verify.py", run_dir / "verify.py")


def _write_environment_and_config_diff(run_dir: Path, mode: str, dtype_profile: str, devices: list[dict[str, Any]]) -> None:
    cfg_dir = run_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    pkgs = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "qwen-vl-utils", "scipy", "opencv-python-headless", "numpy", "pandas", "pillow"):
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = None
    write_json_atomic(
        cfg_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "mode": mode,
            "dtype_profile": dtype_profile,
            "devices": devices,
            "packages": pkgs,
            "recorded_at": utc_now_iso(),
        },
    )
    write_json_atomic(
        cfg_dir / "config_diff.json",
        {
            "upstream_hardware_and_engine": "Single 96GB Ampere+ GPU, vLLM 0.30.0 BF16 for Qwen3-VL-8B, PyTorch 2.13 BF16 for SAM2.1/G-DINO/OWLv2",
            "kaggle_t4x2_adaptation": {
                "stage_1_qwen3_vl_8b": "Transformers Qwen3VLForConditionalGeneration, device_map='balanced' across 2xT4 (15GiB x 2), FP16, sdpa, batch 1, do_sample=False, subprocess exits to free 100% VRAM before Stage 2",
                "stage_2_detectors_and_sam2": "2 parallel worker processes (1 T4 GPU per process, sharded by video); sequential load/unload of GroundingDINO -> OWLv2 -> SAM2.1 with batch 1 and dtype_profile=" + dtype_profile,
                "stage_3_kinematics": "Exact upstream CPU numpy/scipy formulas (geo/measure_v3.py --fix abcdef + 04_geo_chain.sh env flags + xground + sruler)",
                "excluded_non_g_box_components": [
                    "Code-as-World-VL-9B",
                    "Molmo2-8B",
                    "Sa2VA-Qwen3-VL-4B-SAM3",
                    "MoGe-2 / UniDepthV2 / VGGT / DA3 / DepthPro (3D only, unused by 2d_ratio)",
                    "CoTracker3 (g_pt audit branch only, not part of g_box)",
                    "Kimi-K2.5",
                ],
            },
        },
    )


def _stream_subprocesses(procs: list[tuple[subprocess.Popen, Path, str]]) -> list[int]:
    offsets = {log_p: 0 for _, log_p, _ in procs}
    while True:
        alive = False
        for p, log_p, tag in procs:
            alive |= p.poll() is None
            if log_p.is_file():
                with log_p.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(offsets[log_p])
                    chunk = fh.read()
                    offsets[log_p] = fh.tell()
                    if chunk:
                        for ln in chunk.splitlines():
                            print(f"[{tag}] {ln}", flush=True)
        if not alive:
            break
        time.sleep(1.5)
    return [p.returncode for p, _, _ in procs]


def _run_gpu_vlm_subprocess(
    run_dir: Path,
    work_dir: Path,
    run_id: str,
    qwen_dir: Path,
    qwen_rev: str,
    dtype_profile: str,
    parse_rows: list[dict[str, Any]],
    vlm_ground_videos: list[dict[str, Any]],
    devices: list[dict[str, Any]],
    run_llm_parse: bool,
    trace_enabled: bool = True,
    seed: int = 0,
    tag_suffix: str = "main",
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = work_dir / f"vlm_receipt_{tag_suffix}.json"
    work_json = work_dir / f"vlm_work_{tag_suffix}.json"
    log_path = work_dir / f"vlm_worker_{tag_suffix}.log"
    receipt_path.unlink(missing_ok=True)

    write_json_atomic(
        work_json,
        {
            "run_dir": str(run_dir),
            "run_id": run_id,
            "qwen_model_dir": str(qwen_dir),
            "qwen_revision": qwen_rev,
            "dtype_profile": dtype_profile,
            "run_llm_parse": run_llm_parse,
            "parse_rows": parse_rows,
            "vlm_ground_videos": vlm_ground_videos,
            "trace_enabled": trace_enabled,
            "seed": seed,
            "receipt_path": str(receipt_path),
        },
    )
    gpu_uuids = ",".join(d["uuid"] for d in devices[:2]) if devices else "0"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu_uuids,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [sys.executable, "-u", "-m", "solutions.iiixr.kaggle.gpu_stages", "vlm", str(work_json)]
    with log_path.open("wb") as lf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    rcs = _stream_subprocesses([(proc, log_path, f"vlm-{tag_suffix}")])
    if receipt_path.is_file():
        r_json = read_json(receipt_path)
        if rcs[0] != 0 and r_json.get("status") == "ok":
            r_json["status"] = "error"
            r_json.setdefault("errors", []).append({
                "worker": f"vlm-{tag_suffix}",
                "stage": "vlm_stage",
                "error": f"Stage 1 VLM worker exited with non-zero code rc={rcs[0]}",
            })
        return r_json
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.is_file() else ""
    return {
        "status": "error",
        "stage": "vlm_stage",
        "llm_parse_outputs": {},
        "vlm_ground_by_video": {},
        "errors": [{
            "worker": f"vlm-{tag_suffix}",
            "stage": "vlm_stage",
            "error": f"Stage 1 VLM worker exited without receipt (rc={rcs[0]})",
            "log_tail": tail,
        }],
    }


def _run_gpu_track_subprocesses(
    run_dir: Path,
    work_dir: Path,
    run_id: str,
    video_items: list[dict[str, Any]],
    model_dirs: dict[str, Path],
    model_revs: dict[str, str],
    dtype_profile: str,
    devices: list[dict[str, Any]],
    track_cache_dir: Path,
    use_video_cache: bool = True,
    trace_enabled: bool = True,
    seed: int = 0,
    tag_suffix: str = "main",
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    n_workers = min(2, max(1, len(devices)), len(video_items))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(video_items):
        shards[i % n_workers].append(item)

    procs: list[tuple[subprocess.Popen, Path, str]] = []
    receipts: list[tuple[Path, Path, str]] = []
    for w_idx, shard in enumerate(shards):
        if not shard:
            continue
        w_name = f"track-worker-{w_idx}-{tag_suffix}"
        receipt_p = work_dir / f"{w_name}_receipt.json"
        work_p = work_dir / f"{w_name}_work.json"
        log_p = work_dir / f"{w_name}.log"
        receipt_p.unlink(missing_ok=True)
        write_json_atomic(
            work_p,
            {
                "run_dir": str(run_dir),
                "run_id": run_id,
                "worker_name": w_name,
                "dtype_profile": dtype_profile,
                "seed": seed,
                "trace_enabled": trace_enabled,
                "use_video_cache": use_video_cache,
                "track_cache_dir": str(track_cache_dir),
                "models": {
                    "grounding_dino": str(model_dirs["grounding_dino"]),
                    "owlv2": str(model_dirs["owlv2"]),
                    "sam2": str(model_dirs["sam2"]),
                },
                "model_revisions": model_revs,
                "videos": shard,
                "receipt_path": str(receipt_p),
            },
        )
        gpu_uuid = devices[w_idx]["uuid"] if w_idx < len(devices) else "0"
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
        cmd = [sys.executable, "-u", "-m", "solutions.iiixr.kaggle.gpu_stages", "track", str(work_p)]
        with log_p.open("wb") as lf:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((proc, log_p, w_name))
        receipts.append((receipt_p, log_p, w_name))

    _stream_subprocesses(procs)
    merged_videos: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    for receipt_p, log_p, w_name in receipts:
        if not receipt_p.is_file():
            tail = log_p.read_text(encoding="utf-8", errors="replace")[-4000:] if log_p.is_file() else ""
            all_errors.append({"worker": w_name, "error": "Worker exited without receipt", "log_tail": tail})
            continue
        r_data = read_json(receipt_p)
        merged_videos.update(r_data.get("video_results") or {})
        all_errors.extend(r_data.get("errors") or [])
    return {"video_results": merged_videos, "errors": all_errors}


def _seal_manifest(run_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    files_map: dict[str, dict[str, Any]] = {}
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        if rel == "manifest.json":
            continue
        files_map[rel] = {
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    manifest = {
        **meta,
        "file_count": len(files_map),
        "total_bytes": sum(x["bytes"] for x in files_map.values()),
        "files": files_map,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    return manifest


def _package_tar_archives(run_dir: Path, output_dir: Path, base_archive_name: str, budget_bytes: int) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_run_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    archives_written: list[dict[str, Any]] = []

    if total_run_bytes <= int(budget_bytes * 0.92):
        tar_path = output_dir / base_archive_name
        tmp_tar = output_dir / f".{base_archive_name}.tmp"
        with tarfile.open(tmp_tar, "w") as tf:
            tf.add(run_dir, arcname="run")
        os.replace(tmp_tar, tar_path)
        archives_written.append({
            "archive": tar_path.name,
            "path": str(tar_path),
            "bytes": tar_path.stat().st_size,
            "sha256": sha256_file(tar_path),
            "sharded": False,
        })
    else:
        # Split large blobs across numbered shard TARs with a shared manifest TAR (never dropping artifacts!)
        core_tar = output_dir / base_archive_name.replace(".tar", "_core.tar")
        with tarfile.open(core_tar, "w") as tf:
            for p in sorted(run_dir.rglob("*")):
                if p.is_file() and "blobs/" not in p.relative_to(run_dir).as_posix():
                    tf.add(p, arcname=f"run/{p.relative_to(run_dir).as_posix()}")
        archives_written.append({
            "archive": core_tar.name,
            "path": str(core_tar),
            "bytes": core_tar.stat().st_size,
            "sha256": sha256_file(core_tar),
            "sharded": True,
        })
        blob_files = [p for p in sorted((run_dir / "blobs").rglob("*")) if p.is_file()] if (run_dir / "blobs").is_dir() else []
        shard_idx = 0
        cur_batch: list[Path] = []
        cur_sz = 0
        max_shard = max(256 * 1024 * 1024, int(budget_bytes * 0.45))
        for bp in blob_files:
            bsz = bp.stat().st_size
            if cur_batch and cur_sz + bsz > max_shard:
                sh_tar = output_dir / base_archive_name.replace(".tar", f"_blobs_{shard_idx:02d}.tar")
                with tarfile.open(sh_tar, "w") as tf:
                    tf.add(run_dir / "manifest.json", arcname="run/manifest.json")
                    for item_p in cur_batch:
                        tf.add(item_p, arcname=f"run/{item_p.relative_to(run_dir).as_posix()}")
                archives_written.append({
                    "archive": sh_tar.name,
                    "path": str(sh_tar),
                    "bytes": sh_tar.stat().st_size,
                    "sha256": sha256_file(sh_tar),
                    "sharded": True,
                })
                shard_idx += 1
                cur_batch, cur_sz = [], 0
            cur_batch.append(bp)
            cur_sz += bsz
        if cur_batch:
            sh_tar = output_dir / base_archive_name.replace(".tar", f"_blobs_{shard_idx:02d}.tar")
            with tarfile.open(sh_tar, "w") as tf:
                tf.add(run_dir / "manifest.json", arcname="run/manifest.json")
                for item_p in cur_batch:
                    tf.add(item_p, arcname=f"run/{item_p.relative_to(run_dir).as_posix()}")
            archives_written.append({
                "archive": sh_tar.name,
                "path": str(sh_tar),
                "bytes": sh_tar.stat().st_size,
                "sha256": sha256_file(sh_tar),
                "sharded": True,
            })
    return archives_written


def execute_workflow(args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.perf_counter()
    mode = args.mode
    scope = args.scope or ("sanity4" if mode == "sanity" else "target18")
    dtype_profile = args.dtype_profile
    run_id = args.run_id or f"iiixr-{mode}-{int(time.time())}"
    run_dir = args.run_dir or (args.output / "outputs" / "iiixr" / run_id / "run")
    args.output.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        return _execute_workflow_inner(
            args=args,
            t_start=t_start,
            mode=mode,
            scope=scope,
            dtype_profile=dtype_profile,
            run_id=run_id,
            run_dir=run_dir,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        (run_dir / "workflow_exception.txt").write_text(tb, encoding="utf-8")
        print(f"[WORKFLOW FATAL ERROR] mode={mode} run_dir={run_dir}\n{tb}", file=sys.stderr, flush=True)
        archives: list[dict[str, Any]] = []
        if not args.no_tar and run_dir.is_dir():
            try:
                _seal_manifest(
                    run_dir,
                    {
                        "run_id": run_id,
                        "mode": mode,
                        "scope": scope,
                        "status": "ERROR",
                        "created_at": utc_now_iso(),
                    },
                )
                err_prefix = "iiixr_2d_static" if scope in STATIC_2D_SCOPES else "iiixr_box"
                archives = _package_tar_archives(
                    run_dir=run_dir,
                    output_dir=args.output,
                    base_archive_name=f"{err_prefix}_{mode}.tar",
                    budget_bytes=int(args.output_budget_gib * GIB),
                )
            except Exception:
                pass
        err_summary = {
            "run_id": run_id,
            "mode": mode,
            "scope": scope,
            "status": "ERROR",
            "run_dir": str(run_dir),
            "viewer_html": str(run_dir / "index.html"),
            "per_question_csv": str(run_dir / "per_question.csv"),
            "source_commit": _git_head_sha(REPO_ROOT),
            "upstream_commit": None,
            "dtype_profile": dtype_profile,
            "wall_seconds": round(time.perf_counter() - t_start, 3),
            "archives": archives,
            "sanity_parity_report": None,
            "evaluation": {},
            "verification": {"verified": False},
            "stage_errors": [
                {
                    "stage": "workflow",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                    "run_dir": str(run_dir),
                }
            ],
        }
        write_json_atomic(args.output / "iiixr_summary.json", err_summary)
        if scope in STATIC_2D_SCOPES:
            write_json_atomic(args.output / "iiixr_2d_static_summary.json", err_summary)
        write_json_atomic(run_dir / "summary.json", err_summary)
        return err_summary


def _execute_workflow_inner(
    *,
    args: argparse.Namespace,
    t_start: float,
    mode: str,
    scope: str,
    dtype_profile: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    upstream_info = ensure_upstream_tree(UPSTREAM_DIR, auto_clone=args.auto_clone_upstream)
    owl_info = extract_owl_threshold(UPSTREAM_DIR)
    iiixr_fps_map = extract_iiixr_file_fps_val(UPSTREAM_DIR)
    snaps = read_snapshots()
    source_commit = _git_head_sha(REPO_ROOT)

    need_gpu = mode in ("sanity", "full")
    devices = _query_gpu_devices(require_two_t4=(need_gpu and not args.allow_any_gpu))

    work_dir = args.cache / "work" / run_id
    track_cache_dir = args.cache / "track_cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    store = TraceStore(run_dir, run_id, enabled=True)
    _write_environment_and_config_diff(run_dir, mode, dtype_profile, devices)

    # Determine video list first
    prelim_videos_map = {
        "sanity4": ["simulation_0163", "simulation_0089", "internet_0019"],
        "target18": list(TARGET_6_VIDEOS),
        "survey158": list(TARGET_6_VIDEOS),
        "static2d_sanity": ["internet_0001", "internet_0027", "simulation_0163"],
        "static2d_41": list(STATIC_2D_11_VIDEOS),
        "static2d_survey158": list(STATIC_2D_11_VIDEOS),
        "all2d_69": list(ALL_2D_12_VIDEOS),
    }
    prelim_videos = prelim_videos_map.get(scope, list(TARGET_6_VIDEOS))
    assets = _ensure_hf_assets(
        cache_dir=args.cache,
        dataset_dir_arg=args.dataset_dir,
        models_dir_arg=args.models_dir,
        caw_tar_arg=args.caw_tar,
        need_models=need_gpu,
        video_ids=sorted(prelim_videos),
    )

    # Load validation inputs WITH GROUND TRUTH STRIPPED OUT
    inputs = load_validation_inference_inputs(assets["csv_path"], scope=scope)
    selected_rows = inputs["selected_rows"]
    selected_videos = inputs["selected_videos"]

    # Partition by video if requested in full mode
    if args.partition_count > 1:
        if not (0 <= args.partition_index < args.partition_count):
            raise ValueError("partition_index must be in [0, partition_count - 1]")
        selected_videos = selected_videos[args.partition_index :: args.partition_count]
        keep_vset = set(selected_videos)
        selected_rows = [r for r in selected_rows if (not r["in_scope_2d_motion"]) or (r["video_id"] in keep_vset)]

    # Ensure all selected videos are resolved & probed against IIIXR file_fps_val.json
    if set(selected_videos) - set(assets["video_paths"].keys()):
        assets = _ensure_hf_assets(
            cache_dir=args.cache,
            dataset_dir_arg=args.dataset_dir,
            models_dir_arg=args.models_dir,
            caw_tar_arg=args.caw_tar,
            need_models=need_gpu,
            video_ids=selected_videos,
        )

    video_audits: dict[str, dict[str, Any]] = {}
    csv_fps_by_vid = {
        v: float(inputs["video_context_rows"][v][0]["fps"])
        for v in selected_videos
        if inputs["video_context_rows"].get(v)
    }
    for vid in selected_videos:
        va = probe_and_compare_video(
            video_id=vid,
            video_path=assets["video_paths"][vid],
            csv_fps=csv_fps_by_vid.get(vid, 24.0),
            iiixr_fps_entry=iiixr_fps_map.get(vid),
            lock_entry=snaps["dataset"]["target_videos"].get(vid),
        )
        video_audits[vid] = va
        va_ref = store.put_json(va, kind="video_audit")
        store.log_event(
            stage="source_video_verify",
            call_kind="tool_call",
            status="ok",
            video_id=vid,
            consumers=[int(r["id"]) for r in inputs["video_context_rows"].get(vid, [])],
            backend="opencv_sha256",
            device="cpu",
            output_refs={"video_audit": va_ref},
        )

    _copy_videos_into_run(run_dir, {v: assets["video_paths"][v] for v in selected_videos})
    write_json_atomic(run_dir / "configs" / "csv_and_video_audit.json", {"csv": inputs["csv_audit"], "videos": video_audits, "owl_threshold": owl_info})

    # Parse all context rows on selected videos
    all_context_rows = [r for v in selected_videos for r in inputs["video_context_rows"][v]]
    llm_parse_map: dict[int, dict[str, Any]] = {}
    vlm_ground_by_video: dict[str, list[dict[str, Any]]] = {}
    video_track_bundles: dict[str, Any] = {}
    sanity_parity_report: dict[str, Any] | None = None
    stage_errors: list[dict[str, Any]] = []

    if mode == "replay":
        # Pure CPU replay mode: parse questions + run scene_ruler on internet_0001 for formula inspection,
        # but NEVER claim fresh GPU inference ran.
        parsed_bundle = parse_questions_v1_v2(all_context_rows, llm_outputs_by_id={})
        for vid in selected_videos:
            v_recs = [parsed_bundle["v2_records"][r["id"]] for r in inputs["video_context_rows"][vid]]
            sr = run_scene_ruler_on_video(vid, assets["video_paths"][vid], [r["prior_text"] for r in v_recs])
            if sr is not None:
                sr.pop("overlay_rgb", None)
                video_track_bundles[vid] = {"video_id": vid, "scene_ruler": sr}
        predictions_out = []
        for row in selected_rows:
            qid = int(row["id"])
            ptrace = parsed_bundle["traces_by_id"].get(qid)
            if ptrace:
                p_ref = store.put_json(ptrace, kind="parse_trace")
                store.log_event(
                    stage="parse_v2",
                    call_kind="rule_eval",
                    status="ok",
                    video_id=row["video_id"],
                    sample_id=qid,
                    output_refs={"parse_trace": p_ref},
                )
            predictions_out.append({
                "id": qid,
                "video_id": row["video_id"],
                "cat": row["cat"],
                "kind": ptrace["kind"] if ptrace else "",
                "unit": ptrace["unit"] if ptrace else "",
                "in_scope_2d_motion": row["in_scope_2d_motion"],
                "support_reason": row["support_reason"],
                "status": "replay_cache_only" if row["in_scope_2d_motion"] else "unsupported",
                "error_stage": "gpu_fresh_not_executed_in_replay_mode" if row["in_scope_2d_motion"] else "out_of_scope",
                "why": "Replay mode validates evaluator & cached_box vs CAW without running GPU models",
                "fresh_box": None,
                "fresh_box_v2": None,
                "fresh_box_v3nt": None,
                "track_used": "",
                "merge_src": "",
                "mpp": None,
                "px_quantity": None,
                "question": row["question"],
                "prior_text": row["ground_truth_prior"],
            })

    else:
        # =====================================================================
        # TRUE FRESH INFERENCE MODE (Sanity 4Q or Full 18Q on Kaggle 2xT4)
        # =====================================================================
        with FreshAntiCacheGuard(enabled=True):
            # Initial rule parse so Stage 1 VLM knows query phrases for vlm_ground --notext
            prelim_parsed = parse_questions_v1_v2(all_context_rows, llm_outputs_by_id={})
            abort_gpu_pipeline = False

            # Sanity Check: Trace OFF vs Trace ON parity (#2272 on internet_0027 for 2D static, #399 on simulation_0163 for 2D motion)
            parity_vid, parity_qid = (None, None)
            if mode == "sanity" and not args.skip_parity_check:
                if scope == "static2d_sanity" and "internet_0027" in selected_videos:
                    parity_vid, parity_qid = ("internet_0027", 2272)
                elif "simulation_0163" in selected_videos:
                    parity_vid, parity_qid = ("simulation_0163", 399)

            if parity_vid is not None and parity_qid is not None:
                v_par_v1 = [prelim_parsed["v1_records"][r["id"]] for r in inputs["video_context_rows"][parity_vid]]
                v_par_v2 = [prelim_parsed["v2_records"][r["id"]] for r in inputs["video_context_rows"][parity_vid]]
                parity_vals = []
                parity_errors: list[dict[str, Any]] = []
                for tr_flag, sfx in ((False, "trace_off"), (True, "trace_on")):
                    sub_dir = run_dir / "preflight_parity" / sfx
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    p_res = _run_gpu_track_subprocesses(
                        run_dir=sub_dir,
                        work_dir=work_dir / "parity",
                        run_id=f"{run_id}-{sfx}",
                        video_items=[{
                            "video_id": parity_vid,
                            "video_path": str(assets["video_paths"][parity_vid]),
                            "fps": csv_fps_by_vid[parity_vid],
                            "parse_v1_records": v_par_v1,
                            "parse_v2_records": v_par_v2,
                            "vlm_ground_records": [],
                            "attempt": 1,
                        }],
                        model_dirs=assets["model_dirs"],
                        model_revs=assets["model_revs"],
                        dtype_profile=dtype_profile,
                        devices=devices[:1] if devices else [],
                        track_cache_dir=work_dir / f"parity_cache_{sfx}",
                        use_video_cache=False,
                        trace_enabled=tr_flag,
                        seed=0,
                        tag_suffix=sfx,
                    )
                    if p_res["errors"] or parity_vid not in p_res["video_results"]:
                        errs_here = [
                            {"phase": f"preflight_parity_{sfx}", **e}
                            for e in (p_res["errors"] or [{"stage": "preflight_parity", "video_id": parity_vid, "error": f"Missing {parity_vid} result"}])
                        ]
                        parity_errors.extend(errs_here)
                        stage_errors.extend(errs_here)
                        abort_gpu_pipeline = True
                        break
                    vb = p_res["video_results"][parity_vid]
                    active_tj = vb["track_v2"] if vb.get("track_v2") is not None else vb["track_v1"]
                    m_out = measure_video_instrumented(
                        video_recs=v_par_v2,
                        vj=active_tj,
                        t_owl=owl_info["T_owl"],
                        scene_ruler_entry=vb.get("scene_ruler"),
                        target_question_ids={parity_qid},
                    )
                    parity_vals.append(float(m_out["results_by_id"][parity_qid]["geo_value"]))

                if parity_errors:
                    sanity_parity_report = {
                        "sample_id": parity_qid,
                        "video_id": parity_vid,
                        "status": "error",
                        "trace_off_value": parity_vals[0] if len(parity_vals) >= 1 else None,
                        "trace_on_value": parity_vals[1] if len(parity_vals) >= 2 else None,
                        "identical_trace_off_on": False,
                        "errors": parity_errors,
                    }
                else:
                    parity_ok = len(parity_vals) == 2 and math.isclose(parity_vals[0], parity_vals[1], rel_tol=0.0, abs_tol=1e-12)
                    sanity_parity_report = {
                        "sample_id": parity_qid,
                        "video_id": parity_vid,
                        "status": "ok" if parity_ok else "mismatch",
                        "trace_off_value": parity_vals[0],
                        "trace_on_value": parity_vals[1],
                        "identical_trace_off_on": parity_ok,
                    }
                write_json_atomic(run_dir / "configs" / "trace_parity_check.json", sanity_parity_report)

            # Stage 1: Qwen3-VL-8B-Instruct sharded across both T4s (optional LLM parse + vlm_ground_notext)
            if not abort_gpu_pipeline:
                vg_videos_payload = [
                    {
                        "video_id": vid,
                        "video_path": str(assets["video_paths"][vid]),
                        "parse_v2_records": [prelim_parsed["v2_records"][r["id"]] for r in inputs["video_context_rows"][vid]],
                    }
                    for vid in selected_videos
                ]
                vlm_receipt = _run_gpu_vlm_subprocess(
                    run_dir=run_dir,
                    work_dir=work_dir,
                    run_id=run_id,
                    qwen_dir=assets["model_dirs"]["qwen3_vl_8b"],
                    qwen_rev=assets["model_revs"]["qwen3_vl_8b"],
                    dtype_profile=dtype_profile,
                    parse_rows=all_context_rows,
                    vlm_ground_videos=vg_videos_payload,
                    devices=devices,
                    run_llm_parse=not args.rule_parse_only,
                    trace_enabled=True,
                    seed=0,
                    tag_suffix="stage1",
                )
                if vlm_receipt.get("status") != "ok" or vlm_receipt.get("errors"):
                    stage_errors.extend(vlm_receipt.get("errors") or [{"stage": "vlm_stage", "error": "Stage 1 VLM failed"}])
                    abort_gpu_pipeline = True
                else:
                    for k_id, v_llm in (vlm_receipt.get("llm_parse_outputs") or {}).items():
                        llm_parse_map[int(k_id)] = v_llm
                    vlm_ground_by_video = vlm_receipt.get("vlm_ground_by_video") or {}

            # Re-run deterministic v1/v2 parser incorporating LLM fallback outputs (if any)
            parsed_bundle = parse_questions_v1_v2(all_context_rows, llm_outputs_by_id=llm_parse_map)
            for row in selected_rows:
                qid = int(row["id"])
                ptrace = parsed_bundle["traces_by_id"].get(qid)
                if ptrace:
                    p_ref = store.put_json(ptrace, kind="parse_trace")
                    store.log_event(
                        stage="parse_v2",
                        call_kind="rule_eval",
                        status="ok",
                        video_id=row["video_id"],
                        sample_id=qid,
                        output_refs={"parse_trace": p_ref},
                    )

            # Stage 2: 2 parallel single-GPU workers sharded across selected_videos
            if not abort_gpu_pipeline:
                video_items_payload = [
                    {
                        "video_id": vid,
                        "video_path": str(assets["video_paths"][vid]),
                        "fps": csv_fps_by_vid[vid],
                        "parse_v1_records": [parsed_bundle["v1_records"][r["id"]] for r in inputs["video_context_rows"][vid]],
                        "parse_v2_records": [parsed_bundle["v2_records"][r["id"]] for r in inputs["video_context_rows"][vid]],
                        "vlm_ground_records": vlm_ground_by_video.get(vid, []),
                        "attempt": 1,
                    }
                    for vid in selected_videos
                ]
                track_stage_out = _run_gpu_track_subprocesses(
                    run_dir=run_dir,
                    work_dir=work_dir,
                    run_id=run_id,
                    video_items=video_items_payload,
                    model_dirs=assets["model_dirs"],
                    model_revs=assets["model_revs"],
                    dtype_profile=dtype_profile,
                    devices=devices,
                    track_cache_dir=track_cache_dir,
                    use_video_cache=not args.no_cache,
                    trace_enabled=True,
                    seed=0,
                    tag_suffix="stage2",
                )
                video_track_bundles = track_stage_out["video_results"]
                stage_errors.extend(track_stage_out["errors"])

            # Stage 3: CPU Geometric & Kinematic Measurement (both v2 and v3nt branches + xground fusion)
            meas_by_vid: dict[str, dict[str, Any]] = {}
            target_set = {int(r["id"]) for r in selected_rows if r["in_scope_2d_motion"]}
            for vid, vbundle in video_track_bundles.items():
                v2_recs_vid = [parsed_bundle["v2_records"][r["id"]] for r in inputs["video_context_rows"][vid]]
                sr_vid = vbundle.get("scene_ruler")
                tj_primary = vbundle["track_v2"] if vbundle.get("track_v2") is not None else vbundle["track_v1"]
                m_v2 = measure_video_instrumented(
                    video_recs=v2_recs_vid,
                    vj=tj_primary,
                    t_owl=owl_info["T_owl"],
                    scene_ruler_entry=sr_vid,
                    target_question_ids=target_set,
                )
                m_v3nt = None
                if vbundle.get("track_v3nt") is not None:
                    m_v3nt = measure_video_instrumented(
                        video_recs=v2_recs_vid,
                        vj=vbundle["track_v3nt"],
                        t_owl=owl_info["T_owl"],
                        scene_ruler_entry=sr_vid,
                        target_question_ids=target_set,
                    )
                meas_by_vid[vid] = {"v2": m_v2, "v3nt": m_v3nt, "tj_primary": tj_primary}

            predictions_out = []
            for row in selected_rows:
                qid = int(row["id"])
                vid = row["video_id"]
                if not row["in_scope_2d_motion"]:
                    predictions_out.append({
                        "id": qid,
                        "video_id": vid,
                        "cat": row["cat"],
                        "kind": "",
                        "unit": "",
                        "in_scope_2d_motion": False,
                        "support_reason": row["support_reason"],
                        "status": "unsupported",
                        "error_stage": "out_of_scope",
                        "why": row["support_reason"],
                        "fresh_box": None,
                        "fresh_box_v2": None,
                        "fresh_box_v3nt": None,
                        "track_used": "",
                        "merge_src": "",
                        "mpp": None,
                        "px_quantity": None,
                        "question": row["question"],
                        "prior_text": row["ground_truth_prior"],
                    })
                    continue

                if vid not in meas_by_vid:
                    vid_err = next(
                        (e for e in stage_errors if e.get("video_id") in (vid, None)),
                        stage_errors[0] if stage_errors else {},
                    )
                    predictions_out.append({
                        "id": qid,
                        "video_id": vid,
                        "cat": row["cat"],
                        "kind": parsed_bundle["v2_records"][qid]["q"]["kind"],
                        "unit": parsed_bundle["v2_records"][qid]["q"]["unit"],
                        "in_scope_2d_motion": True,
                        "support_reason": row["support_reason"],
                        "status": "error",
                        "error_stage": str(vid_err.get("stage") or vid_err.get("phase") or "track_worker"),
                        "why": str(vid_err.get("error") or "Video track stage failed; see stderr.txt in videos_trace/"),
                        "fresh_box": None,
                        "fresh_box_v2": None,
                        "fresh_box_v3nt": None,
                        "track_used": "",
                        "merge_src": "",
                        "mpp": None,
                        "px_quantity": None,
                        "question": row["question"],
                        "prior_text": row["ground_truth_prior"],
                    })
                    continue

                t_m0 = time.perf_counter()
                m_vid = meas_by_vid[vid]
                res_v2 = m_vid["v2"]["results_by_id"][qid]
                tr_v2 = m_vid["v2"]["traces_by_id"][qid]
                arr_v2 = m_vid["v2"]["arrays_by_id"][qid]

                res_v3 = m_vid["v3nt"]["results_by_id"][qid] if m_vid["v3nt"] else None
                tr_v3 = m_vid["v3nt"]["traces_by_id"][qid] if m_vid["v3nt"] else None
                arr_v3 = m_vid["v3nt"]["arrays_by_id"][qid] if m_vid["v3nt"] else None

                fusion = evaluate_xground_and_fuse(
                    parse_rec=parsed_bundle["v2_records"][qid],
                    res_v2=res_v2,
                    res_v3nt=res_v3,
                    vj_v2=m_vid["tj_primary"],
                    vlm_ground_records=vlm_ground_by_video.get(vid, []),
                )

                # Save lossless .npy arrays & trace_bundle.json per sample
                sample_dir = run_dir / "samples" / f"q_{qid}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                branches_saved: dict[str, Any] = {}
                for b_lbl, b_res, b_tr, b_arr in (("v2", res_v2, tr_v2, arr_v2), ("v3nt", res_v3, tr_v3, arr_v3)):
                    if b_res is None or b_tr is None or b_arr is None:
                        continue
                    arr_refs = {
                        ak: store.put_npy(av, kind="track_array", name=f"q{qid}_{b_lbl}_{ak}")
                        for ak, av in b_arr.items()
                    }
                    tr_ref = store.put_json(b_tr, kind="kinematic_trace")
                    branches_saved[b_lbl] = {
                        "result": b_res,
                        "measure_trace": b_tr,
                        "replay_spec": b_tr["replay_spec"],
                        "array_refs": arr_refs,
                        "trace_json_ref": tr_ref,
                    }

                sample_bundle = {
                    "id": qid,
                    "video_id": vid,
                    "selected_branch": fusion["selected_branch"],
                    "fusion_decision": fusion,
                    "branches": branches_saved,
                }
                write_json_atomic(sample_dir / "trace_bundle.json", sample_bundle)
                sb_ref = store.put_json(
                    {"id": qid, "selected_branch": fusion["selected_branch"], "final_value": fusion["final_value"]},
                    kind="sample_decision",
                )
                store.log_event(
                    stage="measure_v3_and_fuse",
                    call_kind="tool_call",
                    status="ok" if math.isfinite(fusion["final_value"]) else "fail",
                    video_id=vid,
                    sample_id=qid,
                    backend="numpy_polyfit",
                    device="cpu",
                    duration_sec=time.perf_counter() - t_m0,
                    output_refs={"decision": sb_ref},
                )

                chosen_tr = branches_saved.get(fusion["selected_branch"], branches_saved.get("v2", {}))
                chosen_ft = ((chosen_tr.get("measure_trace") or {}).get("formula_trace")) or {}
                final_v = fusion["final_value"]
                is_ok = math.isfinite(final_v) and final_v > 0
                predictions_out.append({
                    "id": qid,
                    "video_id": vid,
                    "cat": row["cat"],
                    "kind": parsed_bundle["v2_records"][qid]["q"]["kind"],
                    "unit": parsed_bundle["v2_records"][qid]["q"]["unit"],
                    "in_scope_2d_motion": True,
                    "support_reason": row["support_reason"],
                    "status": "ok" if is_ok else "fail",
                    "error_stage": "" if is_ok else "measure_v3",
                    "why": "" if is_ok else str(res_v2.get("arbitrate_why") or res_v2.get("why") or "no_valid_measurement"),
                    "fresh_box": final_v if is_ok else None,
                    "fresh_box_v2": fusion["v2_value"] if math.isfinite(fusion["v2_value"]) else None,
                    "fresh_box_v3nt": fusion["v3nt_value"] if math.isfinite(fusion["v3nt_value"]) else None,
                    "track_used": fusion["selected_branch"],
                    "merge_src": fusion["merge_src"],
                    "mpp": chosen_ft.get("mpp"),
                    "px_quantity": chosen_ft.get("px_quantity"),
                    "question": row["question"],
                    "prior_text": row["ground_truth_prior"],
                })

    # Seal predictions BEFORE launching isolated post-inference evaluator subprocess!
    locked_preds_path = run_dir / "predictions_locked.json"
    write_json_atomic(
        locked_preds_path,
        {
            "sealed": True,
            "sealed_at": utc_now_iso(),
            "run_id": run_id,
            "mode": mode,
            "scope": scope,
            "source_commit": source_commit,
            "upstream_commit": upstream_info["pinned_commit"],
            "predictions": predictions_out,
        },
    )

    # Run isolated post-inference evaluation in a separate Python process
    eval_cmd = [
        sys.executable,
        "-m",
        "solutions.iiixr.kaggle.evaluate",
        "--locked-predictions",
        str(locked_preds_path),
        "--validation-csv",
        str(assets["csv_path"]),
        "--output-dir",
        str(run_dir),
    ]
    if args.caw_tar and args.caw_tar.is_file():
        eval_cmd += ["--caw-tar", str(args.caw_tar)]
    subprocess.run(eval_cmd, cwd=str(REPO_ROOT), check=True)
    eval_summary = read_json(run_dir / "evaluation.json")

    # Build Failure-Analysis Offline HTML Viewer (run/index.html)
    with (run_dir / "per_question.csv").open("r", encoding="utf-8", newline="") as fh:
        per_q_rows = list(csv.DictReader(fh))
    all_events = []
    if (run_dir / "events.jsonl").is_file():
        all_events = [json.loads(ln) for ln in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    in_scope_ids_tuple = tuple(
        sorted(int(r["id"]) for r in per_q_rows if str(r.get("in_scope_2d_motion", "")).lower() == "true")
    )
    if scope in STATIC_2D_SCOPES:
        cache_map = load_cached_val_geo_raw(UPSTREAM_DIR, target_ids=in_scope_ids_tuple or STATIC_2D_41_IDS)
        if scope == "all2d_69":
            cache_map.update(load_cached_mot_val(UPSTREAM_DIR, TARGET_18_IDS))
    else:
        cache_map = load_cached_mot_val(UPSTREAM_DIR, TARGET_18_IDS)
    caw_map = (
        read_caw_predictions_from_tar(args.caw_tar, in_scope_ids_tuple or TARGET_18_IDS)
        if (args.caw_tar and args.caw_tar.is_file())
        else {}
    )

    viewer_questions = []
    for erow in per_q_rows:
        if str(erow.get("in_scope_2d_motion", "")).lower() != "true":
            continue
        qid = int(erow["id"])
        vid = erow["video_id"]
        sb_path = run_dir / "samples" / f"q_{qid}" / "trace_bundle.json"
        s_bundle = read_json(sb_path) if sb_path.is_file() else None
        evts_q = [
            e for e in all_events
            if e.get("sample_id") == qid or qid in (e.get("consumers") or [])
        ]
        # Convert numeric strings in erow for clean JS display
        erow_clean: dict[str, Any] = dict(erow)
        for num_k in (
            "cached_box", "fresh_box", "caw", "cached_point", "fresh_box_v2", "fresh_box_v3nt",
            "ground_truth", "abs_err_vs_cache", "rel_err_vs_cache", "abs_err_vs_gt", "rel_err_vs_gt",
            "abs_cache_err_vs_gt", "rel_cache_err_vs_gt", "mra_cached_box", "mra_fresh_box", "mra_caw", "mra_cached_point",
        ):
            if erow_clean.get(num_k, "") not in ("", None):
                try:
                    erow_clean[num_k] = float(erow_clean[num_k])
                except ValueError:
                    pass
        viewer_questions.append(
            build_viewer_payload_for_question(
                eval_row=erow_clean,
                parse_trace=parsed_bundle["traces_by_id"].get(qid),
                sample_bundle=s_bundle,
                video_track_bundle=video_track_bundles.get(vid),
                video_audit=video_audits.get(vid),
                cache_rec=cache_map.get(qid),
                caw_rec=caw_map.get(qid),
                events_for_question=evts_q,
            )
        )

    run_status = "COMPLETE"
    if stage_errors:
        run_status = "ERROR"
    elif mode == "sanity":
        parity_pass = True if sanity_parity_report is None else bool(sanity_parity_report["identical_trace_off_on"])
        if scope == "static2d_sanity":
            no_error_rows = all(p["status"] != "error" for p in predictions_out if p["in_scope_2d_motion"])
            must_ok_ids = {2151, 2152, 2272, 2275} & {int(p["id"]) for p in predictions_out if p["in_scope_2d_motion"]}
            expected_ok = bool(must_ok_ids) and all(
                p["status"] == "ok" for p in predictions_out if int(p["id"]) in must_ok_ids
            )
            run_status = "PASS" if (no_error_rows and expected_ok and parity_pass) else "FAIL"
        else:
            all_ok = all(p["status"] == "ok" for p in predictions_out if p["in_scope_2d_motion"])
            run_status = "PASS" if (all_ok and parity_pass) else "FAIL"

    run_meta = {
        "run_id": run_id,
        "mode": mode,
        "scope": scope,
        "status": run_status,
        "dtype_profile": dtype_profile,
        "source_commit": source_commit,
        "upstream_commit": upstream_info["pinned_commit"],
        "upstream_repo": upstream_info["repo_url"],
        "upstream_license_status": upstream_info["license_status"],
        "model_revisions": assets["model_revs"],
        "sanity_parity_report": sanity_parity_report,
        "created_at": utc_now_iso(),
        "wall_seconds": round(time.perf_counter() - t_start, 3),
    }

    render_failure_analysis_html(
        output_html_path=run_dir / "index.html",
        run_meta=run_meta,
        eval_summary=eval_summary,
        questions_payload=viewer_questions,
    )

    _snapshot_minimal_source(run_dir)
    manifest = _seal_manifest(run_dir, run_meta)

    # Run standalone CPU-only verify.py in an isolated subprocess (CUDA disabled)
    verify_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    verify_out_raw = subprocess.check_output(
        [sys.executable, str(run_dir / "verify.py"), str(run_dir)],
        cwd=str(run_dir),
        env=verify_env,
        text=True,
    )
    verification = json.loads(verify_out_raw)

    archive_prefix = "iiixr_2d_static" if scope in STATIC_2D_SCOPES else "iiixr_box"
    archive_name = f"{archive_prefix}_{mode}.tar"
    archives = []
    if not args.no_tar:
        archives = _package_tar_archives(
            run_dir=run_dir,
            output_dir=args.output,
            base_archive_name=archive_name,
            budget_bytes=int(args.output_budget_gib * GIB),
        )

    summary = {
        "run_id": run_id,
        "mode": mode,
        "scope": scope,
        "status": run_status,
        "run_dir": str(run_dir),
        "viewer_html": str(run_dir / "index.html"),
        "per_question_csv": str(run_dir / "per_question.csv"),
        "source_commit": source_commit,
        "upstream_commit": upstream_info["pinned_commit"],
        "dtype_profile": dtype_profile,
        "wall_seconds": run_meta["wall_seconds"],
        "manifest_total_bytes": manifest["total_bytes"],
        "archives": archives,
        "sanity_parity_report": sanity_parity_report,
        "evaluation": eval_summary,
        "verification": verification,
        "stage_errors": stage_errors,
    }
    write_json_atomic(args.output / "iiixr_summary.json", summary)
    if scope in STATIC_2D_SCOPES:
        write_json_atomic(args.output / "iiixr_2d_static_summary.json", summary)
    write_json_atomic(run_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IIIXR 2D reproduction, trace, and Kaggle 2xT4 runner")
    is_kaggle_linux = os.name != "nt" and Path("/kaggle/working").is_dir()
    ap.add_argument("mode", choices=["replay", "sanity", "full"])
    ap.add_argument("--scope", choices=list(VALID_SCOPES), default=None)
    ap.add_argument("--cache", type=Path, default=Path("/kaggle/temp/iiixr") if is_kaggle_linux else REPO_ROOT / "outputs" / "iiixr_cache")
    ap.add_argument("--output", type=Path, default=Path("/kaggle/working") if is_kaggle_linux else REPO_ROOT / "outputs" / "iiixr")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--caw-tar", type=Path, default=REPO_ROOT / "quantiphy_full.tar" if (REPO_ROOT / "quantiphy_full.tar").is_file() else None)
    ap.add_argument("--dtype-profile", choices=["t4_fp16", "upstream_bf16", "fp32"], default="t4_fp16")
    ap.add_argument("--rule-parse-only", action="store_true", help="Skip Qwen3-VL text parse fallback (all 6 target videos use rule parse)")
    ap.add_argument("--skip-parity-check", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-tar", action="store_true")
    ap.add_argument("--allow-any-gpu", action="store_true")
    ap.add_argument("--auto-clone-upstream", action="store_true")
    ap.add_argument("--partition-count", type=int, default=1)
    ap.add_argument("--partition-index", type=int, default=0)
    ap.add_argument("--output-budget-gib", type=float, default=18.0)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    summary = execute_workflow(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] in ("ERROR", "FAIL"):
        sys.exit(1)


if __name__ == "__main__":
    main()
