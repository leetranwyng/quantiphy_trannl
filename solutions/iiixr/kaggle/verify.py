#!/usr/bin/env python3
"""Standalone CPU-only offline verifier and numerical trace replay for IIIXR g_box runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

MRA_THRESHOLDS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
REQUIRED_EVENT_KEYS = {
    "run_id",
    "event_id",
    "parent_id",
    "video_id",
    "sample_id",
    "consumers",
    "attempt",
    "stage",
    "call_kind",
    "status",
    "backend",
    "device",
    "timestamp",
    "duration_sec",
    "input_refs",
    "output_refs",
    "error",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_mra(pred_str: str, gt_str: str) -> float:
    if pred_str in ("", "nan", "None") or gt_str in ("", "nan", "None"):
        return 0.0
    p = float(pred_str)
    y = float(gt_str)
    if not (math.isfinite(p) and math.isfinite(y)) or p <= 0.0 or y == 0.0:
        return 0.0
    return float(np.mean(abs(p - y) / abs(y) < (1.0 - MRA_THRESHOLDS)))


def verify_run(run_dir: Path, require_complete: bool = False) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add bundled source or workspace root to sys.path for pure-CPU kinematic replay
    bundled_source = run_dir / "source"
    repo_root = Path(__file__).resolve().parents[3]
    for cand in (bundled_source, repo_root):
        if (cand / "solutions" / "iiixr" / "kaggle" / "kinematics.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            break

    from solutions.iiixr.kaggle.kinematics import recompute_from_saved_trace_arrays

    upstream_override = bundled_source / "solutions" / "iiixr" / "upstream"
    if not (upstream_override / "geo" / "measure_v3.py").is_file():
        upstream_override = repo_root / "solutions" / "iiixr" / "upstream"

    # 1. Verify all files listed in manifest["files"]
    verified_files = 0
    for rel_path, meta in manifest.get("files", {}).items():
        fpath = run_dir / rel_path
        if not fpath.is_file():
            raise FileNotFoundError(f"Manifest file missing: {rel_path}")
        actual_bytes = fpath.stat().st_size
        if actual_bytes != int(meta["bytes"]):
            raise ValueError(f"Size mismatch for {rel_path}: {actual_bytes} != {meta['bytes']}")
        actual_sha = _sha256_file(fpath)
        if actual_sha != meta["sha256"]:
            raise ValueError(f"SHA256 mismatch for {rel_path}: {actual_sha} != {meta['sha256']}")
        verified_files += 1

    # 2. Verify events.jsonl schema, refs, and call outcomes
    events_path = run_dir / "events.jsonl"
    verified_events = 0
    verified_blob_refs = 0
    if events_path.is_file():
        with events_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                evt = json.loads(line)
                missing_keys = REQUIRED_EVENT_KEYS - set(evt.keys())
                if missing_keys:
                    raise ValueError(f"Event line {line_no} missing required keys: {sorted(missing_keys)}")
                if evt["status"] == "error" and not evt.get("error"):
                    raise ValueError(f"Event line {line_no} has status='error' without error message")
                if evt["call_kind"] in ("model_call", "tool_call") and evt["status"] == "ok":
                    if not evt.get("output_refs") and not evt.get("extra"):
                        raise ValueError(f"Event line {line_no} ({evt['stage']}) succeeded without output_refs or extra")
                for ref_group in (evt.get("input_refs") or {}, evt.get("output_refs") or {}):
                    for ref_name, ref_obj in ref_group.items():
                        if isinstance(ref_obj, dict) and ref_obj.get("path"):
                            bpath = run_dir / ref_obj["path"]
                            if not bpath.is_file():
                                raise FileNotFoundError(f"Event {evt['event_id']} missing blob {ref_name}: {ref_obj['path']}")
                            if _sha256_file(bpath) != ref_obj["sha256"]:
                                raise ValueError(f"Blob SHA256 mismatch for {ref_obj['path']}")
                            verified_blob_refs += 1
                verified_events += 1

    # 3. CPU-only recomputation of fresh_box from saved numerical trace artifacts (.npy + replay_spec)
    recomputed_samples = 0
    samples_dir = run_dir / "samples"
    if samples_dir.is_dir():
        for sample_folder in sorted(samples_dir.iterdir()):
            if not sample_folder.is_dir():
                continue
            spec_file = sample_folder / "trace_bundle.json"
            if not spec_file.is_file():
                continue
            bundle = json.loads(spec_file.read_text(encoding="utf-8"))
            for branch_name, branch_data in (bundle.get("branches") or {}).items():
                replay_spec = branch_data.get("replay_spec")
                array_refs = branch_data.get("array_refs") or {}
                recorded_res = branch_data.get("result") or {}
                if not replay_spec or not array_refs:
                    continue
                loaded_arrays: dict[str, np.ndarray] = {}
                for arr_key, ref in array_refs.items():
                    arr_path = run_dir / ref["path"]
                    if not arr_path.is_file():
                        raise FileNotFoundError(f"Missing saved array {arr_key} at {ref['path']}")
                    if _sha256_file(arr_path) != ref["sha256"]:
                        raise ValueError(f"Saved array SHA256 mismatch at {ref['path']}")
                    loaded_arrays[arr_key] = np.load(io.BytesIO(arr_path.read_bytes()), allow_pickle=False)
                cpu_res = recompute_from_saved_trace_arrays(
                    replay_spec=replay_spec,
                    arrays=loaded_arrays,
                    upstream_dir=upstream_override,
                )
                rec_val = recorded_res.get("geo_value")
                cpu_val = cpu_res.get("geo_value")
                rec_fin = rec_val is not None and math.isfinite(float(rec_val))
                cpu_fin = cpu_val is not None and math.isfinite(float(cpu_val))
                if rec_fin != cpu_fin:
                    raise ValueError(
                        f"CPU trace replay finiteness mismatch on {sample_folder.name} ({branch_name}): "
                        f"recorded={rec_val} vs cpu={cpu_val}"
                    )
                if rec_fin and not math.isclose(float(rec_val), float(cpu_val), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"CPU trace replay value mismatch on {sample_folder.name} ({branch_name}): "
                        f"recorded={rec_val} vs cpu={cpu_val}"
                    )
                recomputed_samples += 1

    # 4. Replay per_question.csv MRA verification
    per_q_path = run_dir / "per_question.csv"
    eval_path = run_dir / "evaluation.json"
    mra_verified_rows = 0
    if per_q_path.is_file() and eval_path.is_file():
        eval_json = json.loads(eval_path.read_text(encoding="utf-8"))
        with per_q_path.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        in_scope = [r for r in rows if str(r.get("in_scope_2d_motion", "")).lower() == "true"]
        for r in in_scope:
            gt = r["ground_truth"]
            exp_cache = _row_mra(r["cached_box"], gt)
            if not math.isclose(float(r["mra_cached_box"]), exp_cache, abs_tol=1e-9):
                raise ValueError(f"MRA mismatch on cached_box for ID {r['id']}")
            if r.get("mra_fresh_box", "") != "":
                exp_fresh = _row_mra(r["fresh_box"], gt)
                if not math.isclose(float(r["mra_fresh_box"]), exp_fresh, abs_tol=1e-9):
                    raise ValueError(f"MRA mismatch on fresh_box for ID {r['id']}")
            mra_verified_rows += 1
        if in_scope:
            mean_cache = float(np.mean([float(r["mra_cached_box"]) for r in in_scope]))
            if not math.isclose(mean_cache, float(eval_json["mean_per_question_mra"]["cached_box"]), abs_tol=1e-9):
                raise ValueError("Summary mean_per_question_mra.cached_box mismatch")

    if require_complete and manifest.get("status") not in ("COMPLETE", "PASS"):
        raise RuntimeError(f"Run status is {manifest.get('status')}, expected COMPLETE or PASS")

    return {
        "verified": True,
        "run_id": manifest.get("run_id"),
        "mode": manifest.get("mode"),
        "status": manifest.get("status"),
        "verified_files": verified_files,
        "verified_events": verified_events,
        "verified_blob_refs": verified_blob_refs,
        "cpu_recomputed_branch_traces": recomputed_samples,
        "mra_verified_rows": mra_verified_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify IIIXR g_box run directory offline on CPU")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    out = verify_run(args.run_dir, require_complete=args.require_complete)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
