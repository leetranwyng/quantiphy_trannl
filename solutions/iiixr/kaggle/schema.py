"""Dataset resolver, ground-truth isolation, CAW TAR reader, and lossless trace store."""
from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import os
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SOLUTION_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = SOLUTION_ROOT / "upstream"
SNAPSHOTS_PATH = Path(__file__).with_name("snapshots.json")

TARGET_18_IDS: tuple[int, ...] = (
    142, 143,
    398, 399, 400, 401,
    427, 428, 429, 430, 431, 432, 433, 434,
    1896,
    2153,
    2233, 2234,
)

SANITY_4_IDS: tuple[int, ...] = (399, 401, 142, 2234)

EXPECTED_VIDEO_BY_ID: dict[int, str] = {
    142: "simulation_0089",
    143: "simulation_0089",
    398: "simulation_0163",
    399: "simulation_0163",
    400: "simulation_0163",
    401: "simulation_0163",
    427: "simulation_0171",
    428: "simulation_0171",
    429: "simulation_0171",
    430: "simulation_0171",
    431: "simulation_0171",
    432: "simulation_0171",
    433: "simulation_0171",
    434: "simulation_0171",
    1896: "simulation_0098_segmented",
    2153: "internet_0001",
    2233: "internet_0019",
    2234: "internet_0019",
}

TARGET_6_VIDEOS: tuple[str, ...] = (
    "simulation_0089",
    "simulation_0163",
    "simulation_0171",
    "simulation_0098_segmented",
    "internet_0001",
    "internet_0019",
)

STATIC_2D_41_IDS: tuple[int, ...] = (
    139, 140, 141, 144,
    402, 403,
    498, 499, 503, 504, 505, 506, 507, 508, 509,
    513, 514, 515, 516,
    1897, 1898, 1899,
    2151, 2152,
    2161, 2162,
    2165, 2166, 2167, 2168,
    2200, 2201, 2202, 2205,
    2228, 2229, 2230, 2231, 2232,
    2272, 2275,
)

STATIC_2D_SANITY_6_IDS: tuple[int, ...] = (2151, 2152, 2272, 2275, 402, 403)

EXPECTED_STATIC_2D_VIDEO_BY_ID: dict[int, str] = {
    139: "simulation_0089",
    140: "simulation_0089",
    141: "simulation_0089",
    144: "simulation_0089",
    402: "simulation_0163",
    403: "simulation_0163",
    498: "simulation_0193",
    499: "simulation_0193",
    503: "simulation_0193",
    504: "simulation_0193",
    505: "simulation_0193",
    506: "simulation_0193",
    507: "simulation_0193",
    508: "simulation_0193",
    509: "simulation_0193",
    513: "simulation_0196",
    514: "simulation_0196",
    515: "simulation_0196",
    516: "simulation_0196",
    1897: "simulation_0098_segmented",
    1898: "simulation_0098_segmented",
    1899: "simulation_0098_segmented",
    2151: "internet_0001",
    2152: "internet_0001",
    2161: "internet_0004",
    2162: "internet_0004",
    2165: "internet_0005",
    2166: "internet_0005",
    2167: "internet_0005",
    2168: "internet_0005",
    2200: "internet_0013",
    2201: "internet_0013",
    2202: "internet_0013",
    2205: "internet_0013",
    2228: "internet_0019",
    2229: "internet_0019",
    2230: "internet_0019",
    2231: "internet_0019",
    2232: "internet_0019",
    2272: "internet_0027",
    2275: "internet_0027",
}

STATIC_2D_11_VIDEOS: tuple[str, ...] = (
    "internet_0001",
    "internet_0004",
    "internet_0005",
    "internet_0013",
    "internet_0019",
    "internet_0027",
    "simulation_0089",
    "simulation_0098_segmented",
    "simulation_0163",
    "simulation_0193",
    "simulation_0196",
)

ALL_2D_12_VIDEOS: tuple[str, ...] = tuple(sorted(set(STATIC_2D_11_VIDEOS) | set(TARGET_6_VIDEOS)))

VALID_SCOPES: tuple[str, ...] = (
    "sanity4",
    "target18",
    "survey158",
    "static2d_sanity",
    "static2d_41",
    "static2d_survey158",
    "all2d_69",
)

STATIC_2D_SCOPES: tuple[str, ...] = ("static2d_sanity", "static2d_41", "static2d_survey158", "all2d_69")

DEBUG_REFERENCE_ANCHORS: dict[int, dict[str, Any]] = {
    399: {"video_id": "simulation_0163", "cached_g_box": 0.38209679865232415},
    401: {"video_id": "simulation_0163", "cached_g_box": 0.3420306158490744},
}

FORBIDDEN_FRESH_PATTERNS: tuple[str, ...] = (
    "r10/motion/cache/mot_",
    "r10\\motion\\cache\\mot_",
    "work/h/rows.pkl",
    "work\\h\\rows.pkl",
    "val_geo_v4ntrj5sq2srngsr",
    "val_geo_v4ntrj5sq3nsrngsr",
)


def read_snapshots() -> dict[str, Any]:
    return json.loads(SNAPSHOTS_PATH.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def fingerprint(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_upstream_tree(upstream_dir: Path = UPSTREAM_DIR, auto_clone: bool = False) -> dict[str, Any]:
    """Verify pinned IIIXR upstream files exist (and optionally materialize at pinned commit)."""
    lock = read_snapshots()["upstream"]
    required = [
        upstream_dir / "geo" / "measure_v3.py",
        upstream_dir / "geo" / "track_v2.py",
        upstream_dir / "geo" / "track_v3.py",
        upstream_dir / "geo" / "scene_ruler.py",
        upstream_dir / "intermediates" / "intermediates.tar.xz",
        upstream_dir / "team" / "member1" / "trackB" / "package" / "z1s2c" / "2_c2_g3a" / "root" / "analysis" / "v3" / "ag" / "r10" / "MOTION" / "rules.py",
    ]
    missing = [str(p.relative_to(upstream_dir)) for p in required if not p.is_file()]
    if missing and auto_clone:
        upstream_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(upstream_dir)], check=True)
        subprocess.run(["git", "-C", str(upstream_dir), "remote", "add", "origin", lock["repo_url"]], check=False)
        subprocess.run(
            ["git", "-C", str(upstream_dir), "fetch", "--depth", "1", "origin", lock["pinned_commit"]],
            check=True,
        )
        subprocess.run(["git", "-C", str(upstream_dir), "checkout", "--detach", lock["pinned_commit"]], check=True)
        missing = [str(p.relative_to(upstream_dir)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required IIIXR upstream files under {upstream_dir}: {missing}. "
            f"Expected commit {lock['pinned_commit']} from {lock['repo_url']}."
        )
    inter_path = upstream_dir / lock["intermediates_archive"]["path"]
    inter_sha = sha256_file(inter_path)
    inter_bytes = inter_path.stat().st_size
    if inter_sha != lock["intermediates_archive"]["sha256"] or inter_bytes != lock["intermediates_archive"]["bytes"]:
        raise ValueError(
            f"IIIXR intermediates.tar.xz hash/size mismatch: got {inter_sha} ({inter_bytes}B), "
            f"expected {lock['intermediates_archive']['sha256']} ({lock['intermediates_archive']['bytes']}B)."
        )
    return {
        "upstream_dir": str(upstream_dir),
        "pinned_commit": lock["pinned_commit"],
        "repo_url": lock["repo_url"],
        "license_status": lock["license_status"],
        "intermediates_sha256": inter_sha,
        "intermediates_bytes": inter_bytes,
    }


def extract_owl_threshold(upstream_dir: Path = UPSTREAM_DIR) -> dict[str, Any]:
    """Read calibrated OWLv2 support threshold T_owl from IIIXR intermediates.tar.xz."""
    lock = read_snapshots()["upstream"]["owl_threshold_v2"]
    inter_path = upstream_dir / "intermediates" / "intermediates.tar.xz"
    with tarfile.open(inter_path, "r:xz") as tf:
        member = tf.extractfile(lock["member"])
        if member is None:
            raise FileNotFoundError(f"Missing {lock['member']} in {inter_path}")
        raw = member.read()
    digest = sha256_bytes(raw)
    if digest != lock["sha256"]:
        raise ValueError(f"owl_threshold_v2.json SHA256 mismatch: {digest} != {lock['sha256']}")
    parsed = json.loads(raw.decode("utf-8"))
    return {
        "T_owl": float(parsed["T_owl"]),
        "n_conf": int(parsed.get("n_conf", 0)),
        "sha256": digest,
        "bytes": len(raw),
        "source": f"{inter_path.name}:{lock['member']}",
    }


def extract_iiixr_file_fps_val(upstream_dir: Path = UPSTREAM_DIR) -> dict[str, Any]:
    """Read IIIXR's recorded validation video probe metadata (geo/out/file_fps_val.json)."""
    inter_path = upstream_dir / "intermediates" / "intermediates.tar.xz"
    with tarfile.open(inter_path, "r:xz") as tf:
        member = tf.extractfile("geo/out/file_fps_val.json")
        if member is None:
            return {}
        return json.loads(member.read().decode("utf-8"))


class FreshAntiCacheGuard:
    """Context manager blocking any accidental read of mot_val_*, rows.pkl, or cached geo CSVs during fresh mode."""

    _active_depth: int = 0

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._orig_open = builtins.open
        self.blocked_attempts: list[str] = []

    def __enter__(self) -> "FreshAntiCacheGuard":
        if not self.enabled:
            return self
        FreshAntiCacheGuard._active_depth += 1
        orig_open = self._orig_open
        guard = self

        def guarded_open(file: Any, *args: Any, **kwargs: Any):
            path_str = str(file).lower()
            for pat in FORBIDDEN_FRESH_PATTERNS:
                if pat in path_str:
                    guard.blocked_attempts.append(str(file))
                    raise PermissionError(
                        f"Fresh-mode anti-cache guard blocked access to cached answer/intermediate file: {file}"
                    )
            return orig_open(file, *args, **kwargs)

        builtins.open = guarded_open
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.enabled:
            FreshAntiCacheGuard._active_depth = max(0, FreshAntiCacheGuard._active_depth - 1)
            builtins.open = self._orig_open


def _parse_id_column(df: pd.DataFrame) -> str:
    for candidate in ("Unnamed: 0", "id", "ID"):
        if candidate in df.columns:
            return candidate
    if df.columns[0] == "":
        return ""
    raise ValueError(f"Validation CSV lacks canonical 'Unnamed: 0' / 'id' column; got {list(df.columns)}")


def classify_2d_motion_support(row: dict[str, Any]) -> tuple[bool, str]:
    """Determine whether a QuantiPhy validation row belongs to the 18-question 2D motion reference set."""
    qid = int(row["id"])
    vtype = str(row.get("video_type") or "")
    itype = str(row.get("inference_type") or "")
    if qid in EXPECTED_VIDEO_BY_ID:
        return True, "target_18_2d_motion"
    if len(vtype) >= 2 and vtype[1] == "3":
        return False, f"unsupported_3d_video_type({vtype})"
    if len(itype) >= 2 and itype[1] == "S":
        return False, f"unsupported_static_target({itype})"
    return False, "outside_18_reference_motion_subset"


def classify_scope_support(row: dict[str, Any], scope: str) -> tuple[bool, str]:
    """Determine whether a QuantiPhy validation row is in-scope for the requested experiment scope."""
    qid = int(row["id"])
    vtype = str(row.get("video_type") or "")
    itype = str(row.get("inference_type") or "")
    if scope in ("sanity4", "target18", "survey158"):
        return classify_2d_motion_support(row)
    if scope in ("static2d_sanity", "static2d_41", "static2d_survey158"):
        if qid in EXPECTED_STATIC_2D_VIDEO_BY_ID:
            return True, f"target_41_2d_static({itype})"
        if len(vtype) >= 2 and vtype[1] == "3":
            return False, f"unsupported_3d_video_type({vtype})"
        if len(itype) >= 2 and itype[1] == "D":
            return False, f"unsupported_dynamic_target_for_static2d({itype})"
        return False, "outside_41_2d_static_subset"
    if scope == "all2d_69":
        if len(vtype) >= 2 and vtype[1] == "2":
            return True, f"target_69_2d_all({itype})"
        return False, f"unsupported_3d_video_type({vtype})"
    raise ValueError(f"Unknown scope '{scope}'. Expected one of {VALID_SCOPES}.")


def load_validation_inference_inputs(
    csv_path: Path,
    scope: str = "target18",
    explicit_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Load validation questions WITHOUT ground_truth_posterior, keyed strictly by canonical CSV ID."""
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unknown scope '{scope}'. Expected one of {VALID_SCOPES}.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")
    raw_bytes = csv_path.read_bytes()
    csv_sha = sha256_bytes(raw_bytes)
    df = pd.read_csv(io.BytesIO(raw_bytes))
    id_col = _parse_id_column(df)

    all_rows_by_id: dict[int, dict[str, Any]] = {}
    id_mapping: list[dict[str, Any]] = []
    for row_idx, r in df.iterrows():
        qid = int(r[id_col])
        if qid in all_rows_by_id:
            raise ValueError(f"Duplicate question ID {qid} in {csv_path}")
        vid = str(r["video_id"]).strip()
        vtype = str(r["video_type"]).strip()
        itype = str(r["inference_type"]).strip()
        cat = f"{itype[0]}{vtype[1]}" if len(itype) >= 1 and len(vtype) >= 2 else ""
        depth_val = "" if pd.isna(r.get("depth_info")) else str(r.get("depth_info")).strip()
        prior_val = "" if pd.isna(r.get("ground_truth_prior")) else str(r.get("ground_truth_prior")).strip()
        rec = {
            "id": qid,
            "csv_row_index": int(row_idx),
            "video_id": vid,
            "video_source": str(r["video_source"]).strip(),
            "video_type": vtype,
            "inference_type": itype,
            "cat": cat,
            "fps": float(r["fps"]),
            "question": str(r["question"]).strip(),
            "ground_truth_prior": prior_val,
            "depth_info": depth_val,
        }
        in_scope, support_reason = classify_scope_support(rec, scope)
        rec["in_scope"] = in_scope
        rec["in_scope_2d_motion"] = in_scope  # Kept as alias for backward compatibility
        rec["support_reason"] = support_reason
        assert "ground_truth_posterior" not in rec
        all_rows_by_id[qid] = rec
        id_mapping.append({"id": qid, "csv_row_index": int(row_idx), "video_id": vid, "cat": cat})

    # Always verify all 18 canonical motion IDs AND all 41 canonical 2D static IDs against their expected videos
    for expected_map in (EXPECTED_VIDEO_BY_ID, EXPECTED_STATIC_2D_VIDEO_BY_ID):
        for tid, expected_vid in expected_map.items():
            if tid not in all_rows_by_id:
                raise ValueError(f"Canonical target ID {tid} missing from {csv_path}")
            actual_vid = all_rows_by_id[tid]["video_id"]
            if actual_vid != expected_vid:
                raise ValueError(
                    f"ID/video join mismatch for ID {tid}: expected video '{expected_vid}', got '{actual_vid}'"
                )

    if explicit_ids is not None:
        wanted_ids = tuple(int(x) for x in explicit_ids)
    elif scope == "sanity4":
        wanted_ids = SANITY_4_IDS
    elif scope == "target18":
        wanted_ids = TARGET_18_IDS
    elif scope in ("survey158", "static2d_survey158"):
        wanted_ids = tuple(sorted(all_rows_by_id.keys()))
    elif scope == "static2d_sanity":
        wanted_ids = STATIC_2D_SANITY_6_IDS
    elif scope == "static2d_41":
        wanted_ids = STATIC_2D_41_IDS
    elif scope == "all2d_69":
        wanted_ids = tuple(sorted(qid for qid, rec in all_rows_by_id.items() if rec["in_scope"]))
    else:
        raise ValueError(f"Unknown scope '{scope}'.")

    selected_rows = [dict(all_rows_by_id[qid]) for qid in wanted_ids]
    selected_videos = sorted({r["video_id"] for r in selected_rows if r["in_scope"]})
    video_context_rows: dict[str, list[dict[str, Any]]] = {v: [] for v in selected_videos}
    for rec in all_rows_by_id.values():
        if rec["video_id"] in video_context_rows:
            video_context_rows[rec["video_id"]].append(dict(rec))
    for v in video_context_rows:
        video_context_rows[v].sort(key=lambda x: x["id"])

    return {
        "scope": scope,
        "selected_rows": selected_rows,
        "video_context_rows": video_context_rows,
        "selected_videos": selected_videos,
        "csv_audit": {
            "path": str(csv_path),
            "sha256": csv_sha,
            "bytes": len(raw_bytes),
            "total_rows": len(all_rows_by_id),
            "has_hf_extra_row_2607": 2607 in all_rows_by_id,
            "target_18_verified": True,
            "target_6_videos": list(TARGET_6_VIDEOS),
            "id_mapping": id_mapping,
        },
    }


def load_validation_ground_truth(csv_path: Path) -> dict[int, float]:
    """Read ground_truth_posterior ONLY for post-inference evaluation."""
    df = pd.read_csv(csv_path)
    id_col = _parse_id_column(df)
    if "ground_truth_posterior" not in df.columns:
        raise ValueError(f"CSV {csv_path} does not contain ground_truth_posterior")
    gt: dict[int, float] = {}
    for _, r in df.iterrows():
        qid = int(r[id_col])
        raw = r["ground_truth_posterior"]
        if pd.isna(raw) or str(raw).strip() == "":
            gt[qid] = float("nan")
        else:
            tok = str(raw).strip().split()[0]
            gt[qid] = float(tok)
    return gt


def resolve_video_files(video_root: Path, video_ids: list[str]) -> dict[str, Path]:
    """Locate .mp4 files under video_root or video_root/validation_videos (handling leading/trailing spaces)."""
    search_dirs = [
        video_root,
        video_root / "validation_videos",
        video_root / "videos",
        video_root / "run" / "videos",
    ]
    by_stem: dict[str, Path] = {}
    for d in search_dirs:
        if not d.is_dir():
            continue
        for entry in sorted(d.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".mp4":
                stem = entry.stem.strip()
                by_stem.setdefault(stem, entry)
    missing = [v for v in video_ids if v not in by_stem]
    if missing:
        raise FileNotFoundError(f"Missing video .mp4 files under {video_root}: {missing}")
    return {v: by_stem[v] for v in video_ids}


def probe_and_compare_video(
    video_id: str,
    video_path: Path,
    csv_fps: float,
    iiixr_fps_entry: dict[str, Any] | None = None,
    lock_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe MP4 container & OpenCV decode, comparing against IIIXR file_fps_val.json and snapshot lock."""
    digest = sha256_file(video_path)
    size_bytes = video_path.stat().st_size
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")
    cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cv_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cv_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    pts_ms: list[float] = []
    decoded_frames = 0
    while True:
        pts_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
        ok, frame = cap.read()
        if not ok:
            pts_ms.pop()
            break
        if decoded_frames == 0:
            cv_h, cv_w = frame.shape[:2]
        decoded_frames += 1
    cap.release()

    short_side = min(cv_h, cv_w)
    resized_by_tc = short_side > 480
    proc_w = int(round(cv_w * 480 / short_side)) if resized_by_tc else cv_w
    proc_h = int(round(cv_h * 480 / short_side)) if resized_by_tc else cv_h
    stride_160 = max(1, int(np.ceil(decoded_frames / 160)))
    fidx_160 = list(range(0, decoded_frames, stride_160))

    exact_hash_match = bool(lock_entry and lock_entry.get("sha256") == digest and lock_entry.get("bytes") == size_bytes)
    iiixr_frame_match = bool(iiixr_fps_entry and int(iiixr_fps_entry.get("n", -1)) == decoded_frames)
    iiixr_fps_match = bool(
        iiixr_fps_entry
        and abs(float(iiixr_fps_entry.get("file", 0.0)) - cv_fps) <= 0.05
        and abs(float(iiixr_fps_entry.get("csv", 0.0)) - csv_fps) <= 1e-6
    )
    if exact_hash_match and (not iiixr_fps_entry or (iiixr_frame_match and iiixr_fps_match)):
        equivalence = "exact_byte_and_metadata_match"
    elif iiixr_frame_match and iiixr_fps_match:
        equivalence = "content_and_timing_equivalent_hash_differs"
    else:
        equivalence = "divergent_from_iiixr_reference"

    return {
        "video_id": video_id,
        "path": str(video_path),
        "filename_had_whitespace": video_path.stem != video_path.stem.strip(),
        "sha256": digest,
        "bytes": size_bytes,
        "raw_width": cv_w,
        "raw_height": cv_h,
        "decoded_frames": decoded_frames,
        "container_fps": cv_fps,
        "csv_fps": float(csv_fps),
        "fps_rel_diff_container_vs_csv": abs(cv_fps - csv_fps) / max(csv_fps, 1e-9),
        "pts_seconds_head": [round(x / 1000.0, 6) for x in pts_ms[:8]],
        "pts_seconds_tail": [round(x / 1000.0, 6) for x in pts_ms[-4:]],
        "tc_load_video_resized_480p": resized_by_tc,
        "processed_width": proc_w,
        "processed_height": proc_h,
        "coord_scale_x_raw_to_proc": proc_w / max(cv_w, 1),
        "coord_scale_y_raw_to_proc": proc_h / max(cv_h, 1),
        "stride_max160": stride_160,
        "sampled_frame_count": len(fidx_160),
        "iiixr_file_fps_val_ref": iiixr_fps_entry,
        "snapshot_lock_ref": lock_entry,
        "exact_hash_match": exact_hash_match,
        "iiixr_frame_match": iiixr_frame_match,
        "iiixr_fps_match": iiixr_fps_match,
        "equivalence_status": equivalence,
    }


def read_caw_predictions_from_tar(
    tar_path: Path,
    target_ids: tuple[int, ...] = TARGET_18_IDS,
    member_name: str = "run/predictions.csv",
) -> dict[int, dict[str, Any]]:
    """Stream ONLY run/predictions.csv directly from quantiphy_full.tar without extracting the archive."""
    if not tar_path.is_file():
        raise FileNotFoundError(f"CAW archive not found: {tar_path}")
    with tarfile.open(tar_path, "r:*") as tf:
        try:
            member = tf.getmember(member_name)
        except KeyError as exc:
            raise KeyError(f"Member '{member_name}' not found inside {tar_path}") from exc
        extracted = tf.extractfile(member)
        if extracted is None:
            raise ValueError(f"Member '{member_name}' in {tar_path} is not a regular file")
        raw_csv = extracted.read()

    csv_sha = sha256_bytes(raw_csv)
    reader = csv.DictReader(io.StringIO(raw_csv.decode("utf-8")))
    wanted = set(int(x) for x in target_ids)
    out: dict[int, dict[str, Any]] = {}
    for row in reader:
        qid = int(row["id"])
        if qid not in wanted:
            continue
        vid = str(row["video_id"]).strip()
        expected_vid_caw = EXPECTED_VIDEO_BY_ID.get(qid) or EXPECTED_STATIC_2D_VIDEO_BY_ID.get(qid)
        if expected_vid_caw and vid != expected_vid_caw:
            raise ValueError(
                f"CAW archive ID/video mismatch for ID {qid}: expected {expected_vid_caw}, got {vid}"
            )
        pv_raw = str(row.get("parsed_value", "")).strip()
        pv = float(pv_raw) if pv_raw not in ("", "nan", "None") else float("nan")
        out[qid] = {
            "id": qid,
            "video_id": vid,
            "sample_id": row.get("sample_id", ""),
            "status": row.get("status", ""),
            "parsed_value": pv,
            "parsed_unit": row.get("parsed_unit", ""),
            "raw_response": row.get("raw_response", ""),
            "question": str(row.get("question", "")).strip(),
            "source_archive": tar_path.name,
            "source_member": member_name,
            "member_sha256": csv_sha,
        }
    missing = sorted(wanted - set(out.keys()))
    if missing:
        raise ValueError(f"Missing target IDs {missing} in {tar_path}:{member_name}")
    return out


def stage_assets_from_caw_tar(
    tar_path: Path,
    dest_dir: Path,
    video_ids: tuple[str, ...] = TARGET_6_VIDEOS,
) -> dict[str, Path]:
    """Extract ONLY validation CSV and requested MP4s from quantiphy_full.tar for local replay/testing."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = dest_dir / "validation_videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    csv_out = dest_dir / "quantiphy_validation.csv"
    wanted_members = {"run/configs/quantiphy_validation.csv": csv_out}
    for vid in video_ids:
        wanted_members[f"run/videos/{vid}.mp4"] = vid_dir / f"{vid}.mp4"

    remaining = {k: v for k, v in wanted_members.items() if not v.is_file()}
    if remaining:
        with tarfile.open(tar_path, "r:*") as tf:
            for member in tf:
                if member.name in remaining and member.isfile():
                    target = remaining.pop(member.name)
                    src = tf.extractfile(member)
                    if src is not None:
                        target.write_bytes(src.read())
                if not remaining:
                    break
    if remaining:
        raise FileNotFoundError(f"Could not find members {sorted(remaining)} inside {tar_path}")
    return {"csv_path": csv_out, "video_dir": vid_dir}


def load_cached_mot_val(
    upstream_dir: Path = UPSTREAM_DIR,
    target_ids: tuple[int, ...] = TARGET_18_IDS,
) -> dict[int, dict[str, Any]]:
    """Load the 18 IIIXR mot_val_<id>.json reference records (strictly for replay mode or post-inference eval)."""
    cache_dir = (
        upstream_dir
        / "team"
        / "member1"
        / "trackB"
        / "package"
        / "z1s2c"
        / "2_c2_g3a"
        / "root"
        / "analysis"
        / "v3"
        / "ag"
        / "r10"
        / "MOTION"
        / "cache"
    )
    out: dict[int, dict[str, Any]] = {}
    for qid in target_ids:
        p = cache_dir / f"mot_val_{qid}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing IIIXR reference cache file: {p}")
        raw = p.read_bytes()
        rec = json.loads(raw.decode("utf-8"))
        if int(rec["id"]) != qid:
            raise ValueError(f"Cache ID mismatch in {p}: {rec['id']} != {qid}")
        if qid in EXPECTED_VIDEO_BY_ID and str(rec["video"]).strip() != EXPECTED_VIDEO_BY_ID[qid]:
            raise ValueError(f"Cache video mismatch in {p}: {rec['video']} != {EXPECTED_VIDEO_BY_ID[qid]}")
        rec["cache_sha256"] = sha256_bytes(raw)
        rec["cache_relpath"] = p.relative_to(upstream_dir).as_posix()
        out[qid] = rec
    return out


def load_cached_val_geo_raw(
    upstream_dir: Path = UPSTREAM_DIR,
    target_ids: tuple[int, ...] = STATIC_2D_41_IDS,
) -> dict[int, dict[str, Any]]:
    """Load IIIXR reference geometry outputs from preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv (eval/replay ONLY)."""
    if FreshAntiCacheGuard._active_depth > 0:
        raise PermissionError(
            "Fresh-mode anti-cache guard blocked load_cached_val_geo_raw() during fresh inference!"
        )
    lock = read_snapshots()["upstream"]["val_geo_raw"]
    inter_path = upstream_dir / "intermediates" / "intermediates.tar.xz"
    with tarfile.open(inter_path, "r:xz") as tf:
        member = tf.extractfile(lock["member"])
        if member is None:
            raise FileNotFoundError(f"Missing {lock['member']} in {inter_path}")
        raw_bytes = member.read()
    digest = sha256_bytes(raw_bytes)
    if digest != lock["sha256"]:
        raise ValueError(f"val_geo_raw SHA256 mismatch: {digest} != {lock['sha256']}")

    df = pd.read_csv(io.BytesIO(raw_bytes))
    id_col = _parse_id_column(df)
    wanted = set(int(x) for x in target_ids)
    out: dict[int, dict[str, Any]] = {}
    for _, r in df.iterrows():
        qid = int(r[id_col])
        if qid not in wanted:
            continue
        vid = str(r["video_id"]).strip()
        expected_vid = EXPECTED_STATIC_2D_VIDEO_BY_ID.get(qid) or EXPECTED_VIDEO_BY_ID.get(qid)
        if expected_vid and vid != expected_vid:
            raise ValueError(f"val_geo_raw ID/video mismatch for ID {qid}: expected {expected_vid}, got {vid}")
        gv = None if pd.isna(r.get("geo_value")) else float(r["geo_value"])
        mmode = "" if pd.isna(r.get("meas_mode")) else str(r["meas_mode"])
        msrc = "" if pd.isna(r.get("merge_src")) else str(r["merge_src"])
        mtrk = "" if pd.isna(r.get("meas_tracks")) else str(r["meas_tracks"])
        mwhy = "" if pd.isna(r.get("meas_why")) else str(r["meas_why"])
        mkind = "" if pd.isna(r.get("meas_kind")) else str(r["meas_kind"])
        tv_tag = "v3nt" if msrc in ("v", "xg_v") or mmode.startswith("vlm_") else (msrc if msrc in ("sruler", "ruler") else (mtrk or "none"))
        out[qid] = {
            "id": qid,
            "video": vid,
            "g_box": gv,
            "g_pt": None,
            "tv": tv_tag,
            "mode_box": mmode,
            "merge_src": msrc,
            "meas_tracks": mtrk,
            "why": mwhy,
            "kind": mkind,
            "cache_sha256": digest,
            "cache_relpath": f"intermediates/intermediates.tar.xz:{lock['member']}",
        }
    return out


@dataclass
class ArtifactRef:
    path: str
    sha256: str
    bytes: int
    kind: str
    schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "kind": self.kind,
        }
        if self.schema is not None:
            d["schema"] = self.schema
        return d


class TraceStore:
    """Content-addressed lossless numerical/JSON/preview store and structured event logger."""

    def __init__(self, run_dir: Path, run_id: str, enabled: bool = True) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.enabled = enabled
        self.blobs_dir = run_dir / "blobs"
        self.events_path = run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if enabled:
            self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        data: bytes,
        ext: str,
        kind: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        digest = sha256_bytes(data)
        if not self.enabled:
            return {"path": None, "sha256": digest, "bytes": len(data), "kind": kind, "schema": schema}
        ext_clean = ext.lstrip(".")
        rel = Path("blobs") / kind / digest[:2] / f"{digest}.{ext_clean}"
        dest = self.run_dir / rel
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        return ArtifactRef(
            path=rel.as_posix(),
            sha256=digest,
            bytes=len(data),
            kind=kind,
            schema=schema,
        ).to_dict()

    def put_json(self, obj: Any, kind: str = "json", schema: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        return self.put_bytes(raw, "json", kind=kind, schema=schema)

    def put_npy(self, arr: np.ndarray, kind: str = "npy", name: str = "") -> dict[str, Any]:
        arr_np = np.asarray(arr)
        buf = io.BytesIO()
        np.save(buf, arr_np, allow_pickle=False)
        raw = buf.getvalue()
        schema = {
            "name": name,
            "dtype": str(arr_np.dtype),
            "shape": list(arr_np.shape),
        }
        return self.put_bytes(raw, "npy", kind=kind, schema=schema)

    def put_png(self, rgb_or_bgr_bytes: bytes, kind: str = "preview", meta: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.put_bytes(rgb_or_bgr_bytes, "png", kind=kind, schema=meta)

    def put_image_rgb(self, img_rgb: np.ndarray, kind: str = "preview", meta: dict[str, Any] | None = None) -> dict[str, Any]:
        bgr = cv2.cvtColor(np.ascontiguousarray(img_rgb), cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("Failed to encode RGB image to PNG")
        schema = {"height": int(img_rgb.shape[0]), "width": int(img_rgb.shape[1]), **(meta or {})}
        return self.put_png(enc.tobytes(), kind=kind, meta=schema)

    def log_event(
        self,
        *,
        stage: str,
        call_kind: str,
        status: str,
        video_id: str | None = None,
        sample_id: int | str | None = None,
        consumers: list[int] | None = None,
        attempt: int = 1,
        parent_id: str | None = None,
        event_id: str | None = None,
        input_refs: dict[str, Any] | None = None,
        output_refs: dict[str, Any] | None = None,
        backend: str = "cpu",
        device: str = "cpu",
        duration_sec: float = 0.0,
        timestamp: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        eid = event_id or f"evt-{uuid.uuid4().hex[:12]}"
        if not self.enabled:
            return eid
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "event_id": eid,
            "parent_id": parent_id,
            "video_id": video_id,
            "sample_id": sample_id,
            "consumers": consumers if consumers is not None else ([int(sample_id)] if isinstance(sample_id, int) else []),
            "attempt": int(attempt),
            "stage": stage,
            "call_kind": call_kind,
            "status": status,
            "backend": backend,
            "device": device,
            "timestamp": timestamp or utc_now_iso(),
            "duration_sec": round(float(duration_sec), 6),
            "input_refs": input_refs or {},
            "output_refs": output_refs or {},
            "error": error,
        }
        if extra:
            record["extra"] = extra
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return eid

    def timed_event(self, **kwargs: Any) -> "_TimedEventContext":
        return _TimedEventContext(self, **kwargs)


class _TimedEventContext:
    def __init__(self, store: TraceStore, **kwargs: Any) -> None:
        self.store = store
        self.kwargs = kwargs
        self.event_id = kwargs.pop("event_id", None) or f"evt-{uuid.uuid4().hex[:12]}"
        self.output_refs: dict[str, Any] = dict(kwargs.pop("output_refs", None) or {})
        self.extra: dict[str, Any] = dict(kwargs.pop("extra", None) or {})
        self._t0 = 0.0
        self._ts = ""

    def __enter__(self) -> "_TimedEventContext":
        self._t0 = time.perf_counter()
        self._ts = utc_now_iso()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        dur = time.perf_counter() - self._t0
        status = self.kwargs.pop("status", "ok" if exc_type is None else "error")
        err = f"{exc_type.__name__}: {exc_val}" if exc_type is not None else self.kwargs.pop("error", None)
        self.store.log_event(
            event_id=self.event_id,
            timestamp=self._ts,
            duration_sec=dur,
            status=status,
            output_refs=self.output_refs,
            error=err,
            extra=self.extra or None,
            **self.kwargs,
        )
