"""Post-inference isolated evaluator for IIIXR g_box (reads ground truth ONLY after predictions are sealed)."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from solutions.iiixr.kaggle.schema import (
    EXPECTED_VIDEO_BY_ID,
    TARGET_18_IDS,
    UPSTREAM_DIR,
    load_cached_mot_val,
    load_validation_ground_truth,
    read_caw_predictions_from_tar,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json_atomic,
)

MRA_THRESHOLDS: np.ndarray = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
DIAG_ATOL: float = 1e-6
DIAG_RTOL: float = 1e-4


def compute_row_mra(pred: float | None, gt: float | None) -> float:
    """Standard QuantiPhy per-question MRA; missing/non-positive/failed predictions score 0.0 (never dropped)."""
    if pred is None or gt is None:
        return 0.0
    try:
        p = float(pred)
        y = float(gt)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(p) and math.isfinite(y)) or p <= 0.0 or y == 0.0:
        return 0.0
    rel_err = abs(p - y) / abs(y)
    return float(np.mean(rel_err < (1.0 - MRA_THRESHOLDS)))


def _finite_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def evaluate_sealed_predictions(
    *,
    locked_predictions_path: Path,
    validation_csv_path: Path,
    output_dir: Path,
    caw_tar_path: Path | None = None,
    upstream_dir: Path = UPSTREAM_DIR,
) -> dict[str, Any]:
    """Evaluate sealed predictions in isolation and write per_question.csv + evaluation.json."""
    locked = read_json(locked_predictions_path)
    if not locked.get("sealed", False):
        raise RuntimeError(f"Refusing to evaluate unsealed predictions file: {locked_predictions_path}")

    locked_sha = sha256_file(locked_predictions_path)
    mode = str(locked.get("mode", "unknown"))
    scope = str(locked.get("scope", "target18"))
    preds_list: list[dict[str, Any]] = locked["predictions"]

    gt_by_id = load_validation_ground_truth(validation_csv_path)
    cache_by_id = load_cached_mot_val(upstream_dir, TARGET_18_IDS)

    caw_by_id: dict[int, dict[str, Any]] = {}
    caw_source_info: dict[str, Any] = {"available": False, "path": None}
    if caw_tar_path is not None and caw_tar_path.is_file():
        in_scope_ids = tuple(sorted(int(p["id"]) for p in preds_list if int(p["id"]) in EXPECTED_VIDEO_BY_ID))
        if in_scope_ids:
            caw_by_id = read_caw_predictions_from_tar(caw_tar_path, target_ids=in_scope_ids)
            caw_source_info = {
                "available": True,
                "path": str(caw_tar_path),
                "member": "run/predictions.csv",
                "matched_ids": len(caw_by_id),
            }

    csv_rows: list[dict[str, Any]] = []
    exceptions_vs_cache: list[dict[str, Any]] = []

    for p_rec in preds_list:
        qid = int(p_rec["id"])
        vid = str(p_rec["video_id"]).strip()
        in_scope = bool(p_rec.get("in_scope_2d_motion", qid in EXPECTED_VIDEO_BY_ID))
        gt = _finite_or_none(gt_by_id.get(qid))

        cache_rec = cache_by_id.get(qid)
        cached_box = _finite_or_none(cache_rec["g_box"]) if cache_rec else None
        cached_pt = _finite_or_none(cache_rec["g_pt"]) if cache_rec else None
        tv_cache = str(cache_rec["tv"]) if cache_rec else ""

        caw_rec = caw_by_id.get(qid)
        caw_val = _finite_or_none(caw_rec["parsed_value"]) if caw_rec else None

        fresh_val = _finite_or_none(p_rec.get("fresh_box"))
        fresh_v2 = _finite_or_none(p_rec.get("fresh_box_v2"))
        fresh_v3nt = _finite_or_none(p_rec.get("fresh_box_v3nt"))
        status = str(p_rec.get("status", "missing"))
        coverage = 1 if (in_scope and fresh_val is not None and status == "ok") else 0

        abs_vs_cache = abs(fresh_val - cached_box) if (fresh_val is not None and cached_box is not None) else None
        rel_vs_cache = (
            abs_vs_cache / max(abs(cached_box), 1e-12)
            if (abs_vs_cache is not None and cached_box is not None)
            else None
        )
        match_cache = bool(
            fresh_val is not None
            and cached_box is not None
            and math.isclose(fresh_val, cached_box, rel_tol=DIAG_RTOL, abs_tol=DIAG_ATOL)
        )

        abs_vs_gt = abs(fresh_val - gt) if (fresh_val is not None and gt is not None) else None
        rel_vs_gt = abs_vs_gt / max(abs(gt), 1e-12) if (abs_vs_gt is not None and gt is not None) else None

        abs_cache_vs_gt = abs(cached_box - gt) if (cached_box is not None and gt is not None) else None
        rel_cache_vs_gt = abs_cache_vs_gt / max(abs(gt), 1e-12) if (abs_cache_vs_gt is not None and gt is not None) else None

        mra_cache = compute_row_mra(cached_box, gt) if in_scope else None
        mra_pt = compute_row_mra(cached_pt, gt) if in_scope else None
        mra_fresh = compute_row_mra(fresh_val, gt) if (in_scope and mode != "replay") else None
        mra_caw = compute_row_mra(caw_val, gt) if (in_scope and caw_source_info["available"]) else None

        delta_fresh_cache = (mra_fresh - mra_cache) if (mra_fresh is not None and mra_cache is not None) else None
        delta_fresh_caw = (mra_fresh - mra_caw) if (mra_fresh is not None and mra_caw is not None) else None
        delta_cache_caw = (mra_cache - mra_caw) if (mra_cache is not None and mra_caw is not None) else None

        if in_scope and mode != "replay" and not match_cache:
            exceptions_vs_cache.append({
                "id": qid,
                "video_id": vid,
                "status": status,
                "error_stage": p_rec.get("error_stage", ""),
                "tv_cache": tv_cache,
                "track_used": p_rec.get("track_used", ""),
                "cached_box": cached_box,
                "fresh_box": fresh_val,
                "fresh_box_v2": fresh_v2,
                "fresh_box_v3nt": fresh_v3nt,
                "abs_err_vs_cache": abs_vs_cache,
                "rel_err_vs_cache": rel_vs_cache,
                "note": p_rec.get("why", ""),
            })

        csv_rows.append({
            "id": qid,
            "video_id": vid,
            "cat": p_rec.get("cat", ""),
            "kind": p_rec.get("kind", cache_rec["kind"] if cache_rec else ""),
            "unit": p_rec.get("unit", ""),
            "in_scope_2d_motion": in_scope,
            "support_reason": p_rec.get("support_reason", ""),
            "status": status,
            "coverage": coverage,
            "error_stage": p_rec.get("error_stage", ""),
            "why": p_rec.get("why", ""),
            "tv_cache": tv_cache,
            "track_used": p_rec.get("track_used", ""),
            "merge_src": p_rec.get("merge_src", ""),
            "cached_box": cached_box,
            "fresh_box": fresh_val,
            "caw": caw_val,
            "cached_point": cached_pt,
            "fresh_box_v2": fresh_v2,
            "fresh_box_v3nt": fresh_v3nt,
            "ground_truth": gt,
            "abs_err_vs_cache": abs_vs_cache,
            "rel_err_vs_cache": rel_vs_cache,
            "match_cache_tol": match_cache if mode != "replay" else "",
            "abs_err_vs_gt": abs_vs_gt,
            "rel_err_vs_gt": rel_vs_gt,
            "abs_cache_err_vs_gt": abs_cache_vs_gt,
            "rel_cache_err_vs_gt": rel_cache_vs_gt,
            "mra_cached_box": mra_cache,
            "mra_fresh_box": mra_fresh,
            "mra_caw": mra_caw,
            "mra_cached_point": mra_pt,
            "delta_mra_fresh_vs_cache": delta_fresh_cache,
            "delta_mra_fresh_vs_caw": delta_fresh_caw,
            "delta_mra_cache_vs_caw": delta_cache_caw,
            "mpp": _finite_or_none(p_rec.get("mpp")),
            "px_quantity": _finite_or_none(p_rec.get("px_quantity")),
            "question": p_rec.get("question", ""),
            "prior_text": p_rec.get("prior_text", ""),
        })

    per_q_path = output_dir / "per_question.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with per_q_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    in_scope_rows = [r for r in csv_rows if r["in_scope_2d_motion"]]
    denom = len(in_scope_rows)
    mean_mra_cache = float(np.mean([r["mra_cached_box"] for r in in_scope_rows])) if denom else 0.0
    mean_mra_pt = float(np.mean([r["mra_cached_point"] for r in in_scope_rows])) if denom else 0.0
    mean_mra_caw = (
        float(np.mean([r["mra_caw"] for r in in_scope_rows]))
        if (denom and caw_source_info["available"])
        else None
    )
    mean_mra_fresh = (
        float(np.mean([r["mra_fresh_box"] for r in in_scope_rows]))
        if (denom and mode != "replay")
        else None
    )

    n_covered_fresh = sum(int(r["coverage"]) for r in in_scope_rows)
    n_match_cache = sum(1 for r in in_scope_rows if r["match_cache_tol"] is True)

    summary_eval = {
        "evaluated_at": utc_now_iso(),
        "mode": mode,
        "scope": scope,
        "locked_predictions_sha256": locked_sha,
        "benchmark_scope_notice": (
            f"Mean per-question MRA over {denom} 2D motion reference questions (6 videos). "
            "Failed or missing fresh predictions receive MRA=0.0 and remain in the denominator. "
            "Do NOT compare this subset mean directly with 158-question validation macro-MRA or full fusion leaderboard scores."
        ),
        "denominator_questions": denom,
        "survey_total_rows": len(csv_rows),
        "survey_unsupported_rows": sum(1 for r in csv_rows if not r["in_scope_2d_motion"]),
        "fresh_coverage": {
            "generated_ok": n_covered_fresh,
            "denominator": denom,
            "fraction": (n_covered_fresh / denom) if denom else 0.0,
        } if mode != "replay" else None,
        "mean_per_question_mra": {
            "cached_box": mean_mra_cache,
            "fresh_box": mean_mra_fresh,
            "caw": mean_mra_caw,
            "cached_point_reference": mean_mra_pt,
        },
        "paired_mean_mra_deltas": {
            "fresh_minus_cached_box": (mean_mra_fresh - mean_mra_cache) if mean_mra_fresh is not None else None,
            "fresh_minus_caw": (mean_mra_fresh - mean_mra_caw) if (mean_mra_fresh is not None and mean_mra_caw is not None) else None,
            "cached_box_minus_caw": (mean_mra_cache - mean_mra_caw) if mean_mra_caw is not None else None,
        },
        "cache_parity_diagnostics": {
            "atol": DIAG_ATOL,
            "rtol": DIAG_RTOL,
            "matched_count": n_match_cache if mode != "replay" else None,
            "denominator": denom,
            "exceptions": exceptions_vs_cache if mode != "replay" else [],
        },
        "caw_baseline": caw_source_info,
        "per_question_csv": per_q_path.name,
    }
    write_json_atomic(output_dir / "evaluation.json", summary_eval)
    return summary_eval


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated post-inference evaluator for IIIXR g_box")
    ap.add_argument("--locked-predictions", type=Path, required=True)
    ap.add_argument("--validation-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--caw-tar", type=Path, default=None)
    args = ap.parse_args()
    res = evaluate_sealed_predictions(
        locked_predictions_path=args.locked_predictions,
        validation_csv_path=args.validation_csv,
        output_dir=args.output_dir,
        caw_tar_path=args.caw_tar,
    )
    print(f"Evaluated {res['denominator_questions']} 2D motion questions -> {args.output_dir / 'per_question.csv'}")


if __name__ == "__main__":
    main()
