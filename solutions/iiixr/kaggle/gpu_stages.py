"""Instrumented GPU workers for IIIXR g_box on Kaggle 2xT4 (Stage 1 VLM sharded, Stage 2 1-GPU/process)."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import random
import re
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from solutions.iiixr.kaggle.kinematics import (
    _load_upstream_geo_modules,
    check_affected_v2_video,
    measure_video_instrumented,
    run_scene_ruler_on_video,
)
from solutions.iiixr.kaggle.schema import (
    UPSTREAM_DIR,
    FreshAntiCacheGuard,
    TraceStore,
    fingerprint,
    read_json,
    sha256_file,
    utc_now_iso,
    write_json_atomic,
)


def set_deterministic_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _vram_snapshot() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return []
    out = []
    for idx in range(torch.cuda.device_count()):
        out.append({
            "device_index": idx,
            "device_name": torch.cuda.get_device_name(idx),
            "allocated_mib": round(torch.cuda.memory_allocated(idx) / (1024**2), 2),
            "reserved_mib": round(torch.cuda.memory_reserved(idx) / (1024**2), 2),
            "max_allocated_mib": round(torch.cuda.max_memory_allocated(idx) / (1024**2), 2),
        })
    return out


def _reset_vram_peaks() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(idx)


def _free_cuda(*objs: Any) -> None:
    import torch

    for _ in objs:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _resolve_amp_dtype(dtype_profile: str):
    import torch

    if dtype_profile == "fp32":
        return torch.float32
    if dtype_profile == "upstream_bf16":
        return torch.bfloat16
    # Default t4_fp16 for Turing SM 7.5
    return torch.float16


def render_track_overlay_rgb(
    frame_rgb: np.ndarray,
    frame_idx: int,
    fps: float,
    items: list[dict[str, Any]],
) -> np.ndarray:
    """Render bounding boxes, SAM2 mask polygons, and mask centroids onto an RGB frame."""
    canvas = frame_rgb.copy()
    overlay = frame_rgb.copy()
    palette = {
        "target": (56, 189, 248),     # cyan-blue
        "prior": (250, 204, 21),      # amber-gold
        "other": (168, 85, 247),      # purple
        "candidate": (148, 163, 184), # slate
    }
    for it in items:
        role = it.get("role", "other")
        col = palette.get(role, palette["other"])
        poly = it.get("poly")
        if poly:
            for contour in poly:
                pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], col)
                    cv2.polylines(canvas, [pts], True, col, 2)
    canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
    for it in items:
        role = it.get("role", "other")
        col = palette.get(role, palette["other"])
        box = it.get("box")
        if box and len(box) == 4:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 2)
            label = f"{role}:{it.get('label', '')}"
            cv2.putText(canvas, label[:36], (max(2, x0), max(16, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cen = it.get("c")
        if cen and len(cen) == 2:
            cx, cy = int(round(cen[0])), int(round(cen[1]))
            cv2.circle(canvas, (cx, cy), 4, (255, 64, 64) if role == "target" else col, -1)
    hdr = f"frame={frame_idx}  t={frame_idx / max(fps, 1e-6):.3f}s"
    cv2.rectangle(canvas, (4, 4), (250, 26), (15, 23, 42), -1)
    cv2.putText(canvas, hdr, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA)
    return canvas


def _bind_tensors_to_device(batch: Any, device: Any) -> dict[str, Any]:
    return {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in dict(batch).items()
    }


# =============================================================================
# Stage 1 Worker: Qwen3-VL-8B-Instruct (Sharded across 2xT4 in FP16, batch 1)
# =============================================================================
def run_vlm_stage_worker(work_json_path: Path) -> dict[str, Any]:
    """Run optional LLM question parse + vlm_ground (--notext) on 2xT4 via Transformers sharded FP16."""
    spec = read_json(work_json_path)
    run_dir = Path(spec["run_dir"])
    run_id = spec["run_id"]
    trace_enabled = bool(spec.get("trace_enabled", True))
    store = TraceStore(run_dir, run_id, enabled=trace_enabled)
    set_deterministic_seeds(int(spec.get("seed", 0)))

    model = None
    proc = None
    try:
        with FreshAntiCacheGuard(enabled=True):
            import torch
            from PIL import Image
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            mods = _load_upstream_geo_modules(UPSTREAM_DIR)
            parse_v1 = mods["parse_v1"]
            rules_v1 = mods["rules_v1"]
            tc = mods["tc"]

            model_dir = Path(spec["qwen_model_dir"])
            model_rev = spec["qwen_revision"]
            dtype_profile = spec.get("dtype_profile", "t4_fp16")
            torch_dtype = (
                torch.bfloat16
                if dtype_profile == "upstream_bf16"
                else (torch.float32 if dtype_profile == "fp32" else torch.float16)
            )

            _reset_vram_peaks()
            t_load0 = time.perf_counter()
            proc = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            load_kwargs: dict[str, Any] = {
                "dtype": torch_dtype,
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
                "local_files_only": True,
            }
            if n_gpus >= 2:
                load_kwargs["device_map"] = "balanced"
                load_kwargs["max_memory"] = {
                    i: min(13 * 1024**3, max(4 * 1024**3, torch.cuda.mem_get_info(i)[0] - 2 * 1024**3))
                    for i in range(n_gpus)
                }
            elif n_gpus == 1:
                load_kwargs["device_map"] = {"": "cuda:0"}
            else:
                load_kwargs["device_map"] = {"": "cpu"}

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_dir),
                **load_kwargs,
            ).eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            load_sec = time.perf_counter() - t_load0
            load_vram = _vram_snapshot()

            first_dev = model.get_input_embeddings().weight.device
            device_map_repr = str(getattr(model, "hf_device_map", load_kwargs["device_map"]))
            gen_kwargs = {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }

            # 1. Optional LLM fallback parse (geo/parse.py lines 32-72)
            llm_parse_outputs: dict[str, Any] = {}
            if spec.get("run_llm_parse", False):
                for row in spec.get("parse_rows", []):
                    assert "ground_truth_posterior" not in row
                    rid = int(row["id"])
                    dep = rules_v1.parse_depth(row.get("depth_info", "")) if str(row["video_type"])[1] == "3" else []
                    dkeys = sorted({d["key"] for d in dep})
                    user_prompt = parse_v1.build_prompt(
                        rules_v1.clean_text(row["question"]),
                        rules_v1.clean_text(row["ground_truth_prior"]).replace("\n", "; "),
                        dkeys,
                    )
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a precise information extraction assistant."}]},
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                    ]
                    rendered = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    _reset_vram_peaks()
                    t0 = time.perf_counter()
                    inp = _bind_tensors_to_device(proc(text=[rendered], return_tensors="pt"), first_dev)
                    with torch.inference_mode():
                        gen_ids = model.generate(**inp, max_new_tokens=200, **gen_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    dur = time.perf_counter() - t0
                    prompt_ids = inp["input_ids"][0].detach().cpu().tolist()
                    out_ids = gen_ids[0, inp["input_ids"].shape[1]:].detach().cpu().tolist()
                    raw_txt = proc.decode(out_ids, skip_special_tokens=True).strip()
                    parsed_json = parse_v1.parse_json(raw_txt)
                    rec_out = {
                        "id": rid,
                        "model_repo": "Qwen/Qwen3-VL-8B-Instruct",
                        "revision": model_rev,
                        "dtype": str(torch_dtype),
                        "messages": messages,
                        "rendered_prompt": rendered,
                        "prompt_token_ids": prompt_ids,
                        "output_token_ids": out_ids,
                        "raw_response": raw_txt,
                        "parsed": parsed_json,
                        "duration_sec": round(dur, 4),
                        "vram": _vram_snapshot(),
                    }
                    llm_parse_outputs[str(rid)] = rec_out
                    out_ref = store.put_json(rec_out, kind="vlm_parse")
                    store.log_event(
                        stage="parse_llm",
                        call_kind="model_call",
                        status="ok",
                        video_id=row["video_id"],
                        sample_id=rid,
                        backend=f"transformers:{torch_dtype}",
                        device=device_map_repr,
                        duration_sec=dur,
                        output_refs={"llm_parse": out_ref},
                    )

            # 2. VLM visual grounding --notext (geo/vlm_ground.py lines 27-135)
            vlm_ground_by_video: dict[str, list[dict[str, Any]]] = {}
            for v_item in spec.get("vlm_ground_videos", []):
                vid = v_item["video_id"]
                vpath = Path(v_item["video_path"])
                v_recs = v_item["parse_v2_records"]
                consumers = [int(r["id"]) for r in v_recs]

                frames_all = tc.load_video(str(vpath))
                n_all = len(frames_all)
                stride = max(1, int(math.ceil(n_all / 160)))
                fl = list(range(0, n_all, stride))
                frames = [frames_all[i] for i in fl]
                del frames_all
                di = np.unique(np.linspace(0, len(frames) - 1, min(16, len(frames))).round().astype(int)).tolist()
                pick = [di[int(round(x))] for x in np.linspace(0, len(di) - 1, min(4, len(di)))]
                h_img, w_img = frames[0].shape[:2]

                # Build queries_for exactly as in geo/vlm_ground.py
                qs_map: dict[str, dict[str, Any]] = {}

                def _add_vg(ent: dict[str, Any] | None, role: str) -> str | None:
                    if not ent or not ent.get("phrase"):
                        return None
                    rel = ent.get("rel")
                    k = ent["phrase"] + "|" + ",".join(ent.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")
                    if k not in qs_map:
                        qs_map[k] = {
                            "key": k,
                            "phrase": ent["phrase"],
                            "where": list(ent.get("where") or []),
                            "rel": rel,
                            "roles": [],
                            "raw": set(),
                        }
                    if role not in qs_map[k]["roles"]:
                        qs_map[k]["roles"].append(role)
                    if ent.get("raw"):
                        qs_map[k]["raw"].add(ent["raw"])
                    if rel:
                        _add_vg({"phrase": rel[1], "where": [], "raw": rel[1]}, "ref")
                    return k

                for r in v_recs:
                    _add_vg(r["target"], "dist" if r["q"]["kind"] == "dist2" else "target")
                    if r.get("target2"):
                        _add_vg(r["target2"], "dist" if r["q"]["kind"] == "dist2" else "target2")
                    if r.get("prior_obj") and r["prior_obj"].get("phrase"):
                        _add_vg(r["prior_obj"], "prior")

                vid_records: list[dict[str, Any]] = []
                for q in qs_map.values():
                    raws = sorted(q["raw"], key=len, reverse=True)
                    expr = raws[0] if raws else q["phrase"]
                    for wx in [x for x in q["where"] if x != "#second"]:
                        if wx not in expr:
                            expr = f"{wx} {expr}"
                    expr = re.sub(r"\s+", " ", expr).strip()
                    if "#second" in q["where"]:
                        prompt_txt = (
                            f'Locate every instance of "{q["phrase"]}" in the image. Report all of their bounding boxes in JSON format, '
                            f'as a list [{{"bbox_2d": [x1, y1, x2, y2], "label": "{q["phrase"]}"}}, ...]. If there is none, output [].'
                        )
                    else:
                        expr_nt = expr + " (the object itself, not a text label or caption naming it)"
                        prompt_txt = (
                            f'Locate the {expr_nt} in the image. Report its bounding box in JSON format as '
                            f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{q["phrase"]}"}}]. If it is not in the image, output [].'
                        )
                    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_txt}]}]
                    rendered = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

                    for j in pick:
                        _reset_vram_peaks()
                        t0 = time.perf_counter()
                        pil_img = Image.fromarray(frames[j])
                        inp = _bind_tensors_to_device(proc(text=[rendered], images=[pil_img], return_tensors="pt"), first_dev)
                        with torch.inference_mode():
                            gen_ids = model.generate(**inp, max_new_tokens=256, **gen_kwargs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        dur = time.perf_counter() - t0
                        prompt_ids = inp["input_ids"][0].detach().cpu().tolist()
                        out_ids = gen_ids[0, inp["input_ids"].shape[1]:].detach().cpu().tolist()
                        resp_txt = proc.decode(out_ids, skip_special_tokens=True).strip()
                        boxes_raw = [
                            [float(m.group(i)) for i in range(1, 5)]
                            for m in re.finditer(
                                r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]',
                                resp_txt,
                            )
                        ]
                        rel_px = [[x0 / 1000.0 * w_img, y0 / 1000.0 * h_img, x1 / 1000.0 * w_img, y1 / 1000.0 * h_img] for x0, y0, x1, y1 in boxes_raw]
                        boxes_rel1000 = [[round(c, 1) for c in b] for b in rel_px]
                        frame_ref = store.put_image_rgb(frames[j], kind="vlm_input_frame", meta={"video_id": vid, "j": int(j)})
                        vg_rec = {
                            "video_id": vid,
                            "key": q["key"],
                            "phrase": q["phrase"],
                            "where": q["where"],
                            "roles": q["roles"],
                            "expr": expr,
                            "multi": "#second" in q["where"],
                            "j": int(j),
                            "source_fidx": int(fl[j]),
                            "W": int(w_img),
                            "H": int(h_img),
                            "prompt": prompt_txt,
                            "rendered_prompt": rendered,
                            "model_repo": "Qwen/Qwen3-VL-8B-Instruct",
                            "revision": model_rev,
                            "dtype": str(torch_dtype),
                            "pixel_values_shape": list(inp["pixel_values"].shape) if "pixel_values" in inp else None,
                            "image_grid_thw": inp["image_grid_thw"].detach().cpu().tolist() if "image_grid_thw" in inp else None,
                            "prompt_token_ids": prompt_ids,
                            "output_token_ids": out_ids,
                            "n_prompt_tokens": len(prompt_ids),
                            "response": resp_txt,
                            "boxes_raw": boxes_raw,
                            "boxes_rel1000": boxes_rel1000,
                            "duration_sec": round(dur, 4),
                            "vram": _vram_snapshot(),
                        }
                        vid_records.append(vg_rec)
                        vg_ref = store.put_json(vg_rec, kind="vlm_ground")
                        store.log_event(
                            stage="vlm_ground_notext",
                            call_kind="model_call",
                            status="ok",
                            video_id=vid,
                            consumers=consumers,
                            backend=f"transformers:{torch_dtype}",
                            device=device_map_repr,
                            duration_sec=dur,
                            input_refs={"frame_rgb": frame_ref},
                            output_refs={"vlm_ground_record": vg_ref},
                        )
                vlm_ground_by_video[vid] = vid_records

            receipt = {
                "status": "ok",
                "stage": "vlm_stage",
                "load_duration_sec": round(load_sec, 4),
                "load_vram": load_vram,
                "final_vram": _vram_snapshot(),
                "llm_parse_outputs": llm_parse_outputs,
                "vlm_ground_by_video": vlm_ground_by_video,
                "errors": [],
                "completed_at": utc_now_iso(),
            }
            write_json_atomic(Path(spec["receipt_path"]), receipt)
            return receipt
    except Exception as exc:
        tb = traceback.format_exc()
        err_rec = {
            "worker": "vlm-worker",
            "stage": "vlm_stage",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
            "run_dir": str(run_dir),
        }
        store.log_event(
            stage="vlm_stage",
            call_kind="model_call",
            status="error",
            backend=f"transformers:{spec.get('dtype_profile', 't4_fp16')}",
            device="cuda",
            error=err_rec["error"],
        )
        print(f"[FIRST WORKER ERROR] worker=vlm-worker stage=vlm_stage\n{tb}", file=sys.stderr, flush=True)
        receipt = {
            "status": "error",
            "stage": "vlm_stage",
            "llm_parse_outputs": {},
            "vlm_ground_by_video": {},
            "errors": [err_rec],
            "completed_at": utc_now_iso(),
        }
        write_json_atomic(Path(spec["receipt_path"]), receipt)
        return receipt
    finally:
        _free_cuda(model, proc)


# =============================================================================
# Stage 2 Worker: Single-T4 Detector + SAM 2.1 Tracker (1 GPU per worker process)
# =============================================================================
def _run_gdino_instrumented(
    frames: list[np.ndarray],
    det_indices: list[int],
    phrases: list[str],
    model_dir: Path,
    thr: float,
    amp_dtype: Any,
    device: str = "cuda",
) -> tuple[dict[str, list[list[tuple[list[float], float]]]], dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    mods = _load_upstream_geo_modules(UPSTREAM_DIR)
    tc = mods["tc"]

    _reset_vram_peaks()
    t_load0 = time.perf_counter()
    gp = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    gm = GroundingDinoForObjectDetection.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - t_load0
    load_vram = _vram_snapshot()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    dimgs = [frames[i] for i in det_indices]
    H, W = dimgs[0].shape[:2]
    scale = 800.0 / min(H, W)
    if max(H, W) * scale > 1333.0:
        scale = 1333.0 / max(H, W)
    h_in, w_in = int(round(H * scale)), int(round(W * scale))

    dets_by_phrase: dict[str, list[list[tuple[list[float], float]]]] = {}
    raw_candidates_trace: dict[str, list[dict[str, Any]]] = {}
    t_inf0 = time.perf_counter()

    with torch.no_grad():
        for ph in phrases:
            text_query = ph.strip(" .") + " ."
            ph_dets: list[list[tuple[list[float], float]]] = []
            ph_trace: list[dict[str, Any]] = []
            # Batch-1 per frame for T4 memory safety & determinism
            for k_idx, (fi, img) in enumerate(zip(det_indices, dimgs)):
                x = torch.from_numpy(img[None]).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = F.interpolate(x, size=(h_in, w_in), mode="bilinear", align_corners=False, antialias=True)
                x = (x - mean) / std
                tok = gp.tokenizer([text_query], return_tensors="pt", padding=True).to(device)
                if amp_dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast("cuda", dtype=amp_dtype):
                        out = gm(
                            pixel_values=x,
                            input_ids=tok.input_ids,
                            attention_mask=tok.attention_mask,
                            token_type_ids=tok.get("token_type_ids"),
                            pixel_mask=torch.ones((1, h_in, w_in), dtype=torch.long, device=device),
                        )
                else:
                    out = gm(
                        pixel_values=x,
                        input_ids=tok.input_ids,
                        attention_mask=tok.attention_mask,
                        token_type_ids=tok.get("token_type_ids"),
                        pixel_mask=torch.ones((1, h_in, w_in), dtype=torch.long, device=device),
                    )
                out.logits = out.logits.float()
                out.pred_boxes = out.pred_boxes.float()
                res = gp.post_process_grounded_object_detection(
                    outputs=out,
                    input_ids=tok.input_ids,
                    threshold=thr,
                    text_threshold=thr,
                    target_sizes=[(H, W)],
                )[0]
                b_raw = res["boxes"].cpu().numpy().astype(float)
                sc_raw = res["scores"].cpu().numpy().astype(float)
                keep = tc.nms(b_raw, sc_raw, 0.6)
                kept_list = [(b_raw[i].tolist(), float(sc_raw[i])) for i in keep]
                ph_dets.append(kept_list)
                ph_trace.append({
                    "det_slot": k_idx,
                    "frame_idx": int(fi),
                    "text_query": text_query,
                    "pre_nms_candidates": [
                        {"box_xyxy_px": [round(float(v), 2) for v in b_raw[i]], "score": round(float(sc_raw[i]), 5), "kept_nms": i in keep}
                        for i in range(len(sc_raw))
                    ],
                    "post_nms_count": len(kept_list),
                })
            dets_by_phrase[ph] = ph_dets
            raw_candidates_trace[ph] = ph_trace

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_sec = time.perf_counter() - t_inf0
    peak_vram = _vram_snapshot()
    del gm, gp
    _free_cuda()

    meta = {
        "threshold": thr,
        "nms_iou": 0.6,
        "input_hw": [H, W],
        "resized_hw": [h_in, w_in],
        "scale_factor": round(scale, 6),
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
        "amp_dtype": str(amp_dtype),
        "load_duration_sec": round(load_sec, 4),
        "infer_duration_sec": round(inf_sec, 4),
        "load_vram": load_vram,
        "peak_vram": peak_vram,
        "candidates_by_phrase": raw_candidates_trace,
    }
    return dets_by_phrase, meta


def _run_owlv2_instrumented(
    frames: list[np.ndarray],
    det_indices: list[int],
    phrases: list[str],
    model_dir: Path,
    amp_dtype: Any,
    thr: float = 0.05,
    topk: int = 25,
    device: str = "cuda",
) -> tuple[list[dict[str, list[tuple[list[float], float]]]], dict[str, Any]]:
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    mods = _load_upstream_geo_modules(UPSTREAM_DIR)
    tc = mods["tc"]

    _reset_vram_peaks()
    t_load0 = time.perf_counter()
    op = Owlv2Processor.from_pretrained(str(model_dir), local_files_only=True)
    om = Owlv2ForObjectDetection.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - t_load0
    load_vram = _vram_snapshot()

    dimgs = [frames[i] for i in det_indices]
    H, W = dimgs[0].shape[:2]
    S = max(H, W)
    out_all: list[dict[str, list[tuple[list[float], float]]]] = []
    frame_traces: list[dict[str, Any]] = []
    t_inf0 = time.perf_counter()

    with torch.no_grad():
        for k_idx, (fi, img) in enumerate(zip(det_indices, dimgs)):
            inp = op(text=[phrases], images=[img], return_tensors="pt").to(device)
            if amp_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast("cuda", dtype=amp_dtype):
                    o = om(**inp)
            else:
                o = om(**inp)
            logits = o.logits.float().sigmoid().cpu().numpy()[0]
            boxes = o.pred_boxes.float().cpu().numpy()[0]
            xyxy = np.stack(
                [
                    (boxes[:, 0] - boxes[:, 2] / 2) * S,
                    (boxes[:, 1] - boxes[:, 3] / 2) * S,
                    (boxes[:, 0] + boxes[:, 2] / 2) * S,
                    (boxes[:, 1] + boxes[:, 3] / 2) * S,
                ],
                axis=1,
            )
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
            d_frame: dict[str, list[tuple[list[float], float]]] = {}
            tr_frame: dict[str, Any] = {}
            for qi, ph in enumerate(phrases):
                sc = logits[:, qi]
                idx = np.argsort(-sc)[:topk]
                idx_thr = idx[sc[idx] >= thr]
                keep = tc.nms(xyxy[idx_thr], sc[idx_thr], 0.5) if len(idx_thr) else []
                kept_list = [(xyxy[idx_thr[k]].tolist(), float(sc[idx_thr[k]])) for k in keep]
                d_frame[ph] = kept_list
                tr_frame[ph] = [
                    {"box_xyxy_px": [round(float(v), 2) for v in b], "score": round(float(s), 5)}
                    for b, s in kept_list
                ]
            out_all.append(d_frame)
            frame_traces.append({"det_slot": k_idx, "frame_idx": int(fi), "by_phrase": tr_frame})

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_sec = time.perf_counter() - t_inf0
    peak_vram = _vram_snapshot()
    del om, op
    _free_cuda()

    meta = {
        "threshold": thr,
        "nms_iou": 0.5,
        "topk": topk,
        "square_scale_S": int(S),
        "amp_dtype": str(amp_dtype),
        "load_duration_sec": round(load_sec, 4),
        "infer_duration_sec": round(inf_sec, 4),
        "load_vram": load_vram,
        "peak_vram": peak_vram,
        "per_frame_candidates": frame_traces,
    }
    return out_all, meta


def _run_sam2_instrumented(
    frames: list[np.ndarray],
    anchors: list[tuple[int, int, list[float]]],
    model_dir: Path,
    dtype_profile: str,
    keep_all_polys: bool = True,
    device: str = "cuda",
) -> tuple[dict[int, dict[int, tuple[Any, float, Any, Any]]], dict[str, Any]]:
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    mods = _load_upstream_geo_modules(UPSTREAM_DIR)
    tc = mods["tc"]

    sam_dtype = torch.bfloat16 if dtype_profile == "upstream_bf16" else (torch.float32 if dtype_profile == "fp32" else torch.float16)
    _reset_vram_peaks()
    t_load0 = time.perf_counter()
    sp = Sam2VideoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    sm = Sam2VideoModel.from_pretrained(str(model_dir), local_files_only=True).to(device, dtype=sam_dtype).eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - t_load0
    load_vram = _vram_snapshot()

    res: dict[int, dict[int, tuple[Any, float, Any, Any]]] = {}
    groups: dict[int, list[tuple[int, list[float]]]] = {}
    for oid, fi, b in anchors:
        groups.setdefault(int(fi), []).append((int(oid), [float(v) for v in b]))

    def _propagate_group(fi_val: int, ob_list: list[tuple[int, list[float]]], cur_dtype: Any) -> None:
        sess = sp.init_video_session(video=frames, inference_device=device, dtype=cur_dtype)
        ids = [oid + 1 for oid, _ in ob_list]
        sp.add_inputs_to_inference_session(
            inference_session=sess,
            frame_idx=fi_val,
            obj_ids=ids,
            input_boxes=[[b for _, b in ob_list]],
        )
        for rev in (False, True):
            for o in sm.propagate_in_video_iterator(sess, start_frame_idx=fi_val, reverse=rev):
                if rev and o.frame_idx == fi_val:
                    continue
                m = sp.post_process_masks(
                    [o.pred_masks],
                    original_sizes=[[sess.video_height, sess.video_width]],
                    binarize=False,
                )[0]
                m_bin = (m[:, 0] > 0).cpu().numpy()
                lg = o.object_score_logits.float().cpu().numpy().ravel()
                for j_idx, oid_plus1 in enumerate(o.object_ids):
                    if lg[j_idx] > 0:
                        st = tc.mask_stats(m_bin[j_idx])
                        poly = tc.mask_poly(m_bin[j_idx]) if (st and keep_all_polys) else None
                        col = tc.mask_colors(frames[int(o.frame_idx)], m_bin[j_idx]) if (st and int(o.frame_idx) == fi_val) else None
                    else:
                        st, poly, col = None, None, None
                    res.setdefault(int(oid_plus1) - 1, {})[int(o.frame_idx)] = (st, float(lg[j_idx]), poly, col)
        del sess
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_inf0 = time.perf_counter()
    used_dtype_str = str(sam_dtype)
    with torch.no_grad():
        for fi, ob in groups.items():
            try:
                _propagate_group(fi, ob, sam_dtype)
            except (RuntimeError, TypeError, ValueError) as exc:
                # Automatic FP32 fallback on Turing SM 7.5 if any SAM2 op rejects FP16/BF16
                if sam_dtype != torch.float32 and "out of memory" not in str(exc).lower():
                    for oid_clean, _ in ob:
                        res.pop(int(oid_clean), None)
                    sam_dtype = torch.float32
                    used_dtype_str = "torch.float32(fallback_from_half)"
                    sm = sm.to(device, dtype=torch.float32).eval()
                    _propagate_group(fi, ob, torch.float32)
                else:
                    raise

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_sec = time.perf_counter() - t_inf0
    peak_vram = _vram_snapshot()
    del sm, sp
    _free_cuda()

    meta = {
        "sam2_dtype": used_dtype_str,
        "anchor_groups": {str(k): v for k, v in groups.items()},
        "load_duration_sec": round(load_sec, 4),
        "infer_duration_sec": round(inf_sec, 4),
        "load_vram": load_vram,
        "peak_vram": peak_vram,
    }
    return res, meta


def _build_track_v1(
    vid: str,
    frames: list[np.ndarray],
    n_all: int,
    stride: int,
    fidx: list[int],
    fps: float,
    cam: list[list[float]],
    v1_recs: list[dict[str, Any]],
    gdino_dir: Path,
    sam2_dir: Path,
    dtype_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact reproduction of geo/track.py with candidate selection trace."""
    import track as trk_v1_mod
    import track_common as tc

    H, W = frames[0].shape[:2]
    qs = trk_v1_mod.queries_for(v1_recs)
    di = np.unique(np.linspace(0, len(frames) - 1, min(16, len(frames))).round().astype(int)).tolist()
    dimgs = [frames[i] for i in di]
    phrases = sorted({q["phrase"] for q in qs})
    amp_dtype = _resolve_amp_dtype(dtype_profile)

    dets_by_phrase, gd_meta = _run_gdino_instrumented(frames, di, phrases, gdino_dir, thr=0.2, amp_dtype=amp_dtype)

    anchors: list[tuple[int, int, list[float]]] = []
    selection_trace = []
    for qi, q in enumerate(qs):
        dets = dets_by_phrase[q["phrase"]]
        per = [trk_v1_mod.choose(dets[k], dimgs[k], q["phrase"], q["where"], W, H) for k in range(len(di))]
        scores = [c[1] if c else 0.0 for c in per]
        kbest = int(np.argmax(scores)) if scores else 0
        q["det_scores"] = [round(s, 3) for s in scores]
        q["det_boxes"] = [[round(x, 1) for x in c[0]] if c else None for c in per]
        q["det_ngood"] = [c[2] if c else 0 for c in per]
        q["n_det_frames"] = int(sum(s >= 0.25 for s in scores))
        q["best_score"] = float(scores[kbest]) if scores else 0.0
        if scores and scores[kbest] >= 0.15:
            anchors.append((qi, di[kbest], per[kbest][0]))
            q["anchor"] = [di[kbest], [round(x, 1) for x in per[kbest][0]]]
        else:
            q["anchor"] = None
        selection_trace.append({"key": q["key"], "initial_anchor": q["anchor"], "best_score": q["best_score"]})

    consensus_frame = None
    if anchors:
        tot = np.zeros(len(di))
        for qi, _, _ in anchors:
            s = np.array(qs[qi]["det_scores"])
            tot += s / max(s.max(), 1e-6)
        kc = int(np.argmax(tot))
        consensus_frame = int(di[kc])
        new_anchors = []
        for qi, fi, b in anchors:
            s = qs[qi]["det_scores"]
            if s[kc] >= 0.5 * max(s) and s[kc] >= 0.25 and qs[qi]["det_boxes"][kc] is not None:
                new_anchors.append((qi, di[kc], qs[qi]["det_boxes"][kc]))
                qs[qi]["anchor"] = [di[kc], qs[qi]["det_boxes"][kc]]
            else:
                new_anchors.append((qi, fi, b))
        anchors = new_anchors

    uniq: list[tuple[int, int, list[float]]] = []
    alias: dict[int, int] = {}
    for qi, fi, b in anchors:
        hit = next((u for u in uniq if u[1] == fi and tc.iou(u[2], b) > 0.9), None)
        if hit is None:
            uniq.append((qi, fi, b))
            alias[qi] = qi
        else:
            alias[qi] = hit[0]

    tr0, sam_meta = _run_sam2_instrumented(frames, uniq, sam2_dir, dtype_profile=dtype_profile) if uniq else ({}, {"skipped": True})
    tr = {qi: tr0.get(alias[qi], {}) for qi in alias}

    for qi, q in enumerate(qs):
        d = tr.get(qi, {})
        q["frames"] = []
        for j in range(len(frames)):
            st, lg, poly, _ = d.get(j, (None, -99.0, None, None))
            entry: dict[str, Any] = {"st": st, "lg": round(lg, 2)}
            if poly is not None:
                entry["poly"] = poly
            q["frames"].append(entry)
        ious = []
        for k, fi in enumerate(di):
            st = q["frames"][fi]["st"]
            dets = dets_by_phrase[q["phrase"]][k]
            if st is None or not dets:
                continue
            ious.append(max(tc.iou(st["box"], b) for b, _ in dets))
        q["det_iou"] = [round(x, 3) for x in ious]

    vj = {
        "video_id": vid,
        "N": n_all,
        "stride": stride,
        "fidx": fidx,
        "fps": fps,
        "H": H,
        "W": W,
        "det_frames": di,
        "queries": qs,
        "cam": cam,
    }
    return vj, {
        "gdino": gd_meta,
        "sam2": sam_meta,
        "consensus_frame": consensus_frame,
        "selection_trace": selection_trace,
    }


def _build_track_v2(
    vid: str,
    frames: list[np.ndarray],
    n_all: int,
    stride: int,
    fidx: list[int],
    fps: float,
    cam: list[list[float]],
    v2_recs: list[dict[str, Any]],
    gdino_dir: Path,
    owl_dir: Path,
    sam2_dir: Path,
    dtype_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact reproduction of geo/track_v2.py with full candidate/OWL/SAM2 trace."""
    import track_common as tc
    import track_v2 as trk_v2_mod

    H, W = frames[0].shape[:2]
    qs = trk_v2_mod.queries_for(v2_recs)
    di = np.unique(np.linspace(0, len(frames) - 1, min(16, len(frames))).round().astype(int)).tolist()
    dimgs = [frames[i] for i in di]
    phrases = sorted({q["phrase"] for q in qs})
    amp_dtype = _resolve_amp_dtype(dtype_profile)

    dets_by_phrase, gd_meta = _run_gdino_instrumented(frames, di, phrases, gdino_dir, thr=0.15, amp_dtype=amp_dtype)
    owl_by_frame, owl_meta = _run_owlv2_instrumented(frames, di, phrases, owl_dir, amp_dtype=amp_dtype, thr=0.05, topk=25)

    cfg = {}
    for name, use_owl in (("d", False), ("e", True)):
        outs, anchors, resolved = trk_v2_mod.select_anchors(qs, di, dimgs, dets_by_phrase, owl_by_frame, W, H, use_owl)
        cfg[name] = (outs, anchors, resolved)

    uniq: list[tuple[int, int, list[float]]] = []
    alias: dict[tuple[str, int], int] = {}
    for name in ("d", "e"):
        for qi, fi, b in cfg[name][1]:
            hit = next((u for u in uniq if u[1] == fi and tc.iou(u[2], b) > 0.9), None)
            if hit is None:
                uniq.append((len(uniq), fi, b))
                alias[(name, qi)] = uniq[-1][0]
            else:
                alias[(name, qi)] = hit[0]

    tr, sam_meta = _run_sam2_instrumented(frames, uniq, sam2_dir, dtype_profile=dtype_profile) if uniq else ({}, {"skipped": True})

    objs: dict[str, list[dict[str, Any]]] = {}
    for oid, _, _ in uniq:
        d = tr.get(oid, {})
        fr = []
        for j in range(len(frames)):
            st, lg, poly, col = d.get(j, (None, -99.0, None, None))
            e_item: dict[str, Any] = {"st": st, "lg": round(lg, 2)}
            if poly is not None:
                e_item["poly"] = poly
            if col is not None:
                e_item["col"] = col
            fr.append(e_item)
        objs[str(oid)] = fr

    vj: dict[str, Any] = {
        "video_id": vid,
        "N": n_all,
        "stride": stride,
        "fidx": fidx,
        "fps": fps,
        "H": H,
        "W": W,
        "det_frames": di,
        "cam": cam,
        "objs": objs,
        "version": "v2",
    }
    for name, key in (("e", "queries"), ("d", "queries_d")):
        outs, anchors, resolved = cfg[name]
        ql = []
        for qi, q in enumerate(qs):
            o = dict(key=q["key"], phrase=q["phrase"], where=q["where"], rel=q.get("rel"), roles=q["roles"], **outs[qi])
            o["obj"] = alias.get((name, qi))
            ious = []
            if o["obj"] is not None:
                fr = objs[str(o["obj"])]
                for k, fi in enumerate(di):
                    st = fr[fi]["st"]
                    dd = dets_by_phrase[q["phrase"]][k]
                    if st is None or not dd:
                        continue
                    ious.append(max(tc.iou(st["box"], b) for b, _ in dd))
            o["det_iou"] = [round(x, 3) for x in ious]
            ql.append(o)
        vj[key] = ql
        vj["resolved_" + name] = resolved

    return vj, {
        "gdino": gd_meta,
        "owlv2": owl_meta,
        "sam2": sam_meta,
        "resolved_e": vj["resolved_e"],
        "resolved_d": vj["resolved_d"],
    }


def _build_track_v3nt(
    vid: str,
    frames: list[np.ndarray],
    n_all: int,
    stride: int,
    fidx: list[int],
    fps: float,
    cam: list[list[float]],
    vg_records: list[dict[str, Any]],
    sam2_dir: Path,
    dtype_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exact reproduction of geo/track_v3.py from Stage 1 vlm_ground_notext boxes."""
    import track_common as tc

    H, W = frames[0].shape[:2]
    di = np.unique(np.linspace(0, len(frames) - 1, min(16, len(frames))).round().astype(int)).tolist()

    def _valid_box(b: list[float]) -> list[float] | None:
        x0, y0, x1, y1 = b
        x0, x1 = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
        if x1 - x0 < 2 or y1 - y0 < 2 or (x1 - x0) * (y1 - y0) > 0.9 * W * H:
            return None
        return [x0, y0, x1, y1]

    Q: dict[str, dict[str, Any]] = {}
    for x in vg_records:
        q = Q.setdefault(
            x["key"],
            {
                "key": x["key"],
                "phrase": x["phrase"],
                "where": x["where"],
                "roles": x["roles"],
                "expr": x["expr"],
                "multi": x["multi"],
                "boxes": {},
            },
        )
        bl = [_valid_box(b) for b in x.get("boxes_rel1000", [])]
        q["boxes"][int(x["j"])] = [b for b in bl if b is not None]

    vframes = sorted({int(x["j"]) for x in vg_records})
    anchors: dict[str, tuple[int, list[float]]] = {}
    for k, q in Q.items():
        if q["multi"]:
            continue
        bs = [(j, q["boxes"][j][0]) for j in vframes if q["boxes"].get(j)]
        if not bs:
            continue
        ar = np.array([(b[2] - b[0]) * (b[3] - b[1]) for _, b in bs])
        i = int(np.argmin(np.abs(np.log(ar / np.median(ar)))))
        anchors[k] = bs[i]

    second: dict[str, dict[int, list[float] | None]] = {}
    for k, q in Q.items():
        if not q["multi"]:
            continue
        first = next((kk for kk, qq in Q.items() if not qq["multi"] and qq["phrase"] == q["phrase"]), None)
        best, per = None, {}
        for j in vframes:
            bl = q["boxes"].get(j) or []
            ref = (Q[first]["boxes"].get(j) or [None])[0] if first else None
            if ref is None and bl:
                ref = bl[0]
            cand = sorted([b for b in bl if tc.iou(b, ref) < 0.3], key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])) if ref else []
            per[j] = cand[0] if cand else None
            if cand:
                sz = (cand[0][2] - cand[0][0]) * (cand[0][3] - cand[0][1])
                if best is None or sz > best[2]:
                    best = (j, cand[0], sz)
        second[k] = per
        if best:
            anchors[k] = (best[0], best[1])

    cnt = {
        j: sum(1 for k in anchors if (second[k].get(j) if Q[k]["multi"] else (Q[k]["boxes"].get(j) or [None])[0]) is not None)
        for j in vframes
    }
    if cnt:
        jc = max(vframes, key=lambda j: (cnt[j], -abs(j - vframes[len(vframes) // 2])))
        for k in list(anchors):
            bc = second[k].get(jc) if Q[k]["multi"] else (Q[k]["boxes"].get(jc) or [None])[0]
            if bc is None:
                continue
            b0 = anchors[k][1]
            ra = ((bc[2] - bc[0]) * (bc[3] - bc[1])) / max((b0[2] - b0[0]) * (b0[3] - b0[1]), 1e-6)
            if 0.5 <= ra <= 2.0:
                anchors[k] = (jc, bc)

    uniq: list[tuple[int, int, list[float]]] = []
    alias: dict[str, int] = {}
    for k, (fi, b) in anchors.items():
        hit = next((u for u in uniq if u[1] == fi and tc.iou(u[2], b) > 0.9), None)
        if hit is None:
            uniq.append((len(uniq), fi, b))
            alias[k] = uniq[-1][0]
        else:
            alias[k] = hit[0]

    tr, sam_meta = _run_sam2_instrumented(frames, uniq, sam2_dir, dtype_profile=dtype_profile) if uniq else ({}, {"skipped": True})
    objs: dict[str, list[dict[str, Any]]] = {}
    for oid, _, _ in uniq:
        d = tr.get(oid, {})
        fr = []
        for j in range(len(frames)):
            st, lg, poly, col = d.get(j, (None, -99.0, None, None))
            e_item: dict[str, Any] = {"st": st, "lg": round(lg, 2)}
            if poly is not None:
                e_item["poly"] = poly
            if col is not None:
                e_item["col"] = col
            fr.append(e_item)
        objs[str(oid)] = fr

    ql = []
    for k, q in Q.items():
        o: dict[str, Any] = {"key": k, "phrase": q["phrase"], "where": q["where"], "rel": None, "roles": q["roles"], "expr": q["expr"]}
        vb = second[k] if q["multi"] else {j: (q["boxes"].get(j) or [None])[0] for j in vframes}
        o["obj"] = alias.get(k)
        ious = []
        if o["obj"] is not None:
            fr = objs[str(o["obj"])]
            for j in vframes:
                st = fr[j]["st"] if j < len(fr) else None
                if st is None or vb.get(j) is None:
                    continue
                ious.append(tc.iou(st["box"], vb[j]))
        agree = float(sum(x >= 0.5 for x in ious)) / max(len(vframes), 1)
        o.update(
            det_scores=[1.0 if vb.get(j) is not None else 0.0 for j in vframes],
            det_boxes=[[round(c, 1) for c in vb[j]] if vb.get(j) is not None else None for j in vframes],
            det_gd=[agree] * len(vframes),
            det_owl=[agree] * len(vframes),
            det_ngood=[1] * len(vframes),
            best_score=agree,
            best_gd=agree,
            best_owl=agree,
            anchor=[anchors[k][0], [round(c, 1) for c in anchors[k][1]]] if k in anchors else None,
            det_iou=[round(x, 3) for x in ious],
            vlm_frames=vframes,
        )
        ql.append(o)

    vj = {
        "video_id": vid,
        "N": n_all,
        "stride": stride,
        "fidx": fidx,
        "fps": fps,
        "H": H,
        "W": W,
        "det_frames": di,
        "cam": cam,
        "objs": objs,
        "version": "v2",
        "source": "vlm_qwen3vl8b",
        "queries": ql,
        "queries_d": ql,
        "resolved_e": [],
        "resolved_d": [],
    }
    return vj, {"sam2": sam_meta, "vlm_frames": vframes}


def run_track_stage_worker(work_json_path: Path) -> dict[str, Any]:
    """Single-GPU worker executing decode -> track_v1 -> affected_v2 -> track_v2 -> track_v3nt -> scene_ruler."""
    spec = read_json(work_json_path)
    run_dir = Path(spec["run_dir"])
    run_id = spec["run_id"]
    worker_name = spec["worker_name"]
    trace_enabled = bool(spec.get("trace_enabled", True))
    store = TraceStore(run_dir, run_id, enabled=trace_enabled)
    set_deterministic_seeds(int(spec.get("seed", 0)))
    dtype_profile = spec.get("dtype_profile", "t4_fp16")

    video_results: dict[str, Any] = {}
    worker_errors: list[dict[str, Any]] = []

    try:
        # Upstream geo/track.py executes main() at top-level import unless we extract pure helpers via AST;
        # load track_common & measure first, and install safe pure-CPU track_v1/v2 helper modules.
        mods = _load_upstream_geo_modules(UPSTREAM_DIR)
        tc = mods["tc"]
        m1 = mods["m1"]
        _install_safe_track_modules(UPSTREAM_DIR)

        gdino_dir = Path(spec["models"]["grounding_dino"])
        owl_dir = Path(spec["models"]["owlv2"])
        sam2_dir = Path(spec["models"]["sam2"])
        cache_dir = Path(spec["track_cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as setup_exc:
        tb = traceback.format_exc()
        err_rec = {
            "worker": worker_name,
            "stage": "track_worker_setup",
            "error": f"{type(setup_exc).__name__}: {setup_exc}",
            "traceback": tb,
            "run_dir": str(run_dir),
        }
        worker_errors.append(err_rec)
        store.log_event(
            stage="track_worker_setup",
            call_kind="tool_call",
            status="error",
            backend=f"transformers:{dtype_profile}",
            device=worker_name,
            error=err_rec["error"],
        )
        print(f"[FIRST WORKER ERROR] worker={worker_name} stage=track_worker_setup\n{tb}", file=sys.stderr, flush=True)
        receipt = {
            "status": "error",
            "worker_name": worker_name,
            "video_results": {},
            "errors": worker_errors,
            "completed_at": utc_now_iso(),
        }
        write_json_atomic(Path(spec["receipt_path"]), receipt)
        return receipt

    with FreshAntiCacheGuard(enabled=True):
        for v_item in spec["videos"]:
            vid = v_item["video_id"]
            vpath = Path(v_item["video_path"])
            fps = float(v_item["fps"])
            v1_recs = v_item["parse_v1_records"]
            v2_recs = v_item["parse_v2_records"]
            vg_records = v_item.get("vlm_ground_records") or []
            consumers = [int(r["id"]) for r in v2_recs]
            attempt_num = int(v_item.get("attempt", 1))
            attempt_dir = run_dir / "videos_trace" / vid / f"attempt-{attempt_num}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            # Input-keyed cache hash covering video SHA256 + queries + model revisions + dtype profile
            v_sha = sha256_file(vpath)
            cache_key = fingerprint({
                "video_sha256": v_sha,
                "fps": fps,
                "v1_queries": [(r["target"], r.get("target2"), r["prior_obj"]) for r in v1_recs],
                "v2_queries": [(r["target"], r.get("target2"), r["prior_obj"]) for r in v2_recs],
                "vg_records_hash": fingerprint(vg_records),
                "model_revisions": spec["model_revisions"],
                "dtype_profile": dtype_profile,
            })
            cached_bundle_path = cache_dir / f"{vid}_{cache_key[:16]}.json"
            if bool(spec.get("use_video_cache", True)) and cached_bundle_path.is_file():
                cached_data = read_json(cached_bundle_path)
                video_results[vid] = cached_data
                store.log_event(
                    stage="video_track_bundle",
                    call_kind="cache_hit",
                    status="ok",
                    video_id=vid,
                    consumers=consumers,
                    attempt=attempt_num,
                    backend="disk_cache",
                    device=worker_name,
                    extra={"cache_key": cache_key},
                )
                write_json_atomic(attempt_dir / "attempt.json", {"status": "complete", "cache_hit": True, "cache_key": cache_key})
                continue

            try:
                t_vid0 = time.perf_counter()
                # 1. Decode & preprocess
                t_dec0 = time.perf_counter()
                frames_all = tc.load_video(str(vpath))
                n_all = len(frames_all)
                stride = max(1, int(math.ceil(n_all / 160)))
                fidx = list(range(0, n_all, stride))
                frames = [frames_all[i] for i in fidx]
                del frames_all
                cam = tc.camera_motion(frames)
                dec_sec = time.perf_counter() - t_dec0

                preview_refs = {}
                if trace_enabled and frames:
                    for tag_name, idx_f in (("first", 0), ("mid", len(frames) // 2), ("last", len(frames) - 1)):
                        preview_refs[tag_name] = store.put_image_rgb(
                            frames[idx_f], kind="decoded_frame", meta={"video_id": vid, "frame_slot": idx_f, "fidx": fidx[idx_f]}
                        )
                store.log_event(
                    stage="decode_preprocess",
                    call_kind="tool_call",
                    status="ok",
                    video_id=vid,
                    consumers=consumers,
                    attempt=attempt_num,
                    backend="opencv",
                    device="cpu",
                    duration_sec=dec_sec,
                    output_refs=preview_refs,
                    extra={"N": n_all, "stride": stride, "sampled": len(frames), "W": frames[0].shape[1], "H": frames[0].shape[0]},
                )

                # 2. Track v1 (GroundingDINO + SAM 2.1)
                t0 = time.perf_counter()
                vj_v1, v1_meta = _build_track_v1(
                    vid, frames, n_all, stride, fidx, fps, cam, v1_recs, gdino_dir, sam2_dir, dtype_profile
                )
                v1_sec = time.perf_counter() - t0
                v1_ref = store.put_json(vj_v1, kind="track_v1")
                v1_meta_ref = store.put_json(v1_meta, kind="track_v1_meta")
                store.log_event(
                    stage="track_v1",
                    call_kind="model_call",
                    status="ok",
                    video_id=vid,
                    consumers=consumers,
                    attempt=attempt_num,
                    backend=f"transformers:{dtype_profile}",
                    device=worker_name,
                    duration_sec=v1_sec,
                    output_refs={"track_v1": v1_ref, "meta": v1_meta_ref},
                )

                # 3. Measure v1 + affected_v2 check (geo/measure.py + geo/affected_v2.py)
                allf = float(np.median(list(mods["mv3"].FOCAL.values())))
                fdd_v1 = {r["video_source"]: mods["mv3"].FOCAL.get(r["video_source"], allf) * vj_v1["W"] for r in v1_recs}
                m1_outs, _ = m1.measure_video(v1_recs, vj_v1, fdd_v1)
                m1_by_id = {int(x["id"]): x for x in m1_outs}
                aff_info = check_affected_v2_video(vj_v1, v2_recs, m1_by_id)

                # 4. Track v2 (GroundingDINO + OWLv2 + SAM 2.1) when affected (exactly as 02_track.sh line 12)
                vj_v2 = None
                v2_meta = None
                if aff_info["affected"]:
                    t0 = time.perf_counter()
                    vj_v2, v2_meta = _build_track_v2(
                        vid, frames, n_all, stride, fidx, fps, cam, v2_recs, gdino_dir, owl_dir, sam2_dir, dtype_profile
                    )
                    v2_sec = time.perf_counter() - t0
                    v2_ref = store.put_json(vj_v2, kind="track_v2")
                    v2_meta_ref = store.put_json(v2_meta, kind="track_v2_meta")
                    store.log_event(
                        stage="track_v2",
                        call_kind="model_call",
                        status="ok",
                        video_id=vid,
                        consumers=consumers,
                        attempt=attempt_num,
                        backend=f"transformers:{dtype_profile}",
                        device=worker_name,
                        duration_sec=v2_sec,
                        output_refs={"track_v2": v2_ref, "meta": v2_meta_ref},
                        extra={"affected_reasons": aff_info["reasons"]},
                    )
                else:
                    store.log_event(
                        stage="track_v2",
                        call_kind="skipped",
                        status="ok",
                        video_id=vid,
                        consumers=consumers,
                        attempt=attempt_num,
                        backend="rule_gate",
                        device="cpu",
                        extra={"reason": "not_in_affected_v2_video_list_uses_v1_fallback"},
                    )

                # 5. Track v3nt (SAM 2.1 from Stage 1 Qwen3-VL-8B boxes)
                vj_v3nt = None
                v3_meta = None
                if vg_records:
                    t0 = time.perf_counter()
                    vj_v3nt, v3_meta = _build_track_v3nt(
                        vid, frames, n_all, stride, fidx, fps, cam, vg_records, sam2_dir, dtype_profile
                    )
                    v3_sec = time.perf_counter() - t0
                    v3_ref = store.put_json(vj_v3nt, kind="track_v3nt")
                    store.log_event(
                        stage="track_v3nt",
                        call_kind="model_call",
                        status="ok",
                        video_id=vid,
                        consumers=consumers,
                        attempt=attempt_num,
                        backend=f"transformers:{dtype_profile}",
                        device=worker_name,
                        duration_sec=v3_sec,
                        output_refs={"track_v3nt": v3_ref},
                    )

                # 6. Scene ruler (CPU OpenCV Hough lines for lane/court priors)
                t0 = time.perf_counter()
                sr_res = run_scene_ruler_on_video(vid, vpath, [r["prior_text"] for r in v2_recs])
                sr_sec = time.perf_counter() - t0
                sr_clean = None
                if sr_res is not None:
                    ov_rgb = sr_res.pop("overlay_rgb", None)
                    sr_clean = sr_res
                    sr_refs: dict[str, Any] = {"scene_ruler": store.put_json(sr_clean, kind="scene_ruler")}
                    if ov_rgb is not None and trace_enabled:
                        sr_refs["overlay"] = store.put_image_rgb(ov_rgb, kind="scene_ruler_overlay", meta={"video_id": vid})
                    store.log_event(
                        stage="scene_ruler",
                        call_kind="tool_call",
                        status="ok",
                        video_id=vid,
                        consumers=consumers,
                        attempt=attempt_num,
                        backend="opencv_scipy",
                        device="cpu",
                        duration_sec=sr_sec,
                        output_refs=sr_refs,
                    )

                # 7. Render visual track overlays on key frames for failure analysis
                overlay_refs: dict[str, Any] = {}
                active_vj = vj_v2 if vj_v2 is not None else vj_v1
                if trace_enabled and frames and active_vj is not None:
                    key_slots = sorted({0, len(frames) // 2, len(frames) - 1})
                    for sl in key_slots:
                        items_draw = []
                        for q_entry in active_vj.get("queries", []):
                            fr_list_q = (
                                active_vj["objs"].get(str(q_entry.get("obj")), [])
                                if active_vj.get("version") == "v2" and q_entry.get("obj") is not None
                                else q_entry.get("frames", [])
                            )
                            if sl < len(fr_list_q) and fr_list_q[sl] and fr_list_q[sl].get("st"):
                                st_s = fr_list_q[sl]["st"]
                                role_s = "target" if "target" in q_entry.get("roles", []) else ("prior" if "prior" in q_entry.get("roles", []) else "other")
                                items_draw.append({
                                    "role": role_s,
                                    "label": q_entry.get("key", ""),
                                    "box": st_s["box"],
                                    "c": st_s["c"],
                                    "poly": fr_list_q[sl].get("poly"),
                                })
                        ov_img = render_track_overlay_rgb(frames[sl], int(fidx[sl]), fps, items_draw)
                        overlay_refs[f"frame_{fidx[sl]}"] = store.put_image_rgb(
                            ov_img, kind="track_overlay", meta={"video_id": vid, "fidx": int(fidx[sl])}
                        )

                bundle = {
                    "video_id": vid,
                    "cache_key": cache_key,
                    "affected_v2": aff_info,
                    "track_v1": vj_v1,
                    "track_v2": vj_v2,
                    "track_v3nt": vj_v3nt,
                    "scene_ruler": sr_clean,
                    "overlay_refs": overlay_refs,
                    "stage_metrics": {
                        "decode_sec": round(dec_sec, 4),
                        "total_video_sec": round(time.perf_counter() - t_vid0, 4),
                        "v1_meta": {k: v for k, v in v1_meta.items() if k != "candidates_by_phrase"},
                        "v2_meta": {k: v for k, v in (v2_meta or {}).items() if k not in ("candidates_by_phrase", "per_frame_candidates")},
                        "v3_meta": v3_meta,
                    },
                }
                if bool(spec.get("use_video_cache", True)):
                    write_json_atomic(cached_bundle_path, bundle)
                write_json_atomic(
                    attempt_dir / "attempt.json",
                    {
                        "status": "complete",
                        "exit_code": 0,
                        "cache_hit": False,
                        "cache_key": cache_key,
                        "overlay_refs": overlay_refs,
                        "completed_at": utc_now_iso(),
                    },
                )
                video_results[vid] = bundle
                del frames
                _free_cuda()

            except Exception as exc:
                tb = traceback.format_exc()
                err_rec = {
                    "video_id": vid,
                    "consumers": consumers,
                    "stage": "track_worker",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                    "attempt_dir": str(attempt_dir),
                }
                worker_errors.append(err_rec)
                (attempt_dir / "stderr.txt").write_text(tb, encoding="utf-8")
                write_json_atomic(
                    attempt_dir / "attempt.json",
                    {
                        "status": "error",
                        "exit_code": 1,
                        "error": err_rec["error"],
                        "stderr_path": str((attempt_dir / "stderr.txt").relative_to(run_dir).as_posix()),
                        "completed_at": utc_now_iso(),
                    },
                )
                store.log_event(
                    stage="track_worker",
                    call_kind="model_call",
                    status="error",
                    video_id=vid,
                    consumers=consumers,
                    attempt=attempt_num,
                    backend=f"transformers:{dtype_profile}",
                    device=worker_name,
                    error=err_rec["error"],
                )
                print(
                    f"[FIRST WORKER ERROR] worker={worker_name} video={vid} samples={consumers}\n"
                    f"artifact_dir={attempt_dir}\n{tb}",
                    file=sys.stderr,
                    flush=True,
                )

    receipt = {
        "status": "ok" if not worker_errors else "error",
        "worker_name": worker_name,
        "video_results": video_results,
        "errors": worker_errors,
        "completed_at": utc_now_iso(),
    }
    write_json_atomic(Path(spec["receipt_path"]), receipt)
    return receipt


def _extract_pure_ast_module(
    py_path: Path,
    mod_name: str,
    keep_assigns: set[str],
    keep_funcs: set[str],
    extra_globals: dict[str, Any],
) -> types.ModuleType:
    """Compile only specified pure-CPU constants and undecorated functions from an upstream script."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    selected: list[ast.stmt] = []
    found_funcs: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & keep_assigns:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in keep_funcs:
            if node.decorator_list:
                raise ValueError(f"Refusing decorated function {node.name} in {py_path}")
            selected.append(node)
            found_funcs.add(node.name)
    missing = keep_funcs - found_funcs
    if missing:
        raise RuntimeError(f"Missing expected helper functions {sorted(missing)} in {py_path}")
    mod_ast = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mod_ast)
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(py_path)
    mod.__dict__.update(extra_globals)
    exec(compile(mod_ast, str(py_path), "exec"), mod.__dict__)
    return mod


def _install_safe_track_modules(upstream_dir: Path) -> None:
    """Load pure selection helpers from geo/track.py and geo/track_v2.py via AST without executing top-level GPU code."""
    _load_upstream_geo_modules(upstream_dir)
    import track_common as tc

    if "track" not in sys.modules:
        sys.modules["track"] = _extract_pure_ast_module(
            py_path=upstream_dir / "geo" / "track.py",
            mod_name="track",
            keep_assigns={"COLOR_WORDS"},
            keep_funcs={"iou", "nms", "color_frac", "choose", "mask_stats", "camera_motion", "queries_for"},
            extra_globals={"re": re, "math": math, "np": np, "cv2": cv2},
        )

    if "track_v2" not in sys.modules:
        sys.modules["track_v2"] = _extract_pure_ast_module(
            py_path=upstream_dir / "geo" / "track_v2.py",
            mod_name="track_v2",
            keep_assigns={"COLOR_WORDS", "CONTRAST"},
            keep_funcs={"owl_support", "vote", "heads", "contrastive", "rel_pick", "choose", "queries_for", "select_anchors"},
            extra_globals={"re": re, "math": math, "np": np, "tc": tc},
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="IIIXR Kaggle GPU stage worker")
    ap.add_argument("stage", choices=["vlm", "track"])
    ap.add_argument("work_json", type=Path)
    args = ap.parse_args()
    if args.stage == "vlm":
        res = run_vlm_stage_worker(args.work_json)
    else:
        res = run_track_stage_worker(args.work_json)
    if res.get("status") != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
