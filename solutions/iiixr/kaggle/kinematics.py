"""Instrumented IIIXR 2D box/mask-centroid parser, scene ruler, kinematics, and CPU trace replay."""
from __future__ import annotations

import copy
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from solutions.iiixr.kaggle.schema import UPSTREAM_DIR

# Configure upstream environment flags from scripts/pipeline/04_geo_chain.sh BEFORE importing measure_v3.
GEO_CHAIN_ENV: dict[str, str] = {
    "QP_REJ_LO": "0.2",
    "QP_REJ_HI": "5",
    "QP_PRIOR_DRIFT": "1.5",
    "QP_PRIOR_ACCGUARD": "3",
    "QP_SCENE_HEAD": "1",
    "QP_TSHIFT": "1",
    "QP_DEPTH_DEDUP": "1",
    "QP_SCENE_SA2VA": "1",
    "QP_SECOND_FIX": "1",
    "QP_SAMECLASS": "1",
    "QP_SCENE_RULER": "1",
    "QP_FF_ROT": "1",
    "QP_SIDE_FIX": "1",
    "QP_FLOOR": "1",
    "QP_STATIC_CAP": "25",
    "QP_FF_SUB": "1",
    "QP_FF_SUB_SRC": "simulation",
    "QP_PARSE2": "1",
    "QP_NEAREST": "1",
    "QP_BELT": "1",
}


def apply_geo_chain_env() -> dict[str, str]:
    for k, v in GEO_CHAIN_ENV.items():
        os.environ[k] = v
    return dict(GEO_CHAIN_ENV)


def _load_upstream_geo_modules(upstream_dir: Path = UPSTREAM_DIR):
    apply_geo_chain_env()
    geo_dir = str(upstream_dir / "geo")
    tools_dir = str(upstream_dir / "tools")
    for p in (tools_dir, geo_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import measure as m1
    import measure_v3 as mv3
    import parse as parse_v1
    import parse_v2
    import qp_common as qpc
    import rules as rules_v1
    import rules_v2
    import track_common as tc

    mv3.FIX = set("abcdef")
    mv3.MOGE_OPTS = {"fbrej"}
    mv3.HALF_V = 0.25
    mv3.HALF_A = 0.4
    mv3.STATIC_FRAC = 0.25
    mv3.PRIOR_DRIFT = float(os.environ.get("QP_PRIOR_DRIFT", "1.5"))
    mv3.PRIOR_ACCGUARD = float(os.environ.get("QP_PRIOR_ACCGUARD", "3.0"))
    return {
        "rules_v1": rules_v1,
        "rules_v2": rules_v2,
        "parse_v1": parse_v1,
        "parse_v2": parse_v2,
        "qpc": qpc,
        "tc": tc,
        "m1": m1,
        "mv3": mv3,
    }


def _compute_h1_prior_copy(qpc: Any, question: str, inference_type: str, prior_text: str, pr0: dict[str, Any] | None) -> float:
    """Exact reproduction of tools/heuristic.py:h1_prior_copy (tau = 1.0s dimensional conversion)."""
    tu = qpc.target_unit(question, inference_type)
    si, k_from = qpc.prior_value(prior_text)
    if si is None and pr0 and pr0.get("value_si") is not None:
        si = float(pr0["value_si"])
        k_from = {"size": 0, "speed": 1, "accel": 2}.get(str(pr0.get("kind") or ""), 0)
    if si is None:
        return 1.0
    f_to, k_to = qpc.UNIT_TABLE[tu]
    if k_from == k_to:
        conv_si = si
    elif (k_from, k_to) == (0, 1):
        conv_si = si / 1.0
    elif (k_from, k_to) == (1, 0):
        conv_si = si * 1.0
    elif (k_from, k_to) == (2, 0):
        conv_si = 0.5 * si * (1.0**2)
    elif (k_from, k_to) == (0, 2):
        conv_si = 2.0 * si / (1.0**2)
    elif (k_from, k_to) == (1, 2):
        conv_si = si / 1.0
    elif (k_from, k_to) == (2, 1):
        conv_si = si * 1.0
    else:
        conv_si = si
    return abs(float(conv_si / f_to))


def parse_questions_v1_v2(
    rows: list[dict[str, Any]],
    llm_outputs_by_id: dict[int, dict[str, Any]] | None = None,
    upstream_dir: Path = UPSTREAM_DIR,
) -> dict[str, Any]:
    """Run upstream geo/parse.py + geo/parse_v2.py + measure_v3 post-parse rules with full trace."""
    mods = _load_upstream_geo_modules(upstream_dir)
    rules_v1 = mods["rules_v1"]
    rules_v2 = mods["rules_v2"]
    parse_v1 = mods["parse_v1"]
    parse_v2 = mods["parse_v2"]

    llm_map = llm_outputs_by_id or {}
    v1_records: dict[int, dict[str, Any]] = {}
    v2_records: dict[int, dict[str, Any]] = {}
    traces_by_id: dict[int, dict[str, Any]] = {}

    for row in rows:
        assert "ground_truth_posterior" not in row, "Ground truth leaked into parser input!"
        rid = int(row["id"])
        vid = str(row["video_id"]).strip()
        vsrc = str(row["video_source"]).strip().replace("segmentaion", "segmentation")
        vtype = str(row["video_type"]).strip()
        itype = str(row["inference_type"]).strip()
        fps = float(row["fps"])
        cat = f"{itype[0]}{vtype[1]}"
        question = str(row["question"])
        prior_text = str(row["ground_truth_prior"])
        depth_info = str(row.get("depth_info") or "")

        # --- V1 Parse (geo/parse.py) ---
        q1 = rules_v1.parse_question(question, itype)
        pr1 = rules_v1.parse_prior(prior_text)
        dep1 = rules_v1.parse_depth(depth_info) if len(vtype) >= 2 and vtype[1] == "3" else []
        dkeys1 = sorted({d["key"] for d in dep1})
        llm_entry = llm_map.get(rid, {})
        L = llm_entry.get("parsed") if "parsed" in llm_entry else dict(llm_entry)

        tp1, where1 = parse_v1.det_phrase(q1["target"])
        src1 = "rule"
        if (tp1 is None or len(tp1.split()) > 3) and L.get("target"):
            tp1, src1 = str(L["target"]).lower().strip(), "llm"
        if not where1 and L.get("target_where") and str(L["target_where"]).lower() not in ("none", ""):
            where1 = [w for w in parse_v1.SPATIAL if re.search(r"\b" + w + r"\b", str(L["target_where"]).lower())][:2]
        t2_1 = None
        if q1["target2"] == "#second":
            t2_1 = {"phrase": tp1, "where": ["#second"], "src": "rule", "raw": q1["target"], "rel": None}
        elif q1["target2"]:
            p2_1, w2_1 = parse_v1.det_phrase(q1["target2"])
            s2_1 = "rule"
            if (p2_1 is None or len(p2_1.split()) > 3) and L.get("target2"):
                p2_1, s2_1 = str(L["target2"]).lower().strip(), "llm"
            t2_1 = {"phrase": p2_1, "where": w2_1, "src": s2_1, "raw": q1["target2"], "rel": None}
        elif q1["kind"] == "dist2" and L.get("target2"):
            t2_1 = {"phrase": str(L["target2"]).lower().strip(), "where": [], "src": "llm", "raw": None, "rel": None}

        p0_1 = pr1[0] if pr1 else {}
        pp1, pw1 = parse_v1.det_phrase(p0_1.get("obj"))
        psrc1 = "rule"
        if p0_1.get("gravity") and not pp1:
            pp1 = None
        elif (pp1 is None or len(pp1.split()) > 3) and L.get("prior_object") and str(L["prior_object"]).lower() != "gravity":
            pp1, psrc1 = str(L["prior_object"]).lower().strip(), "llm"

        rec_v1 = {
            "id": rid,
            "video_id": vid,
            "video_source": vsrc,
            "video_type": vtype,
            "inference_type": itype,
            "fps": fps,
            "cat": cat,
            "question": question,
            "prior_text": prior_text,
            "q": q1,
            "priors": pr1,
            "depth": dep1,
            "depth_keys": dkeys1,
            "llm": L,
            "target": {"phrase": tp1, "where": where1, "src": src1, "raw": q1["target"], "rel": None},
            "target2": t2_1,
            "prior_obj": {"phrase": pp1, "where": pw1, "src": psrc1, "raw": p0_1.get("obj"), "rel": None},
            "target_dkeys": [],
            "target_dsrc": "none",
            "target2_dkeys": [],
            "prior_dkeys": [],
            "prior_dsrc": "none",
            "target_whole": bool(L.get("target_whole", True)) if L else True,
            "prior_whole": bool(L.get("prior_whole", True)) if L else True,
        }
        v1_records[rid] = rec_v1

        # --- V2 Parse (geo/parse_v2.py) ---
        q2 = rules_v2.parse_question(question, itype)
        pr2 = rules_v2.parse_prior(prior_text)
        dep2 = rules_v2.parse_depth(depth_info) if len(vtype) >= 2 and vtype[1] == "3" else []
        dkeys2 = sorted({d["key"] for d in dep2})
        fixes = list(q2.get("fixes", []))

        tp2, where2 = parse_v2.det_phrase(q2["target"])
        src2 = "rule"
        if (tp2 is None or len(tp2.split()) > 3) and L.get("target"):
            tp2, src2 = str(L["target"]).lower().strip(), "llm"
            fx = rules_v2.fix_object_phrase(tp2)
            if fx["changed"]:
                tp2 = fx["phrase"]
                fixes += ["llm_target:" + c for c in fx["changed"]]
                if fx["rel"] and not q2.get("target_rel"):
                    q2["target_rel"] = [fx["rel"], fx["ref"]]
        if not where2 and L.get("target_where") and str(L["target_where"]).lower() not in ("none", ""):
            where2 = [w for w in parse_v1.SPATIAL if re.search(r"\b" + w + r"\b", str(L["target_where"]).lower())][:2]
        if q2.get("target_rel"):
            drop = {
                "near": ("closest", "nearest", "near", "front", "back"),
                "right_of": ("right",),
                "left_of": ("left",),
                "below": ("lower", "bottom"),
                "above": ("upper", "top"),
            }.get(q2["target_rel"][0], ())
            where2 = [w for w in where2 if w not in drop]
        tgt2_obj = {"phrase": tp2, "where": where2, "src": src2, "raw": q2["target"], "rel": q2.get("target_rel")}

        t2_2 = None
        if q2["target2"] == "#second":
            t2_2 = {"phrase": tp2, "where": ["#second"], "src": "rule", "raw": q2["target"], "rel": None}
        elif q2["target2"]:
            p2_2, w2_2 = parse_v2.det_phrase(q2["target2"])
            s2_2 = "rule"
            if (p2_2 is None or len(p2_2.split()) > 3) and L.get("target2"):
                p2_2, s2_2 = str(L["target2"]).lower().strip(), "llm"
            t2_2 = {"phrase": p2_2, "where": w2_2, "src": s2_2, "raw": q2["target2"], "rel": q2.get("target2_rel")}
        elif q2["kind"] == "dist2" and L.get("target2"):
            t2_2 = {"phrase": str(L["target2"]).lower().strip(), "where": [], "src": "llm", "raw": None, "rel": None}

        p0_2 = pr2[0] if pr2 else {}
        pobj_raw = p0_2.get("obj")
        prel = None
        if pobj_raw:
            fxp = rules_v2.fix_object_phrase(pobj_raw)
            if fxp["changed"]:
                pobj_raw = fxp["phrase"]
                fixes += ["prior:" + c for c in fxp["changed"]]
                prel = [fxp["rel"], fxp["ref"]] if fxp["rel"] else None
        pp2, pw2 = parse_v2.det_phrase(pobj_raw)
        psrc2 = "rule"
        if p0_2.get("gravity") and not pp2:
            pp2 = None
        elif (pp2 is None or len(pp2.split()) > 3) and L.get("prior_object") and str(L["prior_object"]).lower() != "gravity":
            pp2, psrc2 = str(L["prior_object"]).lower().strip(), "llm"
        prior_obj_2 = {"phrase": pp2, "where": pw2, "src": psrc2, "raw": p0_2.get("obj"), "rel": prel}

        def _sig(e: dict[str, Any] | None):
            return None if not e else (e.get("phrase"), tuple(e.get("where") or []), tuple(e.get("rel") or []))

        v2_changed = (
            (_sig(tgt2_obj) != _sig(rec_v1["target"]))
            or (_sig(t2_2) != _sig(rec_v1.get("target2")))
            or (_sig(prior_obj_2) != _sig(rec_v1["prior_obj"]))
            or (q2["kind"] != rec_v1["q"]["kind"])
            or (q2.get("dim") != rec_v1["q"].get("dim"))
            or bool(q2.get("part"))
        )

        rec_v2 = {
            "id": rid,
            "video_id": vid,
            "video_source": vsrc,
            "video_type": vtype,
            "inference_type": itype,
            "fps": fps,
            "cat": cat,
            "question": question,
            "prior_text": prior_text,
            "q": q2,
            "priors": pr2,
            "depth": dep2,
            "depth_keys": dkeys2,
            "llm": L,
            "target": tgt2_obj,
            "target2": t2_2,
            "prior_obj": prior_obj_2,
            "target_dkeys": [],
            "target_dsrc": "none",
            "target2_dkeys": [],
            "prior_dkeys": [],
            "prior_dsrc": "none",
            "target_whole": bool(L.get("target_whole", True)) if L else True,
            "prior_whole": bool(L.get("prior_whole", True)) if L else True,
            "v2_changed": bool(v2_changed),
            "v2_fixes": fixes,
        }
        # Apply load_split_tracks post-parse rules (QP_SIDE_FIX, QP_FLOOR, QP_PARSE2)
        rec_v2 = apply_post_parse_fixes(rec_v2, mods["m1"].SCENE)
        v2_records[rid] = rec_v2

        traces_by_id[rid] = {
            "id": rid,
            "video_id": vid,
            "raw_question": question,
            "cleaned_question": rules_v1.clean_text(question),
            "raw_prior": prior_text,
            "depth_info": depth_info,
            "unit": rec_v2["q"]["unit"],
            "unit_si_factor": float(rules_v1.UNIT_SI[rec_v2["q"]["unit"]]),
            "kind": rec_v2["q"]["kind"],
            "dim": rec_v2["q"].get("dim"),
            "timestamps": {
                "t": rec_v2["q"].get("t"),
                "t1": rec_v2["q"].get("t1"),
                "t2": rec_v2["q"].get("t2"),
                "when": rec_v2["q"].get("when"),
            },
            "target": rec_v2["target"],
            "target2": rec_v2.get("target2"),
            "prior_parsed": rec_v2["priors"][0] if rec_v2["priors"] else None,
            "prior_obj": rec_v2["prior_obj"],
            "rule_vs_llm_sources": {
                "target_src": rec_v2["target"]["src"],
                "target2_src": rec_v2["target2"]["src"] if rec_v2.get("target2") else None,
                "prior_src": rec_v2["prior_obj"]["src"],
                "llm_invoked": bool(llm_entry),
                "llm_meta": {k: v for k, v in llm_entry.items() if k != "parsed"} if llm_entry else None,
            },
            "v2_changed": rec_v2["v2_changed"],
            "v2_fixes": rec_v2["v2_fixes"],
            "post_parse_fixes": [k for k in ("side_fix", "floor_fix", "parse2_fix") if rec_v2.get(k)],
        }

    return {
        "v1_records": v1_records,
        "v2_records": v2_records,
        "traces_by_id": traces_by_id,
    }


def apply_post_parse_fixes(r_in: dict[str, Any], scene_regex: str) -> dict[str, Any]:
    """Reproduce geo/measure_v3.py load_split_tracks() post-parse rules (lines 1501-1548)."""
    r = copy.deepcopy(r_in)
    r["video_source"] = r["video_source"].replace("segmentaion", "segmentation")
    if os.environ.get("QP_SIDE_FIX", "1") == "1" and r.get("q", {}).get("kind") == "size":
        ql_ = r["question"].lower()
        tp_ = ((r.get("target") or {}).get("phrase") or "").strip().lower()
        m_ = re.match(r"^(?:the\s+)?(longer|shorter|long|short|longest|shortest)\s+(?:side|edge)$", tp_)
        mo_ = re.search(
            r"\b(?:longer|shorter|long|short|longest|shortest)\s+(?:side|edge)\s+of\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z \-]*?)\s*(?:\bin\b|,|\?|$)",
            ql_,
        )
        if m_ and mo_ and not re.search(scene_regex, mo_.group(1)):
            r["target"] = dict(r["target"], phrase=mo_.group(1).strip(), raw=mo_.group(1).strip())
            r["q"] = dict(r["q"], dim="length" if m_.group(1).startswith("long") else "width")
            r["side_fix"] = "target"
        elif re.search(r"\bwidth\s*\(\s*(?:the\s+)?long(?:er|est)?\s+side\s*\)", ql_) and r["q"].get("dim") == "width":
            r["q"] = dict(r["q"], dim="length")
            r["side_fix"] = "dim"
        elif tp_ in ("rim", "the rim") and r["q"].get("dim") in ("diameter", "width", "radius"):
            mr_ = re.search(r"\brim of (?:the\s+|a\s+)?([a-z][a-z \-]*?)\s*(?:\bin\b|,|\?|$)", ql_)
            if mr_ and not re.search(scene_regex, mr_.group(1)):
                r["target"] = dict(r["target"], phrase=mr_.group(1).strip(), raw=mr_.group(1).strip())
                r["q"] = dict(r["q"], part=None)
                r["side_fix"] = "rim"
    if os.environ.get("QP_FLOOR", "1") == "1" and r.get("q", {}).get("kind") == "other":
        mf_ = re.search(
            r"\bdistance (?:between|of|from) (?:the )?(?:(?:bottom|lowest point|base)(?: corner)? of (?:the )?)?([a-z][a-z \-]*?) "
            r"(?:and|to|from|above) the (?:floor|ground)(?: at (\d+(?:\.\d+)?)\s*s)?\b",
            r["question"].lower(),
        )
        if mf_ and mf_.group(2) and not re.search(scene_regex, mf_.group(1)):
            r["q"] = dict(r["q"], kind="hfloor", t=float(mf_.group(2)), target=mf_.group(1).strip())
            r["target"] = dict(
                r.get("target") or {},
                phrase=mf_.group(1).strip(),
                raw=mf_.group(1).strip(),
                where=(r.get("target") or {}).get("where") or [],
            )
            r["target2"] = None
            r["target2_dkeys"] = []
            r["floor_fix"] = True
    if os.environ.get("QP_PARSE2", "1") == "1":
        q_, ql_ = r.get("q") or {}, r["question"].lower()
        ob_ = r"(?:the\s+)?([a-z][a-z \-]*?)"
        m1_ = re.search(r"\btotal distance (?:that )?" + ob_ + r" (?:walked|traveled|travelled|moved|ran|flew|swam|covered)\b", ql_)
        m2_ = re.search(r"\btotal length of " + ob_ + r"(?:['’]s)? (?:flight |travel |motion )?path\b", ql_)
        m3_ = re.search(r"\bhorizontal distance of " + ob_ + r" during\b", ql_)
        new_ = None
        if q_.get("kind") in ("path", "other") and (m1_ or m2_):
            new_ = ("path", (m1_ or m2_).group(1))
        elif q_.get("kind") == "dist2" and not q_.get("target") and m3_:
            new_ = ("disp", m3_.group(1))
        elif (
            q_.get("kind") == "other"
            and q_.get("t") is not None
            and r.get("target2")
            and (r.get("target") or {}).get("phrase")
            and re.search(r"\bdistance between\b", ql_)
            and not re.search(r"\b(?:floor|ground)\b", ql_)
        ):
            new_ = ("dist2", None)
        m5_ = re.search(
            r"\b(?:the\s+)?([a-z][a-z \-]*?)(?:['’]s)? moving distance between (\d+(?:\.\d+)?)\s*s?\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*s\b",
            ql_,
        )
        if new_ is None and q_.get("kind") == "dist2" and not ((r.get("target2") or {}).get("phrase")) and m5_:
            q_ = dict(q_, t1=float(m5_.group(2)), t2=float(m5_.group(3)), t=None)
            r["target2"] = None
            r["target2_dkeys"] = []
            new_ = ("path", m5_.group(1))
        if new_ is not None and (new_[1] is None or not re.search(scene_regex, new_[1])):
            r["q"] = dict(q_, kind=new_[0], **({"target": new_[1].strip()} if new_[1] else {}))
            if new_[1]:
                r["target"] = dict(
                    r.get("target") or {},
                    phrase=new_[1].strip(),
                    raw=new_[1].strip(),
                    where=(r.get("target") or {}).get("where") or [],
                )
            r["parse2_fix"] = new_[0]
    return r


def check_affected_v2_video(
    v1_track_json: dict[str, Any] | None,
    v2_parse_recs_for_video: list[dict[str, Any]],
    v1_meas_rows_by_id: dict[int, dict[str, Any]],
    upstream_dir: Path = UPSTREAM_DIR,
) -> dict[str, Any]:
    """Exact reproduction of geo/affected_v2.py for one video."""
    mods = _load_upstream_geo_modules(upstream_dir)
    m1 = mods["m1"]
    reasons: set[str] = set()
    if any(r.get("v2_changed") for r in v2_parse_recs_for_video):
        reasons.add("A1_parse")
    if v1_track_json is None:
        reasons.add("no_v1_track")
        return {"affected": True, "reasons": sorted(reasons)}
    for q in v1_track_json.get("queries", []):
        ious = q.get("det_iou") or []
        if q.get("best_score", 0) < 0.3 or (ious and float(np.median(ious)) < 0.5):
            reasons.add("A2_det")
    if any(str((v1_meas_rows_by_id.get(r["id"]) or {}).get("why")) in ("target=prior_object", "target=target2") for r in v2_parse_recs_for_video):
        reasons.add("A3_coll")
    qs = [q for q in v1_track_json.get("queries", []) if q.get("anchor")]
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            a, b = qs[i]["anchor"], qs[j]["anchor"]
            if a[0] != b[0]:
                continue
            x, y = a[1], b[1]
            ix = max(0, min(x[2], y[2]) - max(x[0], y[0])) * max(0, min(x[3], y[3]) - max(y[1], x[1]))
            ua = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - ix
            pa = qs[i]["phrase"] + " " + " ".join(qs[i]["where"])
            pb = qs[j]["phrase"] + " " + " ".join(qs[j]["where"])
            if ua > 0 and ix / ua > 0.9 and m1.contrastive(pa, pb):
                reasons.add("A3_coll")
    if any(r["q"]["kind"] == "dist2" for r in v2_parse_recs_for_video):
        reasons.add("A4_dist")
    return {"affected": bool(reasons), "reasons": sorted(reasons)}


def run_scene_ruler_on_video(
    video_id: str,
    video_path: Path,
    prior_texts: list[str],
    nframes: int = 8,
    maxdim: int = 1920,
) -> dict[str, Any] | None:
    """Reproduce geo/scene_ruler.py on a single video without requiring caw/analyze_caw.py."""
    kinds = [
        ("lane", r"\blane width\b", r"\blanes?\b"),
        ("tennis_singles", r"\bsingles court width\b", r"court"),
        ("pickleball_width", r"\bwidth of the pickleball court\b|\bpickleball court width\b", r"court"),
        ("basketball_key", r"\b(?:inner )?court width\s*=\s*4\.8[0-9]?\s*m", r"court"),
        ("soccer_circle", r"\bcent(?:er|re) circle radius\b", r"circle"),
    ]
    matched = None
    for ptxt in prior_texts:
        for k, rx, prx in kinds:
            if re.search(rx, str(ptxt), re.I):
                matched = (k, prx)
                break
        if matched is not None:
            break
    if matched is None:
        return None

    kind, prior_re = matched
    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_indices = np.linspace(0, max(n_total - 1, 0), nframes).astype(int).tolist()
    fr_list: list[np.ndarray] = []
    for k in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, fr = cap.read()
        if not ok:
            continue
        h0, w0 = fr.shape[:2]
        s = min(1.0, maxdim / max(h0, w0))
        if s < 1.0:
            fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        fr_list.append(fr)
    cap.release()
    if not fr_list or kind != "lane":
        return {"ok": False, "kind": kind, "prior_re": prior_re, "why": "unsupported_or_empty"}

    h, w = fr_list[0].shape[:2]
    segs_per_frame = []
    for fr in fr_list:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        ksz = max(9, int(round(0.012 * max(g.shape))) | 1)
        th_img = cv2.morphologyEx(
            g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ksz, ksz))
        ).astype(np.float32)
        med = float(np.median(th_img))
        mad = float(np.median(np.abs(th_img - med))) * 1.4826
        thr = max(14.0, med + 5 * mad)
        mk = (th_img > thr).astype(np.uint8)
        lines_p = cv2.HoughLinesP(
            mk * 255,
            1,
            np.pi / 360,
            threshold=12,
            minLineLength=max(8, int(0.008 * max(mk.shape))),
            maxLineGap=3,
        )
        segs_per_frame.append(np.zeros((0, 4)) if lines_p is None else lines_p.reshape(-1, 4).astype(float))

    allseg = np.concatenate(segs_per_frame) if segs_per_frame else np.zeros((0, 4))
    hist = np.zeros(360)
    for x1, y1, x2, y2 in allseg:
        ang = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180) % 180
        hist[int(ang * 2) % 360] += math.hypot(x2 - x1, y2 - y1)
    if hist.sum() == 0:
        return {"ok": False, "kind": "lane", "prior_re": prior_re, "why": "no_lines"}

    hs = sum(np.roll(hist, k) for k in range(-3, 4))
    i_max = int(np.argmax(hs))
    idx_win = np.arange(i_max - 3, i_max + 4)
    w_win = hist[idx_win % 360]
    theta = float((np.sum(idx_win * w_win) / max(w_win.sum(), 1e-9)) / 2.0) % 180

    size = int(math.hypot(w, h)) + 4
    th_rad = math.radians(theta)
    prof = np.zeros(size)
    off = size / 2
    for segs in segs_per_frame:
        for x1, y1, x2, y2 in segs:
            ang = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 180) % 180
            d = abs((ang - theta + 90) % 180 - 90)
            if d > 3.0:
                continue
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            yc = -xm * math.sin(th_rad) + ym * math.cos(th_rad) + off
            if 0 <= yc < size - 1:
                f = yc - int(yc)
                ln = math.hypot(x2 - x1, y2 - y1)
                prof[int(yc)] += ln * (1 - f)
                prof[int(yc) + 1] += ln * f

    p_smooth = gaussian_filter1d(prof, 1.0)
    pk0, _ = find_peaks(p_smooth, prominence=0.04 * p_smooth.max(), distance=4)
    min_gap = max(4.0, 0.006 * max(w, h))
    out_x: list[list[int]] = []
    for x in pk0:
        if out_x and x - out_x[-1][-1] < min_gap:
            out_x[-1].append(int(x))
        else:
            out_x.append([int(x)])
    pk = np.array([np.average(g, weights=[p_smooth[i] for i in g]) for g in out_x]) if out_x else np.zeros(0)
    wt = np.array([sum(p_smooth[i] for i in g) for g in out_x]) if out_x else np.zeros(0)

    cands: list[tuple[tuple[int, float], float, np.ndarray]] = []
    best = None
    min_s = max(10.0, 0.008 * max(w, h))
    pts_l = list(pk)
    for i in range(len(pts_l)):
        for j in range(i + 1, len(pts_l)):
            s = pts_l[j] - pts_l[i]
            if s < min_s:
                continue
            mem = [i, j]
            while True:
                nxt = pts_l[mem[-1]] + s
                c = [k for k in range(mem[-1] + 1, len(pts_l)) if abs(pts_l[k] - nxt) <= 0.08 * s]
                if not c:
                    break
                mem.append(min(c, key=lambda k: abs(pts_l[k] - nxt)))
            while True:
                prv = pts_l[mem[0]] - s
                c = [k for k in range(0, mem[0]) if abs(pts_l[k] - prv) <= 0.08 * s]
                if not c:
                    break
                mem.insert(0, min(c, key=lambda k: abs(pts_l[k] - prv)))
            pos = np.array([pts_l[k] for k in mem], float)
            sp = float(np.median(np.diff(pos)))
            strong = [pts_l[k] for k in range(len(pts_l)) if wt[k] >= 0.25 * max(wt[m] for m in mem)]
            if any(pos[q] + 0.25 * sp < x < pos[q + 1] - 0.25 * sp for q in range(len(pos) - 1) for x in strong):
                continue
            score = (len(mem), -sp)
            cands.append((score, sp, pos))
            if best is None or score > best[0]:
                best = (score, sp, pos)

    if best is None or best[0][0] < 3:
        return {"ok": False, "kind": "lane", "prior_re": prior_re, "why": "no_progression", "theta": theta, "npk": int(len(pk))}

    (cnt, _), sp, pos = best
    d = np.diff(pos)
    bands = [(float(pos[0]), float(pos[-1]))]
    for sc_, sp_, pos_ in sorted(cands, key=lambda c: c[0], reverse=True):
        if sc_[0] < 3 or abs(sp_ - sp) > 0.08 * sp:
            continue
        lo_, hi_ = float(pos_[0]), float(pos_[-1])
        if all(hi_ < b[0] - 0.25 * sp or lo_ > b[1] + 0.25 * sp for b in bands):
            bands.append((lo_, hi_))

    mid_rgb = cv2.cvtColor(fr_list[len(fr_list) // 2].copy(), cv2.COLOR_BGR2RGB)
    for yc in pos:
        c_val = yc - off
        pts_draw = []
        for x_coord in (0, w - 1):
            if abs(math.cos(th_rad)) > 1e-6:
                pts_draw.append((x_coord, (c_val + x_coord * math.sin(th_rad)) / math.cos(th_rad)))
        if abs(math.cos(th_rad)) < 0.5:
            pts_draw = [((y_coord * math.cos(th_rad) - c_val) / math.sin(th_rad), y_coord) for y_coord in (0, h - 1)]
        cv2.line(
            mid_rgb,
            tuple(int(round(v)) for v in pts_draw[0]),
            tuple(int(round(v)) for v in pts_draw[1]),
            (255, 64, 64),
            2,
        )

    return {
        "video_id": video_id,
        "ok": True,
        "kind": "lane",
        "prior_re": prior_re,
        "px": float(sp),
        "W": int(w),
        "H": int(h),
        "frac_w": float(sp / w),
        "n": int(cnt),
        "npk": int(len(pk)),
        "spread": float(np.std(d) / sp),
        "theta": float(theta),
        "lines": [float(x - off) for x in pos],
        "lines_w": [float((x - off) / w) for x in pos],
        "bands_w": [[float((b[0] - off) / w), float((b[1] - off) / w)] for b in bands],
        "sampled_frame_indices": sampled_indices,
        "hough_segment_counts": [int(len(s)) for s in segs_per_frame],
        "overlay_rgb": mid_rgb,
    }


def _track_arrays_dict(trk: Any, vj: dict[str, Any]) -> dict[str, np.ndarray]:
    """Serialize lossless per-timestamp numerical arrays from an upstream Track object."""
    n_total = len(vj["fidx"])
    fidx = np.asarray(vj["fidx"], dtype=np.int64)
    t_all = fidx.astype(np.float64) / float(vj["fps"])
    c_arr = np.full((n_total, 2), np.nan, dtype=np.float64)
    box_arr = np.full((n_total, 4), np.nan, dtype=np.float64)
    rr_arr = np.full((n_total, 3), np.nan, dtype=np.float64)
    area_arr = np.zeros(n_total, dtype=np.float64)
    border_arr = np.full(n_total, 9, dtype=np.int64)
    lg_arr = np.full(n_total, -99.0, dtype=np.float64)

    if trk is not None:
        frames = trk.q.get("frames") or []
        for j in range(min(n_total, len(frames))):
            fr = frames[j] or {}
            lg_arr[j] = float(fr.get("lg", -99.0))
            st = fr.get("st")
            if st is not None:
                c_arr[j] = np.asarray(st["c"], dtype=np.float64)
                box_arr[j] = np.asarray(st["box"], dtype=np.float64)
                rr_arr[j] = np.asarray(st["rr"], dtype=np.float64)
                area_arr[j] = float(st["area"])
                border_arr[j] = int(st.get("border", 0))
    return {
        "fidx": fidx,
        "t_all": t_all,
        "c": c_arr,
        "box": box_arr,
        "rr": rr_arr,
        "area": area_arr,
        "border": border_arr,
        "lg": lg_arr,
        "okc": np.asarray(trk.okc if trk is not None else [], dtype=np.int64),
        "oke": np.asarray(trk.oke if trk is not None else [], dtype=np.int64),
    }


def _extract_query_meta_for_replay(q_dict: dict[str, Any] | None) -> dict[str, Any] | None:
    """Serialize Track.q metadata plus sparse per-slot polygon/color dicts for lossless CPU replay."""
    if q_dict is None:
        return None
    meta = {k: v for k, v in q_dict.items() if k != "frames"}
    frames = q_dict.get("frames") or []
    col_by_slot: dict[str, Any] = {}
    poly_by_slot: dict[str, Any] = {}
    for j, fr in enumerate(frames):
        if not isinstance(fr, dict):
            continue
        if fr.get("col") is not None:
            col_by_slot[str(j)] = fr["col"]
        if fr.get("poly") is not None:
            poly_by_slot[str(j)] = fr["poly"]
    if col_by_slot:
        meta["_replay_col_by_slot"] = col_by_slot
    if poly_by_slot:
        meta["_replay_poly_by_slot"] = poly_by_slot
    return meta


def _reconstruct_query_from_arrays(arrs: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Any]:
    """Rebuild upstream Track query dict purely from lossless numerical arrays + metadata."""
    n = len(arrs["fidx"])
    col_map = meta.get("_replay_col_by_slot") or {}
    poly_map = meta.get("_replay_poly_by_slot") or {}
    frames = []
    for j in range(n):
        if arrs["area"][j] > 0 and np.isfinite(arrs["c"][j]).all():
            st = {
                "area": int(round(float(arrs["area"][j]))),
                "box": [int(round(float(v))) for v in arrs["box"][j]],
                "c": [float(arrs["c"][j, 0]), float(arrs["c"][j, 1])],
                "rr": [float(arrs["rr"][j, 0]), float(arrs["rr"][j, 1]), float(arrs["rr"][j, 2])],
                "border": int(arrs["border"][j]),
                "bottom": float(arrs["box"][j, 3]),
                "top": float(arrs["box"][j, 1]),
            }
        else:
            st = None
        fr_entry: dict[str, Any] = {"st": st, "lg": float(arrs["lg"][j])}
        if str(j) in col_map:
            fr_entry["col"] = col_map[str(j)]
        if str(j) in poly_map:
            fr_entry["poly"] = poly_map[str(j)]
        frames.append(fr_entry)
    q = {k: v for k, v in meta.items() if k not in ("_replay_col_by_slot", "_replay_poly_by_slot")}
    q["frames"] = frames
    return q


def _fit_detail(mv3: Any, m1: Any, ts: np.ndarray, Ps: np.ndarray, t0c: float, half: float, deg: int, minpts: int) -> dict[str, Any] | None:
    """Capture exact local polynomial fit window, frame mask, degree, and coefficients."""
    used_h = None
    m = np.zeros(len(ts), dtype=bool)
    for h in (half, half * 1.6, half * 2.5):
        cand_m = np.abs(ts - t0c) <= h + 1e-9
        if int(cand_m.sum()) >= minpts:
            used_h = float(h)
            m = cand_m
            break
    if used_h is None or int(m.sum()) < minpts:
        return None
    tt = ts[m] - t0c
    eff_deg = int(min(deg, int(m.sum()) - 1))
    coeffs = [np.polyfit(tt, Ps[m, k], eff_deg).tolist() for k in range(Ps.shape[1])]
    return {
        "t0c": float(t0c),
        "half_requested": float(half),
        "half_used": float(used_h),
        "minpts": int(minpts),
        "deg": eff_deg,
        "n_points_in_window": int(m.sum()),
        "window_timestamps": ts[m].tolist(),
        "local_mask_indices": np.where(m)[0].tolist(),
        "coeffs_x": coeffs[0],
        "coeffs_y": coeffs[1],
    }


def _compute_derived_curves(mv3: Any, trk: Any) -> dict[str, Any]:
    """Compute smoothed trajectory, static box/rr extents, and derived velocity/accel curves (marked derived=True)."""
    if trk is None or trk.n < 1:
        return {"derived": True, "available": False}
    t = trk.t
    P = trk.uv
    box_w, box_h, rr_long, rr_short, areas = [], [], [], [], []
    for j in trk.okc:
        st_j = trk.st[int(j)]
        bx = st_j["box"]
        rr = st_j["rr"]
        box_w.append(float(bx[2] - bx[0]))
        box_h.append(float(bx[3] - bx[1]))
        rr_long.append(float(rr[0]))
        rr_short.append(float(rr[1]))
        areas.append(float(st_j["area"]))

    xs_s, ys_s, vx_s, vy_s, sp_s, ax_s, ay_s, ac_s = [], [], [], [], [], [], [], []
    if trk.n >= 3:
        sg = trk.segs()
        for ti in t:
            pos = mv3.pos_at_c(t, P, float(ti), sg)
            vel = mv3.vel_at_c(t, P, float(ti), sg, mv3.HALF_V)
            acc = mv3.acc_at_c(t, P, float(ti), sg, mv3.HALF_A)
            xs_s.append(float(pos[0]) if pos is not None else float("nan"))
            ys_s.append(float(pos[1]) if pos is not None else float("nan"))
            vx_s.append(float(vel[0]) if vel is not None else float("nan"))
            vy_s.append(float(vel[1]) if vel is not None else float("nan"))
            sp_s.append(float(np.linalg.norm(vel)) if vel is not None else float("nan"))
            ax_s.append(float(acc[0]) if acc is not None else float("nan"))
            ay_s.append(float(acc[1]) if acc is not None else float("nan"))
            ac_s.append(float(np.linalg.norm(acc)) if acc is not None else float("nan"))
    else:
        xs_s = P[:, 0].tolist()
        ys_s = P[:, 1].tolist()
        vx_s = [float("nan")] * len(t)
        vy_s = [float("nan")] * len(t)
        sp_s = [float("nan")] * len(t)
        ax_s = [float("nan")] * len(t)
        ay_s = [float("nan")] * len(t)
        ac_s = [float("nan")] * len(t)

    return {
        "derived": True,
        "available": True,
        "note": "Derived from saved track centroids and mask extents via upstream pos_at_c/vel_at_c/acc_at_c/ext_px for visual inspection; not a raw model output.",
        "t": t.tolist(),
        "okc": trk.okc.tolist(),
        "oke": trk.oke.tolist(),
        "x_raw": P[:, 0].tolist(),
        "y_raw": P[:, 1].tolist(),
        "x_smooth": xs_s,
        "y_smooth": ys_s,
        "box_w_px": box_w,
        "box_h_px": box_h,
        "rr_long_px": rr_long,
        "rr_short_px": rr_short,
        "area_px": areas,
        "vx_px_s": vx_s,
        "vy_px_s": vy_s,
        "speed_px_s": sp_s,
        "ax_px_s2": ax_s,
        "ay_px_s2": ay_s,
        "accel_px_s2": ac_s,
    }


def measure_video_instrumented(
    video_recs: list[dict[str, Any]],
    vj: dict[str, Any],
    t_owl: float,
    scene_ruler_entry: dict[str, Any] | None = None,
    target_question_ids: set[int] | None = None,
    upstream_dir: Path = UPSTREAM_DIR,
) -> dict[str, Any]:
    """Run upstream geo/measure_v3.py on one video and extract complete per-question kinematic trace."""
    mods = _load_upstream_geo_modules(upstream_dir)
    mv3 = mods["mv3"]
    m1 = mods["m1"]
    qpc = mods["qpc"]
    rules_v1 = mods["rules_v1"]

    mv3.FIX = set("abcdef")
    mv3.MOGE_OPTS = {"fbrej"}
    mv3.T_OWL = float(t_owl)
    mv3.SCENE_RULER.clear()
    if scene_ruler_entry and scene_ruler_entry.get("ok"):
        sr_clean = {k: v for k, v in scene_ruler_entry.items() if k != "overlay_rgb"}
        mv3.SCENE_RULER[vj["video_id"]] = sr_clean

    # Apply QP_TSHIFT exactly as in load_split_tracks (lines 1577-1589)
    recs = copy.deepcopy(video_recs)
    tshift_applied = False
    if os.environ.get("QP_TSHIFT", "1") == "1" and vj.get("fidx"):
        dur = (max(vj["fidx"]) + 1) / float(vj["fps"])
        ts_all = [float(r["q"][k]) for r in recs for k in ("t", "t1", "t2") if r["q"].get(k) is not None]
        if ts_all and max(ts_all) > dur + 0.5 and (min(ts_all) - dur) >= 0 and (max(ts_all) - dur) <= dur:
            for r in recs:
                for k in ("t", "t1", "t2"):
                    if r["q"].get(k) is not None:
                        r["q"][k] = float(r["q"][k]) - dur
            tshift_applied = True

    allf = float(np.median(list(mv3.FOCAL.values())))
    W = vj["W"]
    fdd = {r["video_source"]: mv3.FOCAL.get(r["video_source"], allf) * W for r in recs}

    raw_outs, V = mv3.measure_video(recs, vj, fdd, "e")

    results_by_id: dict[int, dict[str, Any]] = {}
    traces_by_id: dict[int, dict[str, Any]] = {}
    arrays_by_id: dict[int, dict[str, np.ndarray]] = {}

    for out_row, r in zip(raw_outs, recs):
        rid = int(r["id"])
        q = r["q"]
        u = q["unit"]
        k_dim = {"m": 0, "cm": 0, "mm": 0, "km": 0}.get(u, 2 if "^2" in u else 1)
        x = dict(out_row)
        x["unit"] = u
        x["unit_si_factor"] = float(rules_v1.UNIT_SI[u])
        x["value_before_plaus"] = float(x["val_si"] / rules_v1.UNIT_SI[u]) if x.get("ok") else float("nan")
        x["value"] = x["value_before_plaus"]
        if x.get("ok"):
            lo_p, hi_p = m1.PLAUS[k_dim]
            if not (lo_p <= x["val_si"] <= hi_p):
                x["ok"] = False
                x["why"] = f"implausible({x['val_si']:.3g})"
                x["value"] = float("nan")

        # Arbitrate 2-decade h1 sanity rule (tools/heuristic.py + geo/arbitrate_v2.py lines 35-42)
        pr0 = r["priors"][0] if r["priors"] else None
        h1_val = _compute_h1_prior_copy(
            qpc=qpc,
            question=str(r.get("question") or ""),
            inference_type=str(r.get("inference_type") or "SS"),
            prior_text=str(r.get("prior_text") or ""),
            pr0=pr0,
        )
        arb_ok = bool(x["ok"])
        arb_why = str(x.get("why") or "")
        if arb_ok and np.isfinite(h1_val) and h1_val > 0:
            if abs(math.log10(abs(x["value"]) / h1_val)) > 2.0:
                arb_ok = False
                arb_why = "2-decade sanity rule"
        x["h1_value"] = h1_val
        x["arbitrate_ok"] = arb_ok
        x["arbitrate_why"] = arb_why
        x["geo_value"] = float(x["value"]) if arb_ok else float("nan")
        x["tracks"] = "v2" if vj.get("version") == "v2" else "v1"
        x["video_id"] = vj["video_id"]
        results_by_id[rid] = x

        if target_question_ids is not None and rid not in target_question_ids:
            continue

        # --- Detailed Intermediate Trace for Target Questions ---
        tgt = V.get(r["target"])
        tgt2 = V.get(r["target2"]) if r.get("target2") else None
        tgt2_effective = tgt2
        if (
            os.environ.get("QP_SECOND_FIX") == "1"
            and tgt is not None
            and tgt2 is not None
            and q.get("kind") == "dist2"
            and str(vj.get("video_id", "")).startswith("simulation")
            and (r.get("target2") or {}).get("where") == ["#second"]
            and vj.get("objs")
            and (
                tgt2 is tgt
                or m1.same_object(tgt, tgt2)
                or (tgt.q.get("obj") is not None and tgt.q.get("obj") == tgt2.q.get("obj"))
            )
        ):
            alt_t2 = mv3.second_instance(V, vj, tgt)
            if alt_t2 is not None:
                tgt2_effective = alt_t2
        prior_tr = V.get(r["prior_obj"]) if (r.get("prior_obj") or {}).get("phrase") else None
        gravity = bool(pr0 and pr0.get("gravity", False))
        grav_trace = None
        if gravity and prior_tr is None:
            cands = [trk for trk in V.tr.values() if mv3.track_ok(trk, True)[0]]
            best, bev, g_info = None, None, None
            cand_logs = []
            for trk in cands:
                g_px, inf = mv3.freefall(trk, a=pr0.get("t1"), b=pr0.get("t2"))
                cand_logs.append({
                    "key": trk.q.get("key"),
                    "g_px_s2": float(g_px) if g_px is not None else None,
                    "info": inf if isinstance(inf, dict) else str(inf),
                })
                if g_px is not None:
                    ev = (0 if inf.get("sub") else 1, inf["dur"])
                    if bev is None or ev > bev:
                        best, bev, g_info = trk, ev, (g_px, inf)
            if best is not None:
                prior_tr = best
            grav_trace = {
                "candidates_tested": cand_logs,
                "selected_key": best.q.get("key") if best is not None else None,
                "g_px_s2": float(g_info[0]) if g_info else None,
                "freefall_info": g_info[1] if g_info else None,
                "scale_check": mv3.gravity_scale_check(float(x["mpp"]), prior_tr, V) if np.isfinite(x.get("mpp", np.nan)) else None,
            }

        # Lossless arrays for CPU replay
        q_arrays: dict[str, np.ndarray] = {
            "cam": np.asarray(vj.get("cam") or [[0.0, 0.0, 1.0]], dtype=np.float64),
        }
        tgt_arrs = _track_arrays_dict(tgt, vj)
        for ak, av in tgt_arrs.items():
            q_arrays[f"tgt_{ak}"] = av
        if tgt2 is not None and tgt2 is not tgt:
            t2_arrs = _track_arrays_dict(tgt2, vj)
            for ak, av in t2_arrs.items():
                q_arrays[f"tgt2_{ak}"] = av
        if prior_tr is not None and prior_tr is not tgt and prior_tr is not tgt2:
            pr_arrs = _track_arrays_dict(prior_tr, vj)
            for ak, av in pr_arrs.items():
                q_arrays[f"prior_{ak}"] = av

        # Companion tracks metadata/arrays needed if gravity_scale_check, QP_SAMECLASS, QP_NEAREST, or colour_assign runs
        companion_tracks_meta = []
        referenced_oids: set[int] = set()
        for tr_ref in (tgt, tgt2, prior_tr):
            if tr_ref is not None and tr_ref.q.get("obj") is not None:
                referenced_oids.add(int(tr_ref.q["obj"]))
        for idx_tr, (_tk_name, trk_other) in enumerate(V.tr.items()):
            if trk_other is None or trk_other in (tgt, tgt2, prior_tr):
                continue
            if trk_other.q.get("obj") is not None:
                referenced_oids.add(int(trk_other.q["obj"]))
            oth_arrs = _track_arrays_dict(trk_other, vj)
            for ak, av in oth_arrs.items():
                q_arrays[f"comp{idx_tr}_{ak}"] = av
            companion_tracks_meta.append({
                "slot": f"comp{idx_tr}",
                "query_meta": _extract_query_meta_for_replay(trk_other.q),
            })

        # Also preserve any unreferenced vj["objs"] when dist2 #second might inspect all objects via second_instance()
        extra_objs_meta = []
        if vj.get("version") == "v2" and vj.get("objs") and q.get("kind") == "dist2":
            for oid_str, fr_raw in vj["objs"].items():
                oid_int = int(oid_str)
                if oid_int in referenced_oids:
                    continue
                tmp_trk = mv3.Track({"key": f"_extra_obj_{oid_int}", "obj": oid_int}, vj, fr_raw)
                ex_arrs = _track_arrays_dict(tmp_trk, vj)
                for ak, av in ex_arrs.items():
                    q_arrays[f"exobj{oid_int}_{ak}"] = av
                extra_objs_meta.append({
                    "slot": f"exobj{oid_int}",
                    "oid": oid_int,
                    "query_meta": _extract_query_meta_for_replay(tmp_trk.q),
                })

        arrays_by_id[rid] = q_arrays

        # Prior scale step breakdown
        prior_step: dict[str, Any] = {"prior_parsed": pr0}
        if scene_ruler_entry and scene_ruler_entry.get("ok") and pr0 and pr0.get("kind") == "size" and pr0.get("obj") and re.search(scene_ruler_entry.get("prior_re", r"\blanes?\b"), pr0["obj"]):
            scene_ppx = float(scene_ruler_entry["frac_w"]) * V.W
            prior_step.update({
                "method": "scene_ruler",
                "file_fn": "geo/scene_ruler.py:lane_ruler -> geo/measure_v3.py:measure_video",
                "frac_w": float(scene_ruler_entry["frac_w"]),
                "video_W": int(V.W),
                "prior_px": float(scene_ppx),
                "prior_value_si": float(pr0["value_si"]),
                "mpp": float(pr0["value_si"] / scene_ppx),
                "theta_deg": float(scene_ruler_entry["theta"]),
                "lines_px": scene_ruler_entry.get("lines"),
            })
        elif gravity and grav_trace and grav_trace.get("g_px_s2"):
            prior_step.update({
                "method": "gravity_freefall",
                "file_fn": "geo/measure_v3.py:freefall + robust_quad + gravity_scale_check",
                "prior_px": float(grav_trace["g_px_s2"]),
                "prior_value_si": float(pr0["value_si"]),
                "mpp": float(pr0["value_si"] / grav_trace["g_px_s2"]),
                "gravity_trace": grav_trace,
            })
        elif prior_tr is not None and pr0 is not None:
            pdim = pr0.get("dim") or "length"
            if pr0["kind"] == "size":
                ext_vals, ext_js = prior_tr.ext_px(pdim, r["prior_obj"].get("phrase"))
                ppx_val = mv3.agg_ext(ext_vals, pdim)
                agg_rule = "percentile_75" if pdim in ("diameter", "radius") else "median"
                prior_step.update({
                    "method": "object_size_extent",
                    "file_fn": "geo/measure.py:Track.ext_px + agg_ext",
                    "pdim": pdim,
                    "agg_rule": agg_rule,
                    "oke_frame_indices": [int(vj["fidx"][j]) for j in ext_js],
                    "oke_track_slots": [int(j) for j in ext_js],
                    "per_frame_extents_px": [float(v) for v in ext_vals],
                    "prior_px": float(ppx_val) if np.isfinite(ppx_val) else None,
                    "prior_value_si": float(pr0["value_si"]),
                    "mpp": float(pr0["value_si"] / ppx_val) if np.isfinite(ppx_val) and ppx_val > 0 else None,
                })
            else:
                ppx_val = mv3.prior_kin(prior_tr, pr0)
                prior_step.update({
                    "method": "dynamic_prior_kinematics",
                    "file_fn": "geo/measure_v3.py:prior_kin",
                    "prior_px": float(ppx_val) if ppx_val is not None and np.isfinite(ppx_val) else None,
                    "prior_value_si": float(pr0["value_si"]),
                    "mpp": float(pr0["value_si"] / ppx_val) if ppx_val and np.isfinite(ppx_val) else None,
                })

        # Target kinematic breakdown
        target_step: dict[str, Any] = {"kind": q["kind"]}
        if tgt is not None and tgt.n > 0:
            sg = tgt.segs()
            tt = tgt.t
            PP = tgt.uv
            lo_t, hi_t = float(tt.min()), float(tt.max())
            a_t = lo_t if q["t1"] is None else max(float(q["t1"]), lo_t)
            b_t = hi_t if q["t2"] is None else min(float(q["t2"]), hi_t)
            kind = q["kind"]

            seg_info = {
                "sigma_px": float(sg["sigma"]),
                "segs_okc_slices": [[int(i), int(j)] for i, j in sg["segs"]],
                "segs_time_ranges": [[float(tt[i]), float(tt[j - 1])] for i, j in sg["segs"]],
                "rest_flags": [bool(fl) for fl in sg["rest"]],
            }
            target_step["segmentation"] = seg_info

            if kind == "speed" and q["t"] is not None:
                t0 = m1.clamp_time(tgt, float(q["t"]))
                sub = mv3._sub(sg, tt, PP, t0) if t0 is not None else None
                fit_d = None
                participating_fidx = []
                if sub is not None:
                    ts, Ps, k_seg = sub
                    t0c = min(max(t0, float(ts.min())), float(ts.max()))
                    fit_d = _fit_detail(mv3, m1, ts, Ps, t0c, mv3.HALF_V, 2, 5) or _fit_detail(mv3, m1, ts, Ps, t0c, mv3.HALF_V, 1, 3)
                    if fit_d is not None:
                        i_seg, _ = sg["segs"][k_seg]
                        participating_fidx = [int(vj["fidx"][tgt.okc[i_seg + m_i]]) for m_i in fit_d["local_mask_indices"]]
                v_vec = mv3.vel_at_c(tt, PP, t0, sg, mv3.HALF_V) if t0 is not None else None
                v_px = float(np.linalg.norm(v_vec)) if v_vec is not None else None
                target_step.update({
                    "estimator": "vel_at_c",
                    "file_fn": "geo/measure_v3.py:vel_at_c",
                    "queried_t": float(q["t"]),
                    "clamped_t0": float(t0) if t0 is not None else None,
                    "fit": fit_d,
                    "participating_video_frames": participating_fidx,
                    "velocity_vec_px_s": v_vec.tolist() if v_vec is not None else None,
                    "px_quantity": v_px,
                })
            elif kind == "avgspeed" or (kind == "speed" and q["t"] is None):
                m_idx = np.where((tt >= a_t - 1e-9) & (tt <= b_t + 1e-9))[0]
                per_t_vels = []
                for idx_i in m_idx:
                    vv = mv3.vel_at_c(tt, PP, float(tt[idx_i]), sg, 0.15)
                    if vv is not None:
                        per_t_vels.append({
                            "fidx": int(vj["fidx"][tgt.okc[idx_i]]),
                            "t": float(tt[idx_i]),
                            "vx_px_s": float(vv[0]),
                            "vy_px_s": float(vv[1]),
                            "speed_px_s": float(np.linalg.norm(vv)),
                        })
                v_px = float(np.mean([e["speed_px_s"] for e in per_t_vels])) if per_t_vels else None
                target_step.update({
                    "estimator": "avg_speed_c",
                    "file_fn": "geo/measure_v3.py:avg_speed_c (mean of vel_at_c with half=0.15s)",
                    "interval": [float(a_t), float(b_t)],
                    "participating_video_frames": [e["fidx"] for e in per_t_vels],
                    "per_timestamp_speeds": per_t_vels,
                    "px_quantity": v_px,
                })
            elif kind == "hvel":
                pa = mv3.pos_at_c(tt, PP, a_t, sg)
                pb = mv3.pos_at_c(tt, PP, b_t, sg)
                v_px = abs(float(pb[0] - pa[0])) / (b_t - a_t) if (pa is not None and pb is not None and b_t - a_t >= 0.1) else None
                v2d_avg = mv3.avg_speed_c(tt, PP, a_t, b_t, sg)
                participating_fidx = [
                    int(vj["fidx"][tgt.okc[idx_i]])
                    for idx_i in range(len(tt))
                    if abs(tt[idx_i] - a_t) <= 0.20 + 1e-9 or abs(tt[idx_i] - b_t) <= 0.20 + 1e-9
                ]
                target_step.update({
                    "estimator": "hvel_pos_endpoints",
                    "file_fn": "geo/measure_v3.py:px_quantity(kind='hvel') + pos_at_c(half=0.08s)",
                    "interval": [float(a_t), float(b_t)],
                    "dt_sec": float(b_t - a_t),
                    "pos_start_px": pa.tolist() if pa is not None else None,
                    "pos_end_px": pb.tolist() if pb is not None else None,
                    "dx_abs_px": abs(float(pb[0] - pa[0])) if (pa is not None and pb is not None) else None,
                    "dy_px": float(pb[1] - pa[1]) if (pa is not None and pb is not None) else None,
                    "participating_video_frames": participating_fidx,
                    "px_quantity": v_px,
                    "full_2d_avgspeed_px_s_for_comparison": float(v2d_avg) if v2d_avg is not None else None,
                })
            elif kind == "accel" and q["t"] is not None:
                t0 = m1.clamp_time(tgt, float(q["t"]))
                sub = mv3._sub(sg, tt, PP, t0) if t0 is not None else None
                fit_d = None
                participating_fidx = []
                used_whole_seg_fallback = False
                if sub is not None:
                    ts, Ps, k_seg = sub
                    t0c = min(max(t0, float(ts.min())), float(ts.max()))
                    fit_d = _fit_detail(mv3, m1, ts, Ps, t0c, mv3.HALF_A, 2, 7)
                    i_seg, j_seg = sg["segs"][k_seg]
                    if fit_d is not None and fit_d["deg"] == 2:
                        participating_fidx = [int(vj["fidx"][tgt.okc[i_seg + m_i]]) for m_i in fit_d["local_mask_indices"]]
                    elif len(ts) >= 5:
                        used_whole_seg_fallback = True
                        participating_fidx = [int(vj["fidx"][tgt.okc[k_i]]) for k_i in range(i_seg, j_seg)]
                a_vec = mv3.acc_at_c(tt, PP, t0, sg, mv3.HALF_A) if t0 is not None else None
                a_px = float(np.linalg.norm(a_vec)) if a_vec is not None else None
                # Window sensitivity table for viewer inspection (#401 etc.)
                win_sens = []
                if t0 is not None and np.isfinite(x.get("mpp", np.nan)):
                    for h_try in (0.20, 0.30, 0.40, 0.64, 1.00):
                        av_try = mv3.acc_at_c(tt, PP, t0, sg, h_try)
                        if av_try is not None:
                            apx_try = float(np.linalg.norm(av_try))
                            win_sens.append({
                                "half_s": h_try,
                                "ax_px_s2": float(av_try[0]),
                                "ay_px_s2": float(av_try[1]),
                                "accel_px_s2": apx_try,
                                "accel_unit": float(apx_try * x["mpp"] / rules_v1.UNIT_SI[u]),
                            })
                target_step.update({
                    "estimator": "acc_at_c",
                    "file_fn": "geo/measure_v3.py:acc_at_c",
                    "queried_t": float(q["t"]),
                    "clamped_t0": float(t0) if t0 is not None else None,
                    "fit": fit_d,
                    "used_whole_segment_fallback": used_whole_seg_fallback,
                    "participating_video_frames": participating_fidx,
                    "accel_vec_px_s2": a_vec.tolist() if a_vec is not None else None,
                    "px_quantity": a_px,
                    "window_sensitivity": win_sens,
                })
            elif kind == "size":
                dim_q = q.get("dim") or "length"
                t_phrase = (r.get("target") or {}).get("phrase") or ""
                ext_vals, ext_js = tgt.ext_px(dim_q, t_phrase)
                v_px_raw = mv3.agg_ext(ext_vals, dim_q) if len(ext_vals) else float("nan")
                v_px = float(v_px_raw) if np.isfinite(v_px_raw) else None
                winged_hit = bool(re.search(m1.WINGED, t_phrase))
                agg_rule = "percentile_75" if dim_q in ("diameter", "radius") else "median"
                oke_fidx = [int(vj["fidx"][int(j)]) for j in ext_js]
                target_step.update({
                    "estimator": "ext_px_agg",
                    "file_fn": "geo/measure_v3.py:Track.ext_px + geo/measure.py:agg_ext",
                    "dim": dim_q,
                    "agg_rule": agg_rule,
                    "winged_override": winged_hit,
                    "oke_track_slots": [int(j) for j in ext_js],
                    "participating_video_frames": oke_fidx,
                    "per_frame_extents_px": [float(v) for v in ext_vals],
                    "px_quantity": v_px,
                })
            elif kind == "dist2":
                min_flag = bool(re.search(r"\b(?:minimum|smallest|closest|shortest)\b", str(r.get("question") or "").lower()))
                cands_t2 = mv3.nearest_cands(V, r, tgt2_effective) if tgt2_effective is not None else []
                vs_d2 = [
                    mv3.dist2_px(tgt, t2_, q["t"], q["when"], minimum=min_flag)
                    for t2_ in cands_t2
                ]
                vs_valid = [x_ for x_ in vs_d2 if x_[0] is not None and np.isfinite(x_[0]) and x_[0] > 0] or vs_d2[:1]
                if vs_valid:
                    v_d2, gap_flag = min(vs_valid, key=lambda x_: x_[0] if (x_[0] is not None and np.isfinite(x_[0])) else float("inf"))
                    v_px = float(v_d2) if (v_d2 is not None and np.isfinite(v_d2)) else None
                else:
                    v_px, gap_flag = None, False
                part_fidx = [int(vj["fidx"][int(j)]) for j in tgt.okc]
                target_step.update({
                    "estimator": "dist2_px",
                    "file_fn": "geo/measure_v3.py:dist2_px + _poly_dist",
                    "target2_key": tgt2_effective.q.get("key") if tgt2_effective is not None else None,
                    "second_fix_applied": bool(x.get("grav") == "second_fix"),
                    "when": q.get("when"),
                    "queried_t": float(q["t"]) if q.get("t") is not None else None,
                    "minimum": min_flag,
                    "used_polygon_edge_gap": bool(gap_flag),
                    "participating_video_frames": part_fidx,
                    "px_quantity": v_px,
                })
            elif kind == "orbitdiam":
                v_orb = mv3.px_quantity(tgt, "orbitdiam", q["t"], q["t1"], q["t2"], q["when"])
                v_px = float(v_orb) if (v_orb is not None and np.isfinite(v_orb)) else None
                target_step.update({
                    "estimator": "orbitdiam_px",
                    "file_fn": "geo/measure_v3.py:px_quantity(kind='orbitdiam')",
                    "participating_video_frames": [int(vj["fidx"][int(j)]) for j in tgt.okc],
                    "px_quantity": v_px,
                })
            elif kind == "hfloor":
                fl_px = mv3.floor_level(V)
                v_hf = mv3.hfloor_px(tgt, V, q["t"])
                v_px = float(v_hf) if (v_hf is not None and np.isfinite(v_hf)) else None
                target_step.update({
                    "estimator": "hfloor_px",
                    "file_fn": "geo/measure_v3.py:floor_level + hfloor_px",
                    "floor_level_px": float(fl_px) if fl_px is not None else None,
                    "queried_t": float(q["t"]) if q.get("t") is not None else None,
                    "participating_video_frames": [int(vj["fidx"][int(j)]) for j in tgt.oke],
                    "px_quantity": v_px,
                })
            else:
                v_gen = mv3.px_quantity(tgt, kind, q["t"], q["t1"], q["t2"], q["when"])
                v_px = float(v_gen) if (v_gen is not None and np.isfinite(v_gen)) else None
                target_step.update({
                    "estimator": f"px_quantity_{kind}",
                    "file_fn": "geo/measure_v3.py:px_quantity",
                    "participating_video_frames": [int(vj["fidx"][int(j)]) for j in tgt.okc],
                    "px_quantity": v_px,
                })

        formula_trace = {
            "mode": x.get("mode", ""),
            "symbolic": f"final_value = (px_quantity * (prior_value_si / prior_px)) / UNIT_SI['{u}']",
            "prior_value_si": prior_step.get("prior_value_si"),
            "prior_px": prior_step.get("prior_px"),
            "mpp": float(x["mpp"]) if np.isfinite(x.get("mpp", np.nan)) else None,
            "px_quantity": target_step.get("px_quantity"),
            "val_si": float(x["val_si"]) if np.isfinite(x.get("val_si", np.nan)) else None,
            "unit": u,
            "unit_si_factor": float(rules_v1.UNIT_SI[u]),
            "h1_value": float(h1_val) if np.isfinite(h1_val) else None,
            "final_value": float(x["geo_value"]) if np.isfinite(x.get("geo_value", np.nan)) else None,
        }

        replay_spec = {
            "id": rid,
            "video_id": vj["video_id"],
            "video_meta": {
                "video_id": vj["video_id"],
                "W": int(vj["W"]),
                "H": int(vj["H"]),
                "N": int(vj.get("N", len(vj["fidx"]))),
                "stride": int(vj.get("stride", 1)),
                "fps": float(vj["fps"]),
                "version": vj.get("version", "v1"),
            },
            "t_owl": float(t_owl),
            "tshift_already_applied": tshift_applied,
            "parse_record": r,
            "scene_ruler": {k: v for k, v in scene_ruler_entry.items() if k != "overlay_rgb"} if scene_ruler_entry else None,
            "target_query_meta": _extract_query_meta_for_replay(tgt.q) if tgt is not None else None,
            "target2_query_meta": _extract_query_meta_for_replay(tgt2.q) if tgt2 is not None else None,
            "target2_same_as_target": bool(tgt2 is not None and tgt is not None and tgt2 is tgt),
            "prior_query_meta": _extract_query_meta_for_replay(prior_tr.q) if prior_tr is not None else None,
            "prior_same_as_target": bool(prior_tr is not None and tgt is not None and prior_tr is tgt),
            "prior_same_as_target2": bool(prior_tr is not None and tgt2 is not None and prior_tr is tgt2),
            "companion_tracks_meta": companion_tracks_meta,
            "extra_objs_meta": extra_objs_meta,
        }

        traces_by_id[rid] = {
            "id": rid,
            "video_id": vj["video_id"],
            "result": x,
            "target_track_summary": {
                "key": tgt.q.get("key") if tgt is not None else None,
                "obj_id": tgt.q.get("obj") if tgt is not None else None,
                "anchor": tgt.q.get("anchor") if tgt is not None else None,
                "best_score": tgt.best if tgt is not None else None,
                "best_gd": tgt.best_gd if tgt is not None else None,
                "best_owl": tgt.best_owl if tgt is not None else None,
                "det_iou_median": tgt.iou if tgt is not None else None,
                "n_valid_okc": int(tgt.n) if tgt is not None else 0,
                "n_unclipped_oke": int(len(tgt.oke)) if tgt is not None else 0,
                "n_total_frames": int(len(vj["fidx"])),
                "size_px": float(tgt.size_px) if tgt is not None else 0.0,
                "representative_point": "SAM2 binary mask pixel centroid c = [mean(xs), mean(ys)]",
                "box_derivation": "Derived per frame from SAM2 binary mask bounds [min(xs), min(ys), max(xs)+1, max(ys)+1]",
            },
            "prior_track_summary": {
                "key": prior_tr.q.get("key") if prior_tr is not None else None,
                "obj_id": prior_tr.q.get("obj") if prior_tr is not None else None,
                "anchor": prior_tr.q.get("anchor") if prior_tr is not None else None,
                "n_valid_okc": int(prior_tr.n) if prior_tr is not None else 0,
                "n_unclipped_oke": int(len(prior_tr.oke)) if prior_tr is not None else 0,
                "size_px": float(prior_tr.size_px) if prior_tr is not None else 0.0,
            },
            "prior_step": prior_step,
            "target_step": target_step,
            "formula_trace": formula_trace,
            "derived_curves": _compute_derived_curves(mv3, tgt),
            "replay_spec": replay_spec,
        }

    return {
        "results_by_id": results_by_id,
        "traces_by_id": traces_by_id,
        "arrays_by_id": arrays_by_id,
    }


def recompute_from_saved_trace_arrays(
    replay_spec: dict[str, Any],
    arrays: dict[str, np.ndarray],
    upstream_dir: Path = UPSTREAM_DIR,
) -> dict[str, Any]:
    """Recompute fresh_box strictly on CPU from saved lossless .npy arrays without loading any GPU model."""
    vmeta = replay_spec["video_meta"]
    fidx = arrays["tgt_fidx"].astype(int).tolist()
    queries = []
    objs: dict[str, list[dict[str, Any]]] = {}
    orig_oid_map: dict[int, int] = {}
    next_oid = 0
    seen_keys: set[str | None] = set()

    def _add_query(q_meta: dict[str, Any] | None, prefix: str) -> int | None:
        nonlocal next_oid
        if q_meta is None:
            return None
        qk = q_meta.get("key")
        if qk in seen_keys:
            return None
        seen_keys.add(qk)
        sub = {k[len(prefix) + 1:]: v for k, v in arrays.items() if k.startswith(prefix + "_")}
        if not sub:
            return None
        q_rebuilt = _reconstruct_query_from_arrays(sub, q_meta)
        if vmeta.get("version") == "v2":
            orig_obj = q_meta.get("obj")
            if orig_obj is not None and int(orig_obj) in orig_oid_map:
                oid = orig_oid_map[int(orig_obj)]
                q_rebuilt.pop("frames", None)
            elif orig_obj is not None:
                oid = next_oid
                next_oid += 1
                orig_oid_map[int(orig_obj)] = oid
                objs[str(oid)] = q_rebuilt.pop("frames")
            else:
                oid = None
                q_rebuilt.pop("frames", None)
            q_rebuilt["obj"] = oid
        queries.append(q_rebuilt)
        return len(queries) - 1

    _add_query(replay_spec.get("target_query_meta"), "tgt")
    if not replay_spec.get("target2_same_as_target") and replay_spec.get("target2_query_meta") is not None:
        _add_query(replay_spec.get("target2_query_meta"), "tgt2")
    if (
        not replay_spec.get("prior_same_as_target")
        and not replay_spec.get("prior_same_as_target2")
        and replay_spec.get("prior_query_meta") is not None
    ):
        _add_query(replay_spec.get("prior_query_meta"), "prior")
    for comp in replay_spec.get("companion_tracks_meta", []):
        _add_query(comp["query_meta"], comp["slot"])
    if vmeta.get("version") == "v2":
        for ex in replay_spec.get("extra_objs_meta", []):
            slot_ex = ex["slot"]
            sub_ex = {k[len(slot_ex) + 1:]: v for k, v in arrays.items() if k.startswith(slot_ex + "_")}
            if sub_ex:
                q_ex = _reconstruct_query_from_arrays(sub_ex, ex["query_meta"])
                oid = next_oid
                next_oid += 1
                objs[str(oid)] = q_ex.pop("frames")

    vj_rebuilt: dict[str, Any] = {
        "video_id": vmeta["video_id"],
        "W": int(vmeta["W"]),
        "H": int(vmeta["H"]),
        "N": int(vmeta["N"]),
        "stride": int(vmeta["stride"]),
        "fidx": fidx,
        "fps": float(vmeta["fps"]),
        "cam": arrays["cam"].tolist(),
        "version": vmeta.get("version", "v1"),
        "queries": queries,
        "queries_d": queries,
    }
    if vmeta.get("version") == "v2":
        vj_rebuilt["objs"] = objs

    saved_tshift = os.environ.get("QP_TSHIFT")
    try:
        if replay_spec.get("tshift_already_applied"):
            os.environ["QP_TSHIFT"] = "0"
        out = measure_video_instrumented(
            video_recs=[replay_spec["parse_record"]],
            vj=vj_rebuilt,
            t_owl=float(replay_spec["t_owl"]),
            scene_ruler_entry=replay_spec.get("scene_ruler"),
            target_question_ids={int(replay_spec["id"])},
            upstream_dir=upstream_dir,
        )
    finally:
        if saved_tshift is None:
            os.environ.pop("QP_TSHIFT", None)
        else:
            os.environ["QP_TSHIFT"] = saved_tshift

    rid = int(replay_spec["id"])
    return out["results_by_id"][rid]


def evaluate_xground_and_fuse(
    parse_rec: dict[str, Any],
    res_v2: dict[str, Any] | None,
    res_v3nt: dict[str, Any] | None,
    vj_v2: dict[str, Any] | None,
    vlm_ground_records: list[dict[str, Any]] | None,
    upstream_dir: Path = UPSTREAM_DIR,
) -> dict[str, Any]:
    """Reproduce analysis/v3/fuse_v3.py (rule d_vw) + analysis/v3/xground.py + sruler_apply.py for one row."""
    mods = _load_upstream_geo_modules(upstream_dir)
    mv3 = mods["mv3"]
    tc = mods["tc"]

    gd = float(res_v2["geo_value"]) if (res_v2 and np.isfinite(res_v2.get("geo_value", np.nan))) else float("nan")
    gv = float(res_v3nt["geo_value"]) if (res_v3nt and np.isfinite(res_v3nt.get("geo_value", np.nan))) else float("nan")
    sr_v2 = bool(res_v2 and res_v2.get("sruler"))
    sr_v3 = bool(res_v3nt and res_v3nt.get("sruler"))

    # xground check: does VLM grounding have n >= 2 frames with hit == 0 IoU >= 0.5 against v2 track?
    xg_flagged_roles: list[str] = []
    xg_details: dict[str, Any] = {}
    if vj_v2 is not None and vlm_ground_records:
        V_v2 = mv3.Video(vj_v2, "e")
        by_key: dict[str, list[dict[str, Any]]] = {}
        for x in vlm_ground_records:
            by_key.setdefault(x["key"], []).append(x)
        for role in ("target", "target2", "prior_obj"):
            k = mv3.Video.key(parse_rec.get(role))
            if not k or k not in by_key:
                continue
            trk = V_v2.tr.get(k)
            if trk is None or not trk.n:
                continue
            hit = n = 0
            for x in by_key[k]:
                j = int(x["j"])
                bl = [b for b in x.get("boxes_rel1000", []) if b and len(b) == 4]
                if not bl or j >= len(trk.st) or not trk.st[j]:
                    continue
                tb = trk.st[j]["box"]
                n += 1
                hit += int(max(tc.iou(tb, b) for b in bl) >= 0.5)
            xg_details[role] = {"key": k, "hit_iou_ge_05": hit, "n_checked": n}
            if n >= 2 and hit == 0:
                xg_flagged_roles.append(role)

    # fuse_v3 rule d_vw: take gd if finite, else gv
    if np.isfinite(gd):
        val = gd
        src = "d"
        branch = "v2"
    elif np.isfinite(gv):
        val = gv
        src = "v"
        branch = "v3nt"
    else:
        val = float("nan")
        src = ""
        branch = "none"

    # xground override: if flagged and gv is finite, take gv
    if xg_flagged_roles and np.isfinite(gv):
        val = gv
        src = "xg_v"
        branch = "v3nt"

    # sruler_apply: mark merge_src = sruler if scene ruler produced it
    if (branch == "v2" and sr_v2) or (branch == "v3nt" and sr_v3):
        src = "sruler"

    md_v2 = str((res_v2 or {}).get("mode") or "")
    md_v3 = str((res_v3nt or {}).get("mode") or "")
    if src == "sruler":
        final_mode = "2d_ratio"
    elif src in ("v", "xg_v"):
        final_mode = ("vlm_" + md_v3) if md_v3 and not md_v3.startswith("vlm_") else md_v3
    elif src == "d":
        final_mode = md_v2
    else:
        final_mode = ""

    return {
        "final_value": val,
        "selected_branch": branch,
        "merge_src": src,
        "final_mode": final_mode,
        "v2_value": gd,
        "v3nt_value": gv,
        "xground_flagged_roles": xg_flagged_roles,
        "xground_details": xg_details,
    }
