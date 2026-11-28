"""Run only the inference-producing sections of the pinned validation chain."""
from __future__ import annotations

import argparse
import ast
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

from ..schema import UPSTREAM_DIR, fingerprint, read_json, sha256_file, write_json_atomic

PACKAGE = "solutions.iiixr.kaggle.three_d"
FLAGS = dict(QP_REJ_LO="0.2", QP_REJ_HI="5", QP_PRIOR_DRIFT="1.5", QP_PRIOR_ACCGUARD="3",
             QP_SCENE_HEAD="1", QP_TSHIFT="1", QP_DEPTH_DEDUP="1", QP_SCENE_SA2VA="1",
             QP_SECOND_FIX="1", QP_SAMECLASS="1", QP_SCENE_RULER="1", QP_FF_ROT="1",
             QP_SIDE_FIX="1", QP_FLOOR="1", QP_STATIC_CAP="25", QP_FF_SUB="1",
             QP_FF_SUB_SRC="simulation", QP_PARSE2="1", QP_NEAREST="1", QP_BELT="1")
SELF = dict(RL_SELF="1", RL_SELF_NOGRAV="1", RL_SELF_MODE="2d_selfruler", RL_OVER_VLM="1")
GRAVITY = dict(RL_SELF_GRAV="1", RL_SELF_GRAV_MODE="2d_gself")
PERSON = dict(QP_FPERSON="1", QP_FPERSON_MINN="2")
BOOTSTRAP = "Fresh bootstrap substitutes generated for unreleased rj5pbsrng/rj5plm input tables"
V1 = "geo/out/tracks_val"
V2 = V1 + "_v2"
V3 = V1 + "_v3nt"
DEPTH = "geo/out/moge_val"
PARSE = "geo/out/parse_val_v2.jsonl"
FINAL = "preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv"

SOURCE = (
    "analysis/v3/fuse_v3.py", "analysis/v3/xground.py", "analysis/v3/ruler_build.py",
    "analysis/v3/sruler_apply.py", "tools/qp_common.py", "tools/heuristic.py", "tools/run_vlm.py",
    "caw/analyze_caw.py", "caw/analyze_v2caw.py",
)


def source_files():
    return [*SOURCE, *(p.relative_to(UPSTREAM_DIR).as_posix() for p in sorted((UPSTREAM_DIR / "geo").glob("*.py")))]


def replace_once(source, needle, replacement):
    if source.count(needle) != 1:
        raise ValueError(f"Pinned upstream source changed at {needle!r}")
    return source.replace(needle, replacement, 1)


def adapt(name, source):
    # Exclude validation-only scoring/reporting, preserving the producer statements.
    if name == "analysis/v3/fuse_v3.py":
        tree = ast.parse(source)
        keep = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"load_raw", "qualifier_rows", "merge"}]
        return ("import os, json, sys\nimport numpy as np, pandas as pd\n"
                "R = os.environ['QP_ROOT']\nsys.path.insert(0, f'{R}/tools')\nfrom qp_common import OUT_COLS\n"
                + "\n".join(ast.get_source_segment(source, node) for node in keep) + """
SPLIT, TD, TV, PFX = sys.argv[1:5]
rd, rv = load_raw(TD), load_raw(TV)
out = merge(rd, rv, "d_vw", qualifier_rows(rd.index))
out.to_csv(f"{R}/preds/{SPLIT}_{PFX}_d_vw_raw.csv")
out.reset_index()[OUT_COLS].to_csv(f"{R}/preds/{SPLIT}_{PFX}_d_vw.csv", index=False)
""")
    if name == "analysis/v3/xground.py":
        before = source.split('if split == "val":\n    T =', 1)[0]
        return replace_once(before, 'print(\'\')\nrd =', "json.dump(rows, open(f'{R}/preds/{split}_{XG_OUT}_flags.json', 'w'))\nrd =")
    if name == "analysis/v3/ruler_build.py":
        # stage_q explicitly filters dimension 2. A 3D-only scope has no valid g16 job.
        source = replace_once(source, "    out = out[cols]", "    out = out.reindex(columns=cols)\n    out = out[cols]")
        source = replace_once(source, "    if split == \"val\":\n        d = load_split(\"val\"); d.index = d[\"Unnamed: 0\"].astype(int)\n        y = pd.to_numeric(d.ground_truth_posterior, errors=\"coerce\")\n        S =",
                              "    if False:\n        S =")
        return source
    if name == "geo/person_focal.py":
        a = source.index('raw = pd.read_csv')
        b = source.index('\nvids = {}', a)
        source = source[:a] + "from qp_common import load_split\nvpath = load_split(a.split).groupby('video_id').video_path.first().to_dict()" + source[b:]
    if name == "geo/scene_ruler.py":
        source = replace_once(source, '    sys.path.insert(0, f"{R}/caw")\n    import analyze_caw as A\n    df = A.load_test() if split == "test" else A.load_val()',
                              '    sys.path.insert(0, f"{R}/tools")\n    from qp_common import load_split\n    df = load_split(split)')
    if name == "geo/parse.py":
        source = replace_once(source, "def run_llm(prompts):", "def run_llm(prompts, records):")
        source = replace_once(source, "    outs = llm.generate(texts, sp, use_tqdm=False)",
                              """    requests = [{"prompt": text, "trace_context": {"question_id": row["id"], "video_id": row["video_id"]}}
                for text, row in zip(texts, records)]
    outs = llm.generate(requests, sp, use_tqdm=False)""")
        source = replace_once(source, "raw = run_llm(prompts)", "raw = run_llm(prompts, recs)")
    if name == "geo/vlm_ground.py":
        source = replace_once(source, 'reqs.append({"prompt": pr,',
                              'reqs.append({"prompt": pr, "trace_context": {"video_id": v, "track_key": q["key"], "sampled_index": int(j)},')
    if name == "tools/run_vlm.py":
        source = replace_once(source, "        recs.append((row, text))",
                              '        reqs[-1]["trace_context"] = {"question_id": int(row["Unnamed: 0"]), "video_id": row["video_id"]}\n        recs.append((row, text))')
    if name == "geo/molmo_ground.py":
        source = replace_once(source, "    items = sorted(items)", """    items = sorted(items)
    if not items:
        open(OUT, "w", encoding="utf-8").close()
        return
    if os.environ.get("IIIXR_MOLMO") != "vllm-bnb":
        raise RuntimeError(f"Molmo T4 backend disabled; requested video/phrase pairs={items}")
""")
        source = replace_once(source, "    from vllm import LLM, SamplingParams\n", "")
        source = replace_once(source, "    llm = LLM(model=", "    from vllm import LLM, SamplingParams\n    llm = LLM(dtype='float16', quantization='bitsandbytes', enforce_eager=True, model=")
        source = replace_once(source, 'm.update(points=', 'trace.log("molmo_response", prompt=f"Point to the {m[\'phrase\']}.", response=txt, metadata=m)\n            m.update(points=')
    if name == "geo/moge_run.py":
        source = replace_once(source, '            d = np.where(m &', '            trace.arrays("moge_infer", locals(), image=img, intrinsics=K, depth=d, mask=m, points=o.get("points"), normal=o.get("normal"))\n            d = np.where(m &')
    if name == "geo/measure_v3.py":
        needle = '        if v is None or not np.isfinite(v) or v <= 0:\n            res["why"] = "target_px_fail"; out.append(res); continue\n        res.update(val_si=float(v), f=f_cal'
        source = replace_once(source, needle,
                              '''        trace.geometry(r, tgt, uv=tgt.uv, times=tgt.t, focal=f_cal, principal=[cx, cy],
                       clip_frames=vj["fidx"], fps=vj["fps"], stride=vj["stride"], dimensions=[V.W,V.H],
                       radial_depth=Dfn_t(tgt.t), xyz=to3d(tgt.uv, Dfn_t(tgt.t), f_cal, cx, cy)[0],
                       sample_indices=js if kind == "size" else None, pixel_extent=e if kind == "size" else None, result=v)
''' + needle)
        source = replace_once(source, '            vals = np.array(vals)\n            if np.isfinite(vals).sum()',
                              '''            vals = np.array(vals)
            trace.log("focal_grid", question_id=int(r["id"]), video_id=vj["video_id"],
                      artifact=trace.blob(focals_px=grid, prior_predictions_si=vals, prior_value_si=pr["value_si"]))
            if np.isfinite(vals).sum()''')
    if name in ("geo/track.py", "geo/track_v2.py", "geo/person_focal.py"):
        lines = source.splitlines()
        for ix in reversed(range(len(lines))):
            if 'res = gp.post_process_grounded_object_detection' in lines[ix]:
                pad = lines[ix][:len(lines[ix]) - len(lines[ix].lstrip())]
                lines.insert(ix, pad + 'trace.log("gdino_forward", locals(), input_pixels=trace.blob(pixels=x), artifact=trace.blob(input_ids=tok.input_ids, logits=out.logits, boxes=out.pred_boxes))')
            if 'o = om(**inp)' in lines[ix]:
                pad = lines[ix][:len(lines[ix]) - len(lines[ix].lstrip())]
                lines.insert(ix + 1, pad + 'trace.log("owlv2_forward", locals(), input_pixels=trace.blob(pixels=inp.pixel_values), artifact=trace.blob(input_ids=inp.input_ids, logits=o.logits, boxes=o.pred_boxes))')
        source = "\n".join(lines) + "\n"
    if name.startswith("geo/") and "sp.post_process_masks([o.pred_masks]" in source:
        lines = source.splitlines()
        for ix in reversed(range(len(lines))):
            if 'sp.post_process_masks([o.pred_masks]' in lines[ix]:
                pad = lines[ix][:len(lines[ix]) - len(lines[ix].lstrip())]
                lines.insert(ix, pad + 'trace.arrays("sam2_output", dict(locals(), j=int(o.frame_idx)), logits=o.pred_masks, scores=o.object_score_logits)')
        source = "\n".join(lines) + "\n"
    if name == "geo/sa2va_track.py":
        source = replace_once(
            source,
            "        last = torch.cat(",
            '        trace.log("sa2va_generation", locals(), prompt=text, response=pred, key_frames=key_idx, '
            'artifact=trace.blob(**{k: v for k, v in mm.items() if k != "pixel_values"}), '
            'media_artifact=trace.blob(pixel_values=mm["pixel_values"]) if "pixel_values" in mm else None)\n'
            "        last = torch.cat(",
        )
        source = replace_once(source, "            st.append(tc.mask_stats(m))",
                              '            trace.arrays("sa2va_mask", locals(), mask=np.packbits(m, axis=1), dimensions=m.shape)\n            st.append(tc.mask_stats(m))')
    if name.startswith("geo/"):
        lines = source.splitlines()
        for ix in reversed(range(len(lines))):
            line = lines[ix]
            if ("load_video" in line and "vpath[v]" in line and "=" in line):
                pad = line[:len(line) - len(line.lstrip())]
                lines.insert(ix, pad + "trace.identify(v)")
        source = "\n".join(lines) + "\n"
    return f"from {PACKAGE} import trace\n" + source.replace("torch.bfloat16", "torch.float16")


def stage_sources(root):
    for name in source_files():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = (UPSTREAM_DIR / name).read_text(encoding="utf-8")
        adapted = adapt(name, source)
        compile(adapted, str(path), "exec")
        path.write_text(adapted, encoding="utf-8")


def isolate_inputs(root):
    from .data import CAW_BASELINE

    project = UPSTREAM_DIR.parents[2]
    local_data = root / "data"
    model_paths = tuple(p.resolve() for p in (root / "models").iterdir())

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root):
            if path.is_relative_to(local_data) and path.suffix.lower() not in {".mp4", ".csv"}:
                raise PermissionError(f"Unrecognized worker input: {path.name}")
            return
        if any(path.is_relative_to(model) for model in model_paths):
            return
        if path == CAW_BASELINE or (path.is_relative_to(project) and (
            path.suffix.lower() in {".csv", ".parquet", ".tar", ".xz", ".zip", ".pkl"}
            or path.is_relative_to(UPSTREAM_DIR / "cache")
        )):
            raise PermissionError(f"Worker cannot read prediction, reference or cached stage data: {path.name}")

    sys.addaudithook(audit)


def execute_script(path, argv):
    from . import trace

    isolate_inputs(Path(os.environ["QP_ROOT"]))
    trace.start()
    if path.name in {"parse.py", "vlm_ground.py", "run_vlm.py"}:
        adapter = types.ModuleType("vllm")
        adapter.LLM, adapter.SamplingParams = trace.Qwen, trace.SamplingParams
        sys.modules["vllm"] = adapter
    sys.argv = [str(path), *argv]
    sys.path.insert(0, str(path.parent))
    try:
        runpy.run_path(str(path), run_name="__main__")
        if path.name == "measure_v3.py":
            tag = argv[argv.index("--tag") + 1]
            ensure_measure_columns(Path(os.environ["QP_ROOT"]) / f"geo/out/meas_val_v2{tag}.csv")
    finally:
        trace.write_usage()


class Stages:
    def __init__(self, root, config):
        self.root, self.config = root, config
        self.records = []
        self.latest = []

    def files(self, outputs):
        paths = (self.root / path for path in outputs)
        return sorted({found for path in paths for found in (path.rglob("*") if path.is_dir() else [path]) if found.is_file()})

    def run(self, name, script, args=(), outputs=(), flags=None, models=(), allow_missing=False):
        root, cfg = self.root, self.config
        trace_dir = root / "trace" / name
        receipt = trace_dir / "receipt.json"
        key = fingerprint([cfg["input_key"], name, script, args, flags, models, self.latest])
        if receipt.is_file():
            saved = read_json(receipt)
            if saved["key"] == key and saved["status"] == "ok" and all(
                (root / path).is_file() and sha256_file(root / path) == digest
                for path, digest in saved["files"].items()
            ):
                self.records.append(saved)
                self.latest = [saved["event_id"]]
                return True
        for path in self.files(outputs):
            path.unlink()
        trace_dir.mkdir(parents=True, exist_ok=True)
        event = uuid.uuid4().hex
        attempts = len(list(trace_dir.glob("stdout-*.txt"))) + 1
        log_path = trace_dir / f"stdout-{attempts}.txt"
        err_path = trace_dir / f"stderr-{attempts}.txt"
        event_started = time.time()
        command = [sys.executable, "-u", "-m", f"{PACKAGE}.worker", "--script", str(root / script), "--", *args]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("QP_", "RL_", "XG_"))}
        env.update(QP_ROOT=str(root), QP_PATCH_STRICT="1", PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(root / "hf-offline"),
                   HF_MODULES_CACHE=str(root / "hf-modules"), CUDA_VISIBLE_DEVICES=cfg["gpu"] if models else "",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   PYTHONPATH=str(UPSTREAM_DIR.parents[2]), IIIXR_TRACE_DIR=str(trace_dir),
                   IIIXR_STAGE_EVENT=event, IIIXR_MOLMO=cfg.get("molmo", "disabled"), **(flags or {}))
        record = {"stage": name, "script": script, "args": list(args), "event_id": event,
                  "parent_events": self.latest, "key": key, "seed": 0, "flags": flags or {},
                  "models": {m: cfg["models"].get(m) for m in models}, "gpu": cfg["gpu"] if models else None,
                  "start": event_started, "scope": cfg["scope"], "input": "data/val/validation_dataset.csv",
                  "stdout": log_path.relative_to(root).as_posix(), "stderr": err_path.relative_to(root).as_posix()}
        try:
            with log_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
                unavailable = [m for m in models if m != "molmo" and not cfg["models"].get(m, {}).get("available")]
                if unavailable:
                    raise RuntimeError(f"Model or dependency unavailable: {', '.join(unavailable)}")
                if "molmo" in models and not cfg["models"].get("molmo", {}).get("available"):
                    env["IIIXR_MOLMO"] = "disabled"
                proc = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=err, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"Worker exited {proc.returncode}")
            missing = [item for item in outputs if not (root / item).exists()]
            if missing and not allow_missing:
                raise RuntimeError(f"Missing expected outputs: {missing}")
            record["status"] = "ok"
        except Exception as exc:
            record.update(status="error", error=str(exc))
        report = log_path.read_text(encoding="utf-8", errors="replace")
        error_output = err_path.read_text(encoding="utf-8", errors="replace")
        record["gpu_usage"] = [json.loads(line.removeprefix("IIIXR_GPU_USAGE ")) for line in report.splitlines()
                               if line.startswith("IIIXR_GPU_USAGE ")]
        record.update(end=time.time(), duration_sec=time.time() - event_started, attempt=attempts,
                      oom=("out of memory" in (report + error_output).lower() or "seg 0 cover 0.00 'OOM'" in report),
                      files={p.relative_to(root).as_posix(): sha256_file(p)
                             for p in self.files([*outputs, *[str(trace_dir.relative_to(root) / x) for x in ("arrays", "calls.jsonl")]])})
        write_json_atomic(receipt, record)
        self.records.append(record)
        self.latest = [event]
        print(f"{root.name} {name} {record['status']} ({record['duration_sec']:.1f}s)", flush=True)
        return record["status"] == "ok"


def ensure_measure_columns(path):
    import pandas as pd

    df = pd.read_csv(path)
    changed = False
    for field in ("val_si", "value", "mode", "cam_mag"):
        if field not in df:
            df[field] = float("nan")
            changed = True
    if changed:
        df.to_csv(path, index=False)


def run_chain(root, config):
    stages = Stages(root, config)
    run = stages.run
    branch_issues = []

    def require(result):
        if not result:
            raise RuntimeError(stages.records[-1]["stage"])

    def measure(label, tag, tracks="v2", *, parse_tag="", options=FLAGS):
        output = f"geo/out/meas_val_v2{tag}.csv"
        require(run(label, "geo/measure_v3.py", ("--split", "val", "--fix", "abcdef", "--tag", tag,
                    "--tracks", tracks, "--moge", "--moge_opts", "fbrej", "--parse_tag", parse_tag),
                    outputs=(output,), flags=options))

    def arbitrate(tag, dest, options=FLAGS):
        require(run(f"arbitrate{tag}", "geo/arbitrate_v2.py",
                    ("--split", "val", "--meas_tag", tag, "--tag", dest, "--quiet"),
                    outputs=(f"preds/val_{dest}.csv", f"preds/val_{dest}_raw.csv"), flags=options))

    def ruler_and_scene(tag, flags):
        base = "geo_v4nt" + tag
        require(run(f"ruler_m2_{tag}", "analysis/v3/ruler_build.py", ("val", "m2"),
                    outputs=(f"preds/val_{base}srng_d_vw.csv", f"preds/val_{base}srng_d_vw_raw.csv"),
                    flags={**flags, "RL_BASE_M2": base + "g_d_vw", "RL_OUT": base + "srng"}))
        require(run(f"scene_ruler_restore_{tag}", "analysis/v3/sruler_apply.py", ("val", tag), flags=flags,
                    outputs=(f"preds/val_{base}srngsr_d_vw.csv", f"preds/val_{base}srngsr_d_vw_raw.csv")))

    def fuse(tag, left, right, flags):
        base = "geo_v4nt" + tag
        tm, tv = f"_v3m_r{tag}", f"_v3v_rnt{tag}"
        measure(f"measure{tm}", tm, left, options=flags)
        measure(f"measure{tv}", tv, right, options=flags)
        arbitrate(tm, f"geo{tm}", flags)
        arbitrate(tv, f"geo{tv}", flags)
        require(run(f"fuse_{tag}", "analysis/v3/fuse_v3.py", ("val", tm, tv, base), flags=flags,
                    outputs=(f"preds/val_{base}_d_vw.csv", f"preds/val_{base}_d_vw_raw.csv")))
        out = "preds/val_" + base + "g"
        require(run(f"xground_{tag}", "analysis/v3/xground.py", ("val",), flags={**flags, "XG_VG": "_notext",
                    "XG_RD": f"geo{tm}", "XG_RV": f"geo{tv}", "XG_BASE": base + "_d_vw", "XG_OUT": base + "g"},
                    outputs=tuple(out + suffix for suffix in ("_d_vw.csv", "_d_vw_raw.csv", "d_d_vw.csv",
                                                             "d_d_vw_raw.csv", "_flags.json"))))
        ruler_and_scene(tag, {**flags, **SELF})

    try:
        require(run("parse", "geo/parse.py", models=("qwen",),
                    outputs=("geo/out/parse_val.jsonl", "geo/out/llm_parse_val.jsonl")))
        require(run("parse_v2", "geo/parse_v2.py", outputs=(PARSE,)))
        require(run("track_v1", "geo/track.py", models=("gdino", "sam2"), outputs=(V1,)))
        require(run("measure_v1", "geo/measure.py", outputs=("geo/out/meas_val.csv",)))
        require(run("affected_v2", "geo/affected_v2.py", ("val",), outputs=("geo/out/retrack_val_v2.txt",)))
        require(run("track_v2", "geo/track_v2.py", ("--video_list", "geo/out/retrack_val_v2.txt"),
                    models=("gdino", "owl", "sam2"), outputs=(V2,)))
        require(run("moge", "geo/moge_run.py", models=("moge",), outputs=(DEPTH,)))
        require(run("vlm_ground", "geo/vlm_ground.py", ("--notext", "--out_tag", "_notext"),
                    models=("qwen",), outputs=("geo/out/vlmground_val_notext.jsonl",)))
        videos = ",".join(sorted({r["video_id"] for r in config["scope"]}))
        require(run("track_v3nt", "geo/track_v3.py", ("--vg_tag", "_notext", "--out_tag", "v3nt", "--videos", videos),
                    models=("sam2",), outputs=(V3,)))
        require(run("prior_copy", "tools/heuristic.py",
                    outputs=tuple(f"preds/val_{k}.csv" for k in ("h0_const1", "h1_prior_copy", "h2_prior_dur"))))
        require(run("qwen_fallback", "tools/run_vlm.py", ("--split", "val", "--mode", "video", "--out",
                    "preds/val_qwen3vl8b_video.csv", "--chunk_videos", "1"), models=("qwen",),
                    outputs=tuple(f"preds/val_qwen3vl8b_video{k}" for k in (".csv", "_raw.csv", "_info.json"))))
        measure("bootstrap_movers", "_v3m_rrj5plm")
        arbitrate("_v3m_rrj5plm", "geo_v4ntrj5pbsrng_d_vw")
        if run("molmo", "geo/molmo_ground.py", models=("molmo",), outputs=("geo/out/molmoground_val.jsonl",)):
            for source in ("v2", "v3nt"):
                require(run(f"point_patch_{source}", "geo/sam2_points.py", ("--src", source),
                            models=("sam2",), outputs=(V1 + f"_{source}mo",)))
        else:
            branch_issues.append("Molmo point prompts unavailable; using unpatched tracks")
            for source in ("v2", "v3nt"):
                shutil.copytree(root / (V1 + f"_{source}"), root / (V1 + f"_{source}mo"), dirs_exist_ok=True)
        require(run("mover_scan", "geo/mover_scan.py", ("--split", "val"), outputs=("geo/out/movers_val.json",),
                    flags={"MV_EXTRA_WHYS": "target_low_det,target_no_track,prior_low_det,prior_no_track"}))
        require(run("mover_masks", "geo/sam2_movers.py", ("--split", "val"), models=("sam2",),
                    outputs=("geo/out/movertracks_val.json",)))
        for source, dest in (("v2mo", "v2mv"), ("v3ntmo", "v3ntmv")):
            require(run(f"patch_{dest}", "geo/patch_movers.py", ("--src", source, "--out", dest),
                        flags={"PM_SAMECLASS_SKIP": "0"}, outputs=(V1 + "_" + dest,)))
        if not run("sa2va", "geo/sa2va_track.py", ("--keys", "first", "--out", "_first"), models=("sa2va",),
                   outputs=("geo/out/sa2va_val_first.json",)):
            branch_issues.append("Sa2VA scene candidates unavailable; leaving mover tracks unchanged")
            write_json_atomic(root / "geo/out/sa2va_val_first.json", {})
        elif stages.records[-1]["oom"]:
            branch_issues.append("Sa2VA reported per-query OOM; inspect scene candidates")
        require(run("scene_ruler", "geo/scene_ruler.py", ("--split", "val"), outputs=("geo/out/scene_ruler_val.json",)))
        require(run("scene_patch", "geo/patch_sa2va.py", ("--mode", "scene", "--src", "v2mv", "--out", "v2sc", "--in", "_first"),
                    outputs=(V1 + "_v2sc",)))
        require(run("ruler_requests", "analysis/v3/ruler_build.py", ("val", "q"),
                    flags={"RL_BASE_Q": "geo_v4ntrj5pbsrng_d_vw"}, outputs=("analysis/v3/ruler_val.json", "caw/sc/inputs/val_ruler.csv")))
        if read_json(root / "analysis/v3/ruler_val.json"):
            raise RuntimeError("Unexpected ruler candidates in a strictly 3D worker")
        (root / "caw/sc/out/val_ruler/g16.jsonl").write_text("", encoding="utf-8")
        require(run("ruler_priors", "analysis/v3/ruler_build.py", ("val", "p"),
                    outputs=tuple(f"geo/out/parse_val_v2_rl{k}.jsonl" for k in range(3))))
        for k in range(3):
            measure(f"ruler_measure_{k}", f"_rl{k}", parse_tag=f"_rl{k}", options={})
            arbitrate(f"_rl{k}", f"geo_rl{k}", {})
        fuse("rj5sd", "v2sc", "v3ntmv", FLAGS)
        for pattern in ("preds/val_geo_v4ntrj5sdg_d_vw{}.csv", "geo/out/meas_val_v2_v3m_rrj5sd{}.csv",
                        "geo/out/meas_val_v2_v3v_rntrj5sd{}.csv"):
            for suffix in ("", "_raw") if pattern.startswith("preds/") else ("",):
                source = pattern.format(suffix)
                shutil.copyfile(root / source, root / source.replace("rj5sd", "rj5sdG"))
        ruler_and_scene("rj5sdG", {**FLAGS, **SELF, **GRAVITY})
        require(run("person_focal", "geo/person_focal.py", ("val",), models=("gdino",),
                    outputs=("geo/out/person_focal_val.json",)))
        fuse("rj5sQ", "v2sc", "v3ntmv", {**FLAGS, **GRAVITY, **PERSON})
        require(run("person_focal_merge", "geo/personf_merge.py", ("val", "geo_v4ntrj5sQsrngsr_d_vw",
                    "geo_v4ntrj5sdGsrngsr_d_vw", "geo_v4ntrj5sQ2srngsr_d_vw"), outputs=(FINAL, FINAL.replace("_raw", ""))))
    except Exception as exc:
        return {"stages": stages.records, "error": str(exc), "issues": [BOOTSTRAP, *branch_issues]}
    return {"stages": stages.records, "error": None, "issues": [BOOTSTRAP, *branch_issues]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path)
    parser.add_argument("--lane", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.script:
        execute_script(args.script, args.args[1:] if args.args[:1] == ["--"] else args.args)
    else:
        root = args.lane
        write_json_atomic(root / "worker_receipt.json", run_chain(root, read_json(root / "worker.json")))


if __name__ == "__main__":
    main()
