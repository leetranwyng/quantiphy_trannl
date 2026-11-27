#!/usr/bin/env python3
"""Run from an extracted Kaggle output with Python 3.11+ and no network."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if (HERE / "source" / "code_as_world").is_dir():
    sys.path.insert(0, str(HERE / "source"))

from code_as_world.kaggle.artifacts import (
    GROUPS, INTEGRITY, LABELS, check_reference, fingerprint, read_json, read_prediction_rows,
    relative, require, resolve, sample_id, score_rows,
)
from code_as_world.kaggle.artifacts import parse_answer


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name, value in attrs if name in {"href", "src"} and value)


def verify_html(run, path):
    html = Links()
    html.feed(path.read_text(encoding="utf-8"))
    for link in html.links:
        parsed = urlsplit(link)
        require(not parsed.scheme and not parsed.netloc, f"External link in {path.name}: {link}")
        if parsed.path:
            target = path.parent / unquote(parsed.path)
            resolve(run, relative(run, target))


def check_npy(path, ref, read_ints=False):
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"Invalid NPY header: {path}")
        version = stream.read(2)[0]
        require(version in {1, 2, 3}, f"Unsupported NPY version: {version}")
        size = struct.unpack("<H" if version == 1 else "<I", stream.read(2 if version == 1 else 4))[0]
        header = ast.literal_eval(stream.read(size).decode("utf-8"))
        code = header["descr"]
        kinds = {"f": "float", "i": "int", "u": "uint", "b": "bool"}
        require(code[1] in kinds and code[2:].isdigit(), f"Unexpected dtype: {code}")
        item_size = int(code[2:])
        dtype = "bool" if code[1] == "b" else f"{kinds[code[1]]}{item_size * 8}"
        shape = list(header["shape"])
        require(shape == ref["shape"] and dtype == ref["dtype"], f"NPY metadata mismatch: {ref['path']}")
        require(not header["fortran_order"], f"Unexpected Fortran array order: {ref['path']}")
        element_count = math.prod(shape)
        require(path.stat().st_size - stream.tell() == element_count * item_size, f"Truncated NPY: {ref['path']}")
        if read_ints:
            require(code == "<i8", f"Expected little-endian int64 token IDs: {ref['path']}")
            return list(struct.unpack(f"<{element_count}q", stream.read()))
    return None


def verify_attempt(run, directory, inventory, run_fingerprint):
    result = read_json(directory / "attempt.json")
    refs = read_json(directory / "artifact_refs.json")
    if run_fingerprint is not None:
        require(result["fingerprint"] == run_fingerprint, f"Sample fingerprint mismatch: {directory}")
    paths = {}
    for key, ref in refs.items():
        actual = inventory[ref["path"]]
        require(all(ref[field] == actual[field] for field in ("bytes", "sha256")), f"Artifact ref mismatch: {key}")
        path = resolve(run, ref["path"])
        paths[key] = path
        if path.suffix == ".npy":
            check_npy(path, ref)
    events = [json.loads(line) for line in paths["events.jsonl"].read_text().splitlines()]
    stack, stage_counts = [], Counter()
    for index, event in enumerate(events):
        require(event["id"] == index and event["sample_id"] == result["sample_id"]
                and event["attempt"] == relative(run, directory), f"Broken event identity: {directory}")
        require(all(ref in refs for field in ("input_refs", "output_refs") for ref in event.get(field, [])),
                f"Broken event artifact reference: {directory}")
        if event["phase"] == "end":
            require(stack and stack.pop() == event["call_id"], f"Broken trace nesting: {directory}")
            stage_counts[event["stage"]] += 1
        require(event["parent_call_id"] == (stack[-1] if stack else None), f"Broken trace parent: {directory}")
        if event["phase"] == "begin":
            stack.append(index)
    if result["status"] == "error":
        require("exception.txt" in paths and result.get("stage") and events[-1]["phase"] == "error",
                f"Error attempt has no stage evidence: {directory}")
        return result, None
    require(not stack, f"Unclosed trace calls: {directory}")
    for stage in ("prompt", "fetch_video", "processor", "generate", "parse"):
        require(stage_counts[stage] == 1, f"Missing stage {stage}: {directory}")
    if result["trace_enabled"]:
        for stage in ("model_forward", "vision", "merger", "language_model", "output_head"):
            require(stage_counts[stage] > 0, f"Missing model hook {stage}: {directory}")
        for stage in ("vision", "merger"):
            require(any(k.startswith(stage + ".") and ".output" in k for k in refs),
                    f"Missing activation tensor for {stage}")
        names = read_json(paths["hook_modules.json"])
        called = {e["module"] for e in events if e["phase"] == "end" and "module" in e}
        require(set(names) == called, f"Registered/called model hooks differ: {directory}")
    generation = read_json(paths["generation.json"])
    if result["deep_trace"]:
        lm_steps = [0, 1] if generation["step_count_observed"] > 1 else [0]
        for domain, steps in (("visual", [0]), ("language_model", lm_steps)):
            for step in steps:
                diagnostic_events = [
                    event for event in events if event["phase"] == "end" and event["stage"] == "diagnostic_activation"
                    and f"model.{domain}." in event["module"] and event["step"] == step and event["output_refs"]
                ]
                require(len(diagnostic_events) >= 3, f"Missing {domain} diagnostic tensors at step {step}: {directory}")
    spec = read_json(paths["input_row.json"])
    require(not any(key.casefold() in LABELS for key in [*spec["record"], *spec["source"]]),
            f"Ground truth leaked into worker inputs: {directory}")
    require([m["role"] for m in read_json(paths["messages.json"])] == ["system", "user"],
            f"Incorrect message roles: {directory}")
    video = read_json(paths["video.json"])
    token = read_json(paths["token_mapping.json"])
    value, unit = parse_answer(generation["raw_text"])
    require(read_json(paths["parsed.json"]) == {"value": value, "unit_text": unit},
            f"Parser replay differs: {directory}")
    require(generation["finite_logits"] and generation["generated_token_count"] == len(generation["output_token_ids"]),
            f"Generation count or finite check failed: {directory}")
    pixels = refs["fetch_pixels.npy"]
    count = video["effective_nframes"]
    require(count == pixels["shape"][0] == len(video["frames"]) and pixels["layout"] == "T,C,H,W",
            f"Frame count disagrees with fetch tensor: {directory}")
    expected_pngs = {f"frame_{i:03d}.png" for i in range(count)}
    require(expected_pngs == {key for key in refs if re.fullmatch(r"frame_\d+\.png", key)}, f"Missing PNG: {directory}")
    for i, frame in enumerate(video["frames"]):
        png = paths[f"frame_{i:03d}.png"]
        with png.open("rb") as stream:
            header = stream.read(24)
        require(header.startswith(b"\x89PNG\r\n\x1a\n") and
                list(struct.unpack(">II", header[-8:])) == pixels["shape"][:1:-1],
                f"Preprocessor frame PNG shape mismatch: {png}")
        require(frame["frame_index"] == i and frame["model_index"] == video["processor_metadata"]["frames_indices"][i]
                and math.isclose(frame["processor_time_seconds"], frame["model_index"] / video["processor_fps"]),
                f"Frame timestamp mismatch: {directory}")
    require(bool(video["decoder_pts_source"]) == all(f["decord_pts_seconds"] is not None for f in video["frames"]),
            f"Decoder timestamp provenance mismatch: {directory}")
    require(video["effective_nframes"] == video["requests"][-1]["nframes"], f"Effective frame count mismatch: {directory}")
    rendered = paths["rendered_prompt.txt"].read_text(encoding="utf-8")
    proc_text = paths["processor_text.txt"].read_text(encoding="utf-8")
    repl = token["replacement"]
    start, end = repl["span"]
    require(proc_text[start:end] == repl["text"] and
            rendered == proc_text[:start] + token["vision_prefix"] + repl["text"] + token["vision_suffix"] + proc_text[end:],
            f"Prompt and processor placeholder mapping mismatch: {directory}")
    require(paths["expanded_prompt.txt"].read_text(encoding="utf-8") ==
            proc_text[:start] + repl["replacement"] + proc_text[end:], f"Expanded prompt text mismatch: {directory}")
    ids = token["special_ids"]
    pre, post = token["pre_input_ids"], token["post_input_ids"]
    require(post == check_npy(paths["processor.input_ids"], refs["processor.input_ids"], read_ints=True),
            f"Expanded token IDs differ from processor tensor: {directory}")
    require(token["grid_thw"] == check_npy(paths["processor.video_grid_thw"], refs["processor.video_grid_thw"], read_ints=True),
            f"Visual grid differs from processor tensor: {directory}")
    video_pos = pre.index(ids["video"])
    require(pre.count(ids["video"]) == 1 and pre[video_pos-1:video_pos+2] ==
            [ids["vision_start"], ids["video"], ids["vision_end"]], f"Original placeholder is invalid: {directory}")
    t, h, w = token["grid_thw"]
    span_width = h * w // token["merge_size"]**2
    starts = [i for i, v in enumerate(post) if v == ids["vision_start"]]
    ends = [i for i, v in enumerate(post) if v == ids["vision_end"]]
    require(len(starts) == len(ends) == t and all(
        end - start == span_width + 1 and post[start+1:end] == [ids["video"]] * span_width
        for start, end in zip(starts, ends, strict=True)
    ) and post.count(ids["video"]) == token["visual_token_count"] == t * span_width,
        f"Invalid expanded visual tokens: {directory}")
    indices = [f["model_index"] for f in video["frames"]]
    temporal = token["temporal_patch_size"]
    indices += [indices[-1]] * (-len(indices) % temporal)
    timestamps = [
        f"{(indices[i] / video['processor_fps'] + indices[i+temporal-1] / video['processor_fps']) / 2:.1f}"
        for i in range(0, len(indices), temporal)
    ]
    require(timestamps == token["processor_timestamps_text"], f"Processor time text mismatch: {directory}")
    required_fields = {"processor.input_ids", "processor.attention_mask", "processor.mm_token_type_ids",
                       "processor.video_grid_thw", "processor.pixel_values_videos"}
    require(required_fields <= refs.keys(), f"Missing model input tensors: {directory}")
    return result, generation


def check_official(run, metrics):
    with (run / "official_metrics" / "all_model_results.csv").open(newline="", encoding="utf-8") as handle:
        upstream = next(csv.DictReader(handle))
    comparable = {"mra_average": metrics["mra_macro"], "invalid_percentage": metrics["invalid_percentage"]}
    comparable.update({f"mra_{group}": score for group, score in metrics["mra_by_group"].items()})
    for key, score in comparable.items():
        actual = float(upstream[key]) if upstream.get(key) else None
        require(actual is None if score is None else actual is not None and math.isclose(score, actual, abs_tol=1e-12),
                f"Official evaluator disagrees on {key}: {score} vs {actual}")


def verify(run, require_pass=False):
    run = run.resolve()
    files = read_json(run / INTEGRITY)
    inventory = {ref["path"]: ref for ref in files}
    require(len(files) == len(inventory), "Duplicate inventory paths")
    actual_files = {relative(run, path) for path in run.rglob("*")
                    if path.is_file() and path.name != INTEGRITY and "__pycache__" not in path.parts}
    require(actual_files == inventory.keys(), "Output inventory is incomplete or contains unexpected files")
    for ref in files:
        checked = check_reference(run, ref)
        if checked.suffix == ".html":
            verify_html(run, checked)

    manifest = read_json(run / "manifest.json")
    require(fingerprint(manifest["identity"]) == manifest["fingerprint"], "Run fingerprint does not match identity")
    plan = read_json(run / "configs" / "plan.json")
    rows = read_prediction_rows(run / "predictions.csv")
    expected = [spec["id"] for spec in plan["samples"]]
    require([r["sample_id"] for r in rows] == expected and len(set(expected)) == len(expected),
            "Prediction rows are missing, duplicated or out of order")
    trace_rows = [json.loads(line) for line in (run / "trace_index.jsonl").read_text().splitlines()]
    require([x["sample_id"] for x in trace_rows] == expected, "Trace index does not cover the plan")
    csv_path = run / "configs" / "quantiphy_validation.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        label_column = header.index("ground_truth_posterior")
        gold = {sample_id(manifest["identity"]["csv_sha256"], n): cells[label_column]
                for n, cells in enumerate(reader, start=2) if any(value.strip() for value in cells)}
    require(inventory["configs/quantiphy_validation.csv"]["sha256"] == manifest["identity"]["csv_sha256"],
            "Ground truth CSV checksum mismatch")
    require(len(gold) == manifest["dataset"]["questions"], "Dataset row count mismatch")
    if manifest["mode"] in {"full", "merge"} and manifest["partition_count"] == 1:
        require(set(expected) == set(gold), "Full run is missing validation rows")
    for row, item, spec in zip(rows, trace_rows, plan["samples"], strict=True):
        key = row["sample_id"]
        require(row["ground_truth"] == gold[key], f"Joined ground truth differs for {key}")
        require(row["group"] == spec["record"]["category"] and row["video_id"] == spec["record"]["video_id"],
                f"CSV metadata differs for {key}")
        for attempt in item["attempts"]:
            expected_fingerprint = manifest.get("sample_fingerprints_from_parts", {}).get(key, manifest["fingerprint"])
            status, generation = verify_attempt(run, run / attempt, inventory, expected_fingerprint)
            if attempt == item["selected"]:
                require(row["status"] == status["status"] and status["sample_id"] == key, f"Wrong final attempt: {key}")
                raw = generation["raw_text"] if generation else ""
                require(row["raw_response"] == raw and row["parsed_value"] == parse_answer(raw)[0],
                        f"Prediction CSV does not match raw generation: {key}")
        require(item["selected"] in item["attempts"] if item["selected"] else row["status"] == "deferred",
                f"Missing final-status provenance for {key}")

    for path in sorted((run / "preflight").glob("*/samples/*/attempt-*/attempt.json")):
        verify_attempt(run, path.parent, inventory, None)

    metrics = score_rows(rows)
    require(read_json(run / "metrics.json")["metrics"] == metrics, "Offline MRA replay differs")
    check_official(run, metrics)
    report = {
        "verified_files": len(files), "samples": len(rows), "status_counts": metrics["status_counts"],
        "fingerprint": manifest["fingerprint"], "mode": manifest["mode"],
    }
    if require_pass:
        sanity = read_json(run / "sanity_report.json")
        require(manifest["mode"] == "sanity" and sanity["status"] == "PASS"
                and sanity["fingerprint"] == manifest["fingerprint"], "Sanity outcome is not PASS for this run")
        require(metrics["status_counts"].get("ok", 0) > 0 and
                set(metrics["status_counts"]) <= {"ok", "parse_error"} and set(metrics["counts"]) == set(GROUPS),
                "Saved sanity output has failures or missing groups")
        require(all(value is True for value in sanity["checks"].values()) and len(sanity["checks"]) >= 6,
                "Sanity checks did not all pass")
        require(len(manifest["used_gpu_uuids"]) == 2, "Sanity did not exercise both T4s")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=HERE)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.require_pass), indent=2))
