"""Portable output format and the CPU-only parts of the evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from code_as_world import evaluation as upstream


GROUPS = ("S2", "D2", "S3", "D3")
LABELS = {"ground_truth_posterior", "answer", "ground_truth", "gt"}
SCHEMA = 1
INTEGRITY = "integrity.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_text(path, content):
    atomic_write(path, lambda stream: stream.write(content.encode("utf-8")))


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, entries):
    write_text(path, "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in entries))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(run, path):
    return path.relative_to(run).as_posix()


def reference(run, path, **metadata):
    return {"path": relative(run, path), "sha256": sha256(path), "bytes": path.stat().st_size, **metadata}


def resolve(run, name):
    candidate = (run / name).resolve()
    require(candidate.is_relative_to(run.resolve()), f"Path leaves the output: {name}")
    require(candidate.is_file(), f"Missing output file: {name}")
    return candidate


def check_reference(run, ref):
    path = resolve(run, ref["path"])
    require(path.stat().st_size == ref["bytes"], f"Length mismatch: {ref['path']}")
    require(sha256(path) == ref["sha256"], f"Checksum mismatch: {ref['path']}")
    return path


def blob(run, suffix, writer, **metadata):
    directory = run / "blobs"
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=directory, prefix=".writing-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        digest = sha256(temporary)
        target = directory / digest[:2] / f"{digest}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        size = temporary.stat().st_size
        if not target.exists():
            os.replace(temporary, target)
        return {"path": relative(run, target), "sha256": digest, "bytes": size, **metadata}
    finally:
        temporary.unlink(missing_ok=True)


def copy_file(src, dst):
    def copy(stream):
        with src.open("rb") as original:
            shutil.copyfileobj(original, stream)
    atomic_write(dst, copy)


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def seal(run):
    files = [
        reference(run, path)
        for path in sorted(run.rglob("*"))
        if path.is_file() and path.name != INTEGRITY
        and not path.name.startswith(".writing-") and "__pycache__" not in path.parts
    ]
    write_json(run / INTEGRITY, files)
    return files


def file_bytes(run):
    return sum(path.stat().st_size for path in run.rglob("*") if path.is_file())


def sample_id(csv_sha, source_record_number):
    return f"{csv_sha[:16]}-r{source_record_number:05d}"


def units_in_question(question):
    match = re.search(r"\bin\s+([\w°/%²³^−.]+(?:\s+[\w°/%²³^−.]+){0,3})[?.!]\s*$", question)
    return {"text": match.group(1), "source": "trailing 'in …' phrase in CSV question"} if match else None


def load_dataset(csv_path, videos):
    records, rows, columns = upstream._load_csv(csv_path, Path("."), ".mp4")
    csv_sha = sha256(csv_path)
    specs, labels = [], {}
    for record, cells in zip(records, rows, strict=True):
        source = {name: cells[i] if i < len(cells) else "" for name, i in columns}
        record_number = record["source_record_number"]
        key = sample_id(csv_sha, record_number)
        labels[key] = record.pop("answer")
        require(record["category"] in GROUPS, f"Unrecognized group in CSV record {record_number}")
        record["video"] = videos[record["video_id"]]["path"]
        specs.append({
            "id": key, "row_number": record_number, "line_end": record["source_line_end"],
            "record": record, "source": {k: v for k, v in source.items() if k.casefold() not in LABELS},
            "expected_unit": units_in_question(source["question"]),
            "video": videos[record["video_id"]],
        })
    return specs, labels, csv_sha


def parse_answer(raw):
    parsed = upstream._parse_prediction(raw)
    if parsed is None or not math.isfinite(parsed):
        return None, None
    text = upstream._strip_answer_tags(raw)
    match = upstream.NUMBER_PATTERN.search(text)
    unit = text[match.end():].strip() or None
    return parsed, unit


def score_rows(rows):
    """Replay the QuantiPhy evaluator's per-group and macro metrics in stdlib Python."""
    scores = {group: [] for group in GROUPS}
    invalid = 0
    for row in rows:
        prediction = row["parsed_value"]
        invalid += int(prediction is None or prediction == 0)
        answer = float(row["ground_truth"])
        if answer == 0 or not math.isfinite(answer):
            continue
        score = 0.0
        if prediction is not None:
            error = abs(abs(prediction) - answer) / answer
            score = sum(error < 1 - theta for theta in upstream.MRA_THRESHOLDS) / len(upstream.MRA_THRESHOLDS)
        scores[row["group"]].append(score)
    means = {group: (math.fsum(values) / len(values) if values else None) for group, values in scores.items()}
    return {
        "counts": dict(Counter(row["group"] for row in rows)),
        "mra_denominator_counts": {group: len(values) for group, values in scores.items()},
        "mra_by_group": means,
        "mra_macro": math.fsum(means.values()) / len(GROUPS) if all(x is not None for x in means.values()) else None,
        "invalid_percentage": (invalid / len(rows) * 100) if rows else 0.0,
        "status_counts": dict(Counter(row["status"] for row in rows)),
    }


def choose_sanity(specs):
    def source_frames(s):
        return s["video"]["probe"]["frames"] or float("inf")
    def raw_area(s):
        probe = s["video"]["probe"]
        return probe["height"] * probe["width"]
    short = min(specs, key=source_frames)
    large = max(specs, key=raw_area)
    group_examples = [next(s for s in specs if s["record"]["category"] == group and
                           (group[1] == "2" or s["record"]["depth_info"])) for group in GROUPS]
    same_video = next(s for s in specs if s["record"]["video_id"] == short["record"]["video_id"]
                      and s["id"] != short["id"])
    selected = {s["id"] for s in [short, large, same_video, *group_examples]}
    setup = {
        "shortest_video": short["id"], "largest_source_resolution": large["id"],
        "shared_video": [short["id"], same_video["id"]],
        "deep_examples": [s["id"] for s in group_examples],
    }
    return [s for s in specs if s["id"] in selected], setup


def split_by_video(specs, count):
    grouped = {}
    for spec in specs:
        grouped.setdefault(spec["record"]["video_id"], []).append(spec)
    shards = [[] for _ in range(count)]
    for group in sorted(grouped.values(), key=len, reverse=True):
        min(shards, key=len).extend(group)
    return shards


def latest_attempt(root, key):
    attempts = sorted((root / "samples" / key).glob("attempt-*"), key=lambda p: int(p.name.split("-")[-1]))
    complete = [a for a in attempts if (a / "attempt.json").is_file()]
    return (complete[-1] if complete else None), len(attempts)


def resumable(run, root, spec, run_fingerprint):
    attempt, _ = latest_attempt(root, spec["id"])
    if attempt is None:
        return False
    result = read_json(attempt / "attempt.json")
    require(result["fingerprint"] == run_fingerprint, f"Resume fingerprint mismatch for {spec['id']}")
    if result["status"] not in {"ok", "parse_error"}:
        return False
    for ref in read_json(attempt / "artifact_refs.json").values():
        check_reference(run, ref)
    return True


def wrap_links(directory, targets):
    return {label: os.path.relpath(path, directory) for label, path in targets.items()}


def read_prediction_rows(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({**row, "parsed_value": float(row["parsed_value"]) if row["parsed_value"] else None})
    return rows
