"""Verify row integrity, lineage and offline archive links."""
from __future__ import annotations

import argparse
import collections
import csv
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from ..schema import fingerprint, read_json, sha256_file
from .data import EXPECTED, FIELDS, HERE, SANITY_IDS, finite


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()
        self.panels = 0

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.panels += 1
        for key, value in attrs:
            if key in ("href", "src", "poster") and value:
                self.paths.add(value)


def check(run):
    run = run.resolve()
    manifest = read_json(run / "manifest.json")
    files = manifest["files"]
    for rel, meta in files.items():
        path = (run / rel).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {rel}")
        if path.stat().st_size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
            raise ValueError(f"Corrupted artifact: {rel}")
    cfg = read_json(run / "config.json")
    questions = read_json(run / "questions.json")
    locked = read_json(run / "predictions_locked.json")
    lock = read_json(run / "lock.json")
    if not locked["sealed"] or sha256_file(run / "predictions_locked.json") != lock["sha256"]:
        raise ValueError("Prediction lock mismatch")
    ids = [r["id"] for r in questions]
    expected = read_json(HERE / "validation_inputs.json")["questions"]
    if cfg["scope"] == "sanity":
        wanted = SANITY_IDS
    else:
        groups = sorted({r["video_id"] for r in expected})[cfg["shard"]::cfg["shards"]]
        wanted = {r["id"] for r in expected if r["video_id"] in groups}
    predictions = locked["predictions"]
    if ids != sorted(wanted) or [r["id"] for r in predictions] != ids:
        raise ValueError("Prediction row IDs differ from scope")
    if len(ids) == 90 and collections.Counter(r["category"] for r in questions) != EXPECTED:
        raise ValueError("Full category counts mismatch")
    stages = read_json(run / "stages.json")
    by_event = {s["event_id"]: s for s in stages}
    if len(stages) != len(by_event):
        raise ValueError("Repeated stage IDs")
    for pred, q in zip(predictions, questions, strict=True):
        if q["input_sha256"] != fingerprint({field: q[field] for field in FIELDS}):
            raise ValueError(f"Question fingerprint mismatch: {q['id']}")
        clip = q["clip"]
        if clip.get("path") and clip["sha256"] != files[clip["path"]]["sha256"]:
            raise ValueError(f"Clip fingerprint mismatch: {q['video_id']}")
        success = pred["status"] in {"ok", "provisional"}
        val = finite(pred["parsed_value"])
        if success != (val is not None and val > 0) or (not success and pred["parsed_value"] is not None):
            raise ValueError(f"Invalid success or numeric value: {pred['id']}")
        for event in pred.get("lineage", []):
            if event not in by_event or pred["id"] not in {r["id"] for r in by_event[event]["scope"]}:
                raise ValueError(f"Invalid event {event}: {pred['id']}")
        if cfg["mode"] == "fresh" and success:
            chain = [by_event[event] for event in pred["lineage"]]
            if not chain or chain[0]["stage"] != "parse" or chain[-1]["stage"] != "person_focal_merge":
                raise ValueError(f"Fresh prediction lacks a full chain: {pred['id']}")
            root = pred["worker"]
            inp = f"{root}/data/val/validation_dataset.csv"
            with (run / inp).open(encoding="utf-8", newline="") as handle:
                for item in csv.DictReader(handle):
                    if item["ground_truth_posterior"]:
                        raise ValueError(f"Worker input contains posterior: {inp}")
            if any(s["worker"] != root or s["input"] != "data/val/validation_dataset.csv" for s in chain):
                raise ValueError(f"Fresh prediction has a foreign stage: {pred['id']}")
            if pred["status"] == "ok" and any(s["status"] != "ok" for s in chain):
                raise ValueError(f"Fresh success has a failed stage: {pred['id']}")
    for stage in stages:
        if "worker" not in stage:
            continue
        for rel, digest in stage["files"].items():
            path = f"{stage['worker']}/{rel}"
            # A failed optional producer may be replaced by a documented empty candidate file.
            if stage["status"] == "ok" and (path not in files or files[path]["sha256"] != digest):
                raise ValueError(f"Stage artifact checksum mismatch: {path}")
        if any(event not in by_event for event in stage["parent_events"]):
            raise ValueError(f"Missing stage parent: {stage['stage']}")
    with (run / "per_question.csv").open(newline="", encoding="utf-8") as fh:
        if [int(row["id"]) for row in csv.DictReader(fh)] != ids:
            raise ValueError("Per-question scores differ from the sealed question IDs")
    if cfg["mode"] == "replay" and len(ids) == 90:
        if collections.Counter(r["source"] for r in predictions) != {"meas": 81, "vlm": 9}:
            raise ValueError("Replay route audit mismatch")
    parser = Links()
    parser.feed((run / "index.html").read_text(encoding="utf-8"))
    if parser.panels != len(ids) or parser.paths - files.keys():
        raise ValueError(f"Offline report has bad links or missing panels: {parser.paths - files.keys()}")
    return {"questions": len(ids), "artifacts": len(files), "mode": cfg["mode"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.path.is_file():
        from .workflow import extract_run

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            extract_run(args.path, run)
            print(check(run))
    else:
        print(check(args.path))


if __name__ == "__main__":
    main()
