"""Pinned validation inputs and read-only access to published audit data."""
from __future__ import annotations

import collections
import csv
import io
import math
import shutil
import tarfile
from pathlib import Path

import cv2

from ..schema import UPSTREAM_DIR, fingerprint, read_json, read_snapshots, sha256_file

HERE = Path(__file__).resolve().parent
CAW_BASELINE = HERE / "caw_4b_validation_3d.json"
PINS = read_snapshots()
DATASET = PINS["dataset"]["repo_id"]
REVISION = PINS["dataset"]["iiixr_revision"]
ARCHIVE = UPSTREAM_DIR / PINS["upstream"]["intermediates_archive"]["path"]
REFERENCE = "preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv"
ALTERNATE = "preds/val_geo_v4ntrj5sQ3nsrngsr_d_vw_raw.csv"
EXPECTED = {"S3": 43, "V3": 21, "A3": 26}
SANITY_IDS = {2345, 1090, 2604, 2607}
FIELDS = ("id", "video_id", "video_source", "video_type", "fps", "inference_type",
          "question", "ground_truth_prior", "depth_info")


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rows(stream):
    reader = csv.reader(stream)
    header = next(reader, [])
    if not header or header[0] not in ("", "id", "Unnamed: 0"):
        raise ValueError(f"Expected a question ID in the first column: {header}")
    # The original validation CSV uses an empty ID header and repeats it on
    # trailing spreadsheet columns. Read by position so they cannot replace the ID.
    columns = [(index, field) for index, field in enumerate(header[1:], 1) if field.strip()]
    named_fields = {"id", *(field for _, field in columns)}
    if len(named_fields) != len(columns) + 1:
        raise ValueError(f"Duplicate named CSV columns: {header}")
    last_named = columns[-1][0] if columns else 0
    by_id = {}
    for cells in reader:
        if not cells:
            continue
        if not last_named < len(cells) <= len(header):
            raise ValueError(f"Malformed CSV record ending on line {reader.line_num}")
        try:
            identity = int(cells[0])
        except ValueError as exc:
            raise ValueError(f"Invalid question ID {cells[0]!r} on CSV record ending on line {reader.line_num}") from exc
        if identity in by_id:
            raise ValueError(f"Duplicate question ID {identity}")
        by_id[identity] = {"id": identity, **{field: cells[index] for index, field in columns}}
    return by_id


def pinned_member(member):
    if sha256_file(ARCHIVE) != PINS["upstream"]["intermediates_archive"]["sha256"]:
        raise ValueError("Upstream archive checksum mismatch")
    with tarfile.open(ARCHIVE) as archive:
        return archive.extractfile(member).read()


def csv_bytes(data):
    return rows(io.StringIO(data.decode("utf-8-sig"), newline=""))


def select(by_id):
    return {identity: {"id": identity, **{field: row[field].replace("\r\n", "\n").replace("\r", "\n").strip()
                                        for field in FIELDS[1:]}}
            for identity, row in by_id.items() if row["video_type"].strip()[:2] in EXPECTED}


def resolve_inputs(dataset_dir, cache, download):
    pinned = read_json(HERE / "validation_inputs.json")
    baseline = {row["id"]: row for row in pinned["questions"]}
    if collections.Counter(row["video_type"][:2] for row in baseline.values()) != EXPECTED:
        raise ValueError("Invalid pinned 3D question counts")
    if len(baseline) != 90 or len({row["video_id"] for row in baseline.values()}) != 12:
        raise ValueError("Invalid pinned 3D manifest scope")
    source = {"dataset": DATASET, "revision": REVISION, "upstream_member": REFERENCE,
              "pinned_inputs_sha256": sha256_file(HERE / "validation_inputs.json")}
    official = dataset_dir / "validation_dataset.csv" if dataset_dir else None
    if download and (official is None or not official.is_file()):
        from huggingface_hub import hf_hub_download

        try:
            official = Path(hf_hub_download(DATASET, "validation_dataset.csv", repo_type="dataset",
                                            revision=REVISION, cache_dir=cache))
        except Exception as exc:
            source["official_download_error"] = str(exc)
            print("Official validation CSV unavailable; using the pinned input-only manifest.", flush=True)
    if official is None or not official.is_file():
        source.update(metadata="upstream_allowlisted_inputs", official_compared=False)
        return baseline, source, None
    with official.open(encoding="utf-8-sig", newline="") as handle:
        raw = rows(handle)
    selected = select(raw)
    extra, missing = sorted(selected.keys() - baseline.keys()), sorted(baseline.keys() - selected.keys())
    changed = {}
    for identity in selected.keys() & baseline.keys():
        diff = [field for field in FIELDS if field != "fps" and selected[identity][field] != baseline[identity][field]]
        if finite(selected[identity]["fps"]) != finite(baseline[identity]["fps"]):
            diff.append("fps")
        if diff:
            changed[identity] = diff
    print(f"Official vs upstream 3D IDs: extra={extra}, missing={missing}, metadata_diffs={changed}")
    if extra or missing or changed:
        raise ValueError("Official validation metadata disagrees with the pinned upstream inputs")
    source.update(metadata="official", official_compared=True, official_csv_sha256=sha256_file(official),
                  delta={"extra": extra, "missing": missing, "changed": changed})
    return selected, source, official


def stage_videos(questions, destination, dataset_dir, cache, caw_tar, download):
    destination.mkdir(parents=True, exist_ok=True)
    candidates = {row["video_id"] for row in questions}
    sources = {}
    if dataset_dir:
        for video in sorted(candidates):
            path = next((p for p in (dataset_dir / "validation_videos" / f"{video}.mp4",
                                     dataset_dir / "validation_videos" / f" {video}.mp4",
                                     dataset_dir / f"{video}.mp4",
                                     dataset_dir / f" {video}.mp4") if p.is_file()), None)
            if path:
                sources[video] = (path, "dataset_mount")
    errors = {}
    if download:
        from huggingface_hub import hf_hub_download

        for video in sorted(candidates - sources.keys()):
            # Note: PaulineLi/QuantiPhy-validation stores captured_0041x with a leading space
            # ("validation_videos/ captured_0041x.mp4"); try canonical path first, then spaced fallback.
            last_exc = None
            for hf_rel in (f"validation_videos/{video}.mp4", f"validation_videos/ {video}.mp4"):
                try:
                    path = hf_hub_download(DATASET, hf_rel, repo_type="dataset",
                                           revision=REVISION, cache_dir=cache)
                    sources[video] = (Path(path), f"{DATASET}@{REVISION}:{hf_rel}")
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
            if last_exc is not None:
                errors[video] = str(last_exc)
    if caw_tar and candidates - sources.keys():
        with tarfile.open(caw_tar) as archive:
            available = {m.name: m for m in archive if m.isfile()}
            for video in sorted(candidates - sources.keys()):
                member = next((m for m in (f"run/videos/{video}.mp4", f"run/videos/ {video}.mp4") if m in available), None)
                if member:
                    path = destination / f"{video}.mp4"
                    with archive.extractfile(available[member]) as src, path.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    sources[video] = (path, f"CAW archive:{member}")
    result = {}
    for video in sorted(candidates):
        selected = sources.get(video)
        if selected is None:
            result[video] = {"status": "missing", "error": errors.get(video, "MP4 unavailable")}
            continue
        src, origin = selected
        path = destination / f"{video}.mp4"
        if src.resolve() != path.resolve():
            shutil.copyfile(src, path)
        cap = cv2.VideoCapture(str(path))
        fps = finite(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = cap.read()
        decoded = int(ok)
        while ok:
            ok, _ = cap.read()
            decoded += int(ok)
        cap.release()
        item = {"origin": origin, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "fps_file": fps, "frames_reported": frame_count, "frames": decoded, "path": f"videos/{path.name}"}
        if decoded and fps:
            item.update(status="ok", height=first.shape[0], width=first.shape[1])
            preview = destination / f"{video}.jpg"
            if cv2.imwrite(str(preview), first):
                item["preview"] = f"videos/{preview.name}"
        else:
            item.update(status="error", error="OpenCV could not decode the clip")
        result[video] = item
    return result


def question_manifest(inputs, videos):
    return [{**r, "category": r["video_type"][:2], "input_sha256": fingerprint(r),
             "clip": videos[r["video_id"]]} for r in sorted(inputs, key=lambda r: r["id"])]


def write_worker_inputs(path, questions):
    # qp_common.OUT_COLS requires this header. Its cells stay empty inside all workers.
    columns = ["Unnamed: 0", *FIELDS[1:], "ground_truth_posterior"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[row[field] for field in FIELDS] + [""] for row in questions])
