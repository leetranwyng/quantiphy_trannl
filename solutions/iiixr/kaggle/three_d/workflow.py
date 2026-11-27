"""Kaggle and local entry point for IIIXR validation 3D."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

from ..schema import UPSTREAM_DIR, fingerprint, read_json, sha256_file, utc_now_iso, write_json_atomic
from ..workflow import _git_head_sha, _query_gpu_devices
from . import data, report, worker

PACKAGE = "solutions.iiixr.kaggle.three_d"
ROOT = UPSTREAM_DIR.parents[2]
TITLE = "IIIXR validation 3D – geo chain + fallback"
MODELS = {key: data.PINS["models"][old] for key, old in
          (("gdino", "grounding_dino"), ("owl", "owlv2"), ("sam2", "sam2"), ("qwen", "qwen3_vl_8b"))}
MODELS.update(moge={"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": "cb0e8bbd6b1e243589717c78e750b1ba4c093acf",
                    "allow_patterns": ["model.pt"]},
              molmo={"repo_id": "allenai/Molmo2-8B", "revision": "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"},
              sa2va={"repo_id": "ByteDance/Sa2VA-Qwen3-VL-4B-SAM3", "revision": "3f2bec126b22761a4607eec674781c138f9f9cca"})
THIRD_PARTY = {
    "MoGe": ("https://github.com/microsoft/MoGe.git", "74fbce054ebed49800de42d0ad0e83495065719a"),
    "utils3d-moge": ("https://github.com/EasternJournalist/utils3d-moge.git", "62f09d58509485564e24d5d9f6aac9ee9ebc0c37"),
}


def run_file_info(run):
    return {p.relative_to(run).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(run.rglob("*")) if p.is_file() and "manifest.json" != p.relative_to(run).as_posix()
            and "__pycache__" not in p.parts and not any(part in {"hf-offline", "hf-modules", "models", "third_party"} for part in p.relative_to(run).parts)}


def zip_run(directory, target):
    temp = target.with_suffix(".zip.part")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(directory / "manifest.json", "run/manifest.json")
        for rel in read_json(directory / "manifest.json")["files"]:
            archive.write(directory / rel, f"run/{rel}")
    temp.replace(target)


def save_checkpoint(run, target, workers):
    files = {}
    temporary = target.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
        def store(path):
            rel = path.relative_to(run).as_posix()
            if rel not in files:
                files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                bundle.write(path, f"run/{rel}")

        store(run / "config.json")
        for root in workers:
            for receipt in sorted(root.glob("trace/*/receipt.json")):
                data = read_json(receipt)
                if data["status"] != "ok":
                    continue
                if not all((root / name).is_file() and sha256_file(root / name) == h
                           for name, h in data["files"].items()):
                    continue
                for rel in [*data["files"], data["stdout"], data["stderr"]]:
                    store(root / rel)
                store(receipt)
        bundle.writestr("run/manifest.json", json.dumps({"title": TITLE, "files": files, "checkpoint": True}))
    temporary.replace(target)


def extract_run(path, directory, prefixes=None):
    with zipfile.ZipFile(path) as archive:
        names = set()
        manifest = json.loads(archive.read("run/manifest.json"))
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if (name.is_absolute() or name.parts[:1] != ("run",) or ".." in name.parts
                    or "\\" in info.filename or ":" in info.filename or len(name.parts) < 2 or name in names
                    or info.is_dir()):
                raise ValueError(f"Invalid archive entry: {info.filename}")
            names.add(name)
        expected = {PurePosixPath("run") / rel for rel in [*manifest["files"], "manifest.json"]}
        if names != expected:
            raise ValueError(f"Archive contents differ from its manifest: {path}")
        for name in sorted(names):
            rel = PurePosixPath(*name.parts[1:]).as_posix()
            if prefixes and not rel.startswith(prefixes):
                continue
            dest = directory / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with archive.open(name.as_posix()) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out)
                entry = manifest["files"].get(rel)
                if entry and (tmp.stat().st_size != entry["bytes"] or sha256_file(tmp) != entry["sha256"]):
                    raise ValueError(f"Checksum mismatch for {name}")
                tmp.replace(dest)
            finally:
                tmp.unlink(missing_ok=True)
    return manifest


def environment(run, fresh, allow_other_gpu):
    try:
        gpus = _query_gpu_devices(require_two_t4=fresh and not allow_other_gpu)
        if fresh:
            import torch

            visible = torch.cuda.device_count()
            if visible < (1 if allow_other_gpu else 2):
                raise RuntimeError(f"PyTorch sees {visible} CUDA devices")
            if not allow_other_gpu and (any(int(g["total_mib"]) < 14000 for g in gpus[:2])
                                        or any(torch.cuda.get_device_capability(i) != (7, 5) for i in range(2))):
                raise RuntimeError("Expected two T4s with separate >= 14,000 MiB allocations")
    except Exception as exc:
        gpus, gpu_error = [], str(exc)
    else:
        gpu_error = None
    ffmpeg = shutil.which("ffmpeg")
    version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0] if ffmpeg else None
    disk = shutil.disk_usage(run)
    versions = {}
    for name in ("torch", "torchvision", "transformers", "vllm", "bitsandbytes", "numpy", "scipy", "opencv-python-headless", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "versions": versions, "gpus": gpus, "gpu_error": gpu_error,
            "ffmpeg": version, "ffprobe": shutil.which("ffprobe"), "disk_free_bytes": disk.free}


def pinned_checkout(cache, name, uri, revision):
    dest = cache / "dependencies" / name
    if not (dest / ".git").is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(dest)], check=True, capture_output=True)
    have = _git_head_sha(dest)
    if have != revision:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth=1", uri, revision], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", revision], check=True, capture_output=True)
    return dest


def model_assets(args):
    from huggingface_hub import snapshot_download

    snapshots = {}
    for key, item in MODELS.items():
        repo_id, rev = item["repo_id"], item["revision"]
        alias = repo_id.split("/")[-1]
        config = {"repo_id": repo_id, "revision": rev, "alias": alias, "available": False}
        snapshots[key] = config
        try:
            if key == "molmo" and args.molmo == "disabled":
                raise RuntimeError("Enable --molmo vllm-bnb with a compatible vLLM installation")
            mounted = args.models_dir / alias / rev if args.models_dir else None
            path = mounted if mounted and mounted.is_dir() else Path(snapshot_download(
                repo_id, revision=rev, cache_dir=args.cache / "hf-models", max_workers=2,
                allow_patterns=item.get("allow_patterns"), local_files_only=args.models_local_only))
            config.update(available=True, path=str(path))
            if key == "moge":
                config["third_party"] = {name: str(pinned_checkout(args.cache, name, uri, version))
                                         for name, (uri, version) in THIRD_PARTY.items()}
        except Exception as exc:
            config.update(available=False, error=str(exc))
    return snapshots


def link_asset(target, source):
    if not target.exists():
        target.symlink_to(source, target_is_directory=source.is_dir())


def lane_root(run, lane_id, scope, videos, assets, env, args):
    root = run / "workers" / lane_id
    for name in ("geo/out", "data/val/validation_videos", "preds", "models", "third_party",
                 "caw/sc/inputs", "caw/sc/out/val_ruler"):
        (root / name).mkdir(parents=True, exist_ok=True)
    declared = {key: {k: v for k, v in asset.items() if k not in {"path", "third_party"}} for key, asset in assets.items()}
    # Model and dependency links are not included in archives.
    for asset in assets.values():
        if asset["available"]:
            link_asset(root / "models" / asset["alias"], Path(asset["path"]))
            for name, src in asset.get("third_party", {}).items():
                link_asset(root / "third_party" / name, Path(src))
    for video_id, clip in videos.items():
        dest = root / "data/val/validation_videos" / f"{video_id}.mp4"
        if not dest.is_file() or sha256_file(dest) != clip["sha256"]:
            shutil.copyfile(run / clip["path"], dest)
    data.write_worker_inputs(root / "data/val/validation_dataset.csv", scope)
    worker.stage_sources(root)
    (root / "geo/out/owl_threshold_v2.json").write_bytes(data.pinned_member(data.PINS["upstream"]["owl_threshold_v2"]["member"]))
    hardware = [{k: gpu[k] for k in ("name", "compute_capability", "total_mib")} for gpu in env["gpus"]]
    input_key = fingerprint([scope, videos, declared, args.molmo, env["versions"], hardware,
                             read_json(run / "config.json")["code_hash"]])
    return root, {"models": declared, "scope": [{"id": r["id"], "video_id": r["video_id"]} for r in scope],
                  "input_key": input_key, "molmo": args.molmo}


def read_fresh(root, questions, exit_code):
    receipt = root / "worker_receipt.json"
    info = read_json(receipt) if receipt.is_file() else {"stages": [], "error": f"worker exited {exit_code}", "issues": []}
    ref = root / worker.FINAL
    has_final = exit_code == 0 and not info["error"] and ref.is_file()
    if has_final:
        with ref.open(encoding="utf-8", newline="") as handle:
            computed = data.rows(handle)
        if computed.keys() != {r["id"] for r in questions}:
            raise ValueError(f"Worker {root.name} generated an unexpected question set")
        for label, suffix in (("detector", "v3m_rrj5sd"), ("vlm_boxes", "v3v_rntrj5sd"),
                              ("detector_person_focal", "v3m_rrj5sQ"), ("vlm_boxes_person_focal", "v3v_rntrj5sQ")):
            table = root / f"geo/out/meas_val_v2_{suffix}.csv"
            with table.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    computed[int(row["id"])].setdefault("candidate_measurements", {})[label] = {
                        "value": data.finite(row.get("value")), "mode": row.get("mode"),
                        "valid": row["ok"] == "True", "rejection": row.get("why"),
                        "unit": row.get("unit"), "focal_over_width": data.finite(row.get("f")),
                        "meters_per_pixel": data.finite(row.get("mpp")),
                    }
    else:
        computed = {}
    for stage in info["stages"]:
        stage["worker"] = f"workers/{root.name}"
    return computed, info


def predictions_from_fresh(args, run, processing, clips, env):
    if not env["gpus"]:
        return {row["id"]: {"status": "blocked", "stage": "gpu_preflight", "reason": env["gpu_error"]}
                for row in processing}, []
    groups = sorted({r["video_id"] for r in processing if clips[r["video_id"]]["status"] == "ok"})
    if not groups:
        return {}, []
    assets = model_assets(args)
    write_json_atomic(run / "model_assets.json", assets)
    jobs = []
    gpus = env["gpus"][:2]
    for ix, gpu in enumerate(gpus):
        group = set(groups[ix::len(gpus)])
        if not group:
            continue
        scope = [r for r in processing if r["video_id"] in group]
        lane_id = fingerprint(sorted(group))[:12]
        root, config = lane_root(run, lane_id, scope, {k: clips[k] for k in group}, assets, env, args)
        write_json_atomic(root / "worker.json", {**config, "gpu": gpu["uuid"]})
        (root / "worker_receipt.json").unlink(missing_ok=True)
        command = [sys.executable, "-u", "-m", f"{PACKAGE}.worker", "--lane", str(root)]
        jobs.append((root, scope, subprocess.Popen(command, cwd=ROOT, start_new_session=os.name == "posix")))
    info_by_id, stages = {}, []
    checkpoint = args.output / "iiixr_3d_checkpoint.zip"
    interval = args.checkpoint_seconds
    deadline = time.monotonic() + interval
    interrupted = False
    try:
        while any(proc.poll() is None for _, _, proc in jobs):
            if interval and time.monotonic() >= deadline:
                try:
                    save_checkpoint(run, checkpoint, [root for root, _, _ in jobs])
                except OSError as exc:
                    print(f"Checkpoint write failed: {exc}", file=sys.stderr, flush=True)
                else:
                    print(f"Checkpoint saved to {checkpoint}", flush=True)
                deadline = time.monotonic() + interval
            time.sleep(2)
        for root, scope, proc in jobs:
            outputs, info = read_fresh(root, scope, proc.wait())
            stages.extend(info["stages"])
            for row in scope:
                path = f"workers/{root.name}"
                identity = row["id"]
                if identity in outputs:
                    prediction = report.prediction(outputs[identity])
                    if prediction["status"] == "ok" and info["issues"]:
                        prediction["status"] = "provisional"
                    entry = {**prediction, "issues": info["issues"],
                             "candidate_measurements": outputs[identity]["candidate_measurements"]}
                else:
                    last = info["stages"][-1]["stage"] if info["stages"] else "worker"
                    entry = {"status": "error", "stage": last, "reason": info["error"], "issues": info["issues"]}
                info_by_id[identity] = {**entry, "worker": path, "lineage": [st["event_id"] for st in info["stages"]]}
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for _, _, proc in jobs:
            if proc.poll() is None:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait()
        if interrupted:
            save_checkpoint(run, checkpoint, [root for root, _, _ in jobs])
            print(f"Resume from {checkpoint}", flush=True)
    if interrupted:
        raise KeyboardInterrupt
    if any(record["status"] != "ok" for record in stages):
        save_checkpoint(run, checkpoint, [root for root, _, _ in jobs])
    return info_by_id, stages


def choose(all_inputs, args):
    groups = sorted({r["video_id"] for r in all_inputs.values()})
    if args.shards < 1 or not 0 <= args.shard < args.shards <= len(groups):
        raise ValueError("Expected 0 <= shard < shards <= 12")
    if args.scope == "sanity" and args.shards != 1:
        raise ValueError("Video sharding applies to the full scope")
    if args.scope == "sanity":
        output = [all_inputs[q] for q in sorted(data.SANITY_IDS)]
    else:
        group = set(groups[args.shard::args.shards])
        output = [r for r in all_inputs.values() if r["video_id"] in group]
    video_ids = {r["video_id"] for r in output}
    return output, [r for r in all_inputs.values() if r["video_id"] in video_ids]


def seal(run, manifest, results, mode, scope, stages):
    locked = {"mode": mode, "scope": scope, "sealed": True, "predictions": []}
    for row in manifest:
        identity = row["id"]
        result = results.get(identity, {"status": "error", "stage": "decode", "reason": row["clip"].get("error")})
        locked["predictions"].append({"id": identity, "video_id": row["video_id"], "category": row["category"],
                                      "video_source": row["video_source"], "parsed_value": None,
                                      "geometry_candidate": None, "source": None, **result})
    write_json_atomic(run / "questions.json", manifest)
    write_json_atomic(run / "stages.json", stages)
    write_json_atomic(run / "predictions_locked.json", locked)
    write_json_atomic(run / "lock.json", {"sha256": sha256_file(run / "predictions_locked.json"), "sealed_at": utc_now_iso()})


def finish(run, args, official):
    evaluation = report.evaluate(run, official, args.caw_tar)
    report.viewer(run)
    from .verify import check

    files = run_file_info(run)
    write_json_atomic(run / "manifest.json", {"files": files, "title": TITLE})
    check(run)
    zip_run(run, args.output / "iiixr_3d.zip")
    print(json.dumps({"output": str(args.output / "iiixr_3d.zip"), "mode": evaluation["mode"],
                      "denominator": evaluation["all"]["n"], "coverage": evaluation["all"]["coverage"],
                      "routes": evaluation["routes"], "summary": str(run / "evaluation.json")}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--scope", choices=("sanity", "full"), default="sanity")
    parser.add_argument("--mode", choices=("fresh", "replay"), default="replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/iiixr_3d_cache")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--caw-tar", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--models-local-only", action="store_true")
    parser.add_argument("--molmo", choices=("disabled", "vllm-bnb"), default="disabled")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-seconds", type=int, default=600)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--merge", type=Path, nargs="+")
    args = parser.parse_args(argv)
    for field in ("output", "cache", "dataset_dir", "caw_tar", "models_dir", "resume"):
        if (value := getattr(args, field)) is not None:
            setattr(args, field, value.resolve())
    run = args.output / "run"
    if run.exists():
        raise ValueError(f"Choose a new --output directory (or use --resume with a new output): {run}")
    if args.resume and args.mode != "fresh":
        raise ValueError("--resume applies to fresh stage receipts")
    run.mkdir(parents=True)
    inputs, provenance, official = data.resolve_inputs(args.dataset_dir, args.cache, args.download)
    if args.merge:
        merge(run, args, inputs, official)
        return
    selected, processing = choose(inputs, args)
    clips = data.stage_videos(processing, run / "videos", args.dataset_dir, args.cache, args.caw_tar, args.download)
    env = environment(run, args.mode == "fresh", args.allow_other_gpu)
    write_json_atomic(run / "environment.json", env)
    source = run / "source"
    source.mkdir()
    pkg = Path(__file__).parent
    files = [*pkg.glob("*.py"), pkg / "validation_inputs.json", pkg / "requirements.txt"]
    for path in [*files, data.CAW_BASELINE]:
        shutil.copyfile(path, source / path.name)
    code_hash = fingerprint({p.name: sha256_file(p) for p in files}
                            | {name: sha256_file(UPSTREAM_DIR / name) for name in worker.source_files()})
    bundle = ROOT / "iiixr_3d_source.json"
    cfg = {"code_hash": code_hash, "source_commit": _git_head_sha(),
                       "upstream_commit": data.PINS["upstream"]["pinned_commit"], "dataset": provenance,
                       "scope": args.scope, "shard": args.shard, "shards": args.shards, "mode": args.mode,
                       "models": MODELS if args.mode == "fresh" else {}, "third_party": THIRD_PARTY,
                       "molmo": args.molmo, "qwen_profile": "Transformers NF4 FP16 SDPA, one model per process",
                       "command": shlex.join([sys.executable, "-m", f"{PACKAGE}.workflow", *sys.argv[1:]])}
    if bundle.is_file():
        cfg["source_bundle"] = read_json(bundle)
    write_json_atomic(run / "config.json", cfg)
    if args.resume:
        extract_run(args.resume, run, ("workers/", "config.json"))
        previous = read_json(run / "config.json")
        if any(previous.get(k) != cfg[k] for k in ("code_hash", "mode", "scope", "shard", "shards", "molmo")):
            raise ValueError("Resume requires the same code, mode, scope, shard and model profile")
        write_json_atomic(run / "config.json", cfg)
    if args.mode == "replay":
        results, stages = report.replay(run, [r["id"] for r in selected])
    else:
        results, stages = predictions_from_fresh(args, run, processing, clips, env)
    seal(run, data.question_manifest(selected, clips), results, args.mode, args.scope, stages)
    finish(run, args, official)


def merge(run, args, inputs, official):
    from .verify import check

    config, records, questions, stages, seen = None, {}, [], {}, set()
    for ix, archive in enumerate(args.merge):
        part = args.output / "merge_parts" / str(ix)
        extract_run(archive, part)
        check(part)
        cfg = read_json(part / "config.json")
        identity = cfg["shard"]
        required = ("code_hash", "upstream_commit", "dataset", "mode", "shards", "molmo", "qwen_profile")
        if cfg["scope"] != "full" or (config and any(cfg[k] != config[k] for k in required)) or identity in seen:
            raise ValueError(f"Incompatible or duplicate shard: {archive}")
        config = cfg
        seen.add(identity)
        preds = read_json(part / "predictions_locked.json")["predictions"]
        if records.keys() & {r["id"] for r in preds}:
            raise ValueError(f"Overlapping question IDs: {archive}")
        records.update({r["id"]: r for r in preds})
        questions.extend(read_json(part / "questions.json"))
        stages.update({s["event_id"]: s for s in read_json(part / "stages.json")})
        for folder in ("videos", "workers", "source", "evidence"):
            if (part / folder).is_dir():
                shutil.copytree(part / folder, run / folder, dirs_exist_ok=True)
        for name in ("environment.json", "model_assets.json"):
            if (part / name).is_file():
                (run / "shard_environment").mkdir(exist_ok=True)
                shutil.copyfile(part / name, run / "shard_environment" / f"{identity}-{name}")
        shutil.rmtree(part)
    if records.keys() != inputs.keys() or seen != set(range(config["shards"])):
        raise ValueError(f"Incomplete merge: missing IDs={sorted(inputs.keys() - records.keys())}")
    write_json_atomic(run / "config.json", {**config, "scope": "full", "shards": 1, "shard": 0,
                                            "merged_archives": [str(p) for p in args.merge]})
    seal(run, sorted(questions, key=lambda r: r["id"]), records, config["mode"], "full", list(stages.values()))
    finish(run, args, official)


if __name__ == "__main__":
    main()
