"""Kaggle orchestration for the sanity and validation notebooks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

from code_as_world.kaggle.artifacts import (
    GROUPS, SCHEMA, choose_sanity, copy_file, file_bytes, fingerprint, latest_attempt, load_dataset,
    read_json, reference, require, resumable, seal, sha256, split_by_video, write_json, write_text,
)
from code_as_world.kaggle.report import publish


ROOT = Path(__file__).resolve().parents[2]
LOCK = read_json(Path(__file__).with_name("snapshots.json"))
GIB = 1024**3
ARCHIVES = {"sanity": "quantiphy_sanity.tar", "full": "quantiphy_full.tar", "merge": "quantiphy_merged.tar"}


def run_command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def ensure_git(url, revision, destination):
    if not (destination / ".git").is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        run_command("git", "init", str(destination))
        run_command("git", "-C", str(destination), "remote", "add", "origin", url)
    if run_command("git", "-C", str(destination), "remote", "get-url", "origin") != url:
        raise ValueError(f"Unexpected origin for {destination}")
    try:
        run_command("git", "-C", str(destination), "cat-file", "-e", f"{revision}^{{commit}}")
    except subprocess.CalledProcessError:
        run_command("git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision)
    run_command("git", "-C", str(destination), "checkout", "--detach", revision)
    require(run_command("git", "-C", str(destination), "rev-parse", "HEAD") == revision, "Unexpected benchmark revision")


def expected_files(location, config):
    receipts = {}
    for name, entry in config["files"].items():
        path = location / name
        require(path.is_file() and path.stat().st_size == entry["size"], f"Snapshot file missing or truncated: {name}")
        digest = sha256(path)
        if entry["sha256"]:
            require(digest == entry["sha256"], f"Snapshot SHA256 mismatch: {name}")
        else:
            git_blob = hashlib.sha1(f"blob {entry['size']}\0".encode() + path.read_bytes()).hexdigest()
            require(git_blob == entry["git_blob"], f"Snapshot Git blob mismatch: {name}")
        receipts[name] = {"sha256": digest, "bytes": entry["size"]}
    return receipts


def fetch_assets(args):
    from huggingface_hub import snapshot_download

    benchmark = args.cache / "benchmark"
    ensure_git(LOCK["benchmark"]["repo_url"], LOCK["benchmark"]["revision"], benchmark)
    paths, hashes = {}, {}
    for kind, attach in (("model", args.model_dir), ("dataset", args.dataset_dir)):
        config = LOCK[kind]
        target = attach or (args.cache / kind)
        if attach is None:
            snapshot_download(
                config["repo_id"], repo_type="dataset" if kind == "dataset" else "model",
                revision=config["revision"], local_dir=target, allow_patterns=list(config["files"]),
                max_workers=4, token=False,
            )
        paths[kind] = target
        hashes[kind] = expected_files(target, config)
    return benchmark, paths, hashes


def video_probe(path):
    command = (
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,duration,codec_name", "-of", "json", str(path),
    )
    stream = json.loads(run_command(*command))["streams"][0]
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A").isdigit() else None,
        "avg_frame_rate": stream.get("avg_frame_rate"), "duration": stream.get("duration"),
        "codec": stream.get("codec_name"),
    }


def video_inventory(dataset_path, hashes):
    videos = {}
    for upstream_name, receipt in hashes.items():
        original_file = Path(upstream_name).name
        filename = original_file.strip()
        key = Path(filename).stem
        require(key not in videos, f"Ambiguous canonical video name: {filename}")
        videos[key] = {
            "path": f"videos/{filename}", "snapshot_path": upstream_name,
            "name_normalized": filename != original_file, "probe": video_probe(dataset_path / upstream_name), **receipt,
        }
    return videos


def runtime_devices():
    query = "index,uuid,name,driver_version,compute_cap,memory.total,memory.free"
    lines = run_command("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits")
    keys = ["index", "uuid", "name", "driver", "compute_capability", "total_mib", "free_mib"]
    devices = [dict(zip(keys, row, strict=True)) for row in csv.reader(lines.splitlines(), skipinitialspace=True)]
    require(len(devices) == 2 and all("T4" in d["name"] and d["compute_capability"] == "7.5" for d in devices),
            f"Expected the Kaggle GPU T4 ×2 accelerator, got {devices}")
    return devices


def snapshot_source(run, benchmark, model):
    source = run / "source"
    require(not run_command("git", "status", "--porcelain", "--untracked-files=no"), "Checkout has uncommitted source changes")
    tracked = (
        "LICENSE", "README.md", "code_as_world",
        "notebooks",
    )
    listed = run_command("git", "ls-files", "-z", "--", *tracked)
    for name in listed.split("\0"):
        if name:
            copy_file(ROOT / name, source / name)
    for name in ("evaluator.py", "LICENSE"):
        copy_file(benchmark / name, source / "benchmark" / name)
    copy_file(Path(__file__).with_name("verify.py"), run / "verify.py")
    copy_file(benchmark / "quantiphy_validation.csv", run / "configs" / "quantiphy_validation.csv")
    for name in LOCK["model"]["files"]:
        if name.endswith((".json", ".jinja")):
            copy_file(model / name, run / "configs" / "model" / name)
    diff = run_command("git", "diff", LOCK["upstream_commit"], "HEAD", "--", *tracked)
    write_text(run / "configs" / "instrumentation.patch", diff + "\n")


def configuration(run, args, devices):
    packages = ["torch", "torchvision", "transformers", "qwen-vl-utils", "decord", "accelerate",
                "huggingface-hub", "tokenizers", "numpy", "pandas", "pillow", "jinja2"]
    versions = {package: importlib.metadata.version(package) for package in packages}
    write_json(run / "configs" / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "executable": sys.executable,
        "kaggle_image": os.environ.get("KAGGLE_DOCKER_IMAGE_TAG"),
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "arguments": sys.argv, "packages": versions, "devices": devices,
        "free_disk_bytes": {"run": shutil.disk_usage(run).free, "output": shutil.disk_usage(args.output).free},
        "ffprobe_version": run_command("ffprobe", "-version").splitlines()[0],
    })
    write_text(run / "configs" / "pip_freeze.txt", run_command(sys.executable, "-m", "pip", "freeze") + "\n")
    if devices:
        write_text(run / "configs" / "nvidia-smi.txt", run_command("nvidia-smi", "-q") + "\n")
    write_json(run / "configs" / "config_diff.json", {
        "upstream_engine": "vLLM 0.19.1, BF16, single GPU, engine GPU utilization 0.2 for 4B",
        "engine": "Transformers 5.11.0, eager, FP16, batch 1, inference mode",
        "layout": "one replica per GPU, or a balanced two-GPU model selected by the single-replica smoke test",
        "video": "upstream fetch_video and FPS override, Decord selected with upstream TorchVision fallback",
        "processor": "Use the same pinned processor. Pass sampled video and do_sample_frames=False. Replace the complete wrapper once.",
        "decoding": "Same temperature, top_p, seed, repetition and image/video logit bias. Transformers sampler and RNG differ. Enable KV cache.",
        "eos": "Union of tokenizer and checkpoint EOS IDs. Finish reason inferred from last token and token limit.",
        "max_context": 4608, "max_response_tokens": 512, "video_frames": 16, "fetch_max_pixels": 262144,
        "parity_with_upstream_vllm_bf16": "unverified",
        "model_calls": ["Qwen3.5 vision stack and merger", "Qwen3.5 language model and output head"],
        "components_only_in_demo_or_paper": ["SAM", "DA3", "VGGT", "MuJoCo"],
        "diagnostics": "Vision activations plus one language prefill and one decode per group. No generated image stage exists.",
        "timing": "Per-sample generate synchronized before/after; hook event timings are host wall-clock and include tracing overhead.",
    })


def launch(run, work, specs, mode, profile, devices, model_path, identity, deep, deadline, budget, tag=""):
    place = run / tag if tag else run
    count = 1 if mode == "smoke" or profile == "split_fp16" else 2
    shards = [specs] if count == 1 else split_by_video(specs, count)
    processes = []
    reports = []
    for index, shard in enumerate(shards):
        if not shard:
            continue
        worker = f"worker-{index}"
        gpu_ids = [d["uuid"] for d in devices] if profile == "split_fp16" else [devices[index]["uuid"]]
        folder = place / "shards" / worker
        folder.mkdir(parents=True, exist_ok=True)
        report_file = folder / "worker.json"
        report_file.unlink(missing_ok=True)
        options = {
            "run": str(run), "subdirectory": tag, "worker": worker, "samples": shard, "model": str(model_path),
            "profile": profile, "phase": mode, "fingerprint": identity, "deep_examples": deep,
            "deadline": deadline, "output_budget": budget,
        }
        work_item = work / f"{mode}-{profile}-{index}.json"
        write_json(work_item, options)
        env = {
            **os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids), "FORCE_QWENVL_VIDEO_READER": "decord",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        }
        log_path = folder / "worker.log"
        if log_path.exists():
            log_path.rename(folder / f"worker-{time.time_ns()}.log")
        cmd = [sys.executable, "-m", "code_as_world.kaggle.infer", str(work_item)]
        write_json(folder / "launch.json", {"command": cmd, "gpu_uuids": gpu_ids, "assigned_sample_ids": [s["id"] for s in shard]})
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((proc, log_path, report_file, gpu_ids))
    offsets = {path: 0 for _, path, _, _ in processes}
    while True:
        active = False
        for proc, path, _, _ in processes:
            active |= proc.poll() is None
            with path.open("r", errors="replace") as stream:
                stream.seek(offsets[path])
                output = stream.read()
                offsets[path] = stream.tell()
                if output:
                    print(output, end="", flush=True)
        if not active:
            break
        time.sleep(3)
    for proc, path, receipt, gpu_ids in processes:
        if receipt.exists():
            report = read_json(receipt)
        else:
            report = {"worker_error": "Worker exited without receipt", "model_loaded": False, "results": [],
                      "worker_log": str(path.relative_to(run))}
        report.update(returncode=proc.returncode, gpu_uuids=gpu_ids)
        write_json(receipt, report)
        reports.append(report)
    return reports


def smoke_matches(run, tag, first):
    folder = run / tag / "samples" / first["id"]
    a, b = folder / "attempt-1", folder / "attempt-2"
    token_ids = []
    parsed = []
    for attempt in (a, b):
        generation = attempt / "generation.json"
        if not generation.exists():
            return False
        token_ids.append(read_json(generation)["output_token_ids"])
        parsed.append(read_json(attempt / "parsed.json"))
    return token_ids[0] == token_ids[1] and parsed[0] == parsed[1]


def probe_profile(run, work, specs, devices, model_path, identity, deadline, budget):
    evidence = []
    for profile in ("replica_fp16", "split_fp16"):
        scope = f"preflight/{profile}"
        reports = launch(run, work, specs, "smoke", profile, devices, model_path, fingerprint({**identity, "profile": profile}),
                         [], deadline, budget, scope)
        report = reports[0]
        usable = len(report["results"]) == len(specs) + 1 and all(
            item["status"] in {"ok", "parse_error"} for item in report["results"]
        )
        oom = report.get("oom", False) or any(item.get("oom") for item in report["results"])
        parity = usable and smoke_matches(run, scope, specs[0])
        evidence.append({"profile": profile, "oom": oom, "identical_trace_off_on": parity, "usable": usable,
                         "report": f"{scope}/shards/worker-0/worker.json"})
        if parity:
            return profile, evidence
        if not oom:
            return None, evidence
    return None, evidence


def short_fixture(run, spec):
    path = run / "videos" / "sanity_short_fixture.mp4"
    run_command("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                str(run / spec["video"]["path"]), "-an", "-frames:v", "8", "-c:v", "mpeg4", str(path))
    video = {
        **reference(run, path), "probe": video_probe(path), "derived_from": spec["video"]["path"],
        "purpose": "Exercise the upstream short-clip retry below 16 source frames.",
    }
    return {
        **spec, "id": spec["id"] + "-short-fixture",
        "record": {**spec["record"], "video_id": "sanity_short_fixture", "video": video["path"]},
        "video": video,
    }


def unpack(archive, directory):
    require(archive.is_file(), f"Missing prior Kaggle output: {archive}")
    with tarfile.open(archive, "r:*") as artifact:
        entries = artifact.getmembers()
        for entry in entries:
            target = Path(entry.name)
            require(target.parts and target.parts[0] == "run" and ".." not in target.parts
                    and (entry.isdir() or entry.isfile()), f"Unexpected archive entry: {entry.name}")
        artifact.extractall(directory, members=entries, filter="data")
    return directory / "run"


def offline_check(run, require_pass=False):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-I", str(run / "verify.py"), str(run)]
    if require_pass:
        command.append("--require-pass")
    data = subprocess.check_output(command, cwd=run, env=env, text=True)
    print(data.strip(), flush=True)
    return json.loads(data)


def open_prior(path, directory, require_pass=False):
    # Run the verifier from the checked-out repo against the mounted artifact.
    from code_as_world.kaggle.verify import verify

    target = unpack(path, directory)
    verify(target, require_pass=require_pass)
    return target, read_json(target / "manifest.json")


def model_pass_checks(run, rows, manifest, sanity_selection, scope, evidence, budget):
    counts = Counter(row["group"] for row in rows)
    keymap = {row["sample_id"]: row for row in rows}
    shared_a, shared_b = [latest_attempt(run, key)[0] for key in sanity_selection["shared_video"]]
    shared = False
    if shared_a and shared_b and (shared_a / "video.json").exists() and (shared_b / "video.json").exists():
        refs_a = read_json(shared_a / "artifact_refs.json")
        refs_b = read_json(shared_b / "artifact_refs.json")
        shared = refs_a["fetch_pixels.npy"] == refs_b["fetch_pixels.npy"] and any(
            read_json(path / "video.json")["fetch_cache_hit"] for path in (shared_a, shared_b)
        )
    deep_groups = {}
    for key in sanity_selection["deep_examples"]:
        attempt, _ = latest_attempt(run, key)
        if attempt and read_json(attempt / "attempt.json")["status"] in {"ok", "parse_error"}:
            deep_groups[keymap[key]["group"]] = read_json(attempt / "attempt.json")["deep_trace"]
    timings = []
    for row in rows:
        attempt, _ = latest_attempt(run, row["sample_id"])
        if attempt and (attempt / "generation.json").exists():
            gen = read_json(attempt / "generation.json")
            timings.append({"sample_id": row["sample_id"], "wall_seconds": gen["wall_seconds_cuda_synchronized"],
                            "prefill_peak": gen["prefill_peak"], "decode_peak": gen["decode_peak_since_end_of_prefill"]})
    video_count = len({row["video_id"] for row in rows})
    visual_size = file_bytes(run / "blobs") + file_bytes(run / "videos")
    output_bytes = file_bytes(run)
    manifest_dataset = manifest["dataset"]
    projected = math.ceil(1.25 * (visual_size / video_count * manifest_dataset["videos"]
                + (output_bytes - visual_size) / len(rows) * manifest_dataset["questions"]))
    tputs = 2 if manifest["identity"]["profile"] == "replica_fp16" else 1
    profile = manifest["identity"]["profile"]
    short_attempt, _ = latest_attempt(run / "preflight" / str(profile), sanity_selection["short_fixture"])
    retry_logged = False
    if short_attempt and (short_attempt / "video.json").exists():
        video = read_json(short_attempt / "video.json")
        retry_logged = video["effective_nframes"] == 8 and [r["nframes"] for r in video["requests"]] == [16, 8]
    checks = {
        "single_replica_or_documented_split_smoke": bool(evidence and evidence[-1]["usable"]),
        "trace_off_on_same_token_ids_and_parsed_answer": bool(evidence and evidence[-1]["identical_trace_off_on"]),
        "both_T4s_used": len(manifest["used_gpu_uuids"]) == 2,
        "all_pilot_questions_generated": all(row["status"] in {"ok", "parse_error"} for row in rows),
        "numeric_parser_exercised": any(row["status"] == "ok" for row in rows),
        "four_groups_with_deep_trace": set(counts) == set(GROUPS) and set(deep_groups) == set(GROUPS) and all(deep_groups.values()),
        "short_video_retry_captured": retry_logged,
        "same_video_reused_and_checksum_matches": shared,
        "resume_receipt_checks": all(resumable(run, run, spec, manifest["fingerprint"]) for spec in scope),
        "offline_validation_and_benchmark_metric_replay": False,
    }
    return checks, {
        "output_bytes_pilot": output_bytes,
        "visual_and_source_bytes_per_video_approx": visual_size // video_count,
        "other_bytes_per_question_approx": (output_bytes - visual_size) // len(rows),
        "projected_output_bytes": projected,
        "output_budget_bytes": budget,
        "recommended_partition_count": min(manifest_dataset["videos"], max(1, math.ceil(projected / budget))),
        "free_stage_bytes": shutil.disk_usage(run).free,
        "estimated_full_generation_seconds_excluding_setup": math.fsum(t["wall_seconds"] for t in timings)
              / max(1, len(timings)) * manifest_dataset["questions"] / tputs if timings else None,
        "pilot_generation": timings,
    }


def package(run, output, mode):
    archive = output / ARCHIVES[mode]
    with tempfile.NamedTemporaryFile(prefix=".quantiphy-", dir=output, delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        with tarfile.open(staged, "w") as handle:
            handle.add(run, arcname="run")
        os.replace(staged, archive)
    finally:
        staged.unlink(missing_ok=True)
    return archive


def select_resume(args, run, identity):
    if not args.resume_archive:
        run.mkdir(parents=True)
        return None
    prior, manifest = open_prior(args.resume_archive, run.parent)
    require(prior == run and manifest["mode"] == "full" and manifest["fingerprint"] == fingerprint(identity),
            "Resume archive uses a different full run fingerprint")
    return manifest


def merge_parts(args, work, run, identity_base, specs, partitions):
    require(len(args.merge_archive) == partitions >= 2, "Supply each saved full partition archive exactly once")
    manifests = []
    profile = None
    for index, archive in enumerate(args.merge_archive):
        directory = work / f"part-{index}"
        directory.mkdir()
        part, info = open_prior(archive, directory)
        require(info["mode"] == "full" and info["partition_count"] == partitions and info["state"] == "complete",
                "All merge inputs must be completed parts of the same full run")
        profile = profile or info["identity"]["profile"]
        base = {**identity_base, "profile": profile, "partition_count": partitions, "partition_index": info["partition_index"]}
        require(info["fingerprint"] == fingerprint(base), f"Incompatible partition {archive.name}")
        # Relative paths stay the same. The merged attestation retains the part fingerprint for each sample.
        for name in ("samples", "blobs"):
            src = part / name
            if src.exists():
                shutil.copytree(src, run / name, dirs_exist_ok=True)
        part_receipt = run / "parts" / f"part-{info['partition_index']}"
        shutil.copytree(part / "shards", part_receipt / "shards", dirs_exist_ok=True)
        copy_file(part / "manifest.json", part_receipt / "manifest.json")
        manifests.append(info)
        shutil.rmtree(directory)
    require({m["partition_index"] for m in manifests} == set(range(partitions)), "Duplicated or missing partition")
    used = sorted({gpu for m in manifests for gpu in m["used_gpu_uuids"]})
    origins = {}
    for part in manifests:
        for key in part["selected_sample_ids"]:
            require(key not in origins, f"Duplicate partition sample: {key}")
            origins[key] = part["fingerprint"]
    require(set(origins) == {s["id"] for s in specs}, "Partition sample coverage is incomplete")
    return profile, used, origins, manifests


def workflow(args, work, run, deadline):
    verified_sanity, sanity_manifest = None, None
    if args.mode == "full" and args.sanity_archive is not None:
        prior = work / "verified_sanity"
        prior.mkdir()
        verified_sanity, sanity_manifest = open_prior(args.sanity_archive, prior, require_pass=False)
    benchmark, paths, hashes = fetch_assets(args)
    videos = video_inventory(paths["dataset"], hashes["dataset"])
    specs, labels, csv_sha = load_dataset(benchmark / "quantiphy_validation.csv", videos)
    used_video_ids = {s["record"]["video_id"] for s in specs}
    require(set(videos) == used_video_ids, "Pinned dataset video inventory differs from CSV")
    dataset = {"questions": len(specs), "videos": len(used_video_ids),
               "groups": dict(Counter(s["record"]["category"] for s in specs)), "missing_videos": []}
    pilot, selection = choose_sanity(specs)
    devices = runtime_devices() if args.mode != "merge" else []
    profile = (sanity_manifest["identity"].get("profile") if sanity_manifest else None) or (
        args.profile if args.mode == "full" else None
    )
    output_free = shutil.disk_usage(args.output).free
    budget = min(int(args.output_budget_gib * GIB), max(GIB, output_free - GIB))
    identity_base = {
        "schema": SCHEMA, "source_url": run_command("git", "remote", "get-url", "origin"),
        "source_commit": run_command("git", "rev-parse", "HEAD"), "benchmark": LOCK["benchmark"],
        "csv_sha256": csv_sha, "model_revision": LOCK["model"]["revision"], "weights_and_configs": hashes["model"],
        "data_revision": LOCK["dataset"]["revision"], "dataset_files": hashes["dataset"],
        "python": platform.python_version(),
        "dependencies": {k: importlib.metadata.version(k) for k in (
            "torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils", "decord",
            "numpy", "tokenizers", "pillow", "jinja2", "pandas",
        )},
    }
    if sanity_manifest:
        shutil.rmtree(verified_sanity.parent)
    require(1 <= args.partition_count <= len(used_video_ids) and 0 <= args.partition_index < args.partition_count,
            "Invalid partition count or index")
    if args.mode == "full":
        scope = sorted(split_by_video(specs, args.partition_count)[args.partition_index], key=lambda s: s["row_number"])
    else:
        scope = pilot if args.mode == "sanity" else specs

    identity = {**identity_base, "profile": profile, "partition_count": args.partition_count, "partition_index": args.partition_index}
    previous_manifest = select_resume(args, run, identity)
    if previous_manifest:
        checkpoint = run / "resume"
        checkpoint.mkdir(exist_ok=True)
        copy_file(run / "manifest.json", checkpoint / f"manifest-{time.time_ns()}.json")
    snapshot_source(run, benchmark, paths["model"])
    configuration(run, args, devices)
    for key in {s["record"]["video_id"] for s in scope}:
        video = videos[key]
        target = run / video["path"]
        if not target.exists():
            copy_file(paths["dataset"] / video["snapshot_path"], target)
    preflight, reports, origins, parts = [], [], {}, []
    if args.mode == "sanity":
        by_key = {s["id"]: s for s in scope}
        smoke_keys = list(dict.fromkeys([selection["shortest_video"], selection["largest_source_resolution"]]))
        short = short_fixture(run, by_key[selection["shortest_video"]])
        selection["short_fixture"] = short["id"]
        profile, preflight = probe_profile(run, work, [*[by_key[key] for key in smoke_keys], short], devices, paths["model"],
                                           identity_base, deadline, budget)
        identity["profile"] = profile
    write_json(run / "configs" / "plan.json", {"samples": scope, "sanity_selection": selection})
    write_json(run / "configs" / "snapshot_lock.json", LOCK)
    if args.mode == "merge":
        profile, used, origins, parts = merge_parts(args, work, run, identity_base, specs, args.partition_count)
        identity.update(profile=profile, partition_count=1, partition_index=0)
    fp = fingerprint(identity)
    if args.mode != "merge":
        if profile:
            reports = launch(run, work, scope, "evaluation", profile, devices, paths["model"],
                             fp, selection["deep_examples"], deadline, budget)
        used = sorted({gpu for worker in reports if worker["model_loaded"] and worker["results"] for gpu in worker["gpu_uuids"]}
                      | set(previous_manifest.get("used_gpu_uuids", []) if previous_manifest else []))
    manifest = {
        "schema": SCHEMA, "mode": args.mode, "fingerprint": fp, "identity": identity,
        "partition_index": identity["partition_index"], "partition_count": identity["partition_count"],
        "selected_sample_ids": [s["id"] for s in scope], "dataset": dataset,
        "devices": devices, "used_gpu_uuids": used, "workers": reports, "preflight": preflight,
        "sample_fingerprints_from_parts": origins, "part_fingerprints": [m["fingerprint"] for m in parts],
        "sanity_fingerprint": sanity_manifest["fingerprint"] if sanity_manifest else None,
        "sanity_archive_sha256": sha256(args.sanity_archive) if sanity_manifest else None,
    }
    write_json(run / "manifest.json", manifest)
    rows, metrics = publish(run, scope, labels, manifest)
    manifest["state"] = "complete" if not ({"error", "deferred"} & set(metrics["status_counts"])) else "partial"
    write_json(run / "manifest.json", manifest)
    if args.mode == "sanity":
        checks, estimate = model_pass_checks(run, rows, manifest, selection, scope, preflight, budget)
        write_json(run / "sanity_report.json", {
            "status": "PENDING_VERIFICATION", "fingerprint": fp, "checks": checks, "estimate": estimate,
        })
    seal(run)
    verification = offline_check(run)
    if args.mode == "sanity":
        checks["offline_validation_and_benchmark_metric_replay"] = True
        passed = all(checks.values())
        write_json(run / "sanity_report.json", {
            "status": "PASS" if passed else "FAIL", "fingerprint": fp, "checks": checks, "estimate": estimate,
            "local_offline_verification": verification,
        })
        seal(run)
    archive = package(run, args.output, args.mode)
    if args.mode == "sanity":
        copied = unpack(archive, work / "sanity_readback")
        offline_check(copied, require_pass=passed)
        if not passed:
            print("Sanity FAIL. Full remains gated.", flush=True)
    status = ("PASS" if passed else "FAIL") if args.mode == "sanity" else manifest["state"].upper()
    return {
        "status": status, "mode": args.mode, "archive": archive.name, "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size, "run_dir": str(run),
        "fingerprint": fp, "dataset": dataset, "verification": verification,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(ARCHIVES))
    parser.add_argument("--cache", type=Path, default=Path("/kaggle/quantiphy-cache"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--sanity-archive", type=Path)
    parser.add_argument("--profile", choices=("replica_fp16", "split_fp16"), default="replica_fp16")
    parser.add_argument("--resume-archive", type=Path)
    parser.add_argument("--merge-archive", type=Path, action="append", default=[])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--minutes", type=int, default=650)
    parser.add_argument("--output-budget-gib", type=float, default=16.0)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.minutes * 60
    work = Path(tempfile.mkdtemp(prefix=f"{args.mode}-", dir=args.cache))
    run = work / "run"
    try:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        summary = workflow(args, work, run, deadline)
    except Exception:
        run.mkdir(exist_ok=True)
        write_text(run / "workflow_exception.txt", traceback.format_exc())
        if (run / "integrity.json").exists():
            seal(run)
        print(traceback.format_exc(), flush=True)
        archive = package(run, args.output, args.mode)
        summary = {
            "status": "ERROR", "mode": args.mode, "archive": archive.name, "archive_sha256": sha256(archive),
            "archive_bytes": archive.stat().st_size, "run_dir": str(run),
            "error_file": "run/workflow_exception.txt",
        }
    write_json(args.output / "quantiphy_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
