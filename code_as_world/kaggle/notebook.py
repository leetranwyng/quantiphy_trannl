import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

WORKING = Path("/kaggle/working")
CACHE = Path("/kaggle/temp/quantiphy")
ERROR_LOG = WORKING / "quantiphy_launch_error.txt"
SUMMARY_PATH = WORKING / "quantiphy_summary.json"


def check_runtime(mode):
    commands = ["git", "ffprobe"]
    if mode != "merge":
        commands += ["nvidia-smi", "ffmpeg"]
    missing = [name for name in commands if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Kaggle runtime is missing {', '.join(missing)}. Check accelerator settings.")
    if mode != "merge":
        devices = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        if len(devices) != 2 or any("T4" not in name for name in devices):
            raise RuntimeError(f"Select GPU T4 ×2 in Kaggle settings. Found {devices}.")


def run_workflow(source, mode, options, config):
    command = [
        sys.executable, "-u", "-m", "code_as_world.kaggle.workflow", mode,
        "--cache", str(CACHE / "assets"), "--output", str(WORKING),
        "--minutes", str(config["WORK_MINUTES"]), "--output-budget-gib", str(config["OUTPUT_BUDGET_GIB"]), *options,
    ]
    if config["MODEL_DIR"]:
        command += ["--model-dir", config["MODEL_DIR"]]
    if config["DATASET_DIR"]:
        command += ["--dataset-dir", config["DATASET_DIR"]]
    subprocess.run(command, cwd=source, check=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["mode"] != mode:
        raise RuntimeError(f"Expected a {mode} run receipt, got {summary['mode']}.")
    return summary

def full_options(config):
    if not 0 <= config["PARTITION_INDEX"] < config["PARTITION_COUNT"]:
        raise ValueError("PARTITION_INDEX must be between 0 and PARTITION_COUNT - 1.")
    options = ["--partition-count", str(config["PARTITION_COUNT"]), "--partition-index", str(config["PARTITION_INDEX"])]
    if config["MERGE_ARCHIVES"]:
        if config["PARTITION_COUNT"] < 2 or len(config["MERGE_ARCHIVES"]) != config["PARTITION_COUNT"]:
            raise ValueError("Supply one archive per partition in MERGE_ARCHIVES.")
        for archive in config["MERGE_ARCHIVES"]:
            options += ["--merge-archive", archive]
        return "merge", options
    candidates = [Path(config["SANITY_ARCHIVE"])] if config["SANITY_ARCHIVE"] else sorted(
        path for path in Path("/kaggle/input").rglob("quantiphy_sanity.tar") if path.is_file()
    )
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError("Attach one saved sanity output via Add Input or set SANITY_ARCHIVE to its path.")
    options += ["--sanity-archive", str(candidates[0])]
    if config["RESUME_ARCHIVE"]:
        options += ["--resume-archive", config["RESUME_ARCHIVE"]]
    return "full", options




def show_result(SUMMARY):
    import csv
    from html import escape

    from IPython.display import HTML, Image, display
    print(f"{SUMMARY['mode']} · {SUMMARY['status']}")
    print(f"Archive: {WORKING / SUMMARY['archive']}")
    print("Giải nén archive rồi mở run/index.html.")
    run = Path(SUMMARY["run_dir"])
    for name in ("sanity_report.json", "metrics.json", "workflow_exception.txt"):
        path = run / name
        if path.is_file():
            print(f"\n{name}\n{path.read_text(encoding='utf-8')}")
    if SUMMARY["status"] in {"ERROR", "FAIL", "PARTIAL"}:
        for pattern in (
            "preflight/*/shards/*/worker.json",
            "preflight/*/shards/*/worker_exception.txt",
            "preflight/*/shards/*/worker.log",
            "shards/*/worker.json",
            "shards/*/worker_exception.txt",
            "shards/*/worker.log",
            "preflight/*/samples/*/attempt-*/exception.txt",
            "samples/*/attempt-*/exception.txt",
        ):
            for extra in sorted(run.glob(pattern)):
                rel = extra.relative_to(run).as_posix()
                text = extra.read_text(encoding="utf-8", errors="replace")
                if extra.name == "worker.json" and '"worker_error"' not in text:
                    continue
                print(f"\n{rel}\n{text[-4000:]}")

    predictions = run / "predictions.csv"
    if predictions.is_file():
        with predictions.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for group in ("S2", "D2", "S3", "D3"):
            row = next((r for r in rows if r["group"] == group and r["status"] == "ok"), None)
            if row is None:
                row = next((r for r in rows if r["group"] == group), None)
            if row is None:
                continue
            sample = row["sample_id"]
            display(HTML(
                f"<h4>{group} · {escape(row['question'])}</h4>"
                f"<p>Dự đoán {escape(row['parsed_value'])} · Đáp án {escape(row['ground_truth'])}"
                f" · {escape(row['status'])}</p><p><code>run/viewer/{escape(sample)}.html</code></p>"
            ))
            attempts = sorted((run / "samples" / sample).glob("attempt-*/artifact_refs.json"),
                              key=lambda p: int(p.parent.name.split("-")[-1]))
            if attempts:
                frame = json.loads(attempts[-1].read_text(encoding="utf-8")).get("filmstrip.png")
                if frame:
                    display(Image(filename=str(run / frame["path"]), width=650))
def launch(source, mode, config):
    WORKING.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.unlink(missing_ok=True)
    try:
        mode, options = full_options(config) if mode == "full" else (mode, [])
        check_runtime(mode)
        summary = run_workflow(source, mode, options, config)
        show_result(summary)
        if summary["status"] in {"ERROR", "FAIL"}:
            raise RuntimeError(f"Workflow ended with {summary['status']}. See the saved archive.")
        return summary
    except Exception:
        ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise
