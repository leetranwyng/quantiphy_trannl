"""Launch the same module from either Kaggle notebook or a local notebook smoke test."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

def launch(source, config):
    output = Path(config["OUTPUT"])
    if (output / "run").exists():
        output = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    cmd = [sys.executable, "-u", "-m", f"{__package__}.workflow",
           "--scope", config["SCOPE"], "--mode", config["MODE"], "--output", str(output),
           "--cache", config["CACHE"], "--shards", str(config["SHARDS"]), "--shard", str(config["SHARD"]),
           "--molmo", config["MOLMO"], "--download"]
    if config["RESUME"]:
        cmd.extend(["--resume", config["RESUME"]])
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=source, check=True)
    print(f"Archive  {output / 'iiixr_3d.zip'}")
    print("Extract the archive and open run/index.html.")
    evaluation = json.loads((output / "run/evaluation.json").read_text(encoding="utf-8"))
    print(json.dumps({key: evaluation[key] for key in ("mode", "all", "by_type", "routes")}, indent=2))
    questions = json.loads((output / "run/questions.json").read_text(encoding="utf-8"))
    failed_clips = {row["video_id"]: row["clip"].get("error")
                    for row in questions if row["clip"]["status"] != "ok"}
    if failed_clips:
        raise RuntimeError(f"Validation clips unavailable. Check Kaggle Internet and the pinned Hugging Face files: {failed_clips}")
    if config["MODE"] == "fresh" and not evaluation["all"]["covered"]:
        predictions = json.loads((output / "run/predictions_locked.json").read_text(encoding="utf-8"))
        first = predictions["predictions"][0]
        raise RuntimeError(f"Fresh inference produced no predictions. First failure in {first['stage']}: "
                           f"{first.get('reason')}. See the archived environment and worker logs.")
    return evaluation
