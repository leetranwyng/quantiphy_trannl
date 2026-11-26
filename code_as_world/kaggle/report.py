"""Build the labelled CSV, metrics and offline viewer after inference."""
from __future__ import annotations

import csv
import html
import io
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from code_as_world.kaggle.artifacts import (
    GROUPS, latest_attempt, parse_answer, read_json, relative, score_rows, wrap_links,
    write_json, write_jsonl, write_text,
)


def html_page(title, body):
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font:16px system-ui;max-width:1120px;margin:36px auto;padding:0 18px;color:#172633}}
a{{color:#0a5a96}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #dadfe8}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f8;padding:12px}}
video,img{{display:block;max-width:100%;margin:12px 0}}li{{margin:4px 0}}
</style><h1>{html.escape(title)}</h1>{body}</html>"""


def esc(value):
    return html.escape(str(value))


def hyperlink(href, label):
    return f'<a href="{quote(href, safe="/._-")}">{esc(label)}</a>'


def link_list(root, targets):
    return "<ul>" + "".join(f"<li>{hyperlink(path, label)}</li>" for label, path in wrap_links(root, targets).items()) + "</ul>"


def viewer_page(run, spec, item, row):
    path = run / "viewer" / f"{spec['id']}.html"
    links = {"Back to index": run / "index.html", "Source video": run / spec["video"]["path"]}
    body = f"<p>{esc(row['group'])} · {esc(row['video_id'])} · {esc(row['status'])}</p>"
    body += f"<p>{esc(row['question'])}</p><p>Prediction {esc(row['parsed_value'])} · Reference answer {esc(row['ground_truth'])}</p>"
    video_url = quote(wrap_links(path.parent, links)["Source video"], safe="/._-")
    body += f'<video controls preload="none" src="{video_url}"></video>' + link_list(path.parent, links)
    for attempt in item["attempts"]:
        folder = run / attempt
        status = read_json(folder / "attempt.json")
        refs = read_json(folder / "artifact_refs.json")
        body += f"<h2>{esc(folder.name)} · {esc(status['status'])}</h2>"
        if "filmstrip.png" in refs:
            sheet = wrap_links(path.parent, {"preview": run / refs["filmstrip.png"]["path"]})["preview"]
            body += f'<img src="{quote(sheet, safe="/._-")}" alt="Sampled frames with processor timestamps">'
        for key, label in (("rendered_prompt.txt", "Rendered prompt"), ("generation.json", "Raw completion"), ("exception.txt", "Exception")):
            if key not in refs:
                continue
            file = run / refs[key]["path"]
            content = read_json(file)["raw_text"] if key == "generation.json" else file.read_text(encoding="utf-8")
            body += f"<details><summary>{label}</summary><pre>{esc(content)}</pre></details>"
        targets = {"Attempt receipt": folder / "attempt.json",
                   "Artifact references": folder / "artifact_refs.json", **{key: run / ref["path"] for key, ref in refs.items()}}
        body += f"<details><summary>Files and tensor downloads</summary>{link_list(path.parent, targets)}</details>"
    write_text(path, html_page(spec["id"], body))


def write_predictions(path, source_names, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=source_names + [
        "sample_id", "row_number", "group", "status", "raw_response", "parsed_value", "parsed_unit",
        "ground_truth", "model",
    ], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def publish(run, specs, labels, manifest):
    rows, index, failures = [], [], []
    for spec in specs:
        key = spec["id"]
        final, _ = latest_attempt(run, key)
        attempts = sorted((run / "samples" / key).glob("attempt-*/attempt.json"))
        item = {"sample_id": key, "group": spec["record"]["category"], "video_id": spec["record"]["video_id"],
                "selected": relative(run, final) if final else None,
                "attempts": [relative(run, path.parent) for path in attempts]}
        status = read_json(final / "attempt.json") if final else {"status": "deferred"}
        generation_path = final / "generation.json" if final else None
        raw = read_json(generation_path)["raw_text"] if generation_path and generation_path.exists() else ""
        value, unit = parse_answer(raw)
        row = {
            **spec["source"], "sample_id": key, "row_number": spec["row_number"],
            "group": item["group"], "status": status["status"], "raw_response": raw,
            "parsed_value": value, "parsed_unit": unit, "ground_truth": labels[key], "model": "Code-as-World-VL-4B",
        }
        if status["status"] != "ok":
            failures.append({"sample_id": key, "attempt": item["selected"], **status})
        index.append(item)
        rows.append(row)
        viewer_page(run, spec, item, row)
    write_predictions(run / "predictions.csv", list(specs[0]["source"]), rows)
    write_jsonl(run / "trace_index.jsonl", index)
    write_jsonl(run / "failures.jsonl", failures)
    metrics = score_rows(rows)
    write_json(run / "metrics.json", {
        "mode": manifest["mode"], "coverage": {
            "selected": len(specs), "generated": metrics["status_counts"].get("ok", 0) + metrics["status_counts"].get("parse_error", 0),
            "dataset_total": manifest["dataset"]["questions"],
        },
        "metrics": metrics,
        "method": "QuantiPhy official evaluator. Missing predictions score zero when the reference is valid.",
    })
    official_in = run / "official_inputs"
    official_in.mkdir(exist_ok=True)
    # The original evaluator reads the numeric benchmark ID from the first CSV column.
    write_text(official_in / "caw_fp16.csv", (run / "predictions.csv").read_text(encoding="utf-8"))
    command = [sys.executable, str(run / "source" / "benchmark" / "evaluator.py"), str(official_in),
               str(run / "official_metrics"), "--gt_file", str(run / "configs" / "quantiphy_validation.csv")]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    write_text(run / "official_metrics.log", result.stdout)
    result.check_returncode()
    build_index(run, rows, metrics)
    return rows, metrics


def build_index(run, rows, metrics):
    def cell(value):
        return f"<td>{esc(value)}</td>"
    links = {name: run / name for name in (
        "manifest.json", "metrics.json", "predictions.csv", "failures.jsonl", "trace_index.jsonl",
        "verify.py", "configs/config_diff.json", "official_metrics/all_model_results.csv",
    )}
    body = link_list(run, links)
    body += "<p>" + " · ".join(f"{group} MRA {esc(metrics['mra_by_group'][group])}" for group in GROUPS) + "</p>"
    body += "<table><thead><tr>" + "".join(f"<th>{label}</th>" for label in (
        "Question", "Group", "Video", "Status", "Prediction", "Reference"
    )) + "</tr></thead><tbody>"
    for row in rows:
        body += "<tr><td>" + hyperlink(f"viewer/{row['sample_id']}.html", row["question"]) + "</td>"
        body += "".join(cell(row[k]) for k in ("group", "video_id", "status", "parsed_value", "ground_truth")) + "</tr>"
    write_text(run / "index.html", html_page("QuantiPhy · Code-as-World-VL-4B", body + "</tbody></table>"))
