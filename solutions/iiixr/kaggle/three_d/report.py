"""Replay, post-seal scoring, and an offline question-by-question report."""
from __future__ import annotations

import collections
import csv
import html
import io
import json
import math
import tarfile
from pathlib import Path

import cv2
import numpy as np

from ..evaluate import compute_row_mra
from ..schema import fingerprint, read_json, sha256_bytes, sha256_file, write_json_atomic
from . import data

RAW_FIELDS = ("id", "parsed_value", "answer_source", "geo_value", "meas_value", "vlm_value",
              "h1_value", "meas_valid", "meas_why", "meas_mode", "meas_kind", "meas_grav",
              "meas_gap", "meas_tracks", "meas_f", "meas_mpp", "meas_owlvote", "meas_nseg",
              "meas_colswap", "merge_src")
ATOL, RTOL = 1e-6, 1e-4


def prediction(row):
    value = data.finite(row.get("parsed_value"))
    valid = value is not None and value > 0 and row["answer_source"] in {"meas", "vlm", "h1"}
    return {"parsed_value": value if valid else None, "source": row["answer_source"] if valid else None,
            "status": "ok" if valid else "error", "stage": "person_focal_merge",
            "geometry_candidate": data.finite(row.get("meas_value")),
            "accepted_geometry": data.finite(row.get("geo_value")),
            "vlm_candidate": data.finite(row.get("vlm_value")),
            "heuristic_candidate": data.finite(row.get("h1_value")),
            "decision": {key: row.get(key, "") for key in RAW_FIELDS[7:]}}


def replay(run, ids):
    raw = data.pinned_member(data.REFERENCE)
    table = data.csv_bytes(raw)
    evidence = {str(q): {key: table[q][key] for key in RAW_FIELDS} for q in ids}
    identity = fingerprint(sorted(ids))[:12]
    path = run / "evidence" / f"published_rows_{identity}.json"
    write_json_atomic(path, {"member": data.REFERENCE, "member_sha256": sha256_bytes(raw),
                             "columns": RAW_FIELDS, "rows": evidence,
                             "unavailable": ["validation tracks", "validation MoGe", "rendered model calls"]})
    stage = {"stage": "published_table_read", "event_id": f"published-3d-{identity}", "status": "ok",
             "input": data.REFERENCE, "kind": "stored answers", "recomputed": False,
             "input_sha256": sha256_bytes(raw), "output": path.relative_to(run).as_posix(),
             "scope": [{"id": q, "video_id": table[q]["video_id"].strip()} for q in ids]}
    results = {q: {**prediction(table[q]), "stage": stage["stage"], "lineage": [stage["event_id"]],
                   "artifact": stage["output"]} for q in ids}
    if len(ids) == 90 and collections.Counter(r["source"] for r in results.values()) != {"meas": 81, "vlm": 9}:
        raise ValueError("The pinned replay route counts differ from 81 meas / 9 vlm")
    return results, [stage]


def aggregate(records, column):
    valid = [r for r in records if r.get(column) is not None]
    scored = [r for r in records if r["gt"] is not None and r["gt"] > 0]
    return {"n": len(records), "scored_n": len(scored), "covered": len(valid),
            "coverage": len(valid) / len(records) if records else 0,
            "mra": sum(compute_row_mra(r.get(column), r["gt"]) for r in scored) / len(scored) if scored else None}


def grouped(rows, field):
    return {value: aggregate([r for r in rows if r[field] == value], "parsed_value")
            for value in sorted({r[field] for r in rows})}


def evaluate(run, official=None, caw_tar=None):
    lock = read_json(run / "lock.json")
    predictions_path = run / "predictions_locked.json"
    if sha256_file(predictions_path) != lock["sha256"]:
        raise ValueError("Locked prediction checksum mismatch")
    locked = read_json(predictions_path)
    if not locked["sealed"]:
        raise ValueError("Predictions have not been sealed")
    source = data.pinned_member(data.REFERENCE)
    reference = data.csv_bytes(source)
    alternate_source = data.pinned_member(data.ALTERNATE)
    alternate = data.csv_bytes(alternate_source)
    if official:
        with official.open(encoding="utf-8-sig", newline="") as handle:
            truth = data.rows(handle)
    else:
        truth = reference
    member = data.PINS["caw_baseline"]["predictions_member"]
    if caw_tar:
        with tarfile.open(caw_tar) as archive:
            content = archive.extractfile(member).read()
        caw = {qid: data.finite(row["parsed_value"]) for qid, row in data.csv_bytes(content).items()}
        caw_info = {"kind": "archive", "member": member, "sha256": sha256_bytes(content)}
    else:
        snapshot = read_json(data.CAW_BASELINE)
        caw = {int(qid): data.finite(value) for qid, value in snapshot["predictions"].items()}
        caw_info = {"kind": "repository_snapshot", "filename": data.CAW_BASELINE.name,
                    "sha256": sha256_file(data.CAW_BASELINE), "archive_member": snapshot["archive_member"],
                    "archive_member_sha256": snapshot["archive_member_sha256"]}
    caw_info["3d_reference_missing_in_caw"] = sorted(data.select(reference).keys() - caw.keys())
    table, diffs = [], []
    stage_list = read_json(run / "stages.json")
    for pred in locked["predictions"]:
        q = pred["id"]
        gold = data.finite(truth[q]["ground_truth_posterior"])
        if gold is not None and gold <= 0:
            gold = None
        ref = prediction(reference[q])
        other = prediction(alternate[q])
        agrees = pred["parsed_value"] is not None and ref["parsed_value"] is not None and math.isclose(
            pred["parsed_value"], ref["parsed_value"], abs_tol=ATOL, rel_tol=RTOL)
        same_route = pred["source"] == ref["source"]
        baseline = caw.get(q)
        row = {key: pred[key] for key in ("id", "video_id", "category", "video_source", "status", "stage", "parsed_value",
                                          "geometry_candidate", "source")}
        table.append({**row, "gt": gold, "ref": ref["parsed_value"], "ref_source": ref["source"], "caw": baseline,
                      "secondary": other["parsed_value"], "secondary_source": other["source"],
                      "route_matches": same_route, "value_matches": agrees,
                      "row_mra": compute_row_mra(pred["parsed_value"], gold) if gold else None,
                      "mra_reference": compute_row_mra(ref["parsed_value"], gold) if gold else None,
                      "mra_caw": compute_row_mra(baseline, gold) if gold and q in caw else None})
        if not same_route or not agrees:
            matched_stages = [s for s in stage_list if any(item["id"] == q for item in s.get("scope", []))]
            earliest = next((s["stage"] for s in matched_stages if s["status"] != "ok"), pred["stage"])
            diffs.append({"id": q, "route": pred["source"], "reference_route": ref["source"],
                          "value": pred["parsed_value"], "reference_value": ref["parsed_value"],
                          "first_observed_difference": earliest,
                          "earliest_numeric_comparison": "final row; intermediate validation tracks/measures unpublished"})
    with (run / "per_question.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    paired = [r for r in table if r["id"] in caw]
    route_counts = collections.Counter(r["source"] or "unavailable" for r in table)
    evaluation = {"locked_sha256": lock["sha256"], "mode": locked["mode"], "scope": locked["scope"],
                  "reference": {"member": data.REFERENCE, "sha256": sha256_bytes(source),
                                "score": aggregate(table, "ref")},
                  "secondary_published_table": {"member": data.ALTERNATE, "sha256": sha256_bytes(alternate_source),
                                                  "score": aggregate(table, "secondary"), "generator": "unverified"},
                  "gt_source": str(official) if official else data.REFERENCE,
                  "scorer": "QuantiPhy strict relative-error thresholds 0.1 through 0.9 plus 0.95; missing predictions score zero",
                  "all": aggregate(table, "parsed_value"), "by_type": grouped(table, "category"),
                  "by_video_source": grouped(table, "video_source"),
                  "by_answer_source": grouped([{**r, "source": r["source"] or "unavailable"} for r in table], "source"),
                  "by_status": grouped(table, "status"), "routes": dict(route_counts),
                  "paired_caw": {"source": caw_info, "ids": [r["id"] for r in paired], "n": len(paired),
                                 "iiixr": aggregate(paired, "parsed_value"), "caw": aggregate(paired, "caw")},
                  "parity": {"rtol": RTOL, "atol": ATOL, "matches": sum(r["route_matches"] and r["value_matches"] for r in table),
                             "differences": diffs}}
    if locked["mode"] == "fresh" and not evaluation["all"]["covered"]:
        evaluation["all"]["mra"] = None
        for group in ("by_type", "by_video_source", "by_answer_source", "by_status"):
            for row in evaluation[group].values():
                row["mra"] = None
        evaluation["paired_caw"]["iiixr"]["mra"] = None
    if locked["mode"] == "fresh":
        timings = [s["duration_sec"] for s in stage_list]
        ids = sum(len(s.get("scope", [])) for s in stage_list if s["stage"] == "parse")
        evaluation["runtime"] = {"sum_worker_stage_seconds": sum(timings), "processed_context_rows": ids,
                                 "rough_full_worker_seconds_at_same_rate": sum(timings) * 90 / ids
                                 if ids and len(table) < 90 and evaluation["all"]["covered"] else None,
                                 "estimate_excludes": "Download and model cache overhead; timing sums both GPU lanes"}
    write_json_atomic(run / "evaluation.json", evaluation)
    return evaluation


def track_key(entity):
    if not entity or not entity.get("phrase"):
        return None
    relation = entity.get("rel")
    return entity["phrase"] + "|" + ",".join(entity.get("where") or []) + ("|" + ":".join(relation) if relation else "")


def track_data(run, pred, row):
    if not pred.get("worker"):
        return [], []
    root = run / pred["worker"]
    video = row["video_id"]
    parse_path = root / "geo/out/parse_val_v2.jsonl"
    details = []
    overlay = []
    if parse_path.exists():
        rec = next((rec for line in parse_path.read_text(encoding="utf-8").splitlines()
                    if (rec := json.loads(line))["id"] == row["id"]), {})
        merge = pred.get("decision", {}).get("merge_src")
        variants = ("_v3ntmv", "_v2sc", "_v2", "") if str(merge).startswith(("v", "xg_")) else ("_v2sc", "_v2", "", "_v3ntmv")
        for variant in variants:
            path = root / "geo/out" / f"tracks_val{variant}" / f"{video}.json"
            if not path.is_file():
                continue
            details.append((path.relative_to(run).as_posix(), f"Track output {variant or 'v1'}"))
            if overlay:
                continue
            track = read_json(path)
            lookup = {q["key"]: q for q in track["queries"]}
            for role, color in (("target", "#29ddf1"), ("target2", "#ea84ff"), ("prior_obj", "#ffa64e")):
                query = lookup.get(track_key(rec.get(role)))
                if not query:
                    continue
                track_id = query.get("obj")
                frames = (track.get("objs") or {}).get(str(track_id)) or query.get("frames") or []
                boxes = [{"frame": ix, "time": ix / float(track["fps"]), "box": obj["st"]["box"]}
                         for ix, obj in zip(track["fidx"], frames) if obj and obj.get("st")]
                if boxes:
                    overlay.append({"label": f"{role} · {query['phrase']} · track {track_id if track_id is not None else query['key']}",
                                    "color": color, "width": track["W"], "height": track["H"], "boxes": boxes})
    for path, label in ((root / "geo/out/moge_val" / f"{video}.npz", "MoGe depth and focal arrays"),
                        (parse_path, "Parsed questions and priors")):
        if path.is_file():
            details.append((path.relative_to(run).as_posix(), label))
    return overlay, details


def depth_preview(run, pred, row):
    if not pred.get("worker"):
        return None
    video = row["video_id"]
    source = run / pred["worker"] / "geo/out/moge_val" / f"{video}.npz"
    if not source.is_file():
        return None
    preview = run / "previews" / f"{video}_moge.jpg"
    if preview.is_file():
        return preview.relative_to(run).as_posix()
    with np.load(source, allow_pickle=False) as arrays:
        depth = arrays["depth"][0].astype(np.float32)
    valid = np.isfinite(depth)
    if not valid.any():
        return None
    lo, hi = np.percentile(depth[valid], [5, 95])
    image = np.zeros(depth.shape, dtype=np.uint8)
    image[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    preview.parent.mkdir(exist_ok=True)
    if cv2.imwrite(str(preview), colored):
        return preview.relative_to(run).as_posix()
    return None


def viewer(run):
    questions = read_json(run / "questions.json")
    locked = read_json(run / "predictions_locked.json")
    stages = {s["event_id"]: s for s in read_json(run / "stages.json")}
    panels = []
    esc = lambda value: html.escape(str(value), quote=True)
    for row, pred in zip(questions, locked["predictions"], strict=True):
        clip = row["clip"]
        tracks, links = track_data(run, pred, row)
        if pred.get("artifact"):
            links.append((pred["artifact"], "Published row evidence"))
        media = ""
        if clip.get("path") and clip["status"] == "ok":
            media = (f'<div class="frame"><video controls preload="none" src="{esc(clip["path"])}"'
                     f' poster="{esc(clip.get("preview", ""))}"></video><svg aria-hidden="true"></svg></div>'
                     f'<output>0.000 s</output>')
        depth = depth_preview(run, pred, row)
        depth_img = (f'<figure><img loading="lazy" src="{esc(depth)}" alt="MoGe model depth"><figcaption>'
                     f'MoGe depth at its first sample · preview normalized to the 5th–95th percentile</figcaption></figure>') if depth else ""
        event_links = []
        for event in pred.get("lineage", []):
            stage = stages[event]
            if "worker" in stage:
                for field in ("stderr", "stdout"):
                    event_links.append((f"{stage['worker']}/{stage[field]}", f"{stage['stage']} {field}"))
                calls = f"{stage['worker']}/trace/{stage['stage']}/calls.jsonl"
                if (run / calls).exists():
                    event_links.append((calls, f"{stage['stage']} detailed calls and array paths"))
                for artifact in stage["files"]:
                    if artifact.endswith((".csv", ".jsonl", "_flags.json")) and not artifact.startswith("trace/"):
                        event_links.append((f"{stage['worker']}/{artifact}", Path(artifact).name))
        status = str(pred["status"])
        selected_source = str(pred["source"] or "unavailable")
        legend = "".join(f'<li style="color:{t["color"]}">{esc(t["label"])}</li>' for t in tracks)
        trace = "".join(f'<li><a href="{esc(path)}">{esc(label)}</a></li>' for path, label in links + event_links)
        track_json = json.dumps(tracks, ensure_ascii=True).replace("<", "\\u003c")
        decision = {key: value for key, value in pred.items() if key not in ("lineage", "video_id", "video_source", "category")}
        panels.append(f'''<article data-type="{row['category']}" data-status="{status}" data-source="{selected_source}">
<h2>#{row['id']} · {esc(row['video_id'])} · {row['category']}</h2>
<p>{esc(row['question'])}</p>
<p><b>Prior</b> {esc(row['ground_truth_prior'])}</p><p class="depth">{esc(row['depth_info'])}</p>
<div class="status">{esc(status)} · {esc(selected_source)} · {esc(pred['parsed_value'])}</div>
{media}<div class="legend"><ul>{legend}</ul>{'Viewer overlay from track output' if tracks else 'Track overlay unavailable'}</div>
<script type="application/json">{track_json}</script>
<details><summary>Decision and stage artifacts</summary>{depth_img}<pre>{esc(json.dumps(decision, ensure_ascii=False, indent=2))}</pre><ul>{trace}</ul></details>
</article>''')
    body = "\n".join(panels)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>IIIXR validation 3D</title>
<style>
:root{color-scheme:dark} body{max-width:78rem;margin:auto;padding:1rem;font:16px system-ui;background:#10151e;color:#edf2fa}
a{color:#7ccaff} article{border:1px solid #34415a;padding:1rem;margin:1rem 0;border-radius:10px}
label{margin-right:1rem} .frame{position:relative;max-width:720px} video{width:100%;display:block}
svg{inset:0;position:absolute;width:100%;height:100%;pointer-events:none} .depth,pre{white-space:pre-wrap}
.legend li{display:inline-block;margin-right:1rem} .status{font-weight:700;margin:.55rem 0} pre{overflow-wrap:anywhere}
</style>
<h1>IIIXR validation 3D · geo chain + fallback</h1>
<p>MODE_SCOPE · <a href="evaluation.json">Evaluation</a> · <a href="per_question.csv">Per question</a> · <a href="stages.json">Stage manifest</a></p>
<form><label>Type <select id="type"></select></label>
<label>Status <select id="status"></select></label>
<label>Answer source <select id="source"></select></label></form>
''' + body + '''
<script>
const items = [...document.querySelectorAll('article')];
for (const key of ['type', 'status', 'source']) {
  const select = document.getElementById(key);
  select.add(new Option('All', ''));
  for (const value of [...new Set(items.map(a => a.dataset[key]))].sort()) select.add(new Option(value));
}
document.querySelector('form').addEventListener('change', () => {
  for (const item of items) item.hidden = ['type', 'status', 'source'].some(key => {
    const value = document.getElementById(key).value;
    return value && item.dataset[key] !== value;
  });
});
for (const item of items) {
  const vid = item.querySelector('video');
  if (!vid) continue;
  const svg = item.querySelector('svg'), output = item.querySelector('output');
  const tracks = JSON.parse(item.querySelector('script').textContent);
  const update = () => {
    output.textContent = vid.currentTime.toFixed(3) + ' s';
    svg.replaceChildren();
    if (!tracks.length) return;
    svg.setAttribute('viewBox', '0 0 ' + tracks[0].width + ' ' + tracks[0].height);
    for (const track of tracks) {
      const previous = track.boxes.filter(f => f.time <= vid.currentTime);
      if (previous.length > 1) {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        path.setAttribute('points', previous.map(f => [(f.box[0]+f.box[2])/2,(f.box[1]+f.box[3])/2].join(',')).join(' '));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', track.color);
        path.setAttribute('opacity', '.6');
        svg.append(path);
      }
      const frame = track.boxes.reduce((best, f) => Math.abs(f.time - vid.currentTime) < Math.abs(best.time - vid.currentTime) ? f : best);
      if (Math.abs(frame.time - vid.currentTime) > 0.16) continue;
      const [x,y,x2,y2] = frame.box;
      const box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      for (const [key,val] of Object.entries({x,y,width:x2-x,height:y2-y,fill:'none',stroke:track.color,'stroke-width':3})) box.setAttribute(key,val);
      svg.append(box);
      output.textContent += ' · frame ' + frame.frame + ' · ' + track.label;
    }
  };
  vid.addEventListener('loadedmetadata', update);
  vid.addEventListener('timeupdate', update);
  vid.addEventListener('seeked', update);
}
</script>'''
    (run / "index.html").write_text(page.replace("MODE_SCOPE", esc(f"{locked['mode']} · {locked['scope']}"), 1), encoding="utf-8")
