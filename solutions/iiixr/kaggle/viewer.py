"""Standalone offline HTML failure-analysis viewer for IIIXR g_box (never modifies QuantiPhy_Review.html)."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def build_viewer_payload_for_question(
    *,
    eval_row: dict[str, Any],
    parse_trace: dict[str, Any] | None,
    sample_bundle: dict[str, Any] | None,
    video_track_bundle: dict[str, Any] | None,
    video_audit: dict[str, Any] | None,
    cache_rec: dict[str, Any] | None,
    caw_rec: dict[str, Any] | None,
    events_for_question: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble compact, self-contained JSON payload for one question card in index.html."""
    qid = int(eval_row["id"])
    vid = str(eval_row["video_id"])

    # Extract per-frame overlay tracks for active branch (and alternate branch on #142)
    branches_viz: dict[str, Any] = {}
    if sample_bundle and sample_bundle.get("branches"):
        for bname, bdata in sample_bundle["branches"].items():
            mtrace = bdata.get("measure_trace") or {}
            branches_viz[bname] = {
                "result": bdata.get("result"),
                "target_track_summary": mtrace.get("target_track_summary"),
                "prior_track_summary": mtrace.get("prior_track_summary"),
                "prior_step": mtrace.get("prior_step"),
                "target_step": mtrace.get("target_step"),
                "formula_trace": mtrace.get("formula_trace"),
                "derived_curves": mtrace.get("derived_curves"),
                "array_refs": bdata.get("array_refs"),
                "trace_json_ref": bdata.get("trace_json_ref"),
            }

    # Build lightweight frame-indexed overlay geometry per track variant (v2 / v1 / v3nt)
    track_overlays_by_variant: dict[str, Any] = {}
    if video_track_bundle and parse_trace:
        tgt_ent = parse_trace.get("target") or {}
        pr_ent = parse_trace.get("prior_obj") or {}

        def _ent_key(e: dict[str, Any] | None) -> str | None:
            if not e or not e.get("phrase"):
                return None
            rel = e.get("rel")
            return e["phrase"] + "|" + ",".join(e.get("where") or []) + ("|" + rel[0] + ":" + rel[1] if rel else "")

        t_key = _ent_key(tgt_ent)
        p_key = _ent_key(pr_ent)

        for var_name, tj_key in (("v2", "track_v2"), ("v1", "track_v1"), ("v3nt", "track_v3nt")):
            tj = video_track_bundle.get(tj_key)
            if not tj:
                continue
            fidx = tj.get("fidx") or []
            fps = float(tj.get("fps") or 24.0)
            W, H = int(tj.get("W") or 854), int(tj.get("H") or 480)
            q_map = {q.get("key"): q for q in tj.get("queries", [])}

            def _extract_obj_frames(q_obj: dict[str, Any] | None) -> list[dict[str, Any] | None]:
                if not q_obj:
                    return [None] * len(fidx)
                raw_fr = (
                    tj["objs"].get(str(q_obj.get("obj")), [])
                    if tj.get("version") == "v2" and q_obj.get("obj") is not None
                    else q_obj.get("frames", [])
                )
                out_fr: list[dict[str, Any] | None] = []
                for idx_slot in range(len(fidx)):
                    fr = raw_fr[idx_slot] if idx_slot < len(raw_fr) else None
                    st = fr.get("st") if isinstance(fr, dict) else None
                    if st:
                        out_fr.append({
                            "box": st["box"],
                            "c": [round(float(st["c"][0]), 2), round(float(st["c"][1]), 2)],
                            "rr": [round(float(v), 2) for v in st["rr"]],
                            "area": int(st["area"]),
                            "border": int(st.get("border", 0)),
                            "lg": round(float(fr.get("lg", 0.0)), 2),
                            "poly": fr.get("poly"),
                        })
                    else:
                        out_fr.append(None)
                return out_fr

            # For gravity videos (e.g., internet_0019 where prior phrase is None), find target as reference too
            tq = q_map.get(t_key) if t_key else None
            pq = q_map.get(p_key) if p_key else None
            track_overlays_by_variant[var_name] = {
                "W": W,
                "H": H,
                "fps": fps,
                "fidx": fidx,
                "target_key": t_key,
                "prior_key": p_key,
                "target_anchor": tq.get("anchor") if tq else None,
                "prior_anchor": pq.get("anchor") if pq else None,
                "target_frames": _extract_obj_frames(tq),
                "prior_frames": _extract_obj_frames(pq) if (pq and p_key != t_key) else None,
            }

    return {
        "id": qid,
        "video_id": vid,
        "video_relpath": f"videos/{vid}.mp4" if video_audit else None,
        "video_audit": video_audit,
        "eval": eval_row,
        "parse_trace": parse_trace,
        "cache_ref": cache_rec,
        "caw_ref": caw_rec,
        "selected_branch": (sample_bundle or {}).get("selected_branch", ""),
        "fusion_decision": (sample_bundle or {}).get("fusion_decision"),
        "branches": branches_viz,
        "track_overlays": track_overlays_by_variant,
        "scene_ruler": (video_track_bundle or {}).get("scene_ruler"),
        "overlay_png_refs": (video_track_bundle or {}).get("overlay_refs") or {},
        "events": events_for_question,
    }


def render_failure_analysis_html(
    *,
    output_html_path: Path,
    run_meta: dict[str, Any],
    eval_summary: dict[str, Any],
    questions_payload: list[dict[str, Any]],
) -> Path:
    """Write standalone offline HTML viewer with synchronized video+canvas overlay, derived plots, and formulas."""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    embedded_json = json.dumps(
        {
            "run_meta": run_meta,
            "eval_summary": eval_summary,
            "questions": questions_payload,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")

    title_str = f"IIIXR g_box Failure Analysis · {escape(str(run_meta.get('mode', 'run')).upper())} · {escape(str(run_meta.get('run_id', '')))}"

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title_str}</title>
<style>
  :root {{
    --bg: #0b1120;
    --panel: #111827;
    --panel-alt: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --prior: #facc15;
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
    --purple: #c084fc;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code, pre {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
  }}
  header {{
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(11, 17, 32, 0.96);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
  }}
  .hdr-top {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }}
  .hdr-title {{ font-size: 17px; font-weight: 700; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--panel-alt);
  }}
  .badge-good {{ color: var(--good); border-color: rgba(52,211,153,0.4); }}
  .badge-warn {{ color: var(--warn); border-color: rgba(251,191,36,0.4); }}
  .badge-bad {{ color: var(--bad); border-color: rgba(248,113,113,0.4); }}
  .badge-derived {{ color: var(--purple); border-color: rgba(192,132,252,0.45); }}
  .controls {{
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }}
  input[type="search"] {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    min-width: 240px;
    font-size: 13px;
  }}
  button.flt, button.btn {{
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
  }}
  button.flt.active {{
    background: #0284c7;
    border-color: #38bdf8;
    color: #fff;
  }}
  main {{ max-width: 1560px; margin: 0 auto; padding: 18px 20px 60px; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }}
  .stat-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .stat-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat-val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .stat-sub {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
  .notice {{
    background: rgba(251, 191, 36, 0.09);
    border: 1px solid rgba(251, 191, 36, 0.35);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    margin-bottom: 18px;
  }}
  .q-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 24px;
    overflow: hidden;
  }}
  .q-hdr {{
    background: var(--panel-alt);
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
  }}
  .q-body {{
    display: grid;
    grid-template-columns: 1.08fr 1fr;
    gap: 16px;
    padding: 16px;
  }}
  @media (max-width: 1100px) {{
    .q-body {{ grid-template-columns: 1fr; }}
  }}
  .video-stage {{
    position: relative;
    width: 100%;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  .video-stage video {{
    display: block;
    width: 100%;
    height: auto;
  }}
  .video-stage canvas {{
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }}
  .scrub-bar {{
    margin-top: 8px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .scrub-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }}
  .scrub-row input[type="range"] {{ flex: 1; min-width: 160px; }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 12px;
    margin-top: 6px;
    color: var(--muted);
  }}
  .dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
  }}
  .cmp-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    margin-bottom: 12px;
  }}
  .cmp-table th, .cmp-table td {{
    border: 1px solid var(--border);
    padding: 6px 8px;
    text-align: left;
  }}
  .cmp-table th {{ background: var(--panel-alt); color: var(--muted); font-weight: 600; }}
  .plot-box {{
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 10px;
  }}
  .plot-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    margin-bottom: 4px;
  }}
  svg.chart {{
    width: 100%;
    height: 155px;
    background: #090e1a;
    border-radius: 6px;
    cursor: crosshair;
    display: block;
  }}
  .formula-box {{
    background: #090e1a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 12px;
    font-size: 12.5px;
  }}
  .special-box {{
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 12px;
    font-size: 12.5px;
  }}
  .missing-trace {{
    background: rgba(248, 113, 113, 0.1);
    border: 1px dashed rgba(248, 113, 113, 0.5);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 8px;
    font-size: 12.5px;
    color: #fecaca;
  }}
  details {{
    margin-top: 8px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: #0f172a;
    padding: 6px 10px;
  }}
  summary {{ cursor: pointer; font-weight: 600; font-size: 12.5px; color: var(--accent); }}
  .evt-list {{ list-style: none; padding-left: 0; margin: 8px 0 0; font-size: 12px; }}
  .evt-list li {{
    padding: 5px 6px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.5);
  }}
</style>
</head>
<body>
<header>
  <div class="hdr-top">
    <div>
      <div class="hdr-title" id="hdrTitle">{title_str}</div>
      <div style="font-size:12px;color:var(--muted)" id="hdrSub"></div>
    </div>
    <div id="hdrBadges"></div>
  </div>
  <div class="controls">
    <input type="search" id="searchBox" placeholder="Tìm theo ID (vd: 401, 142, 2234), video_id hoặc câu hỏi..." />
    <button class="flt active" data-filter="all">Tất cả (<span id="cntAll">0</span>)</button>
    <button class="flt" data-filter="error">Lỗi / Thiếu trace (<span id="cntErr">0</span>)</button>
    <button class="flt" data-filter="cache_diff">Lệch cache (&gt;1e-4) (<span id="cntDiff">0</span>)</button>
    <button class="flt" data-filter="better_caw">Tốt hơn CAW (<span id="cntBetter">0</span>)</button>
    <button class="flt" data-filter="worse_caw">Xấu hơn CAW (<span id="cntWorse">0</span>)</button>
    <button class="flt" data-filter="special">Mốc chẩn đoán (#142, #399, #401, #2234)</button>
  </div>
</header>

<main>
  <div class="summary-grid" id="summaryGrid"></div>
  <div class="notice" id="scopeNotice"></div>
  <div id="cardsContainer"></div>
</main>

<script id="viewerData" type="application/json">{embedded_json}</script>
<script>
(function() {{
  const DATA = JSON.parse(document.getElementById("viewerData").textContent);
  const meta = DATA.run_meta || {{}};
  const evSum = DATA.eval_summary || {{}};
  const questions = DATA.questions || [];

  function fmt(v, d = 6) {{
    if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "—";
    const n = Number(v);
    if (!Number.isFinite(n)) return "—";
    return Math.abs(n) >= 100 ? n.toFixed(4) : n.toFixed(d);
  }}

  function esc(s) {{
    return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }}

  // Populate Header & Summary
  document.getElementById("hdrSub").textContent =
    `Source SHA: ${{meta.source_commit || "uncommitted"}} · Upstream IIIXR: ${{(meta.upstream_commit || "").slice(0, 12)}} · Mode: ${{meta.mode}}`;
  const mraMap = evSum.mean_per_question_mra || {{}};
  const sGrid = document.getElementById("summaryGrid");
  const cardsSummary = [
    ["Mode & Scope", `${{(meta.mode || "").toUpperCase()}} (${{evSum.denominator_questions || questions.length}} câu 2D)`, `Dtype profile: ${{meta.dtype_profile || "cpu_replay"}}`],
    ["Mean MRA · cached_box", fmt(mraMap.cached_box, 4), "18 g_box từ cache IIIXR (tham chiếu ~0.8389)"],
    ["Mean MRA · fresh_box", mraMap.fresh_box !== null && mraMap.fresh_box !== undefined ? fmt(mraMap.fresh_box, 4) : "Chưa chạy GPU fresh", evSum.fresh_coverage ? `Coverage: ${{evSum.fresh_coverage.generated_ok}}/${{evSum.fresh_coverage.denominator}}` : "Chế độ replay cache"],
    ["Mean MRA · CAW baseline", mraMap.caw !== null && mraMap.caw !== undefined ? fmt(mraMap.caw, 4) : "Chưa gắn TAR", "Từ quantiphy_full.tar (tham chiếu ~0.4278)"],
    ["Khớp cache (atol=1e-6, rtol=1e-4)", evSum.cache_parity_diagnostics && evSum.cache_parity_diagnostics.matched_count !== null ? `${{evSum.cache_parity_diagnostics.matched_count}} / ${{evSum.denominator_questions}}` : "N/A (replay)", "Mốc chẩn đoán sai lệch số học"]
  ];
  sGrid.innerHTML = cardsSummary.map(([lbl, val, sub]) => `
    <div class="stat-card">
      <div class="stat-label">${{esc(lbl)}}</div>
      <div class="stat-val">${{esc(val)}}</div>
      <div class="stat-sub">${{esc(sub)}}</div>
    </div>
  `).join("");
  document.getElementById("scopeNotice").textContent = evSum.benchmark_scope_notice || "";

  // Filter helpers
  function hasMissingTrace(q) {{
    const hasBranch = q.branches && Object.keys(q.branches).length > 0;
    return q.eval.status !== "ok" || !hasBranch;
  }}
  function hasCacheDiff(q) {{
    const rel = q.eval.rel_err_vs_cache;
    return rel !== null && rel !== undefined && rel !== "" && Number(rel) > 1e-4;
  }}
  function primaryMra(q) {{
    return (q.eval.mra_fresh_box !== null && q.eval.mra_fresh_box !== undefined && q.eval.mra_fresh_box !== "")
      ? Number(q.eval.mra_fresh_box)
      : Number(q.eval.mra_cached_box || 0);
  }}
  function isBetterCaw(q) {{
    if (q.eval.mra_caw === null || q.eval.mra_caw === undefined || q.eval.mra_caw === "") return false;
    return primaryMra(q) > Number(q.eval.mra_caw) + 1e-9;
  }}
  function isWorseCaw(q) {{
    if (q.eval.mra_caw === null || q.eval.mra_caw === undefined || q.eval.mra_caw === "") return false;
    return primaryMra(q) + 1e-9 < Number(q.eval.mra_caw);
  }}

  document.getElementById("cntAll").textContent = questions.length;
  document.getElementById("cntErr").textContent = questions.filter(hasMissingTrace).length;
  document.getElementById("cntDiff").textContent = questions.filter(hasCacheDiff).length;
  document.getElementById("cntBetter").textContent = questions.filter(isBetterCaw).length;
  document.getElementById("cntWorse").textContent = questions.filter(isWorseCaw).length;

  function renderSvgPlot(svgEl, xs, seriesList, vLineX, winBand, partTimes, yLabel, onPickTime) {{
    const W = 560, H = 155, padL = 48, padR = 14, padT = 14, padB = 24;
    svgEl.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
    if (!xs || xs.length < 2) {{
      svgEl.innerHTML = `<text x="${{W/2}}" y="${{H/2}}" fill="#94a3b8" font-size="12" text-anchor="middle">Chưa có chuỗi số (chế độ replay cache hoặc track rỗng)</text>`;
      return;
    }}
    let xMin = Math.min(...xs), xMax = Math.max(...xs);
    if (xMax <= xMin) xMax = xMin + 1;
    let vals = [];
    seriesList.forEach(s => (s.y || []).forEach(v => {{ if (v !== null && Number.isFinite(v)) vals.push(v); }}));
    if (!vals.length) vals = [0, 1];
    let yMin = Math.min(...vals), yMax = Math.max(...vals);
    if (yMax <= yMin) {{ yMin -= 1; yMax += 1; }}
    const padY = 0.08 * (yMax - yMin);
    yMin -= padY; yMax += padY;

    const sx = x => padL + ((x - xMin) / (xMax - xMin)) * (W - padL - padR);
    const sy = y => H - padB - ((y - yMin) / (yMax - yMin)) * (H - padT - padB);

    let out = `<rect x="${{padL}}" y="${{padT}}" width="${{W-padL-padR}}" height="${{H-padT-padB}}" fill="#0b1324" stroke="#1e293b"/>`;
    if (winBand && winBand.length === 2 && Number.isFinite(winBand[0]) && Number.isFinite(winBand[1])) {{
      const bx0 = sx(Math.max(xMin, winBand[0])), bx1 = sx(Math.min(xMax, winBand[1]));
      if (bx1 > bx0) out += `<rect x="${{bx0}}" y="${{padT}}" width="${{bx1-bx0}}" height="${{H-padT-padB}}" fill="rgba(56,189,248,0.13)"/>`;
    }}
    if (vLineX !== null && vLineX !== undefined && Number.isFinite(vLineX)) {{
      const vx = sx(Math.min(Math.max(vLineX, xMin), xMax));
      out += `<line x1="${{vx}}" y1="${{padT}}" x2="${{vx}}" y2="${{H-padB}}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="4,3"/>`;
    }}
    // Axes labels
    out += `<text x="6" y="${{padT+10}}" fill="#94a3b8" font-size="10">${{fmt(yMax, 1)}}</text>`;
    out += `<text x="6" y="${{H-padB}}" fill="#94a3b8" font-size="10">${{fmt(yMin, 1)}}</text>`;
    out += `<text x="${{padL}}" y="${{H-6}}" fill="#94a3b8" font-size="10">t=${{xMin.toFixed(2)}}s</text>`;
    out += `<text x="${{W-padR}}" y="${{H-6}}" fill="#94a3b8" font-size="10" text-anchor="end">t=${{xMax.toFixed(2)}}s · ${{esc(yLabel)}}</text>`;

    const partSet = new Set((partTimes || []).map(t => Number(t).toFixed(4)));
    seriesList.forEach(s => {{
      if (!s.show) return;
      const pts = [];
      (s.y || []).forEach((yv, i) => {{
        if (yv !== null && Number.isFinite(yv)) pts.push([sx(xs[i]), sy(yv), xs[i]]);
      }});
      if (s.mode === "line" && pts.length >= 2) {{
        const d = pts.map((p, i) => `${{i===0?'M':'L'}}${{p[0].toFixed(1)}},${{p[1].toFixed(1)}}`).join(" ");
        out += `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="${{s.width || 1.8}}" ${{s.dash ? `stroke-dasharray="${{s.dash}}"` : ""}}/>`;
      }}
      if (s.dots) {{
        pts.forEach(p => {{
          const inWin = partSet.has(Number(p[2]).toFixed(4));
          out += `<circle cx="${{p[0].toFixed(1)}}" cy="${{p[1].toFixed(1)}}" r="${{inWin ? 3.4 : 2.1}}" fill="${{inWin ? '#facc15' : s.color}}"/>`;
        }});
      }}
    }});
    svgEl.innerHTML = out;
    svgEl.onclick = function(ev) {{
      const rect = svgEl.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * W;
      const tClick = xMin + Math.min(1, Math.max(0, (relX - padL) / (W - padL - padR))) * (xMax - xMin);
      if (onPickTime) onPickTime(tClick);
    }};
  }}

  const container = document.getElementById("cardsContainer");

  questions.forEach((q, qIdx) => {{
    const ev = q.eval || {{}};
    const qid = q.id;
    const card = document.createElement("section");
    card.className = "q-card";
    card.dataset.id = String(qid);
    card.dataset.video = q.video_id;
    card.dataset.search = `${{qid}} ${{q.video_id}} ${{ev.question || ""}} ${{ev.prior_text || ""}} ${{ev.kind || ""}}`.toLowerCase();

    const statusBadge = ev.status === "ok"
      ? `<span class="badge badge-good">status: ok</span>`
      : `<span class="badge badge-warn">status: ${{esc(ev.status)}}</span>`;
    const tagBadge = `<span class="badge">cache tv=${{esc(ev.tv_cache || "—")}}</span>` +
      (ev.track_used ? ` <span class="badge">fresh branch=${{esc(ev.track_used)}}</span>` : "");

    const branchNames = Object.keys(q.branches || {{}});
    let activeBranchName = q.selected_branch && q.branches[q.selected_branch] ? q.selected_branch : (branchNames[0] || "");
    const overlayVariants = Object.keys(q.track_overlays || {{}});
    let activeOverlayVar = overlayVariants.includes(activeBranchName) ? activeBranchName : (overlayVariants[0] || "");

    // Special diagnostic panels (#401, #142, #2234)
    let specialHtml = "";
    if (qid === 401) {{
      const bAct = q.branches ? (q.branches[activeBranchName] || {{}}) : {{}};
      const winSens = ((bAct.target_step || {{}}).window_sensitivity) || [];
      const sensRows = winSens.map(w =>
        `<tr><td>±${{w.half_s.toFixed(2)}} s</td><td>${{fmt(w.ax_px_s2, 2)}}</td><td>${{fmt(w.ay_px_s2, 2)}}</td><td>${{fmt(w.accel_px_s2, 2)}}</td><td><b>${{fmt(w.accel_unit, 6)}}</b></td></tr>`
      ).join("");
      specialHtml += `
        <div class="special-box">
          <b>Chẩn đoán đặc biệt #401 (simulation_0163 · gia tốc tại t=1.5s):</b>
          <div style="margin-top:4px">
            <code>cached_box = ${{fmt(ev.cached_box, 6)}} m/s²</code> (MRA=${{fmt(ev.mra_cached_box, 2)}}) ·
            <code>fresh_box = ${{fmt(ev.fresh_box, 6)}} m/s²</code> ·
            <code>cached_point (g_pt) = ${{fmt(ev.cached_point, 6)}} m/s²</code> (MRA=${{fmt(ev.mra_cached_point, 2)}}) ·
            <code>GT = ${{fmt(ev.ground_truth, 4)}} m/s²</code>
          </div>
          <div style="margin-top:4px;color:var(--warn)">
            Lưu ý nguồn: <code>cached_point (g_pt)</code> là output đọc từ file cache <code>mot_val_401.json</code> (nhánh point tracker không chạy trong thí nghiệm nhánh box này). Dưới đây là bảng độ nhạy thực nghiệm của phép fit bậc 2 theo bán kính cửa sổ thời gian <code>half_a</code> trên quỹ đạo hiện có để tự kiểm tra (không suy đoán nguyên nhân từ đáp số):
          </div>
          ${{sensRows ? `<table class="cmp-table" style="margin-top:6px"><thead><tr><th>Cửa sổ ±half_a</th><th>a_x (px/s²)</th><th>a_y (px/s²)</th><th>||a|| (px/s²)</th><th>Gia tốc quy đổi (m/s²)</th></tr></thead><tbody>${{sensRows}}</tbody></table>` : `<div style="margin-top:4px;color:var(--muted)">Bảng độ nhạy cửa sổ sẽ tự động điền khi có quỹ đạo fresh/track.</div>`}}
        </div>`;
    }}
    if (qid === 142) {{
      specialHtml += `
        <div class="special-box">
          <b>Chẩn đoán khác biệt tag #142 (simulation_0089 · tv=v3nt vs tv=v2):</b>
          <div style="margin-top:4px">
            Trong <code>mot_val_142.json</code>, bản ghi dùng <code>tv="v3nt"</code> (<code>g_box = ${{fmt(ev.cached_box, 6)}}</code>, khớp với <code>s1_g = 13.008680</code> trong ngưỡng <code>|log(g_box/s1_g)| &lt; 0.01</code> của <code>r10/MOTION/rules.py</code>), trong khi nhánh <code>v2</code> trong <code>preds/val_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv</code> cho <code>13.552076 m/s</code>.
            Kết quả fresh ghi nhận song song cả hai nhánh: <code>fresh_box_v2 = ${{fmt(ev.fresh_box_v2, 6)}}</code> và <code>fresh_box_v3nt = ${{fmt(ev.fresh_box_v3nt, 6)}}</code>. Dùng nút chọn nhánh bên dưới video để đổi trực tiếp overlay/đồ thị giữa <code>v2</code> và <code>v3nt</code>.
          </div>
        </div>`;
    }}
    if (qid === 2234) {{
      const bAct = q.branches ? (q.branches[activeBranchName] || {{}}) : {{}};
      const ts = bAct.target_step || {{}};
      specialHtml += `
        <div class="special-box">
          <b>Chẩn đoán vận tốc ngang #2234 (internet_0019 · kind=hvel):</b>
          <div style="margin-top:4px">
            Công thức upstream (<code>geo/measure_v3.py:px_quantity(kind="hvel")</code>) lấy độ dịch chuyển trục ngang <code>|x(b) - x(a)| / (b - a)</code> bằng fit tuyến tính cục bộ <code>pos_at_c(half=0.08s)</code> ở hai đầu quỹ đạo, quy đổi tỉ lệ bằng gia tốc rơi tự do <code>g = 9.8 m/s²</code>.
            ${{ts.dx_abs_px !== undefined && ts.dx_abs_px !== null ? `<br/><code>|Δx| = ${{fmt(ts.dx_abs_px, 2)}} px</code>, <code>Δy = ${{fmt(ts.dy_px, 2)}} px</code>, <code>Δt = ${{fmt(ts.dt_sec, 4)}} s</code> ⇒ <code>v_x = ${{fmt(ts.px_quantity, 3)}} px/s</code> (so với tốc độ 2D toàn phần <code>${{fmt(ts.full_2d_avgspeed_px_s_for_comparison, 3)}} px/s</code>).` : ""}}
          </div>
        </div>`;
    }}

    const missingHtml = hasMissingTrace(q) ? `
      <div class="missing-trace">
        <b>Vị trí thiếu trace trực tiếp:</b> Lượt chạy hiện tại đang ở chế độ <code>${{esc(meta.mode)}}</code> (status câu hỏi: <code>${{esc(ev.status)}}</code>${{ev.error_stage ? `, error_stage: <code>${{esc(ev.error_stage)}}</code>` : ""}}).
        Các bước model GPU (GroundingDINO / OWLv2 / Qwen3-VL / SAM 2.1) chưa được thực thi trên câu này nên overlay mask/box và mảng quỹ đạo <code>.npy</code> được đánh dấu trống thay vì giả lập.
      </div>` : "";

    // Build events list HTML
    const evtItems = (q.events || []).map(e => {{
      const links = [];
      Object.entries(e.output_refs || {{}}).forEach(([k, ref]) => {{
        if (ref && ref.path) links.push(`<a href="${{esc(ref.path)}}" download>${{esc(k)}} (${{esc(ref.kind || "file")}})</a>`);
      }});
      return `<li><code>[${{esc(e.stage)}} · ${{esc(e.call_kind)}}]</code> status=<b>${{esc(e.status)}}</b> · backend=${{esc(e.backend)}} · dev=${{esc(e.device)}} · ${{Number(e.duration_sec || 0).toFixed(3)}}s ${{links.length ? " · " + links.join(" | ") : ""}} ${{e.error ? `<span style="color:var(--bad)">err=${{esc(e.error)}}</span>` : ""}}</li>`;
    }}).join("");

    const va = q.video_audit || {{}};
    const videoAuditSummary = va.sha256
      ? `SHA256: <code>${{va.sha256.slice(0, 16)}}…</code> · ${{va.raw_width}}×${{va.raw_height}} · frames=${{va.decoded_frames}} · container_fps=${{fmt(va.container_fps, 3)}} vs csv_fps=${{fmt(va.csv_fps, 1)}} · 480p_resized=${{va.tc_load_video_resized_480p}} · status=<code>${{esc(va.equivalence_status)}}</code>`
      : "Video file chưa gắn vào thư mục run/videos/";

    card.innerHTML = `
      <div class="q-hdr">
        <div>
          <div style="font-size:15px;font-weight:700">#${{qid}} · <code>${{esc(q.video_id)}}</code> · <span style="color:var(--accent)">${{esc(ev.cat)}} / ${{esc(ev.kind)}}</span></div>
          <div style="margin-top:4px;font-size:13.5px">${{esc(ev.question)}}</div>
          <div style="margin-top:2px;font-size:12px;color:var(--muted)">Prior: <code>${{esc(ev.prior_text)}}</code> · Đơn vị: <code>${{esc(ev.unit)}}</code></div>
        </div>
        <div>${{statusBadge}} ${{tagBadge}}</div>
      </div>
      <div class="q-body">
        <div>
          <div class="video-stage">
            ${{q.video_relpath ? `<video id="vid_${{qid}}" src="${{esc(q.video_relpath)}}" preload="metadata" muted playsinline></video>` : `<div style="padding:50px 20px;text-align:center;color:var(--muted)">Không có file video MP4 trong run/videos/</div>`}}
            <canvas id="cv_${{qid}}"></canvas>
          </div>
          <div class="scrub-bar">
            <div class="scrub-row">
              <button class="btn" id="prev_${{qid}}">◀ -1f</button>
              <button class="btn" id="play_${{qid}}">▶ Play</button>
              <button class="btn" id="next_${{qid}}">+1f ▶</button>
              <input type="range" id="slider_${{qid}}" min="0" max="${{Math.max((va.decoded_frames || 1) - 1, 0)}}" step="1" value="0" />
              <span style="font-family:monospace;font-size:12px" id="finfo_${{qid}}">frame=0 · PTS=0.000s</span>
            </div>
            <div class="scrub-row" style="margin-top:6px">
              <label style="font-size:12px"><input type="checkbox" id="chkRaw_${{qid}}" checked /> Quỹ đạo raw c(t)</label>
              <label style="font-size:12px"><input type="checkbox" id="chkSmooth_${{qid}}" checked /> Quỹ đạo làm mượt/segmented</label>
              <label style="font-size:12px"><input type="checkbox" id="chkMask_${{qid}}" checked /> Mask/Box overlay</label>
              ${{overlayVariants.length > 1 ? `<select id="selVar_${{qid}}" style="background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 6px;font-size:12px">${{overlayVariants.map(v => `<option value="${{esc(v)}}" ${{v===activeOverlayVar?'selected':''}}>Track: ${{esc(v)}}</option>`).join("")}}</select>` : ""}}
              <button class="btn" id="jmp_${{qid}}">Tới thời điểm đo</button>
            </div>
            <div class="legend">
              <span><i class="dot" style="background:#38bdf8"></i>Target object (box + mask + tâm c)</span>
              <span><i class="dot" style="background:#facc15"></i>Reference / Prior (hoặc vạch làn đường)</span>
              <span><i class="dot" style="background:#f87171"></i>Thời điểm câu hỏi yêu cầu</span>
              <span><i class="dot" style="background:#c084fc"></i>Quỹ đạo smoothed (derived)</span>
            </div>
            <div style="margin-top:6px;font-size:11px;color:var(--muted)">${{videoAuditSummary}}</div>
          </div>

          <div class="plot-box">
            <div class="plot-hdr"><span><b>Quỹ đạo tâm x(t), y(t)</b> (click để tua frame)</span><span class="badge badge-derived">DERIVED PLOT</span></div>
            <svg class="chart" id="svgXY_${{qid}}"></svg>
          </div>
          <div class="plot-box">
            <div class="plot-hdr"><span><b>Kích thước vật / box &amp; minAreaRect (px)</b></span><span class="badge badge-derived">DERIVED PLOT</span></div>
            <svg class="chart" id="svgSz_${{qid}}"></svg>
          </div>
          <div class="plot-box">
            <div class="plot-hdr"><span><b>Vận tốc v(t) (px/s) &amp; Gia tốc a(t) (px/s²)</b></span><span class="badge badge-derived">DERIVED PLOT</span></div>
            <svg class="chart" id="svgVA_${{qid}}"></svg>
          </div>
        </div>

        <div>
          <table class="cmp-table">
            <thead>
              <tr>
                <th>Nguồn dự đoán</th>
                <th>Giá trị (${{esc(ev.unit)}})</th>
                <th>Sai số tương đối vs GT</th>
                <th>MRA câu hỏi</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><b>cached_box</b> (IIIXR cache)</td>
                <td><code>${{fmt(ev.cached_box, 6)}}</code></td>
                <td>${{fmt(ev.rel_cache_err_vs_gt, 4)}}</td>
                <td><b>${{fmt(ev.mra_cached_box, 2)}}</b></td>
              </tr>
              <tr>
                <td><b>fresh_box</b> (Chạy mới)</td>
                <td><code>${{fmt(ev.fresh_box, 6)}}</code></td>
                <td>${{fmt(ev.rel_err_vs_gt, 4)}}</td>
                <td><b>${{ev.mra_fresh_box !== "" && ev.mra_fresh_box !== null ? fmt(ev.mra_fresh_box, 2) : "—"}}</b></td>
              </tr>
              <tr>
                <td><b>caw</b> (Baseline CAW-4B)</td>
                <td><code>${{fmt(ev.caw, 6)}}</code></td>
                <td>${{ev.caw && ev.ground_truth ? fmt(Math.abs(ev.caw - ev.ground_truth) / Math.abs(ev.ground_truth), 4) : "—"}}</td>
                <td><b>${{ev.mra_caw !== "" && ev.mra_caw !== null ? fmt(ev.mra_caw, 2) : "—"}}</b></td>
              </tr>
              <tr>
                <td><b>Ground Truth</b> (Chỉ đọc sau inference)</td>
                <td><code>${{fmt(ev.ground_truth, 6)}}</code></td>
                <td>0.0000</td>
                <td>1.00</td>
              </tr>
            </tbody>
          </table>

          ${{specialHtml}}
          ${{missingHtml}}

          <div class="formula-box" id="fbox_${{qid}}"></div>

          <details open>
            <summary>Chi tiết Parse, Vật chuẩn (Scale mpp) &amp; Tham số Cửa sổ Động học</summary>
            <pre id="preDetail_${{qid}}" style="white-space:pre-wrap;color:#cbd5e1;margin:8px 0 0;max-height:300px;overflow:auto"></pre>
          </details>

          <details>
            <summary>Luồng Model / Tool Calls theo thứ tự &amp; Link tải Raw Artifacts (${{(q.events || []).length}} events)</summary>
            <div id="rawLinks_${{qid}}" style="margin-top:6px;font-size:12px"></div>
            <ul class="evt-list">${{evtItems || "<li>Không có event GPU/tool trực tiếp.</li>"}}</ul>
          </details>
        </div>
      </div>
    `;
    container.appendChild(card);

    // Interactive wiring per question
    const vidEl = document.getElementById(`vid_${{qid}}`);
    const cvEl = document.getElementById(`cv_${{qid}}`);
    const sliderEl = document.getElementById(`slider_${{qid}}`);
    const finfoEl = document.getElementById(`finfo_${{qid}}`);
    const chkRaw = document.getElementById(`chkRaw_${{qid}}`);
    const chkSmooth = document.getElementById(`chkSmooth_${{qid}}`);
    const chkMask = document.getElementById(`chkMask_${{qid}}`);
    const selVar = document.getElementById(`selVar_${{qid}}`);

    const fpsVal = Number(va.csv_fps || (q.parse_trace && q.parse_trace.fps) || 24.0);
    let curFrame = 0;

    function currentBranchObj() {{
      if (q.branches && q.branches[activeOverlayVar]) return q.branches[activeOverlayVar];
      if (q.branches && q.branches[activeBranchName]) return q.branches[activeBranchName];
      const ks = Object.keys(q.branches || {{}});
      return ks.length ? q.branches[ks[0]] : null;
    }}

    function updateFormulaAndCharts() {{
      const bObj = currentBranchObj();
      const fbox = document.getElementById(`fbox_${{qid}}`);
      const preD = document.getElementById(`preDetail_${{qid}}`);
      const rawLinks = document.getElementById(`rawLinks_${{qid}}`);
      if (!bObj) {{
        fbox.innerHTML = `<b>Công thức thay số:</b> Chưa có dữ liệu nhánh đo trực tiếp (chế độ replay cache: <code>cached_box = ${{fmt(ev.cached_box, 6)}} ${{esc(ev.unit)}}</code>).`;
        preD.textContent = JSON.stringify({{ parse: q.parse_trace, cache_record: q.cache_ref }}, null, 2);
        renderSvgPlot(document.getElementById(`svgXY_${{qid}}`), [], [], null, null, [], "px", null);
        renderSvgPlot(document.getElementById(`svgSz_${{qid}}`), [], [], null, null, [], "px", null);
        renderSvgPlot(document.getElementById(`svgVA_${{qid}}`), [], [], null, null, [], "px/s", null);
        return;
      }}
      const ft = bObj.formula_trace || {{}};
      const ps = bObj.prior_step || {{}};
      const ts = bObj.target_step || {{}};
      fbox.innerHTML = `
        <div><b>Công thức vật lý &amp; thay số trực tiếp (${{esc(ts.file_fn || "geo/measure_v3.py")}}):</b></div>
        <div style="margin-top:4px"><code>${{esc(ft.symbolic || "")}}</code></div>
        <div style="margin-top:4px">
          • Vật chuẩn (<code>${{esc(ps.method || "prior")}}</code>): <code>prior_value_si = ${{fmt(ft.prior_value_si, 4)}}</code> / <code>prior_px = ${{fmt(ft.prior_px, 4)}} px</code> ⇒ <code>mpp = ${{fmt(ft.mpp, 7)}}</code><br/>
          • Đại lượng ảnh đích (<code>${{esc(ts.estimator || ev.kind)}}</code>): <code>px_quantity = ${{fmt(ft.px_quantity, 5)}}</code> (${{ (ts.participating_video_frames || []).length }} frames tham gia)<br/>
          • Đổi đơn vị: <code>${{fmt(ft.px_quantity, 5)}} × ${{fmt(ft.mpp, 7)}} / ${{fmt(ft.unit_si_factor, 4)}} = <b>${{fmt(ft.final_value, 6)}} ${{esc(ft.unit)}}</b></code>
        </div>
      `;
      preD.textContent = JSON.stringify({{
        target_track: bObj.target_track_summary,
        prior_step: ps,
        target_step: {{ ...ts, per_timestamp_speeds: ts.per_timestamp_speeds ? `[${{ts.per_timestamp_speeds.length}} timestamps]` : undefined }},
        fusion_decision: q.fusion_decision
      }}, null, 2);

      // Raw artifact download links
      const dl = [];
      if (bObj.trace_json_ref && bObj.trace_json_ref.path) {{
        dl.push(`<a href="${{esc(bObj.trace_json_ref.path)}}" download>trace_bundle.json</a>`);
      }}
      Object.entries(bObj.array_refs || {{}}).forEach(([ak, ref]) => {{
        if (ref && ref.path) dl.push(`<a href="${{esc(ref.path)}}" download>${{esc(ak)}}.npy</a>`);
      }});
      rawLinks.innerHTML = dl.length ? `<b>Tải mảng số lossless (.npy / .json):</b> ${{dl.join(" · ")}}` : "";

      const dc = bObj.derived_curves || {{}};
      const tArr = dc.t || [];
      const qTime = ts.clamped_t0 !== undefined && ts.clamped_t0 !== null ? Number(ts.clamped_t0) : (q.parse_trace && q.parse_trace.timestamps ? q.parse_trace.timestamps.t : null);
      let winBand = null;
      if (ts.fit && Number.isFinite(ts.fit.t0c) && Number.isFinite(ts.fit.half_used)) {{
        winBand = [ts.fit.t0c - ts.fit.half_used, ts.fit.t0c + ts.fit.half_used];
      }} else if (ts.interval) {{
        winBand = ts.interval;
      }}
      const partFrames = new Set(ts.participating_video_frames || []);
      const ov = q.track_overlays ? q.track_overlays[activeOverlayVar] : null;
      const partTimes = (dc.okc || []).map((slotIdx, ii) => {{
        const realF = ov && ov.fidx ? ov.fidx[slotIdx] : slotIdx;
        return partFrames.has(realF) ? tArr[ii] : null;
      }}).filter(v => v !== null);

      const seekToTime = (tSec) => {{
        setFrame(Math.round(tSec * fpsVal));
      }};

      renderSvgPlot(
        document.getElementById(`svgXY_${{qid}}`),
        tArr,
        [
          {{ y: dc.x_raw, color: "#38bdf8", mode: "none", dots: true, show: chkRaw.checked }},
          {{ y: dc.y_raw, color: "#34d399", mode: "none", dots: true, show: chkRaw.checked }},
          {{ y: dc.x_smooth, color: "#38bdf8", mode: "line", dots: false, show: chkSmooth.checked }},
          {{ y: dc.y_smooth, color: "#34d399", mode: "line", dots: false, show: chkSmooth.checked }}
        ],
        qTime, winBand, partTimes, "x(lam), y(lục) [px]", seekToTime
      );

      // Box size series from overlay frames
      let szT = [], bwS = [], bhS = [], rr0S = [];
      if (ov && ov.target_frames) {{
        ov.target_frames.forEach((fr, idx) => {{
          szT.push(ov.fidx[idx] / ov.fps);
          if (fr && fr.box) {{
            bwS.push(fr.box[2] - fr.box[0]);
            bhS.push(fr.box[3] - fr.box[1]);
            rr0S.push(fr.rr ? fr.rr[0] : null);
          }} else {{
            bwS.push(null); bhS.push(null); rr0S.push(null);
          }}
        }});
      }}
      renderSvgPlot(
        document.getElementById(`svgSz_${{qid}}`),
        szT,
        [
          {{ y: bwS, color: "#38bdf8", mode: "line", dots: true, show: true }},
          {{ y: bhS, color: "#facc15", mode: "line", dots: true, show: true }},
          {{ y: rr0S, color: "#c084fc", mode: "line", dash: "3,2", dots: false, show: true }}
        ],
        qTime, winBand, [], "box_w(lam), box_h(vàng), rr0(tím) [px]", seekToTime
      );

      renderSvgPlot(
        document.getElementById(`svgVA_${{qid}}`),
        tArr,
        [
          {{ y: ev.kind === "hvel" ? dc.vx_px_s : dc.speed_px_s, color: "#38bdf8", mode: "line", dots: true, show: true }},
          {{ y: dc.accel_px_s2, color: "#c084fc", mode: "line", dash: "4,2", dots: false, show: ev.kind === "accel" }}
        ],
        qTime, winBand, partTimes, ev.kind === "accel" ? "v(lam) [px/s], a(tím) [px/s²]" : "v [px/s]", seekToTime
      );
    }}

    function drawOverlay() {{
      const ov = q.track_overlays ? q.track_overlays[activeOverlayVar] : null;
      const W = (ov && ov.W) || va.raw_width || 854;
      const H = (ov && ov.H) || va.raw_height || 480;
      if (cvEl.width !== W || cvEl.height !== H) {{
        cvEl.width = W; cvEl.height = H;
      }}
      const ctx = cvEl.getContext("2d");
      ctx.clearRect(0, 0, W, H);
      finfoEl.textContent = `frame=${{curFrame}} · PTS=${{(curFrame / fpsVal).toFixed(3)}}s`;
      if (!ov) return;

      // Scene ruler lane lines (#2153)
      if (chkMask.checked && q.scene_ruler && q.scene_ruler.ok && q.scene_ruler.lines) {{
        const th = (Number(q.scene_ruler.theta || 0) * Math.PI) / 180.0;
        ctx.save();
        ctx.strokeStyle = "rgba(250, 204, 21, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        q.scene_ruler.lines.forEach(cVal => {{
          ctx.beginPath();
          if (Math.abs(Math.cos(th)) >= 0.5) {{
            const y0 = (cVal + 0 * Math.sin(th)) / Math.cos(th);
            const y1 = (cVal + (W - 1) * Math.sin(th)) / Math.cos(th);
            ctx.moveTo(0, y0); ctx.lineTo(W - 1, y1);
          }} else {{
            const x0 = (0 * Math.cos(th) - cVal) / Math.sin(th);
            const x1 = ((H - 1) * Math.cos(th) - cVal) / Math.sin(th);
            ctx.moveTo(x0, 0); ctx.lineTo(x1, H - 1);
          }}
          ctx.stroke();
        }});
        ctx.restore();
      }}

      // Raw trajectory trail
      if (chkRaw.checked && ov.target_frames) {{
        ctx.save();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        let started = false;
        ov.target_frames.forEach(fr => {{
          if (fr && fr.c) {{
            if (!started) {{ ctx.moveTo(fr.c[0], fr.c[1]); started = true; }}
            else ctx.lineTo(fr.c[0], fr.c[1]);
          }}
        }});
        ctx.stroke();
        ctx.restore();
      }}

      // Smoothed trajectory trail
      const bObj = currentBranchObj();
      if (chkSmooth.checked && bObj && bObj.derived_curves && bObj.derived_curves.x_smooth) {{
        const xs = bObj.derived_curves.x_smooth, ys = bObj.derived_curves.y_smooth;
        ctx.save();
        ctx.strokeStyle = "rgba(192, 132, 252, 0.9)";
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        let st = false;
        for (let i = 0; i < xs.length; i++) {{
          if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) {{
            if (!st) {{ ctx.moveTo(xs[i], ys[i]); st = true; }}
            else ctx.lineTo(xs[i], ys[i]);
          }}
        }}
        ctx.stroke();
        ctx.restore();
      }}

      if (!chkMask.checked) return;

      // Find nearest slot in ov.fidx
      let slot = 0, bestD = 1e9;
      (ov.fidx || []).forEach((f, idx) => {{
        const d = Math.abs(f - curFrame);
        if (d < bestD) {{ bestD = d; slot = idx; }}
      }});

      function drawEntity(frList, color, roleLbl) {{
        if (!frList || !frList[slot] || bestD > 1) return;
        const fr = frList[slot];
        ctx.save();
        if (fr.poly) {{
          ctx.fillStyle = color.replace("1)", "0.24)");
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          fr.poly.forEach(cnt => {{
            if (!cnt || cnt.length < 3) return;
            ctx.beginPath();
            ctx.moveTo(cnt[0][0], cnt[0][1]);
            for (let i = 1; i < cnt.length; i++) ctx.lineTo(cnt[i][0], cnt[i][1]);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          }});
        }}
        if (fr.box) {{
          const [x0, y0, x1, y1] = fr.box;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
          ctx.fillStyle = "rgba(15,23,42,0.82)";
          ctx.fillRect(x0, Math.max(0, y0 - 18), 155, 18);
          ctx.fillStyle = color;
          ctx.font = "11px monospace";
          ctx.fillText(roleLbl, x0 + 4, Math.max(12, y0 - 5));
        }}
        if (fr.c) {{
          ctx.fillStyle = "#ef4444";
          ctx.beginPath();
          ctx.arc(fr.c[0], fr.c[1], 4, 0, Math.PI * 2);
          ctx.fill();
        }}
        ctx.restore();
      }}

      drawEntity(ov.prior_frames, "rgba(250, 204, 21, 1)", `ref: ${{ov.prior_key || ""}}`);
      drawEntity(ov.target_frames, "rgba(56, 189, 248, 1)", `tgt: ${{ov.target_key || ""}}`);
    }}

    function setFrame(fIdx) {{
      const maxF = Number(sliderEl.max || 0);
      curFrame = Math.max(0, Math.min(maxF, fIdx));
      sliderEl.value = String(curFrame);
      if (vidEl) {{
        vidEl.pause();
        vidEl.currentTime = (curFrame + 0.05) / fpsVal;
      }}
      drawOverlay();
    }}

    sliderEl.oninput = () => setFrame(Number(sliderEl.value));
    document.getElementById(`prev_${{qid}}`).onclick = () => setFrame(curFrame - 1);
    document.getElementById(`next_${{qid}}`).onclick = () => setFrame(curFrame + 1);
    document.getElementById(`play_${{qid}}`).onclick = () => {{
      if (!vidEl) return;
      if (vidEl.paused) vidEl.play(); else vidEl.pause();
    }};
    document.getElementById(`jmp_${{qid}}`).onclick = () => {{
      const qT = q.parse_trace && q.parse_trace.timestamps && q.parse_trace.timestamps.t !== null
        ? Number(q.parse_trace.timestamps.t)
        : 0.5 * (Number(sliderEl.max || 0) / fpsVal);
      setFrame(Math.round(qT * fpsVal));
    }};
    if (vidEl) {{
      vidEl.addEventListener("timeupdate", () => {{
        curFrame = Math.round(vidEl.currentTime * fpsVal);
        sliderEl.value = String(curFrame);
        drawOverlay();
      }});
    }}
    [chkRaw, chkSmooth, chkMask].forEach(el => el.onchange = () => {{ updateFormulaAndCharts(); drawOverlay(); }});
    if (selVar) {{
      selVar.onchange = () => {{
        activeOverlayVar = selVar.value;
        updateFormulaAndCharts();
        drawOverlay();
      }};
    }}

    updateFormulaAndCharts();
    drawOverlay();
  }});

  // Search & Filter Bar
  let curFilter = "all";
  function applyFilters() {{
    const needle = document.getElementById("searchBox").value.trim().toLowerCase();
    Array.from(container.children).forEach((cardEl, idx) => {{
      const q = questions[idx];
      let show = true;
      if (needle && !cardEl.dataset.search.includes(needle)) show = false;
      if (curFilter === "error" && !hasMissingTrace(q)) show = false;
      if (curFilter === "cache_diff" && !hasCacheDiff(q)) show = false;
      if (curFilter === "better_caw" && !isBetterCaw(q)) show = false;
      if (curFilter === "worse_caw" && !isWorseCaw(q)) show = false;
      if (curFilter === "special" && ![142, 399, 401, 2234].includes(q.id)) show = false;
      cardEl.style.display = show ? "" : "none";
    }});
  }}
  document.getElementById("searchBox").oninput = applyFilters;
  document.querySelectorAll("button.flt").forEach(btn => {{
    btn.onclick = () => {{
      document.querySelectorAll("button.flt").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      curFilter = btn.dataset.filter;
      applyFilters();
    }};
  }});
}})();
</script>
</body>
</html>
"""
    output_html_path.write_text(html, encoding="utf-8")
    return output_html_path
