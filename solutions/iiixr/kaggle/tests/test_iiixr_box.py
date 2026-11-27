"""Comprehensive CPU tests for IIIXR g_box resolver, kinematics, anti-cache, replay, and verifier."""
from __future__ import annotations

import csv
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solutions.iiixr.kaggle.evaluate import compute_row_mra
from solutions.iiixr.kaggle.kinematics import (
    evaluate_xground_and_fuse,
    measure_video_instrumented,
    parse_questions_v1_v2,
    recompute_from_saved_trace_arrays,
)
from solutions.iiixr.kaggle.schema import (
    EXPECTED_VIDEO_BY_ID,
    REPO_ROOT,
    TARGET_6_VIDEOS,
    TARGET_18_IDS,
    UPSTREAM_DIR,
    FreshAntiCacheGuard,
    TraceStore,
    load_validation_inference_inputs,
    stage_assets_from_caw_tar,
    write_json_atomic,
)
from solutions.iiixr.kaggle.verify import verify_run


def _make_synthetic_track_frames(
    fidx: list[int],
    fps: float,
    fn_xy,
    box_w: float = 40.0,
    box_h: float = 100.0,
    drop_slots: set[int] | None = None,
) -> list[dict]:
    drop = drop_slots or set()
    frames = []
    area = int(round(box_w * box_h))
    rr_long = float(max(box_w, box_h))
    rr_short = float(min(box_w, box_h))
    for slot, fi in enumerate(fidx):
        if slot in drop:
            frames.append({"st": None, "lg": -10.0})
            continue
        t = fi / fps
        cx, cy = fn_xy(t)
        x0 = cx - box_w / 2.0
        y0 = cy - box_h / 2.0
        x1 = cx + box_w / 2.0
        y1 = cy + box_h / 2.0
        st = {
            "area": area,
            "box": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
            "c": [float(cx), float(cy)],
            "rr": [rr_long, rr_short, 0.0],
            "pca": [rr_long, rr_short],
            "border": 0,
            "bottom": float(y1),
            "top": float(y0),
        }
        frames.append({"st": st, "lg": 8.5})
    return frames


def _make_synthetic_vj(
    video_id: str,
    fidx: list[int],
    fps: float,
    W: int,
    H: int,
    queries_spec: list[tuple[str, str, list[str], list[str], list[dict]]],
) -> dict:
    objs = {}
    queries = []
    for oid, (key, phrase, where, roles, fr_list) in enumerate(queries_spec):
        objs[str(oid)] = fr_list
        first_valid = next((i for i, f in enumerate(fr_list) if f["st"] is not None), 0)
        b0 = fr_list[first_valid]["st"]["box"] if fr_list[first_valid]["st"] else [0, 0, 10, 10]
        queries.append({
            "key": key,
            "phrase": phrase,
            "where": where,
            "rel": None,
            "roles": roles,
            "obj": oid,
            "best_score": 0.85,
            "best_gd": 0.65,
            "best_owl": 0.45,
            "det_iou": [0.92, 0.94, 0.91],
            "anchor": [first_valid, b0],
        })
    return {
        "video_id": video_id,
        "N": max(fidx) + 1 if fidx else 0,
        "stride": (fidx[1] - fidx[0]) if len(fidx) >= 2 else 1,
        "fidx": fidx,
        "fps": fps,
        "H": H,
        "W": W,
        "det_frames": [0, len(fidx) // 2, len(fidx) - 1],
        "cam": [[0.0, 0.0, 1.0] for _ in fidx],
        "objs": objs,
        "version": "v2",
        "queries": queries,
        "queries_d": queries,
    }


class TestIIIXRBoxPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.caw_tar = REPO_ROOT / "quantiphy_full.tar"
        cls._tmp_obj = tempfile.TemporaryDirectory(prefix="iiixr-tests-")
        cls.tmp_dir = Path(cls._tmp_obj.name)
        if cls.caw_tar.is_file():
            staged = stage_assets_from_caw_tar(cls.caw_tar, cls.tmp_dir / "staged", TARGET_6_VIDEOS)
            cls.val_csv = staged["csv_path"]
            cls.vid_dir = staged["video_dir"]
        else:
            cls.val_csv = None
            cls.vid_dir = None

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_obj.cleanup()

    def test_01_resolver_18_ids_6_videos_shuffled_and_gt_isolated(self) -> None:
        self.assertIsNotNone(self.val_csv, "quantiphy_full.tar required for validation CSV test")
        raw_lines = self.val_csv.read_text(encoding="utf-8").splitlines()
        reader = list(csv.reader(io.StringIO("\n".join(raw_lines))))
        header, body = reader[0], reader[1:]
        # Reverse row order so positional index != original row order
        shuffled_csv = self.tmp_dir / "shuffled_val.csv"
        with shuffled_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for row in reversed(body):
                w.writerow(row)

        orig = load_validation_inference_inputs(self.val_csv, scope="target18")
        shuf = load_validation_inference_inputs(shuffled_csv, scope="target18")

        self.assertEqual([r["id"] for r in orig["selected_rows"]], list(TARGET_18_IDS))
        self.assertEqual([r["id"] for r in shuf["selected_rows"]], list(TARGET_18_IDS))
        self.assertEqual(sorted(orig["selected_videos"]), sorted(TARGET_6_VIDEOS))

        for r_o, r_s in zip(orig["selected_rows"], shuf["selected_rows"]):
            self.assertEqual(r_o["id"], r_s["id"])
            self.assertEqual(r_o["video_id"], EXPECTED_VIDEO_BY_ID[r_o["id"]])
            self.assertEqual(r_o["question"], r_s["question"])
            self.assertEqual(r_o["ground_truth_prior"], r_s["ground_truth_prior"])
            self.assertNotIn("ground_truth_posterior", r_o)
            self.assertNotIn("ground_truth_posterior", r_s)

        survey = load_validation_inference_inputs(self.val_csv, scope="survey158")
        self.assertEqual(len(survey["selected_rows"]), 158)
        in_scope_ids = [r["id"] for r in survey["selected_rows"] if r["in_scope_2d_motion"]]
        self.assertEqual(sorted(in_scope_ids), sorted(TARGET_18_IDS))

    def test_02_synthetic_static_constant_vel_and_constant_accel(self) -> None:
        fps = 24.0
        fidx = list(range(48))  # 2.0 seconds
        W, H = 1000, 600
        # Prior: height = 1.0 m, box_h = 100 px => mpp = 0.01 m/px
        rows = [
            {
                "id": 9001, "video_id": "syn_vid", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the speed of the static block at 1.0s in m/s?",
                "ground_truth_prior": "height of the wooden post = 1.0 m", "depth_info": "",
            },
            {
                "id": 9002, "video_id": "syn_vid", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the speed of the glider at 1.0s in m/s?",
                "ground_truth_prior": "height of the wooden post = 1.0 m", "depth_info": "",
            },
            {
                "id": 9003, "video_id": "syn_vid", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the average speed of the glider in m/s?",
                "ground_truth_prior": "height of the wooden post = 1.0 m", "depth_info": "",
            },
            {
                "id": 9004, "video_id": "syn_vid", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the average horizontal velocity of the glider in m/s?",
                "ground_truth_prior": "height of the wooden post = 1.0 m", "depth_info": "",
            },
            {
                "id": 9005, "video_id": "syn_vid", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the acceleration of the rocket cart at 1.0s in m/s^2?",
                "ground_truth_prior": "height of the wooden post = 1.0 m", "depth_info": "",
            },
        ]
        parsed = parse_questions_v1_v2(rows)
        v2_recs = [parsed["v2_records"][r["id"]] for r in rows]

        fr_post = _make_synthetic_track_frames(fidx, fps, lambda t: (100.0, 300.0), box_w=20.0, box_h=100.0)
        fr_static = _make_synthetic_track_frames(fidx, fps, lambda t: (250.0, 300.0), box_w=40.0, box_h=40.0)
        # Constant velocity: vx=120 px/s, vy=50 px/s => speed=130 px/s => 1.30 m/s, hvel=1.20 m/s
        fr_glider = _make_synthetic_track_frames(fidx, fps, lambda t: (150.0 + 120.0 * t, 200.0 + 50.0 * t), box_w=36.0, box_h=36.0)
        # Constant accel: ax=80 px/s^2, ay=60 px/s^2 => ||a||=100 px/s^2 => 1.00 m/s^2
        fr_cart = _make_synthetic_track_frames(
            fidx, fps, lambda t: (120.0 + 20.0 * t + 0.5 * 80.0 * t * t, 150.0 + 10.0 * t + 0.5 * 60.0 * t * t), box_w=36.0, box_h=36.0
        )

        vj = _make_synthetic_vj(
            "syn_vid", fidx, fps, W, H,
            [
                ("wooden post|", "wooden post", [], ["prior"], fr_post),
                ("static block|", "static block", [], ["target"], fr_static),
                ("glider|", "glider", [], ["target"], fr_glider),
                ("rocket cart|", "rocket cart", [], ["target"], fr_cart),
            ],
        )
        out = measure_video_instrumented(v2_recs, vj, t_owl=0.1)
        res = out["results_by_id"]

        # Static object rejected for dynamic speed query
        self.assertFalse(res[9001]["ok"])
        self.assertIn("static_track", res[9001]["why"])
        # Constant velocity & horizontal velocity
        self.assertAlmostEqual(res[9002]["geo_value"], 1.30, places=5)
        self.assertAlmostEqual(res[9003]["geo_value"], 1.30, places=5)
        self.assertAlmostEqual(res[9004]["geo_value"], 1.20, places=5)
        # Constant acceleration
        self.assertAlmostEqual(res[9005]["geo_value"], 1.00, places=5)

    def test_03_resize_invariance_and_missing_frames_true_dt(self) -> None:
        fps = 30.0
        # Non-unit stride=2 with missing frames in the middle
        fidx = list(range(0, 90, 2))  # 45 sampled frames across 3.0s, true dt = 2/30s
        drop_slots = {5, 6, 14, 15, 28}
        rows = [
            {
                "id": 9101, "video_id": "syn_scale", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the speed of the ball at 1.2s in m/s?",
                "ground_truth_prior": "height of the pole = 2.0 m", "depth_info": "",
            },
            {
                "id": 9102, "video_id": "syn_scale", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": fps,
                "question": "What is the acceleration of the ball at 1.2s in m/s^2?",
                "ground_truth_prior": "height of the pole = 2.0 m", "depth_info": "",
            },
        ]
        parsed = parse_questions_v1_v2(rows)
        v2_recs = [parsed["v2_records"][r["id"]] for r in rows]

        vals_by_scale = {}
        mpps_by_scale = {}
        for scale in (1.0, 2.0, 0.5):
            # At scale=1: pole height=100px => mpp=0.02 m/px; x(t)=80+90t+25t^2 (ax=50 px/s^2 = 1.0 m/s^2, vx(1.2)=150 px/s = 3.0 m/s)
            fr_pole = _make_synthetic_track_frames(
                fidx, fps, lambda t, s=scale: (80.0 * s, 250.0 * s), box_w=20.0 * scale, box_h=100.0 * scale
            )
            fr_ball = _make_synthetic_track_frames(
                fidx, fps,
                lambda t, s=scale: ((80.0 + 90.0 * t + 25.0 * t * t) * s, (120.0 * s)),
                box_w=24.0 * scale, box_h=24.0 * scale,
                drop_slots=drop_slots,
            )
            vj = _make_synthetic_vj(
                "syn_scale", fidx, fps, int(1200 * scale), int(700 * scale),
                [
                    ("pole|", "pole", [], ["prior"], fr_pole),
                    ("ball|", "ball", [], ["target"], fr_ball),
                ],
            )
            out = measure_video_instrumented(v2_recs, vj, t_owl=0.1)
            vals_by_scale[scale] = (out["results_by_id"][9101]["geo_value"], out["results_by_id"][9102]["geo_value"])
            mpps_by_scale[scale] = out["results_by_id"][9101]["mpp"]

        # Verify exact analytical velocity (3.0 m/s) & acceleration (1.0 m/s^2) despite missing frames & stride=2
        self.assertAlmostEqual(vals_by_scale[1.0][0], 3.0, places=4)
        self.assertAlmostEqual(vals_by_scale[1.0][1], 1.0, places=4)
        # Verify simultaneous target+prior resize invariance
        for sc in (2.0, 0.5):
            self.assertAlmostEqual(vals_by_scale[sc][0], vals_by_scale[1.0][0], places=3)
            self.assertAlmostEqual(vals_by_scale[sc][1], vals_by_scale[1.0][1], places=3)
            self.assertAlmostEqual(mpps_by_scale[sc] * sc, mpps_by_scale[1.0], places=5)

    def test_04_anti_cache_guard_and_fresh_without_mot_val_files(self) -> None:
        mot_401 = UPSTREAM_DIR / "team/member1/trackB/package/z1s2c/2_c2_g3a/root/analysis/v3/ag/r10/MOTION/cache/mot_val_401.json"
        with FreshAntiCacheGuard(enabled=True):
            with self.assertRaises(PermissionError):
                open(mot_401, "r", encoding="utf-8")

        # Create stripped upstream geo copy WITHOUT any cache/mot_val_*.json or rows.pkl
        clean_up = self.tmp_dir / "upstream_no_cache"
        shutil.copytree(UPSTREAM_DIR / "geo", clean_up / "geo", dirs_exist_ok=True)
        shutil.copytree(UPSTREAM_DIR / "tools", clean_up / "tools", dirs_exist_ok=True)
        self.assertFalse((clean_up / "team").exists())

        with FreshAntiCacheGuard(enabled=True):
            rows = [{
                "id": 399, "video_id": "simulation_0163", "video_source": "simulation", "video_type": "S2MX",
                "inference_type": "SD", "fps": 24.0,
                "question": "What is the acceleration of the ball at 1.0s in m/s^2?",
                "ground_truth_prior": "diameter of the ball = 0.21 m", "depth_info": "",
            }]
            parsed = parse_questions_v1_v2(rows, upstream_dir=clean_up)
            fidx = list(range(40))
            fr_ball = _make_synthetic_track_frames(
                fidx, 24.0, lambda t: (100.0 + 60.0 * t + 10.0 * t * t, 240.0), box_w=36.0, box_h=36.0
            )
            vj = _make_synthetic_vj("simulation_0163", fidx, 24.0, 854, 480, [("ball|", "ball", [], ["target", "prior"], fr_ball)])
            out = measure_video_instrumented([parsed["v2_records"][399]], vj, t_owl=0.1, upstream_dir=clean_up)
            fused = evaluate_xground_and_fuse(parsed["v2_records"][399], out["results_by_id"][399], None, vj, [], upstream_dir=clean_up)
            self.assertTrue(math.isfinite(fused["final_value"]) and fused["final_value"] > 0)

    def test_05_cpu_trace_replay_and_offline_verifier(self) -> None:
        run_dir = self.tmp_dir / "cpu_trace_run"
        store = TraceStore(run_dir, "test-cpu-replay", enabled=True)
        rows = [{
            "id": 401, "video_id": "simulation_0163", "video_source": "simulation", "video_type": "S2MX",
            "inference_type": "SD", "fps": 24.0,
            "question": "What is the acceleration of the ball at 1.5s in m/s^2?",
            "ground_truth_prior": "diameter of the ball = 0.21 m", "depth_info": "",
        }]
        parsed = parse_questions_v1_v2(rows)
        fidx = list(range(51))
        fr_ball = _make_synthetic_track_frames(
            fidx, 24.0, lambda t: (80.0 + 110.0 * t + 14.5 * t * t, 220.0 + 5.0 * t), box_w=36.0, box_h=36.0
        )
        vj = _make_synthetic_vj("simulation_0163", fidx, 24.0, 854, 480, [("ball|", "ball", [], ["target", "prior"], fr_ball)])
        out = measure_video_instrumented([parsed["v2_records"][401]], vj, t_owl=0.0980793)

        res_401 = out["results_by_id"][401]
        tr_401 = out["traces_by_id"][401]
        arr_401 = out["arrays_by_id"][401]

        # Direct CPU recomputation check
        recomputed = recompute_from_saved_trace_arrays(tr_401["replay_spec"], arr_401)
        self.assertAlmostEqual(float(res_401["geo_value"]), float(recomputed["geo_value"]), places=11)

        # Save into run_dir structure & verify via standalone verify_run()
        arr_refs = {ak: store.put_npy(av, kind="track_array", name=ak) for ak, av in arr_401.items()}
        tr_ref = store.put_json(tr_401, kind="kinematic_trace")
        store.log_event(
            stage="measure_v3",
            call_kind="tool_call",
            status="ok",
            video_id="simulation_0163",
            sample_id=401,
            output_refs={"trace": tr_ref},
        )
        sdir = run_dir / "samples" / "q_401"
        sdir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            sdir / "trace_bundle.json",
            {
                "id": 401,
                "video_id": "simulation_0163",
                "selected_branch": "v2",
                "branches": {
                    "v2": {
                        "result": res_401,
                        "replay_spec": tr_401["replay_spec"],
                        "array_refs": arr_refs,
                        "trace_json_ref": tr_ref,
                    }
                },
            },
        )
        from solutions.iiixr.kaggle.workflow import _seal_manifest, _snapshot_minimal_source

        _snapshot_minimal_source(run_dir)
        _seal_manifest(run_dir, {"run_id": "test-cpu-replay", "mode": "sanity", "status": "PASS"})
        v_report = verify_run(run_dir, require_complete=True)
        self.assertTrue(v_report["verified"])
        self.assertEqual(v_report["cpu_recomputed_branch_traces"], 1)

    def test_06_end_to_end_replay_mra_and_viewer(self) -> None:
        self.assertTrue(self.caw_tar.is_file(), "quantiphy_full.tar required for end-to-end replay check")
        out_root = self.tmp_dir / "e2e_replay_out"
        cmd = [
            sys.executable,
            "-m",
            "solutions.iiixr.kaggle.workflow",
            "replay",
            "--caw-tar",
            str(self.caw_tar),
            "--cache",
            str(self.tmp_dir / "cache"),
            "--output",
            str(out_root),
            "--run-id",
            "replay-check",
            "--no-tar",
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        summary = json.loads((out_root / "iiixr_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertTrue(summary["verification"]["verified"])

        mra = summary["evaluation"]["mean_per_question_mra"]
        self.assertAlmostEqual(mra["cached_box"], 0.8388888888888889, places=6)
        self.assertAlmostEqual(mra["caw"], 0.4277777777777778, places=6)
        self.assertIsNone(mra["fresh_box"])

        pq_path = Path(summary["per_question_csv"])
        with pq_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 18)
        by_id = {int(r["id"]): r for r in rows}
        self.assertAlmostEqual(float(by_id[399]["cached_box"]), 0.38209679865232415, places=12)
        self.assertAlmostEqual(float(by_id[399]["ground_truth"]), 0.359, places=6)
        self.assertAlmostEqual(float(by_id[401]["cached_box"]), 0.3420306158490744, places=12)
        self.assertAlmostEqual(float(by_id[401]["ground_truth"]), 0.339, places=6)
        self.assertAlmostEqual(float(by_id[401]["cached_point"]), 2.2860761843131794, places=12)
        self.assertEqual(by_id[142]["tv_cache"], "v3nt")
        self.assertEqual(by_id[2234]["kind"], "hvel")

        viewer_path = Path(summary["viewer_html"])
        self.assertTrue(viewer_path.is_file())
        html_txt = viewer_path.read_text(encoding="utf-8")
        self.assertIn("mot_val_401.json", html_txt)
        self.assertIn("DERIVED PLOT", html_txt)
        self.assertAlmostEqual(compute_row_mra(0.3420306158490744, 0.339), 1.0, places=6)

    def test_07_safe_track_modules_and_track_worker_orchestration(self) -> None:
        from unittest.mock import patch
        from solutions.iiixr.kaggle import gpu_stages

        # Force fresh AST compilation of upstream track & track_v2 pure helpers
        sys.modules.pop("track", None)
        sys.modules.pop("track_v2", None)
        gpu_stages._install_safe_track_modules(UPSTREAM_DIR)

        import track as trk_v1_mod
        import track_v2 as trk_v2_mod

        self.assertNotIn("torch", trk_v1_mod.__dict__)
        self.assertNotIn("gdino", trk_v1_mod.__dict__)
        self.assertNotIn("sam2_track", trk_v1_mod.__dict__)
        self.assertNotIn("gdino", trk_v2_mod.__dict__)

        self.assertIsNotNone(self.val_csv)
        self.assertIsNotNone(self.vid_dir)
        inputs = load_validation_inference_inputs(self.val_csv, scope="sanity4")
        rows_0163 = inputs["video_context_rows"]["simulation_0163"]
        parsed = parse_questions_v1_v2(rows_0163)
        v1_recs = [parsed["v1_records"][r["id"]] for r in rows_0163]
        v2_recs = [parsed["v2_records"][r["id"]] for r in rows_0163]

        qs_v1 = trk_v1_mod.queries_for(v1_recs)
        qs_v2 = trk_v2_mod.queries_for(v2_recs)
        self.assertGreaterEqual(len(qs_v1), 1)
        self.assertGreaterEqual(len(qs_v2), 1)

        def fake_gdino(frames, det_indices, phrases, model_dir, thr, amp_dtype, device="cuda"):
            dets = {
                ph: [[([100.0 + k * 4.0, 200.0, 140.0 + k * 4.0, 240.0], 0.24)] for k in range(len(det_indices))]
                for ph in phrases
            }
            return dets, {"threshold": thr, "candidates_by_phrase": {}}

        def fake_owlv2(frames, det_indices, phrases, model_dir, amp_dtype, thr=0.05, topk=25, device="cuda"):
            out = [
                {ph: [([100.0 + k * 4.0, 200.0, 140.0 + k * 4.0, 240.0], 0.42)] for ph in phrases}
                for k in range(len(det_indices))
            ]
            return out, {"threshold": thr, "per_frame_candidates": []}

        def fake_sam2(frames, anchors, model_dir, dtype_profile, keep_all_polys=True, device="cuda"):
            res = {}
            for oid, _fi, _box in anchors:
                per_f = {}
                for j in range(len(frames)):
                    cx = 120.0 + 3.5 * j + 0.04 * j * j
                    cy = 220.0
                    st = {
                        "area": 1296,
                        "box": [int(cx - 18), int(cy - 18), int(cx + 18), int(cy + 18)],
                        "c": [float(cx), float(cy)],
                        "rr": [36.0, 36.0, 0.0],
                        "pca": [36.0, 36.0],
                        "border": 0,
                        "bottom": float(cy + 18),
                        "top": float(cy - 18),
                    }
                    poly = [[[int(cx - 18), int(cy - 18)], [int(cx + 18), int(cy - 18)], [int(cx + 18), int(cy + 18)], [int(cx - 18), int(cy + 18)]]]
                    per_f[j] = (st, 7.5, poly, {"white": 0.8})
                res[int(oid)] = per_f
            return res, {"sam2_dtype": "mock_fp16"}

        vg_mock = [
            {
                "video_id": "simulation_0163",
                "key": qs_v2[0]["key"],
                "phrase": qs_v2[0]["phrase"],
                "where": qs_v2[0]["where"],
                "roles": qs_v2[0]["roles"],
                "expr": qs_v2[0]["phrase"],
                "multi": False,
                "j": 0,
                "boxes_rel1000": [[102.0, 202.0, 138.0, 238.0]],
            }
        ]
        work_root = self.tmp_dir / "mock_worker"
        work_root.mkdir(parents=True, exist_ok=True)
        work_json = work_root / "work.json"
        receipt_json = work_root / "receipt.json"
        write_json_atomic(
            work_json,
            {
                "run_dir": str(work_root / "run"),
                "run_id": "mock-track-worker",
                "worker_name": "track-worker-0-mock",
                "dtype_profile": "t4_fp16",
                "seed": 0,
                "trace_enabled": True,
                "use_video_cache": False,
                "track_cache_dir": str(work_root / "cache"),
                "models": {"grounding_dino": "mock_gd", "owlv2": "mock_owl", "sam2": "mock_sam"},
                "model_revisions": {"grounding_dino": "r1", "owlv2": "r2", "sam2": "r3"},
                "videos": [
                    {
                        "video_id": "simulation_0163",
                        "video_path": str(self.vid_dir / "simulation_0163.mp4"),
                        "fps": 24.0,
                        "parse_v1_records": v1_recs,
                        "parse_v2_records": v2_recs,
                        "vlm_ground_records": vg_mock,
                        "attempt": 1,
                    }
                ],
                "receipt_path": str(receipt_json),
            },
        )
        with (
            patch.object(gpu_stages, "_run_gdino_instrumented", side_effect=fake_gdino),
            patch.object(gpu_stages, "_run_owlv2_instrumented", side_effect=fake_owlv2),
            patch.object(gpu_stages, "_run_sam2_instrumented", side_effect=fake_sam2),
        ):
            receipt = gpu_stages.run_track_stage_worker(work_json)

        self.assertEqual(receipt["status"], "ok")
        self.assertEqual(receipt["errors"], [])
        self.assertIn("simulation_0163", receipt["video_results"])
        vb = receipt["video_results"]["simulation_0163"]
        self.assertIsNotNone(vb["track_v1"])
        self.assertIsNotNone(vb["track_v2"])  # best_score=0.24 < 0.3 triggers A2_det -> track_v2 runs!
        self.assertIsNotNone(vb["track_v3nt"])
        self.assertGreaterEqual(len(vb["overlay_refs"]), 1)

    def test_08_sanity_workflow_graceful_error_summary_on_worker_failure(self) -> None:
        from unittest.mock import patch
        from solutions.iiixr.kaggle import workflow

        self.assertIsNotNone(self.val_csv)
        self.assertIsNotNone(self.vid_dir)
        # Stage a mini dataset dir with validation_dataset.csv + 3 sanity videos
        ds_dir = self.tmp_dir / "mock_ds"
        ds_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.val_csv, ds_dir / "validation_dataset.csv")
        for vid in ("simulation_0163", "simulation_0089", "internet_0019"):
            shutil.copy2(self.vid_dir / f"{vid}.mp4", ds_dir / f"{vid}.mp4")
        models_dir = self.tmp_dir / "mock_models"
        for sub in ("grounding-dino-base", "owlv2-base-patch16-ensemble", "sam2.1-hiera-large", "Qwen3-VL-8B-Instruct"):
            (models_dir / sub).mkdir(parents=True, exist_ok=True)

        out_dir = self.tmp_dir / "sanity_err_out"
        parser = workflow.build_arg_parser()
        args = parser.parse_args([
            "sanity",
            "--allow-any-gpu",
            "--dataset-dir",
            str(ds_dir),
            "--models-dir",
            str(models_dir),
            "--caw-tar",
            str(self.caw_tar),
            "--cache",
            str(self.tmp_dir / "sanity_err_cache"),
            "--output",
            str(out_dir),
            "--run-id",
            "sanity-err-check",
            "--no-tar",
        ])

        def failing_track_subproc(**_kwargs):
            return {
                "video_results": {},
                "errors": [{"worker": "track-worker-0-trace_off", "stage": "track_worker", "video_id": "simulation_0163", "error": "Simulated worker crash"}],
            }

        with patch.object(workflow, "_run_gpu_track_subprocesses", side_effect=failing_track_subproc):
            summary = workflow.execute_workflow(args)

        self.assertEqual(summary["status"], "ERROR")
        self.assertTrue((out_dir / "iiixr_summary.json").is_file())
        self.assertGreaterEqual(len(summary["stage_errors"]), 1)
        self.assertIn("Simulated worker crash", summary["stage_errors"][0]["error"])
        self.assertTrue(summary["verification"]["verified"])

    def test_09_sanity_workflow_mocked_happy_path_full_trace(self) -> None:
        from unittest.mock import patch
        from solutions.iiixr.kaggle import workflow

        self.assertIsNotNone(self.val_csv)
        self.assertIsNotNone(self.vid_dir)
        ds_dir = self.tmp_dir / "mock_ds_ok"
        ds_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.val_csv, ds_dir / "validation_dataset.csv")
        for vid in ("simulation_0163", "simulation_0089", "internet_0019"):
            shutil.copy2(self.vid_dir / f"{vid}.mp4", ds_dir / f"{vid}.mp4")
        models_dir = self.tmp_dir / "mock_models_ok"
        for sub in ("grounding-dino-base", "owlv2-base-patch16-ensemble", "sam2.1-hiera-large", "Qwen3-VL-8B-Instruct"):
            (models_dir / sub).mkdir(parents=True, exist_ok=True)

        out_dir = self.tmp_dir / "sanity_ok_out"
        parser = workflow.build_arg_parser()
        args = parser.parse_args([
            "sanity",
            "--allow-any-gpu",
            "--dataset-dir",
            str(ds_dir),
            "--models-dir",
            str(models_dir),
            "--caw-tar",
            str(self.caw_tar),
            "--cache",
            str(self.tmp_dir / "sanity_ok_cache"),
            "--output",
            str(out_dir),
            "--run-id",
            "sanity-ok-check",
            "--no-tar",
        ])

        def fake_vlm_subproc(**_kwargs):
            return {"status": "ok", "stage": "vlm_stage", "llm_parse_outputs": {}, "vlm_ground_by_video": {}, "errors": []}

        def fake_track_subproc(**kwargs):
            vid_res = {}
            for item in kwargs["video_items"]:
                vid = item["video_id"]
                fps = float(item["fps"])
                fidx = list(range(48))
                q_specs = []
                seen_k = set()
                for r in item["parse_v2_records"]:
                    for role_name, ent in (("target", r["target"]), ("prior", r["prior_obj"])):
                        if ent and ent.get("phrase"):
                            k = ent["phrase"] + "|" + ",".join(ent.get("where") or [])
                            if k not in seen_k:
                                seen_k.add(k)
                                fr = _make_synthetic_track_frames(
                                    fidx, fps, lambda t: (100.0 + 80.0 * t, 60.0 + 140.0 * t * t), box_w=36.0, box_h=36.0
                                )
                                q_specs.append((k, ent["phrase"], list(ent.get("where") or []), [role_name], fr))
                vj = _make_synthetic_vj(vid, fidx, fps, 854, 480, q_specs)
                vid_res[vid] = {
                    "video_id": vid,
                    "track_v1": vj,
                    "track_v2": vj,
                    "track_v3nt": vj,
                    "scene_ruler": None,
                    "overlay_refs": {},
                }
            return {"video_results": vid_res, "errors": []}

        with (
            patch.object(workflow, "_run_gpu_vlm_subprocess", side_effect=fake_vlm_subproc),
            patch.object(workflow, "_run_gpu_track_subprocesses", side_effect=fake_track_subproc),
        ):
            summary = workflow.execute_workflow(args)

        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["sanity_parity_report"]["identical_trace_off_on"])
        self.assertTrue(summary["verification"]["verified"])
        self.assertEqual(summary["evaluation"]["fresh_coverage"]["generated_ok"], 4)
        self.assertGreaterEqual(summary["verification"]["cpu_recomputed_branch_traces"], 4)


if __name__ == "__main__":
    unittest.main()
