"""CPU checks for IIIXR 3D input isolation, real routing and stage resume."""
from __future__ import annotations

import csv
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from solutions.iiixr.kaggle.schema import REPO_ROOT, read_json, write_json_atomic
from solutions.iiixr.kaggle.three_d import data, report, trace, verify, worker, workflow


class ThreeDTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_pinned_replay_and_paired_clip(self):
        inputs, provenance, _ = data.resolve_inputs(None, self.root, False)
        self.assertFalse(provenance["official_compared"])
        self.assertEqual(len(inputs), 90)
        self.assertEqual(inputs[2607]["video_id"], inputs[2604]["video_id"])
        self.assertEqual(len(inputs[2607]["depth_info"].splitlines()), 4)
        results, _ = report.replay(self.root, sorted(inputs))
        self.assertEqual(sum(r["source"] == "meas" for r in results.values()), 81)
        self.assertEqual(sum(r["source"] == "vlm" for r in results.values()), 9)
        self.assertEqual(results[2607]["decision"]["meas_why"], "part_target")
        self.assertEqual(results[2607]["parsed_value"], 44.0)

    def test_legacy_validation_csv_keeps_question_ids_and_scores_replay(self):
        reference = data.csv_bytes(data.pinned_member(data.REFERENCE))
        csv_path = self.root / "validation_dataset.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", *data.FIELDS[1:], "ground_truth_posterior", *([""] * 17)])
            for row in reference.values():
                values = [str(row[field]).replace("\n", "\r\n") for field in data.FIELDS]
                writer.writerow([*values, row["ground_truth_posterior"], "", "#NAME?", *([""] * 15)])

        inputs, provenance, official = data.resolve_inputs(self.root, self.root, False)
        self.assertTrue(provenance["official_compared"])
        self.assertEqual(len(inputs), 90)
        self.assertEqual(len(inputs[2607]["depth_info"].splitlines()), 4)
        self.assertEqual(official, csv_path)
        questions = [inputs[q] for q in data.SANITY_IDS]
        clips = {row["video_id"]: {"status": "missing"} for row in questions}
        manifest = data.question_manifest(questions, clips)
        results, stages = report.replay(self.root, sorted(data.SANITY_IDS))
        workflow.seal(self.root, manifest, results, "replay", "sanity", stages)
        evaluation = report.evaluate(self.root, official)
        self.assertEqual(evaluation["all"]["scored_n"], 4)
        self.assertAlmostEqual(evaluation["all"]["mra"], 0.675)

    def test_clean_worker_runs_actual_rejection_and_arbitration(self):
        for name in ("geo/out/tracks_val", "preds", "models", "data/val/validation_videos",
                     "caw/sc/inputs", "caw/sc/out/val_ruler"):
            (self.root / name).mkdir(parents=True)
        worker.stage_sources(self.root)
        inputs, _, _ = data.resolve_inputs(None, self.root, False)
        questions = [inputs[2604], inputs[2607]]
        path = self.root / "data/val/validation_dataset.csv"
        data.write_worker_inputs(path, questions)
        self.assertTrue(all(not row["ground_truth_posterior"] for row in data.csv_bytes(path.read_bytes()).values()))
        (self.root / "data/val/validation_videos/captured_0027s.mp4").touch()
        (self.root / "geo/out/llm_parse_val.jsonl").touch()
        (self.root / "geo/out/vlmground_val_notext.jsonl").touch()
        write_json_atomic(self.root / "geo/out/tracks_val/captured_0027s.json", {
            "video_id": "captured_0027s", "H": 480, "W": 854, "N": 3, "stride": 1, "fidx": [0, 1, 2],
            "fps": 24, "cam": [[0, 0], [0, 0]], "queries": [],
        })
        for tag, values in (("qwen3vl8b_video", (9, 44)), ("h1_prior_copy", (7, 7))):
            with (self.root / f"preds/val_{tag}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Unnamed: 0", "parsed_value"])
                writer.writerows(zip([2604, 2607], values))
        stages = worker.Stages(self.root, {"scope": questions, "models": {}, "gpu": "", "input_key": "cpu_fixture"})
        fixture = (
            ("parse", "geo/parse.py", ("--no_llm",)),
            ("parse_v2", "geo/parse_v2.py", ()),
            ("measure", "geo/measure_v3.py", ("--tag", "_fixture", "--fix", "abcdef")),
            ("arbitrate", "geo/arbitrate_v2.py", ("--meas_tag", "_fixture", "--tag", "geo_fixture")),
            ("fuse", "analysis/v3/fuse_v3.py", ("val", "_fixture", "_fixture", "geo_cpu")),
            ("xground", "analysis/v3/xground.py", ("val",)),
            ("ruler_q", "analysis/v3/ruler_build.py", ("val", "q")),
        )
        for name, script, args in fixture:
            flags = worker.FLAGS
            if name == "xground":
                flags = {**flags, "XG_BASE": "geo_cpu_d_vw", "XG_RD": "geo_fixture", "XG_RV": "geo_fixture", "XG_VG": "_notext"}
            if name == "ruler_q":
                flags = {**flags, "RL_BASE_Q": "geo_v4xg_d_vw"}
            ok = stages.run(name, script, args, flags=flags)
            self.assertTrue(ok, (self.root / stages.records[-1]["stderr"]).read_text(encoding="utf-8"))
        output = data.csv_bytes((self.root / "preds/val_geo_v4xg_d_vw_raw.csv").read_bytes())
        self.assertEqual(output[2607]["meas_why"], "part_target")
        self.assertEqual(output[2607]["answer_source"], "vlm")
        self.assertEqual(float(output[2607]["parsed_value"]), 44)
        self.assertEqual(read_json(self.root / "analysis/v3/ruler_val.json"), {})

    def test_worker_failure_and_resume_checkpoint(self):
        root = self.root / "run"
        lane = root / "workers/lane"
        (lane / "models").mkdir(parents=True)
        write_json_atomic(root / "config.json", {"mode": "fresh"})
        script = lane / "probe.py"
        script.write_text("from pathlib import Path\nPath('result.txt').write_text('partial')\nraise RuntimeError('crash')",
                          encoding="utf-8")
        config = {"scope": [], "gpu": "", "models": {}, "input_key": "test"}
        self.assertFalse(worker.Stages(lane, config).run("probe", "probe.py", outputs=("result.txt",)))
        script.write_text("from pathlib import Path\nassert not Path('result.txt').exists()\nPath('result.txt').write_text('done')",
                          encoding="utf-8")
        self.assertTrue(worker.Stages(lane, config).run("probe", "probe.py", outputs=("result.txt",)))
        checkpoint = self.root / "checkpoint.zip"
        workflow.save_checkpoint(root, checkpoint, [lane])
        restored = self.root / "restored"
        workflow.extract_run(checkpoint, restored)
        resumed = worker.Stages(restored / "workers/lane", config)
        self.assertTrue(resumed.run("probe", "probe.py", outputs=("result.txt",)))
        self.assertEqual(resumed.records[0]["attempt"], 2)
        self.assertEqual((restored / "workers/lane/result.txt").read_text(), "done")
        checkpoint.write_bytes(checkpoint.read_bytes()[:40])
        with self.assertRaisesRegex(Exception, "zip file"):
            workflow.extract_run(checkpoint, self.root / "bad")

    def test_qwen_video_metadata_sanitization_and_media_dedup(self):
        import torch
        from transformers.feature_extraction_utils import BatchFeature
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

        trace_dir = self.root / "qwen_trace"
        trace_dir.mkdir(parents=True)
        video_proc = Qwen3VLVideoProcessor()

        class RealVideoProcWrapper:
            def __call__(self, *, text, videos=None, video_metadata=None, do_sample_frames=None, **kwargs):
                # Exercises actual transformers 5.x Qwen3VLVideoProcessor -> make_batched_metadata -> VideoMetadata
                vid_out = video_proc(
                    videos=videos,
                    video_metadata=video_metadata,
                    do_sample_frames=do_sample_frames,
                    return_tensors="pt",
                )
                tok_len = 8 + len(text[0]) % 5
                return BatchFeature(
                    data={
                        "input_ids": torch.arange(tok_len, dtype=torch.long).unsqueeze(0),
                        "attention_mask": torch.ones((1, tok_len), dtype=torch.long),
                        **vid_out,
                    },
                    tensor_type="pt",
                )

            def decode(self, ids, skip_special_tokens=True):
                return "2.5 m"

        class DummyModel:
            device = torch.device("cpu")

            def generate(self, **kwargs):
                inp = kwargs["input_ids"]
                extra = torch.tensor([[42, 43]], dtype=torch.long)
                return torch.cat([inp, extra], dim=1)

        qwen = trace.Qwen.__new__(trace.Qwen)
        qwen.config = {}
        qwen._video_blob_cache = {}
        qwen.processor = RealVideoProcWrapper()
        qwen.model = DummyModel()

        frames = np.zeros((4, 240, 320, 3), dtype=np.uint8)
        # Exact metadata dict shape produced by upstream/tools/run_vlm.py (includes 'do_sample_frames': False)
        vllm_meta = {
            "fps": 24.0,
            "duration": 2.0,
            "total_num_frames": 48,
            "frames_indices": [0, 12, 24, 36],
            "video_backend": "opencv",
            "do_sample_frames": False,
        }
        reqs = [
            {
                "prompt": "Question 1 about video",
                "multi_modal_data": {"video": [(frames, vllm_meta)]},
                "trace_context": {"question_id": 1001, "video_id": "captured_0027s"},
            },
            {
                "prompt": "Different question 2 about the very same video",
                "multi_modal_data": {"video": [(frames, vllm_meta)]},
                "trace_context": {"question_id": 1002, "video_id": "captured_0027s"},
            },
        ]
        sp = trace.SamplingParams(temperature=0.0, max_tokens=16)
        with patch.dict(os.environ, {"IIIXR_TRACE_DIR": str(trace_dir), "IIIXR_STAGE_EVENT": "evt-qwen-test"}):
            outs = qwen.generate(reqs, sp)

        self.assertEqual([o.outputs[0].text for o in outs], ["2.5 m", "2.5 m"])
        calls = [json.loads(ln) for ln in (trace_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(calls), 2)
        # Different prompts produce distinct token input artifacts, but share 1 deduplicated video pixel artifact
        self.assertNotEqual(calls[0]["artifact"], calls[1]["artifact"])
        self.assertIsNotNone(calls[0]["media_artifact"])
        self.assertEqual(calls[0]["media_artifact"], calls[1]["media_artifact"])
        npz_files = list((trace_dir / "arrays").glob("*.npz"))
        self.assertEqual(len(npz_files), 3)  # 2 small token blobs + 1 shared video pixel blob

    def test_spaced_captured_0041x_staging_fail_fast_and_full_3d_replay(self):
        caw_tar = REPO_ROOT / "quantiphy_full.tar"
        self.assertTrue(caw_tar.is_file(), "quantiphy_full.tar required for clip staging test")

        # Extract real captured_0041x.mp4 from quantiphy_full.tar to simulate HuggingFace spaced filename
        real_mp4 = self.root / "real_0041x.mp4"
        with tarfile.open(caw_tar) as archive:
            real_mp4.write_bytes(archive.extractfile("run/videos/captured_0041x.mp4").read())

        inputs, _, _ = data.resolve_inputs(None, self.root / "cache", False)
        q_0041x = [r for r in inputs.values() if r["video_id"] == "captured_0041x"]
        self.assertGreaterEqual(len(q_0041x), 1)

        # 1. Simulate Hugging Face returning 404 on 'validation_videos/captured_0041x.mp4'
        #    and succeeding on 'validation_videos/ captured_0041x.mp4'
        calls_seen = []

        def fake_hf_download(repo_id, filename, **kwargs):
            calls_seen.append(filename)
            if filename == "validation_videos/captured_0041x.mp4":
                raise RuntimeError("404 Client Error: Entry Not Found for validation_videos/captured_0041x.mp4")
            if filename == "validation_videos/ captured_0041x.mp4":
                return str(real_mp4)
            raise AssertionError(f"Unexpected HF filename: {filename}")

        with patch("huggingface_hub.hf_hub_download", side_effect=fake_hf_download):
            staged = data.stage_videos(q_0041x, self.root / "videos_hf", None, self.root / "cache", None, True)

        self.assertEqual(
            calls_seen,
            ["validation_videos/captured_0041x.mp4", "validation_videos/ captured_0041x.mp4"],
        )
        self.assertEqual(staged["captured_0041x"]["status"], "ok")
        self.assertEqual(staged["captured_0041x"]["path"], "videos/captured_0041x.mp4")
        self.assertEqual(
            staged["captured_0041x"]["sha256"],
            "f096debad0a8cabda4addbd216c22fc4d0a5c487417a4807149b6b13cf75b29c",
        )

        # 2. Verify fail-fast in fresh mode BEFORE launching GPU preflight/workers when clips are missing
        with patch.object(workflow, "predictions_from_fresh") as mock_fresh:
            with self.assertRaisesRegex(RuntimeError, "Validation clips unavailable"):
                workflow.main([
                    "--scope", "sanity",
                    "--mode", "fresh",
                    "--output", str(self.root / "fail_fast_out"),
                    "--cache", str(self.root / "cache"),
                ])
            mock_fresh.assert_not_called()

        # 3. Verify full 90-question 3D replay with all 12 real videos (including captured_0041x) + verify.check()
        replay_out = self.root / "full_3d_replay"
        workflow.main([
            "--scope", "full",
            "--mode", "replay",
            "--caw-tar", str(caw_tar),
            "--output", str(replay_out),
            "--cache", str(self.root / "cache"),
        ])
        chk = verify.check(replay_out / "run")
        self.assertEqual(chk["questions"], 90)
        ev = read_json(replay_out / "run/evaluation.json")
        self.assertEqual(ev["all"]["n"], 90)
        self.assertEqual(ev["all"]["covered"], 90)
        self.assertAlmostEqual(ev["all"]["mra"], 0.6488888888888888, places=6)
        qs_manifest = read_json(replay_out / "run/questions.json")
        self.assertTrue(all(q["clip"]["status"] == "ok" for q in qs_manifest))


if __name__ == "__main__":
    unittest.main()
