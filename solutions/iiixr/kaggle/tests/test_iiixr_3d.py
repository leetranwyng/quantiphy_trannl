"""CPU checks for IIIXR 3D input isolation, real routing and stage resume."""
from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from solutions.iiixr.kaggle.schema import read_json, write_json_atomic
from solutions.iiixr.kaggle.three_d import data, report, worker, workflow


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


if __name__ == "__main__":
    unittest.main()
