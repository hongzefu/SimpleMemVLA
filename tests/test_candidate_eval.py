"""固定候选的反例测试；不加载模型或启动仿真 GPU。"""
import copy
import locale
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robomme_sim import candidate_plan as plan
from robomme_sim.robomme_env import candidate_demo_info, _setup_robomme_path
from robomme_sim.video_writer import StreamingVideoRecorder, verify_video


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = plan.REPO / "third_party/robomme_benchmark"
        cls.header, cls.rows = plan.read_candidates(cls.root)

    def test_seven_hundred_identity_and_balanced_lanes(self):
        self.assertEqual(len({plan.key(r) for r in self.rows}), 700)
        self.assertLess(len({(r["task"], r["episode"]) for r in self.rows}), 700)
        lanes = plan.partition(self.rows)
        left = {k for c in lanes["hold01"] for k in c}
        right = {k for c in lanes["hold02"] for k in c}
        self.assertFalse(left & right)
        self.assertEqual(left | right, {plan.key(r) for r in self.rows})
        for lane in ("hold01", "hold02"):
            self.assertEqual(len(lanes[lane]), 25)
            self.assertTrue(all(len(c) == 14 and len({k[:2] for k in c}) == 14 for c in lanes[lane]))
        self.assertEqual(lanes["local-smoke"][0][0][:2], ("BinFill", "hard"))

    def test_chinese_error_is_durable_after_native_locale_change(self):
        previous = locale.setlocale(locale.LC_CTYPE)
        try:
            locale.setlocale(locale.LC_CTYPE, "C")
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "error.json"
                value = {"status": "error", "error": "环境重置失败，保留原 seed"}
                plan.atomic_json(path, value)
                self.assertEqual(plan.read_json(path), value)
        finally:
            locale.setlocale(locale.LC_CTYPE, previous)

    def test_controller_revision_cannot_change_policy_or_inputs(self):
        old = dict(source_commit="old", source_files={"robomme_sim/candidate_launcher.py": "a",
                   "robomme_sim/candidate_eval.py": "b"}, config={"max_steps": 2000}, assets={"model": "same"})
        new = copy.deepcopy(old)
        new["source_commit"] = "new"
        new["source_files"]["robomme_sim/candidate_launcher.py"] = "fixed"
        new["previous_manifests"] = [old]
        self.assertEqual(plan.manifest_tokens(new), {plan.digest(old), plan.digest(new)})
        for section, field, value in (("source_files", "robomme_sim/candidate_eval.py", "changed"),
                                      ("config", "max_steps", 1300), ("assets", "model", "other")):
            bad = copy.deepcopy(new)
            bad[section][field] = value
            with self.assertRaises(ValueError):
                plan.manifest_tokens(bad)

    def test_duplicate_extra_or_wrong_manifest_rejected(self):
        manifest = {"benchmark_root": str(self.root)}
        row_key = plan.key(self.rows[0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plan.json"
            for keys, token in (([row_key, row_key], plan.digest(manifest)),
                                ([("BinFill", "xhard", -1)], plan.digest(manifest)),
                                ([row_key], "wrong")):
                plan.atomic_json(path, {"keys": keys, "manifest_sha256": token})
                with self.assertRaises(ValueError):
                    plan.load_plan(path, manifest)

    def test_same_episode_different_difficulty_does_not_collide(self):
        a = dict(task="BinFill", difficulty="easy", episode=153)
        b = dict(a, difficulty="hard")
        self.assertNotEqual(plan.result_dir("out", a), plan.result_dir("out", b))

    def test_binfill_initial_observation_is_not_demo(self):
        env = SimpleNamespace(unwrapped=SimpleNamespace(difficulty="hard", task_list=[{"demonstration": False}]))
        self.assertEqual(candidate_demo_info("BinFill", env, {"front_rgb_list": [0]}, {"difficulty": "hard"}),
                         {"demo_frames": 0, "demo_tasks": 0})
        with self.assertRaises(RuntimeError):
            candidate_demo_info("BinFill", env, {"front_rgb_list": [0, 1]}, {"difficulty": "hard"})
        env.unwrapped.task_list[0]["demonstration"] = True
        with self.assertRaises(RuntimeError):
            candidate_demo_info("BinFill", env, {"front_rgb_list": [0]}, {"difficulty": "hard"})
        self.assertEqual(candidate_demo_info("RouteStick", env, {"front_rgb_list": [0, 1]}, {"difficulty": "hard"})["demo_frames"], 1)
        with self.assertRaises(RuntimeError):
            candidate_demo_info("RouteStick", env, {"front_rgb_list": [0]}, {"difficulty": "hard"})

    def test_import_cannot_silently_switch_sources(self):
        fake = SimpleNamespace(__file__="/other/robomme/__init__.py")
        with patch.dict("sys.modules", {"robomme": fake}):
            with self.assertRaises(RuntimeError):
                _setup_robomme_path(str(self.root))

    def test_attempt_identity_and_no_retry_of_normal_outcome(self):
        row, manifest = self.rows[0], {"test": True}
        with tempfile.TemporaryDirectory() as d:
            target = plan.result_dir(d, row)
            r = dict(key=plan.key(row), seed=row["seed"], spec_sha256=row["spec_sha256"],
                     manifest_sha256=plan.digest(manifest), attempt=1, status="error")
            plan.atomic_json(target / "attempt-1.json", r)
            self.assertEqual(len(plan.attempts(d, row, manifest)), 1)
            bad = dict(r, seed=r["seed"] + 1)
            plan.atomic_json(target / "attempt-1.json", bad)
            with self.assertRaises(ValueError):
                plan.attempts(d, row, manifest)
            plan.atomic_json(target / "attempt-1.json", dict(r, status="fail"))
            plan.atomic_json(target / "attempt-2.json", dict(r, attempt=2))
            with self.assertRaises(ValueError):
                plan.attempts(d, row, manifest)

    def test_error_stays_in_full_denominator(self):
        manifest = {"benchmark_root": str(self.root)}
        with tempfile.TemporaryDirectory() as d:
            row = self.rows[0]
            r = dict(key=plan.key(row), seed=row["seed"], spec_sha256=row["spec_sha256"],
                     manifest_sha256=plan.digest(manifest), attempt=1, status="error")
            plan.atomic_json(plan.result_dir(Path(d) / "hold01", row) / "attempt-1.json", r)
            with patch.object(plan, "verify_manifest", return_value=manifest):
                result = plan.summarize(d, decode=False)
            self.assertFalse(result["complete"])
            self.assertEqual(result["planned"], 700)
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(len(result["missing"]), 699)

    def test_streaming_video_decode_and_damage(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.mp4"
            writer = StreamingVideoRecorder(path)
            for i in range(7):
                frame = np.full((32, 32, 3), i * 30, np.uint8)
                writer.add({"front": frame, "wrist": frame})
            writer.close()
            self.assertEqual(verify_video(path, 7), 7)
            self.assertFalse(hasattr(writer, "frames"))
            with self.assertRaises(ValueError):
                verify_video(path, 8)
            path.write_bytes(path.read_bytes()[:40])
            with self.assertRaises(Exception):
                verify_video(path, 7)

    def test_rollout_errors_timeout_and_resume(self):
        from robomme_sim import candidate_eval as evaluator
        frame = {k: np.full((32, 32, 3), 127, dtype=np.uint8) for k in ("front", "wrist")}
        state = np.zeros(8, np.float32)
        row = self.rows[0]
        buffer = SimpleNamespace(image_keys=["front", "wrist"], native_fps=20,
                                 reset=lambda: None, observe=lambda _: None, _prepare_inputs=lambda _: {})
        model = SimpleNamespace(device="cpu", generate_batch=lambda *_: [(np.zeros((30, 8)), "test")])

        class Pool:
            status = "error"
            services = [SimpleNamespace(close=lambda: None)]
            def __init__(self, *args, **kwargs): pass
            def start(self): pass
            def close(self): pass
            def reset(self, _):
                return [dict(ok=True, frames=[frame], states=[state], instruction="test", demo_frames=0, demo_tasks=0)]
            def step(self, chunks, _):
                return [dict(frames=[frame], states=[state], consumed=len(chunks[0]),
                             status=self.status, done=self.status != "ongoing", error_message=None)]

        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "hold01"
            manifest = dict(benchmark_root=str(self.root), checkpoint="/test-checkpoint")
            pp = Path(d) / "batch.json"
            plan.atomic_json(pp, dict(keys=[plan.key(row)], lane="hold01", manifest_sha256=plan.digest(manifest)))
            args = SimpleNamespace(**plan.CONFIG, run_dir=str(run), benchmark_root=str(self.root),
                                   episode_plan=str(pp), pretrained_checkpoint="/test-checkpoint", resume=True,
                                   pipelined=False, tasks=None, reset_timeout=10, step_timeout=10)
            with patch.object(evaluator, "verify_manifest", return_value=manifest), \
                    patch.object(evaluator, "InProcSimPool", Pool), \
                    patch("robomme_sim.eval_success.build_policy", return_value=(model, lambda: buffer, None)) as build:
                self.assertEqual(evaluator.evaluate_candidates(args), 1)
                self.assertEqual(plan.attempts(run, row, manifest)[-1]["status"], "error")
                Pool.status = "ongoing"
                self.assertEqual(evaluator.evaluate_candidates(args), 0)
                record = plan.attempts(run, row, manifest)[-1]
                self.assertEqual(record["status"], "timeout")
                self.assertEqual(record["steps"], 2000)
                self.assertEqual(record["attempt"], 2)
                self.assertTrue(plan.check_normal(record, run, decode=True))
                before = build.call_count
                self.assertEqual(evaluator.evaluate_candidates(args), 0)
                self.assertEqual(build.call_count, before)


if __name__ == "__main__":
    unittest.main()
