"""Bounded regressions for retries and durable experiment evidence."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reproduction.recall_generate import CONDITIONS, QUERIES
from reproduction.recall_spec import EVAL_EXAMPLES, MANIFEST_SHA256, STEPS, config
from reproduction.train_wikitext import restore_training_curve
from scripts.check_reproduction import check_arm
import scripts.check_reproduction as checker
import scripts.run_experiments as runner


class RetryTest(unittest.TestCase):
    def test_bad_cache_and_launch_failure_leave_other_jobs_running(self):
        for failure in ("cached", "launch"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "out"
                if failure == "cached":
                    bad = output / "recall/bad"
                    bad.mkdir(parents=True)
                    (bad / "result.json").write_text("{}")
                processes = []

                class Process:
                    def __init__(self, cmd, **kwargs):
                        self.arm = cmd[cmd.index("--arm") + 1]
                        if failure == "launch" and self.arm == "bad":
                            raise OSError("cannot launch")
                        self.terminated = False
                        self.polls = 0
                        processes.append(self)

                    def poll(self):
                        self.polls += 1
                        return None if self.polls == 1 else 0

                    def terminate(self):
                        self.terminated = True

                    def wait(self):
                        return 0

                def validate(path, bench, arm, **kwargs):
                    if arm == "bad":
                        raise KeyError("malformed cached result")

                argv = ["run", "--benchmark", "recall", "--gpus", "0,1", "--output", str(output),
                        "--data", str(root / "data")]
                with patch("sys.argv", argv), patch.object(runner, "selected_arms", return_value=("live", "bad", "queued")), \
                        patch("scripts.verify_vendor.validate_vendor"), patch("scripts.build_extensions.verify"), \
                        patch("scripts.prepare_recall.check"), patch.object(runner, "check_arm", side_effect=validate), \
                        patch.object(runner.subprocess, "Popen", Process), patch.object(runner.time, "sleep"), \
                        redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "recall/bad"):
                        runner.main()
                self.assertEqual([p.arm for p in processes], ["live", "queued"])
                self.assertFalse(any(p.terminated for p in processes))

    def test_restore_curve_replaces_uncommitted_or_truncated_attempt_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_curve.jsonl"
            path.write_text('{"step": 6, "train_nll": 9}\n{"step": 7}\n{"step":')
            restored = [{"step": 1, "train_nll": 5}, {"step": 6, "train_nll": 4}]
            restore_training_curve(path, restored, 6)
            with path.open("a") as handle:
                handle.write('{"step": 7, "train_nll": 3}\n')
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()],
                             restored + [{"step": 7, "train_nll": 3}])
            with self.assertRaises(ValueError):
                restore_training_curve(path, [{"step": 6}, {"step": 6}], 6)


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "recall/fixed_k8"
        self.root.mkdir(parents=True)
        self.labels = np.zeros((EVAL_EXAMPLES, QUERIES), dtype=np.uint8)
        self.data = SimpleNamespace(evaluation=lambda split, condition: (None, self.labels))
        prediction = self.labels.copy()
        prediction[::2, 0] = 1
        rows = []
        for condition in CONDITIONS:
            path = self.root / (condition["id"] + ".npy")
            np.save(path, prediction, allow_pickle=False)
            rows.append({"condition_id": condition["id"], "family": condition["family"],
                         "examples": EVAL_EXAMPLES, "loss_sum": float(EVAL_EXAMPLES * QUERIES),
                         "correct_queries": EVAL_EXAMPLES * QUERIES - EVAL_EXAMPLES // 2,
                         "exact_sets": EVAL_EXAMPLES // 2,
                         "predictions": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})

        def summarize(selected):
            n = len(selected) * EVAL_EXAMPLES
            return {"examples": n, "queries": n * QUERIES, "query_loss": 1.,
                    "query_accuracy": 31 / 32, "exact_set_accuracy": .5}

        split = {"by_condition": rows, "aggregate": summarize(rows),
                 "by_family": {family: summarize([r for r in rows if r["family"] == family])
                               for family in {r["family"] for r in rows}}}
        self.row = {"status": "terminal_verified", "arm": "fixed_k8", "config": config("fixed_k8"),
                    "manifest_sha256": MANIFEST_SHA256, "steps": STEPS,
                    "checkpoint_roundtrip_inference_passed": True,
                    "validation": copy.deepcopy(split), "test": copy.deepcopy(split)}
        self.checkpoint = {"schema": "access-curriculum-recall-checkpoint-v1", "config": config("fixed_k8"),
                           "manifest_sha256": MANIFEST_SHA256, "step": STEPS, "mode": "experiment",
                           "model": {"weight": torch.zeros(1)}}
        self.save_checkpoint()

    def save_checkpoint(self):
        path = self.root / "terminal-model.pt"
        torch.save(self.checkpoint, path)
        self.row["terminal_checkpoint"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def check(self):
        (self.root / "result.json").write_text(json.dumps(self.row))
        return check_arm(self.root, "recall", "fixed_k8", recall_data=self.data)

    def test_complete_evidence_recomputes_accuracy(self):
        self.assertTrue(self.check()["within_expected_range"])

    def test_missing_or_changed_prediction_is_rejected(self):
        path = self.root / self.row["test"]["by_condition"][0]["predictions"]["path"]
        path.unlink()
        with self.assertRaisesRegex(ValueError, "absent"):
            self.check()
        np.save(path, self.labels, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.check()

    def test_matching_hash_does_not_hide_wrong_prediction_accuracy(self):
        record = self.row["validation"]["by_condition"][0]["predictions"]
        path = self.root / record["path"]
        np.save(path, self.labels, allow_pickle=False)
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "reported accuracy"):
            self.check()

    def test_missing_condition_and_wrong_family_total_are_rejected(self):
        removed = self.row["test"]["by_condition"].pop()
        with self.assertRaisesRegex(ValueError, "condition coverage"):
            self.check()
        self.row["test"]["by_condition"].append(removed)
        self.row["test"]["by_family"]["pointer_chase"]["exact_set_accuracy"] = 1.
        with self.assertRaisesRegex(ValueError, "aggregate differs"):
            self.check()

    def test_checkpoint_config_is_checked_even_with_a_matching_digest(self):
        self.checkpoint["config"]["seed"] = 1
        self.save_checkpoint()
        with self.assertRaisesRegex(ValueError, "experiment identity"):
            self.check()

    def test_partial_suite_fails_before_loading_inputs(self):
        self.check()
        argv = ["check", str(self.root.parents[1]), "--suite", "study", "--benchmark", "recall"]
        with patch("sys.argv", argv), self.assertRaisesRegex(SystemExit, "missing results"):
            checker.main()


if __name__ == "__main__":
    unittest.main()
