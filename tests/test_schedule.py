import json
from pathlib import Path
import unittest

from access_curriculum.schedule import stages, access_at_step, CELLS, ALIASES, HEADLINE_ARMS, POLICIES
from scripts.run_experiments import selected_arms, command


class ScheduleTest(unittest.TestCase):
    def test_every_recorded_step_has_one_access_pair(self):
        for benchmark, total in (("wikitext", 21603), ("recall", 30000)):
            for arm in (*CELLS, *(a for a in HEADLINE_ARMS if a != "dense_attention")):
                rows = stages(arm, benchmark)
                self.assertEqual(rows[0][0], 1)
                self.assertEqual(rows[-1], (rows[-1][0], total, 8, 8))
                self.assertEqual(sum(b - a + 1 for a, b, _, _ in rows), total)
                for left, right in zip(rows, rows[1:]):
                    self.assertEqual(left[1] + 1, right[0])
                    self.assertEqual(access_at_step(arm, left[1], benchmark), left[2:])
                    self.assertEqual(access_at_step(arm, right[0], benchmark), right[2:])
                for step in (0, total + 1):
                    with self.assertRaises(ValueError):
                        access_at_step(arm, step, benchmark)

    def test_fractional_transfer_preserves_recorded_rounding(self):
        wiki = stages("wiki_forward", "wikitext")
        recall = stages("wiki_forward", "recall")
        self.assertEqual([r[1] for r in recall], [round(w[1] * 30000 / 21603) for w in wiki])
        self.assertEqual([r[2:] for r in recall], [w[2:] for w in wiki])

    def test_full_suite_does_not_repeat_selected_factorial_cells(self):
        arms = selected_arms("full")
        study = selected_arms("study")
        self.assertEqual({ALIASES.get(a, a) for a in study}, {"fixed_k8", "C", "D", "G", "H"})
        self.assertEqual(len(study), 5)
        self.assertTrue(set(study).issubset(arms))
        self.assertEqual(selected_arms("topology"), selected_arms("headline"))
        self.assertEqual(len(selected_arms("headline")), 11)
        self.assertEqual(len(arms), 16)
        self.assertEqual(len({ALIASES.get(a, a) for a in arms}), 16)
        for arm in arms:
            cmd = command("recall", arm, Path("inputs/manifest.json"), Path("outputs") / arm)
            self.assertEqual(cmd[cmd.index("--reads") + 1], "8")
            self.assertEqual(cmd[cmd.index("--writes") + 1], "8")

    def test_topology_pairs_preserve_recorded_policy_and_parameters(self):
        from reproduction.spec import ARMS
        from reproduction.recall_spec import PARAMETERS
        expected = {"native_k8": (15038880, 2212256), "hybrid_native_k8": (15029660, 2203036),
                    "fixed_k8": (15104928, 2278304), "hybrid_fixed_k8": (15087452, 2260828)}
        for name, counts in expected.items():
            self.assertEqual((ARMS[name].expected_trainable_parameters, PARAMETERS[name]), counts)
        for policy in POLICIES:
            base, hybrid = ARMS[policy], ARMS["hybrid_" + policy]
            self.assertEqual(base.profile.layout, "BBBBBBBB")
            self.assertEqual(hybrid.profile.layout, "BBBBBBBA")
            self.assertEqual(base.router, hybrid.router)
            for benchmark in ("wikitext", "recall"):
                self.assertEqual(stages(policy, benchmark), stages("hybrid_" + policy, benchmark))

    def test_project_identity_contains_exact_stages(self):
        root = Path(__file__).resolve().parents[1]
        identity = json.loads((root / "PROJECT_IDENTITY.json").read_text())
        for benchmark in ("wikitext", "recall"):
            for arm, rows in identity["resolved_stages"][benchmark].items():
                self.assertEqual(rows, [list(r) for r in stages(arm, benchmark)])

    def test_recall_controls_use_matched_learning_rates(self):
        from reproduction.recall_spec import ARMS, STEPS, learning_rate
        for arm in ARMS:
            for step in range(1, STEPS + 1):
                self.assertEqual(learning_rate(arm, step), 3e-4 * min(step / 100, 1))


if __name__ == "__main__":
    unittest.main()
