from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from q_lab_hl.autoresearch import (
    AcceptancePolicy,
    ExperimentSpec,
    ExpressFilterConfig,
    RecordingConfig,
    append_leaderboard_entry,
    evaluate_acceptance,
    load_experiment_spec,
    load_leaderboard,
    run_experiment,
)
from q_lab_hl.data import DataStore
from q_lab_hl.research_objects import load_research_policy


class AutoResearchTests(unittest.TestCase):
    def test_load_experiment_spec_applies_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "exp_a",
                        "candidate_id": "cand_a",
                        "hypothesis": "test hypothesis",
                    }
                )
            )
            spec = load_experiment_spec(config_path)
            self.assertEqual(spec.experiment_id, "exp_a")
            self.assertEqual(spec.candidate_id, "cand_a")
            self.assertEqual(spec.strategy_path, "strategy.py")
            self.assertEqual(spec.strategy_family, "linear_cross_section_v1")
            self.assertEqual(spec.research_policy_path, "autoresearch/research_policy.json")
            self.assertEqual(spec.evaluation_periods, ("inner", "outer"))
            self.assertTrue(spec.express_filter.enabled)
            self.assertIsNone(spec.strategy_spec)

    def test_load_research_policy_applies_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy_path = Path(tmp) / "policy.json"
            policy_path.write_text(json.dumps({"policy_id": "custom_policy", "version": 3}))
            policy = load_research_policy(policy_path)
            self.assertEqual(policy.policy_id, "custom_policy")
            self.assertEqual(policy.version, 3)
            self.assertIn("q_lab_hl/backtest.py", policy.fixed_paths)

    def test_leaderboard_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            leaderboard_path = Path(tmp) / "leaderboard.jsonl"
            append_leaderboard_entry({"experiment_id": "exp_a", "primary_metric_value": 1.23}, leaderboard_path)
            records = load_leaderboard(leaderboard_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["experiment_id"], "exp_a")

    def test_evaluate_acceptance_compares_against_reference(self):
        result = {
            "periods": {"outer": {"active_sharpe_annualized": 0.2, "beta_to_market": 0.05, "turnover": 0.3}},
        }
        policy = AcceptancePolicy(
            primary_metric="periods.outer.active_sharpe_annualized",
            compare_to_best=True,
            min_primary_lift=0.02,
        )
        leaderboard = [{"experiment_id": "baseline", "candidate_id": "baseline", "periods": {"outer": {"active_sharpe_annualized": 0.25}}}]
        decision = evaluate_acceptance(result, policy, leaderboard)
        self.assertEqual(decision["status"], "rejected")
        self.assertIn("reference_comparison", decision["failed_checks"])

    def test_run_experiment_writes_structured_result_and_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore.synthetic(n_assets=16, periods=24 * 25, seed=7)
            spec = ExperimentSpec(
                experiment_id="exp_structured",
                candidate_id="cand_structured",
                hypothesis="Synthetic smoke test for the autoresearch runner.",
                strategy_spec={
                    "position_bucket": 3,
                    "model": {"family": "ols", "l2_reg": 0.0, "prediction_clip": 2.0},
                },
                execution_overrides={"rebalance_every_bars": 12},
                synthetic=True,
                evaluation_periods=("inner",),
                express_filter=ExpressFilterConfig(
                    period="outer",
                    trailing_bars=24 * 20,
                    max_assets=8,
                    bootstrap_samples=10,
                    primary_min=-10.0,
                    min_active_sharpe=-10.0,
                    max_beta_abs=10.0,
                    max_turnover=10.0,
                ),
                acceptance=AcceptancePolicy(
                    primary_metric="periods.inner.score_inner",
                    primary_min=-10.0,
                    min_active_sharpe=-10.0,
                    max_beta_abs=10.0,
                    max_turnover=10.0,
                ),
                recording=RecordingConfig(
                    leaderboard_path=str(Path(tmp) / "leaderboard.jsonl"),
                    results_dir=str(Path(tmp) / "results"),
                    append_leaderboard=True,
                    write_result=True,
                ),
            )
            result = run_experiment(spec, data_store=store)
            self.assertEqual(result["acceptance"]["status"], "accepted")
            self.assertEqual(result["express_filter"]["status"], "passed")
            self.assertTrue(result["promotion_eligibility"]["paper_eligible"])
            self.assertTrue(Path(result["result_path"]).exists())
            self.assertEqual(result["model_fit"]["strategy_spec"]["position_bucket"], 3)
            self.assertEqual(result["model_fit"]["model_fit"]["family"], "ols")
            self.assertEqual(result["execution"]["rebalance_every_bars"], 12)
            records = load_leaderboard(spec.recording.leaderboard_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["experiment_id"], "exp_structured")
            self.assertEqual(records[0]["express_filter"]["status"], "passed")
            self.assertTrue(records[0]["promotion_eligibility"]["paper_eligible"])

    def test_run_experiment_filters_before_full_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore.synthetic(n_assets=16, periods=24 * 25, seed=7)
            spec = ExperimentSpec(
                experiment_id="exp_filtered",
                candidate_id="cand_filtered",
                synthetic=True,
                evaluation_periods=("inner",),
                express_filter=ExpressFilterConfig(
                    period="outer",
                    trailing_bars=24 * 20,
                    max_assets=8,
                    bootstrap_samples=10,
                    primary_min=100.0,
                ),
                acceptance=AcceptancePolicy(
                    primary_metric="periods.inner.score_inner",
                    primary_min=-10.0,
                    min_active_sharpe=-10.0,
                    max_beta_abs=10.0,
                    max_turnover=10.0,
                ),
                recording=RecordingConfig(
                    leaderboard_path=str(Path(tmp) / "leaderboard.jsonl"),
                    results_dir=str(Path(tmp) / "results"),
                    append_leaderboard=True,
                    write_result=True,
                ),
            )
            result = run_experiment(spec, data_store=store)
            self.assertEqual(result["acceptance"]["status"], "filtered")
            self.assertEqual(result["express_filter"]["status"], "filtered")
            self.assertFalse(result["promotion_eligibility"]["paper_eligible"])
            self.assertEqual(result["periods"], {})
            records = load_leaderboard(spec.recording.leaderboard_path)
            self.assertEqual(records[0]["status"], "filtered")


if __name__ == "__main__":
    unittest.main()
