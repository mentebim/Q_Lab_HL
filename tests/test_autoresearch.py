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
            store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=7)
            spec = ExperimentSpec(
                experiment_id="exp_structured",
                candidate_id="cand_structured",
                hypothesis="Synthetic smoke test for the autoresearch runner.",
                strategy_spec={
                    "position_bucket": 5,
                    "model": {"family": "ols", "l2_reg": 0.0, "prediction_clip": 2.0},
                },
                execution_overrides={"rebalance_every_bars": 12},
                synthetic=True,
                evaluation_periods=("inner",),
                express_filter=ExpressFilterConfig(
                    period="outer",
                    trailing_bars=24 * 40,
                    max_assets=20,
                    bootstrap_samples=10,
                    primary_min=-10.0,

                    max_beta_abs=10.0,
                    max_turnover=10.0,
                ),
                acceptance=AcceptancePolicy(
                    primary_metric="periods.inner.score_inner",
                    primary_min=-10.0,

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
            self.assertIn(result["acceptance"]["status"], ("accepted", "rejected"))
            self.assertEqual(result["express_filter"]["status"], "passed")
            self.assertTrue(Path(result["result_path"]).exists())
            self.assertEqual(result["model_fit"]["strategy_spec"]["position_bucket"], 5)
            self.assertEqual(result["model_fit"]["model_fit"]["family"], "ols")
            self.assertEqual(result["execution"]["rebalance_every_bars"], 12)
            self.assertIn("walk_forward", result)
            self.assertIn("periods", result)
            self.assertIn("inner", result["periods"])
            records = load_leaderboard(spec.recording.leaderboard_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["experiment_id"], "exp_structured")
            self.assertEqual(records[0]["express_filter"]["status"], "passed")
            self.assertIn("promotion_eligibility", records[0])

    def test_run_experiment_filters_before_full_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=7)
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


class PolicyConsistencyTests(unittest.TestCase):
    def test_default_policy_matches_absolute_sharpe(self):
        policy = AcceptancePolicy()
        self.assertEqual(policy.primary_metric, "periods.outer.sharpe_annualized")
        self.assertEqual(policy.primary_min, 0.3)
        self.assertFalse(hasattr(policy, "min_active_sharpe"))

    def test_default_express_filter_matches_absolute_sharpe(self):
        config = ExpressFilterConfig()
        self.assertEqual(config.primary_metric, "sharpe_annualized")
        self.assertEqual(config.primary_min, -0.5)
        self.assertEqual(config.max_assets, 20)
        self.assertFalse(hasattr(config, "min_active_sharpe"))

    def test_default_evaluation_periods(self):
        spec = ExperimentSpec()
        self.assertEqual(spec.evaluation_periods, ("inner", "outer"))
        self.assertFalse(spec.enable_walk_forward)


class WalkForwardInvariantTests(unittest.TestCase):
    def test_walk_forward_preserves_candidate_spec(self):
        from q_lab_hl.backtest import load_strategy
        from q_lab_hl.config import ExecutionConfig
        from q_lab_hl.evaluate import walk_forward_evaluate
        strategy = load_strategy("strategy.py")
        strategy.apply_runtime_overrides(
            strategy_spec={"position_bucket": 5, "model": {"family": "ridge", "l2_reg": 2.0}},
            execution_overrides={"max_net_deviation": 0.20, "min_history_bars": 24, "min_dollar_volume": 0.0, "listing_cooldown_bars": 0},
        )
        spec_before = strategy.SPEC
        execution_before = strategy.EXECUTION
        store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=7)
        walk_forward_evaluate(strategy, store, execution_before, runway_bars=200, eval_bars=100, step_bars=100, bootstrap_samples=5)
        self.assertEqual(strategy.SPEC.position_bucket, spec_before.position_bucket)
        self.assertEqual(strategy.SPEC.model.family, spec_before.model.family)
        self.assertEqual(strategy.SPEC.model.l2_reg, spec_before.model.l2_reg)
        self.assertEqual(strategy.EXECUTION, execution_before)

    def test_walk_forward_restores_runtime_state(self):
        from q_lab_hl.backtest import load_strategy
        from q_lab_hl.config import ExecutionConfig
        from q_lab_hl.evaluate import walk_forward_evaluate
        strategy = load_strategy("strategy.py")
        strategy.apply_runtime_overrides(
            execution_overrides={"max_net_deviation": 0.20, "min_history_bars": 24, "min_dollar_volume": 0.0, "listing_cooldown_bars": 0},
        )
        strategy._STATE["marker"] = "before_wf"
        store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=7)
        walk_forward_evaluate(strategy, store, strategy.EXECUTION, runway_bars=200, eval_bars=100, step_bars=100, bootstrap_samples=5)
        self.assertEqual(strategy._STATE.get("marker"), "before_wf")


class ModelFitArtifactTests(unittest.TestCase):
    def test_run_experiment_records_judged_period_model_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=7)
            spec = ExperimentSpec(
                experiment_id="exp_fit", candidate_id="cand_fit", synthetic=True,
                evaluation_periods=("inner", "outer"),
                express_filter=ExpressFilterConfig(
                    trailing_bars=24 * 20, max_assets=20, bootstrap_samples=5,
                    primary_min=-10.0, max_beta_abs=10.0, max_turnover=10.0,
                ),
                acceptance=AcceptancePolicy(
                    primary_metric="periods.outer.score_inner", primary_min=-10.0,
                    max_beta_abs=10.0, max_turnover=10.0,
                ),
                recording=RecordingConfig(
                    leaderboard_path=str(Path(tmp) / "lb.jsonl"),
                    results_dir=str(Path(tmp) / "results"),
                ),
            )
            result = run_experiment(spec, data_store=store)
            self.assertIn("period_model_fit", result)
            self.assertIn("outer", result["period_model_fit"])
            self.assertIn("inner", result["period_model_fit"])
            self.assertEqual(result["model_fit"], result["period_model_fit"]["outer"])

    def test_result_model_fit_is_candidate_fit_not_walk_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=7)
            spec = ExperimentSpec(
                experiment_id="exp_wf_fit", candidate_id="cand_wf_fit", synthetic=True,
                evaluation_periods=("inner",),
                express_filter=ExpressFilterConfig(
                    trailing_bars=24 * 20, max_assets=20, bootstrap_samples=5,
                    primary_min=-10.0, max_beta_abs=10.0, max_turnover=10.0,
                ),
                acceptance=AcceptancePolicy(
                    primary_metric="periods.inner.score_inner", primary_min=-10.0,
                    max_beta_abs=10.0, max_turnover=10.0,
                ),
                recording=RecordingConfig(
                    leaderboard_path=str(Path(tmp) / "lb.jsonl"),
                    results_dir=str(Path(tmp) / "results"),
                ),
            )
            result = run_experiment(spec, data_store=store)
            self.assertIsNotNone(result["model_fit"])
            self.assertEqual(result["model_fit"], result["period_model_fit"]["inner"])


if __name__ == "__main__":
    unittest.main()
