from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from execution.portfolio_live import build_trade_instructions, derive_target_gross_notional
from execution.risk import validate_margin_plan, validate_requested_leverages
from execution.exchange_hl import HyperliquidExecutionClient, VenueConfig
from execution.run_live import is_rebalance_bar
from execution.select_champion import (
    build_champion_payload,
    build_promotion_record,
    select_record,
    select_stage_record,
    validate_champion_payload,
    validate_result_for_stage,
    validate_stage_candidate_match,
)
from execution.state import last_fill_time_ms, load_state, record_reconciliation, record_run, save_state, seen_signal_bar
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore
from q_lab_hl.promotion_objects import StagePromotionPolicy


class ExecutionTests(unittest.TestCase):
    def test_build_trade_instructions_respects_caps_and_thresholds(self):
        target = pd.Series({"BTC": 0.05, "ETH": -0.02}, dtype=float)
        prices = pd.Series({"BTC": 50000.0, "ETH": 2500.0}, dtype=float)
        current = {"BTC": 0.0, "ETH": 0.0}
        instructions = build_trade_instructions(
            target,
            prices,
            current,
            account_value=10_000.0,
            size_decimals={"BTC": 4, "ETH": 3},
            min_trade_notional_usd=25.0,
            max_single_order_notional_usd=600.0,
        )
        by_coin = {item.coin: item for item in instructions}
        self.assertEqual(by_coin["BTC"].status, "trade")
        self.assertEqual(by_coin["ETH"].status, "trade")
        self.assertAlmostEqual(by_coin["BTC"].target_size, 0.01)
        self.assertAlmostEqual(by_coin["ETH"].target_size, -0.08)

    def test_build_trade_instructions_splits_oversized_delta_and_flip(self):
        target = pd.Series({"BTC": -0.50}, dtype=float)
        prices = pd.Series({"BTC": 50000.0}, dtype=float)
        current = {"BTC": 0.02}
        instructions = build_trade_instructions(
            target,
            prices,
            current,
            account_value=10_000.0,
            size_decimals={"BTC": 4},
            min_trade_notional_usd=25.0,
            max_single_order_notional_usd=1000.0,
        )
        trade_instructions = [item for item in instructions if item.status == "trade"]
        self.assertGreaterEqual(len(trade_instructions), 2)
        self.assertEqual(trade_instructions[0].execution_stage, "flip_close")
        self.assertTrue(all(abs(item.delta_notional_usd) <= 1000.0 + 1e-9 for item in trade_instructions))
        self.assertEqual(trade_instructions[-1].target_size, -0.1)

    def test_state_tracks_processed_signal_bars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = load_state(state_path)
            self.assertFalse(seen_signal_bar(state, "2026-01-01 00:00:00"))
            next_state = record_run(state, "2026-01-01 00:00:00", {"ok": True})
            save_state(state_path, next_state)
            loaded = load_state(state_path)
            self.assertTrue(seen_signal_bar(loaded, "2026-01-01 00:00:00"))

    def test_record_reconciliation_updates_fill_watermark(self):
        state = load_state(Path(tempfile.gettempdir()) / "nonexistent_state.json")
        updated = record_reconciliation(
            state,
            {
                "open_orders": [{"coin": "BTC", "oid": 1}],
                "recent_fills": [{"coin": "BTC", "time": 123}, {"coin": "ETH", "time": 456}],
                "snapshot_time_ms": 789,
                "account_value": 10.0,
                "positions": {},
                "mode": "paper",
                "account_address": None,
            },
        )
        self.assertEqual(last_fill_time_ms(updated), 456)
        self.assertEqual(len(updated["reconciliation"]["open_orders"]), 1)
        self.assertEqual(updated["reconciliation"]["snapshot"]["snapshot_time_ms"], 789)

    def test_select_record_and_build_champion_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "r2.json"
            result = {
                "experiment_id": "exp",
                "candidate_id": "cand",
                "result_path": str(result_path),
                "acceptance": {"accepted": True, "status": "accepted"},
                "promotion_eligibility": {
                    "accepted_result": True,
                    "express_filter_passed": True,
                    "paper_eligible": True,
                    "live_eligible": True,
                    "reason": "eligible",
                },
                "strategy_path": "strategy.py",
                "spec": {
                    "strategy_path": "strategy.py",
                    "strategy_spec": {"model": {"family": "ridge"}},
                    "execution_overrides": {"rebalance_every_bars": 24},
                    "data_dir": "data/market_cache_1h",
                },
            }
            result_path.write_text(json.dumps(result))
            rows = [
                {"accepted": False, "primary_metric_value": -1.0},
                {"accepted": True, "primary_metric_value": 0.5, "result_path": str(Path(tmpdir) / "r1.json")},
                {"accepted": True, "primary_metric_value": 1.2, "result_path": str(result_path)},
            ]
            selected = select_record(rows, "best-accepted")
            self.assertEqual(selected["primary_metric_value"], 1.2)
            promotion = build_promotion_record(
                stage="paper",
                policy_path="autoresearch/promotion_policy.json",
                policy_id="promotion_policy_v1",
                selector_policy="best-accepted",
                promoted_from="autoresearch/leaderboard.jsonl",
                promoted_at="2026-03-12T00:00:00+00:00",
            )
            payload = build_champion_payload(
                result,
                promotion=promotion,
                existing={
                    "source": {"note": "keep me"},
                    "live": {
                        "mode": "live",
                        "target_gross_notional_usd": 1000.0,
                        "default_leverage": 3,
                        "leverage_overrides": {"BTC": 4},
                    },
                },
            )
            self.assertEqual(payload["promotion"]["stage"], "paper")
            self.assertEqual(payload["promotion"]["selector_policy"], "best-accepted")
            self.assertEqual(payload["source"]["candidate_id"], "cand")
            self.assertEqual(payload["source"]["note"], "keep me")
            self.assertEqual(payload["live"]["mode"], "live")
            self.assertEqual(payload["live"]["target_gross_notional_usd"], 1000.0)
            self.assertEqual(payload["live"]["default_leverage"], 3)
            self.assertEqual(validate_champion_payload(payload), result_path)

    def test_validate_stage_candidate_match_rejects_live_mismatch(self):
        stage_policy = StagePromotionPolicy(require_matching_paper_candidate=True)
        with self.assertRaises(ValueError):
            validate_stage_candidate_match(
                record={"candidate_id": "cand-live"},
                stage_policy=stage_policy,
                paper_candidate_id="cand-paper",
            )

    def test_select_stage_record_requires_stage_eligibility(self):
        stage_policy = StagePromotionPolicy()
        record = select_stage_record(
            [
                {
                    "accepted": True,
                    "candidate_id": "cand",
                    "primary_metric_value": 1.0,
                    "promotion_eligibility": {"paper_eligible": False, "live_eligible": False},
                }
            ],
            "paper",
            stage_policy,
        )
        self.assertIsNone(record)

    def test_validate_result_for_stage_rejects_ineligible_artifact(self):
        with self.assertRaises(ValueError):
            validate_result_for_stage(
                result={
                    "acceptance": {"accepted": True},
                    "promotion_eligibility": {"paper_eligible": False, "live_eligible": False},
                },
                stage="paper",
                stage_policy=StagePromotionPolicy(),
            )

    def test_validate_champion_payload_rejects_missing_artifact(self):
        payload = {
            "promotion": {
                "stage": "paper",
                "policy_path": "autoresearch/promotion_policy.json",
                "policy_id": "promotion_policy_v1",
                "selector_policy": "best-accepted",
                "promoted_at": "2026-03-12T00:00:00+00:00",
                "promoted_from": "autoresearch/leaderboard.jsonl",
            },
            "source": {
                "experiment_id": "exp",
                "candidate_id": "cand",
                "result_path": "/tmp/does-not-exist.json",
            }
        }
        with self.assertRaises(FileNotFoundError):
            validate_champion_payload(payload)

    def test_validate_margin_plan_enforces_headroom(self):
        target = pd.Series({"BTC": 0.10}, dtype=float)
        prices = pd.Series({"BTC": 50000.0}, dtype=float)
        instructions = build_trade_instructions(
            target,
            prices,
            {},
            account_value=10_000.0,
            size_decimals={"BTC": 4},
            min_trade_notional_usd=25.0,
            max_single_order_notional_usd=1000.0,
        )
        with self.assertRaises(ValueError):
            validate_margin_plan(
                instructions,
                account_value=1000.0,
                leverage_by_coin={"BTC": 2},
                max_margin_usage_ratio=0.5,
                min_margin_headroom_usd=100.0,
            )

    def test_validate_requested_leverages_checks_asset_caps(self):
        with self.assertRaises(ValueError):
            validate_requested_leverages(
                leverage_by_coin={"BTC": 5},
                metadata={"BTC": {"maxLeverage": 3}},
            )

    def test_derive_target_gross_notional_from_margin_usage(self):
        sizing = derive_target_gross_notional(
            pd.Series({"BTC": 0.5, "ETH": -0.5}, dtype=float),
            account_value=250.0,
            leverage_by_coin={"BTC": 4, "ETH": 4},
            target_margin_usage_ratio=0.5,
            max_margin_usage_ratio=0.8,
            min_margin_headroom_usd=50.0,
        )
        self.assertAlmostEqual(sizing["target_initial_margin_usd"], 125.0)
        self.assertAlmostEqual(sizing["effective_margin_usage_ratio"], 0.5)
        self.assertAlmostEqual(sizing["target_gross_notional_usd"], 500.0)

    def test_paper_reconciliation_snapshot_uses_state(self):
        venue = VenueConfig(mode="paper", account_address="0xabc", paper_account_value=321.0)
        client = HyperliquidExecutionClient(venue)
        snapshot = client.reconciliation_snapshot(
            {
                "paper_positions": {"BTC": 0.1},
                "paper_fills": [{"coin": "BTC", "time": 100}, {"coin": "ETH", "time": 200}],
            },
            fills_since_ms=150,
        )
        self.assertEqual(snapshot.account_value, 321.0)
        self.assertEqual(snapshot.positions["BTC"], 0.1)
        self.assertEqual(len(snapshot.recent_fills), 1)
        self.assertEqual(snapshot.recent_fills[0]["coin"], "ETH")

    def test_is_rebalance_bar_matches_backtest_schedule(self):
        store = DataStore.synthetic(n_assets=8, periods=24 * 40, seed=7)
        execution = ExecutionConfig(rebalance_every_bars=24, min_history_bars=24 * 14)
        self.assertTrue(is_rebalance_bar(store, execution, store.index[24 * 14]))
        self.assertFalse(is_rebalance_bar(store, execution, store.index[24 * 14 + 1]))


if __name__ == "__main__":
    unittest.main()
