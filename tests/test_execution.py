from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from execution.portfolio_live import build_trade_instructions
from execution.select_champion import build_champion_payload, select_record
from execution.state import load_state, record_run, save_state, seen_signal_bar


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

    def test_state_tracks_processed_signal_bars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            state = load_state(state_path)
            self.assertFalse(seen_signal_bar(state, "2026-01-01 00:00:00"))
            next_state = record_run(state, "2026-01-01 00:00:00", {"ok": True})
            save_state(state_path, next_state)
            loaded = load_state(state_path)
            self.assertTrue(seen_signal_bar(loaded, "2026-01-01 00:00:00"))

    def test_select_record_and_build_champion_payload(self):
        rows = [
            {"accepted": False, "primary_metric_value": -1.0},
            {"accepted": True, "primary_metric_value": 0.5, "result_path": "/tmp/r1.json"},
            {"accepted": True, "primary_metric_value": 1.2, "result_path": "/tmp/r2.json"},
        ]
        selected = select_record(rows, "best-accepted")
        self.assertEqual(selected["primary_metric_value"], 1.2)
        result = {
            "experiment_id": "exp",
            "candidate_id": "cand",
            "result_path": "/tmp/r2.json",
            "strategy_path": "strategy.py",
            "spec": {
                "strategy_path": "strategy.py",
                "strategy_spec": {"model": {"family": "ridge"}},
                "execution_overrides": {"rebalance_every_bars": 24},
                "data_dir": "data/market_cache_1h",
            },
        }
        payload = build_champion_payload(result)
        self.assertEqual(payload["source"]["candidate_id"], "cand")
        self.assertEqual(payload["live"]["mode"], "paper")


if __name__ == "__main__":
    unittest.main()
