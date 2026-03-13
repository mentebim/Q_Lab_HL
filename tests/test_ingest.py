from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from q_lab_hl.cache import CacheBuildConfig, build_hyperliquid_cache, validate_market_cache_dir
from q_lab_hl.ingest import fetch_candles_chunked, fetch_funding_chunked


class FakeClient:
    def __init__(self):
        self.candle_calls = []
        self.funding_calls = []

    def meta(self):
        return {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}]}

    def meta_and_asset_ctxs(self):
        return [
            {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}]},
            [{"name": "BTC", "funding": "0.001", "openInterest": "100", "dayNtlVlm": "1000000"}],
        ]

    def candle_snapshot(self, coin, interval, start_ms, end_ms):
        self.candle_calls.append((coin, interval, start_ms, end_ms))
        data = [
            {"t": 1_700_000_000_000, "T": 1_700_003_599_999, "o": "100", "h": "110", "l": "95", "c": "105", "v": "10", "n": 5},
            {"t": 1_700_003_600_000, "T": 1_700_007_199_999, "o": "105", "h": "111", "l": "100", "c": "109", "v": "12", "n": 6},
        ]
        return [row for row in data if start_ms <= row["T"] and row["t"] <= end_ms]

    def funding_history(self, coin, start_ms, end_ms):
        self.funding_calls.append((coin, start_ms, end_ms))
        data = [
            {"time": 1_700_000_000_000, "fundingRate": "0.0001", "premium": "0.0002"},
            {"time": 1_700_003_600_000, "fundingRate": "-0.0002", "premium": "0.0001"},
        ]
        return [row for row in data if start_ms <= row["time"] <= end_ms]


class IngestTests(unittest.TestCase):
    def test_fetch_candles_chunked_shapes_rows(self):
        client = FakeClient()
        df = fetch_candles_chunked(client, "BTC", "1h", "2023-11-14", "2023-11-15", max_bars_per_call=1)
        self.assertEqual(list(df.columns), ["date", "open", "high", "low", "close", "volume", "trades"])
        self.assertEqual(len(df), 2)
        self.assertGreaterEqual(len(client.candle_calls), 2)

    def test_fetch_funding_chunked_shapes_rows(self):
        client = FakeClient()
        df = fetch_funding_chunked(client, "BTC", "2023-11-14", "2023-11-15", max_hours_per_call=1)
        self.assertEqual(list(df.columns), ["date", "funding_rate", "premium"])
        self.assertEqual(len(df), 2)
        self.assertGreaterEqual(len(client.funding_calls), 2)

    def test_build_hyperliquid_cache_writes_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = FakeClient()
            summary = build_hyperliquid_cache(
                output_dir=out,
                config=CacheBuildConfig(start="2023-11-14", end="2023-11-15", interval="1h", top_n=1),
                client=client,
            )
            self.assertEqual(summary["coins"], 1)
            self.assertTrue((out / "open.parquet").exists())
            self.assertTrue((out / "funding.parquet").exists())
            self.assertTrue((out / "tradable.parquet").exists())
            self.assertTrue((out / "metadata.json").exists())
            close = pd.read_parquet(out / "close.parquet")
            self.assertEqual(list(close.columns), ["BTC"])
            validation = validate_market_cache_dir(out, interval="1h")
            self.assertEqual(validation["assets"], 1)
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertIn("listing_start", metadata["BTC"])


if __name__ == "__main__":
    unittest.main()
