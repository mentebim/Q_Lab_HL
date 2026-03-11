from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from q_lab_hl.cache import CacheBuildConfig, build_hyperliquid_cache
from q_lab_hl.ingest import fetch_candles_chunked, fetch_funding_chunked


class FakeClient:
    def meta(self):
        return {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}]}

    def meta_and_asset_ctxs(self):
        return [
            {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}]},
            [{"name": "BTC", "funding": "0.001", "openInterest": "100", "dayNtlVlm": "1000000"}],
        ]

    def candle_snapshot(self, coin, interval, start_ms, end_ms):
        data = [
            {"t": 1_700_000_000_000, "T": 1_700_000_359_999, "o": "100", "h": "110", "l": "95", "c": "105", "v": "10", "n": 5},
            {"t": 1_700_000_360_000, "T": 1_700_000_719_999, "o": "105", "h": "111", "l": "100", "c": "109", "v": "12", "n": 6},
        ]
        return [row for row in data if start_ms <= row["T"] and row["t"] <= end_ms]

    def funding_history(self, coin, start_ms, end_ms):
        data = [
            {"time": 1_700_000_000_000, "fundingRate": "0.0001", "premium": "0.0002"},
            {"time": 1_700_000_360_000, "fundingRate": "-0.0002", "premium": "0.0001"},
        ]
        return [row for row in data if start_ms <= row["time"] <= end_ms]


class IngestTests(unittest.TestCase):
    def test_fetch_candles_chunked_shapes_rows(self):
        df = fetch_candles_chunked(FakeClient(), "BTC", "1h", "2023-11-14", "2023-11-15", max_bars_per_call=1)
        self.assertEqual(list(df.columns), ["date", "open", "high", "low", "close", "volume", "trades"])
        self.assertEqual(len(df), 2)

    def test_fetch_funding_chunked_shapes_rows(self):
        df = fetch_funding_chunked(FakeClient(), "BTC", "2023-11-14", "2023-11-15", max_hours_per_call=1)
        self.assertEqual(list(df.columns), ["date", "funding_rate", "premium"])
        self.assertEqual(len(df), 2)

    def test_build_hyperliquid_cache_writes_required_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = build_hyperliquid_cache(
                output_dir=out,
                config=CacheBuildConfig(start="2023-11-14", end="2023-11-15", interval="1h", top_n=1),
                client=FakeClient(),
            )
            self.assertEqual(summary["coins"], 1)
            self.assertTrue((out / "open.parquet").exists())
            self.assertTrue((out / "funding.parquet").exists())
            self.assertTrue((out / "tradable.parquet").exists())
            self.assertTrue((out / "metadata.json").exists())
            close = pd.read_parquet(out / "close.parquet")
            self.assertEqual(list(close.columns), ["BTC"])


if __name__ == "__main__":
    unittest.main()
