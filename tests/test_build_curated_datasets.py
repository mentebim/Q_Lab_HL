from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "build_curated_datasets.py"
SPEC = importlib.util.spec_from_file_location("build_curated_datasets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD_CURATED_DATASETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_CURATED_DATASETS)


class CuratedDatasetsTests(unittest.TestCase):
    def test_research_eligibility_requires_history_and_selects_top_assets(self):
        periods = (
            BUILD_CURATED_DATASETS.RESEARCH_LOOKBACK_HOURS
            + BUILD_CURATED_DATASETS.RESEARCH_REFRESH_HOURS
        )
        index = pd.date_range("2025-01-01", periods=periods, freq="h", name="date")
        assets = [f"A{i:02d}" for i in range(25)]
        close = pd.DataFrame(1.0, index=index, columns=assets)
        volume = pd.DataFrame(
            {asset: np.full(periods, 1_000.0 - i) for i, asset in enumerate(assets)},
            index=index,
        )
        tradable = pd.DataFrame(True, index=index, columns=assets)
        has_ohlcv = pd.DataFrame(True, index=index, columns=assets)

        late_asset = assets[-1]
        late_start = periods - 100
        close.loc[index[:late_start], late_asset] = np.nan
        volume.loc[index[:late_start], late_asset] = np.nan
        tradable.loc[index[:late_start], late_asset] = False
        has_ohlcv.loc[index[:late_start], late_asset] = False
        volume.loc[index[late_start + 1 :], late_asset] = 50_000.0

        eligibility = BUILD_CURATED_DATASETS._build_research_eligibility(
            {
                "close": close,
                "volume": volume,
                "tradable": tradable,
            },
            has_ohlcv,
        )

        self.assertFalse(
            bool(
                eligibility.iloc[
                    : BUILD_CURATED_DATASETS.RESEARCH_LOOKBACK_HOURS - 1
                ].any().any()
            )
        )

        last_row = eligibility.iloc[-1]
        self.assertEqual(int(last_row.sum()), BUILD_CURATED_DATASETS.ACTIVE_ASSET_COUNT)
        self.assertTrue(bool(last_row["A00"]))
        self.assertTrue(bool(last_row["A19"]))
        self.assertFalse(bool(last_row["A20"]))
        self.assertFalse(bool(last_row[late_asset]))

    def test_research_eligibility_does_not_accumulate_when_archive_is_shallow(self):
        periods = (
            BUILD_CURATED_DATASETS.RESEARCH_LOOKBACK_HOURS
            + BUILD_CURATED_DATASETS.RESEARCH_REFRESH_HOURS * 3
        )
        asset_count = BUILD_CURATED_DATASETS.ACTIVE_ASSET_COUNT + 1
        index = pd.date_range("2025-01-01", periods=periods, freq="h", name="date")
        assets = [f"A{i:02d}" for i in range(asset_count)]
        close = pd.DataFrame(1.0, index=index, columns=assets)
        base_levels = {asset: float(asset_count - i) for i, asset in enumerate(assets)}
        volume = pd.DataFrame(
            {asset: np.full(periods, base_levels[asset]) for asset in assets},
            index=index,
        )
        tradable = pd.DataFrame(True, index=index, columns=assets)
        has_ohlcv = pd.DataFrame(True, index=index, columns=assets)

        swap_asset = assets[-1]
        swap_start = BUILD_CURATED_DATASETS.RESEARCH_LOOKBACK_HOURS + BUILD_CURATED_DATASETS.RESEARCH_REFRESH_HOURS
        volume.loc[index[swap_start:], swap_asset] = base_levels[assets[0]] + 10.0

        eligibility = BUILD_CURATED_DATASETS._build_research_eligibility(
            {
                "close": close,
                "volume": volume,
                "tradable": tradable,
            },
            has_ohlcv,
        )

        self.assertEqual(int(eligibility.iloc[-1].sum()), BUILD_CURATED_DATASETS.ACTIVE_ASSET_COUNT)
        self.assertTrue(bool(eligibility.iloc[-1][swap_asset]))
        self.assertFalse(bool(eligibility.iloc[-1][assets[-2]]))


if __name__ == "__main__":
    unittest.main()
