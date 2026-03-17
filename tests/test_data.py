from __future__ import annotations

import unittest

from q_lab_hl.data import DataStore


class DataStoreTests(unittest.TestCase):
    def test_tradable_universe_respects_listing_cooldown(self):
        store = DataStore.synthetic(n_assets=8, periods=24 * 20, seed=3)
        late_asset = store.assets[-1]
        first_ts = store.index[24 * 4]
        early = store.tradable_universe(
            first_ts,
            min_history_bars=0,
            min_dollar_volume=0.0,
            min_price=0.0,
            listing_cooldown_bars=24 * 7,
        )
        self.assertNotIn(late_asset, early)

    def test_tradable_universe_respects_research_eligibility(self):
        store = DataStore.synthetic(n_assets=8, periods=24 * 20, seed=9)
        ts = store.index[-1]
        blocked_asset = store.assets[0]
        store.research_eligible_panel.loc[:, blocked_asset] = False
        universe = store.tradable_universe(
            ts,
            min_history_bars=0,
            min_dollar_volume=0.0,
            min_price=0.0,
            listing_cooldown_bars=0,
        )
        self.assertTrue(universe)
        self.assertNotIn(blocked_asset, universe)


if __name__ == "__main__":
    unittest.main()
