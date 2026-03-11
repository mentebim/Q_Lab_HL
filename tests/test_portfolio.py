from __future__ import annotations

import unittest

import pandas as pd

from q_lab_hl.portfolio import exposure_diagnostics, normalize_long_short_weights


class PortfolioTests(unittest.TestCase):
    def test_normalize_long_short_hits_targets_and_caps(self):
        raw = pd.Series({"A": 3.0, "B": 2.0, "C": -2.5, "D": -1.5})
        groups = pd.Series({"A": "L1", "B": "L2", "C": "L1", "D": "L2"})
        weights = normalize_long_short_weights(
            raw,
            gross_target=1.0,
            net_target=0.0,
            max_abs_weight=0.35,
            groups=groups,
            max_group_gross=0.60,
        )
        self.assertAlmostEqual(float(weights.sum()), 0.0, places=6)
        self.assertAlmostEqual(float(weights.abs().sum()), 1.0, places=6)
        self.assertLessEqual(float(weights.abs().max()), 0.35 + 1e-9)
        diag = exposure_diagnostics(weights, groups=groups)
        self.assertLessEqual(diag["max_group_gross"], 0.60 + 1e-9)


if __name__ == "__main__":
    unittest.main()
