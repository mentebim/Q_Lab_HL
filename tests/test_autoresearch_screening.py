from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from q_lab_hl.autoresearch import ExperimentSpec, RecordingConfig, run_experiment
from q_lab_hl.data import DataStore


class AutoResearchScreeningTests(unittest.TestCase):
    def test_screening_window_and_liquidity_subset_reduce_dataset(self):
        store = DataStore.synthetic(n_assets=20, periods=24 * 80, seed=13)
        with tempfile.TemporaryDirectory() as tmp:
            spec = ExperimentSpec(
                experiment_id="screening_smoke",
                candidate_id="screening_smoke",
                synthetic=True,
                evaluation_periods=("outer",),
                data_window_bars=24 * 20,
                liquidity_top_n=8,
                recording=RecordingConfig(
                    leaderboard_path=str(Path(tmp) / "leaderboard.jsonl"),
                    results_dir=str(Path(tmp) / "results"),
                    append_leaderboard=False,
                    write_result=False,
                ),
            )
            result = run_experiment(spec, data_store=store)
            self.assertEqual(result["data"]["bars"], 24 * 20)
            self.assertEqual(result["data"]["assets"], 8)
            self.assertEqual(result["data"]["data_window_bars"], 24 * 20)
            self.assertEqual(result["data"]["liquidity_top_n"], 8)


if __name__ == "__main__":
    unittest.main()
