from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from q_lab_hl.artifacts import package_candidate_artifact
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.promotion import evaluate_for_paper
from q_lab_hl.registries import audit_registry, research_registry


class PromotionTests(unittest.TestCase):
    def test_promotion_approves_strong_candidate(self):
        inner = {
            "score_inner": 0.5,
            "active_sharpe_annualized": 1.2,
            "turnover": 0.4,
            "beta_to_market": 0.02,
            "max_drawdown": -0.08,
        }
        audit = {
            "DSR": 0.8,
            "bootstrap_sharpe_ci": (0.2, 1.4),
            "active_sharpe_annualized": 0.9,
        }
        decision = evaluate_for_paper(inner, audit)
        self.assertTrue(decision["approved"])
        self.assertEqual(decision["status"], "approved_for_paper")

    def test_artifact_packaging_writes_manifest_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strategy_path = root / "strategy.py"
            strategy_path.write_text("def signals(data, ts):\n    return None\n")
            artifact_dir = package_candidate_artifact(
                candidate_id="cand_test",
                strategy_path=strategy_path,
                inner_metrics={"score_inner": 0.1},
                audit_metrics={"DSR": 0.7},
                execution=ExecutionConfig(),
                artifact_root=root / "artifacts",
            )
            self.assertTrue((artifact_dir / "strategy.py").exists())
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(manifest["artifact_id"], "cand_test")
            metrics = json.loads((artifact_dir / "metrics.json").read_text())
            self.assertEqual(metrics["audit"]["DSR"], 0.7)

    def test_registries_round_trip_latest_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            research = research_registry(tmp)
            audit = audit_registry(tmp)
            research.upsert(
                {
                    "candidate_id": "cand_a",
                    "created_at": "2026-03-10T00:00:00+00:00",
                    "strategy_hash": "sha256:1",
                    "git_commit": "abc123",
                    "period": "inner",
                    "score_inner": 0.3,
                    "active_sharpe_annualized": 1.0,
                    "turnover": 0.4,
                    "beta_to_market": 0.01,
                    "max_drawdown": -0.05,
                    "status": "backtested",
                    "notes": "test",
                }
            )
            audit.upsert(
                {
                    "candidate_id": "cand_a",
                    "audited_at": "2026-03-10T01:00:00+00:00",
                    "strategy_hash": "sha256:1",
                    "git_commit": "abc123",
                    "period": "outer",
                    "DSR": 0.7,
                    "bootstrap_sharpe_ci_low": 0.1,
                    "bootstrap_sharpe_ci_high": 1.1,
                    "active_sharpe_annualized": 0.8,
                    "turnover": 0.4,
                    "beta_to_market": 0.01,
                    "max_drawdown": -0.06,
                    "promotable": False,
                    "status": "audited",
                }
            )
            self.assertEqual(research.latest_for_candidate("cand_a")["status"], "backtested")
            self.assertEqual(audit.latest_for_candidate("cand_a")["DSR"], 0.7)


if __name__ == "__main__":
    unittest.main()
