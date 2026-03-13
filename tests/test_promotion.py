from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from execution.select_champion import (
    build_champion_payload,
    build_promotion_record,
    select_stage_record,
    validate_champion_payload,
    validate_result_for_stage,
    validate_stage_candidate_match,
)
from q_lab_hl.promotion_objects import StagePromotionPolicy


class PromotionTests(unittest.TestCase):
    def test_build_champion_payload_and_validate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
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
            promotion = build_promotion_record(
                stage="paper",
                policy_path="autoresearch/promotion_policy.json",
                policy_id="promotion_policy_v1",
                selector_policy="best-accepted",
                promoted_from="autoresearch/leaderboard.jsonl",
                promoted_at="2026-03-13T00:00:00+00:00",
            )
            payload = build_champion_payload(result, promotion=promotion, existing={"source": {"note": "keep"}})
            self.assertEqual(payload["source"]["note"], "keep")
            self.assertEqual(payload["promotion"]["stage"], "paper")
            self.assertEqual(validate_champion_payload(payload), result_path)

    def test_select_stage_record_requires_eligibility(self):
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
            StagePromotionPolicy(),
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

    def test_validate_stage_candidate_match_rejects_live_mismatch(self):
        with self.assertRaises(ValueError):
            validate_stage_candidate_match(
                record={"candidate_id": "cand-live"},
                stage_policy=StagePromotionPolicy(require_matching_paper_candidate=True),
                paper_candidate_id="cand-paper",
            )


if __name__ == "__main__":
    unittest.main()
