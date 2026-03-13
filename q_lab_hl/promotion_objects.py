from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMOTION_POLICY_PATH = "autoresearch/promotion_policy.json"


@dataclass(frozen=True)
class StagePromotionPolicy:
    selector_policy: str = "best-accepted"
    require_accepted: bool = True
    require_result_artifact: bool = True
    require_matching_paper_candidate: bool = False


@dataclass(frozen=True)
class PromotionPolicy:
    policy_id: str = "promotion_policy_v1"
    version: int = 1
    paper: StagePromotionPolicy = StagePromotionPolicy()
    live: StagePromotionPolicy = StagePromotionPolicy(require_matching_paper_candidate=True)

    def summary(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "paper": asdict(self.paper),
            "live": asdict(self.live),
        }


@dataclass(frozen=True)
class PromotionRecord:
    stage: str
    policy_path: str
    policy_id: str
    selector_policy: str
    promoted_at: str
    promoted_from: str

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionEligibility:
    accepted_result: bool
    express_filter_passed: bool
    paper_eligible: bool
    live_eligible: bool
    reason: str

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def load_promotion_policy(path: str | Path = DEFAULT_PROMOTION_POLICY_PATH) -> PromotionPolicy:
    target = Path(path)
    if not target.exists():
        return PromotionPolicy()
    payload = json.loads(target.read_text())
    paper = StagePromotionPolicy(**payload.get("paper", {}))
    live = StagePromotionPolicy(**payload.get("live", {}))
    defaults = PromotionPolicy()
    return PromotionPolicy(
        policy_id=str(payload.get("policy_id") or defaults.policy_id),
        version=int(payload.get("version", defaults.version)),
        paper=paper,
        live=live,
    )


def build_promotion_eligibility(result: dict[str, Any]) -> PromotionEligibility:
    acceptance = result.get("acceptance") or {}
    express_filter = result.get("express_filter") or {}
    accepted_result = bool(acceptance.get("accepted"))
    express_filter_passed = bool(express_filter.get("passed", True))
    stage_eligible = accepted_result and express_filter_passed
    if not express_filter_passed:
        reason = "express_filter_failed"
    elif not accepted_result:
        reason = "judge_not_accepted"
    else:
        reason = "eligible"
    return PromotionEligibility(
        accepted_result=accepted_result,
        express_filter_passed=express_filter_passed,
        paper_eligible=stage_eligible,
        live_eligible=stage_eligible,
        reason=reason,
    )


def stage_is_eligible(payload: dict[str, Any], stage: str) -> bool:
    eligibility = payload.get("promotion_eligibility") or {}
    if not eligibility:
        return bool(payload.get("accepted"))
    return bool(eligibility.get(f"{stage}_eligible"))
