from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_lab_hl.promotion_objects import (
    DEFAULT_PROMOTION_POLICY_PATH,
    PromotionRecord,
    StagePromotionPolicy,
    stage_is_eligible,
    load_promotion_policy,
)


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def select_record(rows: list[dict], policy: str) -> dict | None:
    accepted = [row for row in rows if bool(row.get("accepted"))]
    if policy == "latest-accepted":
        return accepted[-1] if accepted else None
    if policy == "best-accepted":
        if not accepted:
            return None
        return max(accepted, key=lambda row: float(row.get("primary_metric_value", float("-inf"))))
    if policy == "best-any":
        return max(rows, key=lambda row: float(row.get("primary_metric_value", float("-inf"))), default=None)
    raise ValueError(f"Unsupported policy '{policy}'")


def select_stage_record(
    rows: list[dict],
    stage: str,
    stage_policy: StagePromotionPolicy,
) -> dict | None:
    record = select_record(rows, stage_policy.selector_policy)
    if record is None:
        return None
    if stage_policy.require_accepted and not bool(record.get("accepted")):
        return None
    if not stage_is_eligible(record, stage):
        return None
    return record


def build_promotion_record(
    *,
    stage: str,
    policy_path: str,
    policy_id: str,
    selector_policy: str,
    promoted_from: str,
    promoted_at: str | None = None,
) -> PromotionRecord:
    return PromotionRecord(
        stage=stage,
        policy_path=policy_path,
        policy_id=policy_id,
        selector_policy=selector_policy,
        promoted_at=promoted_at or datetime.now(timezone.utc).isoformat(),
        promoted_from=promoted_from,
    )


def build_champion_payload(
    result: dict,
    *,
    promotion: PromotionRecord | dict,
    existing: dict | None = None,
) -> dict:
    existing = existing or {}
    live = dict(existing.get("live", {}))
    source = dict(existing.get("source", {}))
    promotion_payload = promotion.summary() if isinstance(promotion, PromotionRecord) else dict(promotion)
    spec = result.get("spec", {})
    live_defaults = {
        "mode": "paper",
        "network": "mainnet",
        "account_address": None,
        "vault_address": None,
        "secret_key_env": "HL_SECRET_KEY",
        "paper_account_value": 10000.0,
        "min_trade_notional_usd": 25.0,
        "max_single_order_notional_usd": 500.0,
        "slippage": 0.01,
        "state_path": "execution/state.json",
        "log_dir": "execution/logs",
        "kill_switch_path": "execution/STOP",
        "max_data_lag_hours": 3.0,
        "default_leverage": 2,
        "leverage_overrides": {"BTC": 3, "ETH": 3},
        "target_gross_notional_usd": None,
        "target_margin_usage_ratio": None,
        "max_margin_usage_ratio": 0.80,
        "min_margin_headroom_usd": 50.0,
    }
    return {
        "promotion": promotion_payload,
        "source": {
            "experiment_id": result.get("experiment_id"),
            "candidate_id": result.get("candidate_id"),
            "result_path": result.get("result_path"),
            **({"note": source["note"]} if "note" in source else {}),
        },
        "strategy_path": spec.get("strategy_path", result.get("strategy_path", "strategy.py")),
        "strategy_spec": spec.get("strategy_spec"),
        "execution_overrides": spec.get("execution_overrides"),
        "data_dir": spec.get("data_dir", "data/market_cache_1h"),
        "live": {**live_defaults, **live},
    }


def validate_champion_payload(champion: dict) -> Path:
    promotion = champion.get("promotion") or {}
    if promotion:
        for field in ("stage", "policy_path", "policy_id", "selector_policy", "promoted_at", "promoted_from"):
            if not promotion.get(field):
                raise ValueError(f"Champion config is missing promotion.{field}.")
        if promotion["stage"] not in {"paper", "live"}:
            raise ValueError(f"Champion promotion stage must be 'paper' or 'live', got {promotion['stage']!r}.")
    source = champion.get("source") or {}
    result_path = source.get("result_path")
    if not result_path:
        raise ValueError("Champion config is missing source.result_path.")
    artifact = Path(result_path)
    if not artifact.exists():
        raise FileNotFoundError(f"Champion result artifact does not exist: {artifact}")
    result = json.loads(artifact.read_text())
    for field in ("experiment_id", "candidate_id"):
        expected = source.get(field)
        actual = result.get(field)
        if expected and actual and expected != actual:
            raise ValueError(
                f"Champion source {field} mismatch: champion={expected!r} artifact={actual!r}"
            )
    return artifact


def load_current_candidate_id(path: str | Path) -> str | None:
    champion_path = Path(path)
    if not champion_path.exists():
        return None
    payload = json.loads(champion_path.read_text())
    validate_champion_payload(payload)
    return (payload.get("source") or {}).get("candidate_id")


def validate_stage_candidate_match(
    *,
    record: dict,
    stage_policy: StagePromotionPolicy,
    paper_candidate_id: str | None,
) -> None:
    if not stage_policy.require_matching_paper_candidate:
        return
    if not paper_candidate_id:
        raise ValueError("Live promotion requires an existing paper champion candidate.")
    if record.get("candidate_id") != paper_candidate_id:
        raise ValueError(
            "Live promotion candidate does not match the current paper champion candidate: "
            f"selected={record.get('candidate_id')!r} paper={paper_candidate_id!r}"
        )


def validate_result_for_stage(
    *,
    result: dict,
    stage: str,
    stage_policy: StagePromotionPolicy,
) -> None:
    if stage_policy.require_accepted and not bool((result.get("acceptance") or {}).get("accepted")):
        raise ValueError(f"Selected result artifact is not accepted and cannot be promoted to {stage}.")
    if not stage_is_eligible(result, stage):
        raise ValueError(f"Selected result artifact is not promotion-eligible for stage {stage}.")


def infer_stage_from_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    if ".live." in name or "live" in name:
        return "live"
    if ".paper." in name or "paper" in name:
        return "paper"
    return "paper"


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a paper or live champion from autoresearch results.")
    parser.add_argument("--leaderboard", default="autoresearch/leaderboard.jsonl")
    parser.add_argument("--policy", choices=["latest-accepted", "best-accepted", "best-any"], default=None)
    parser.add_argument("--stage", choices=["paper", "live"], default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--promotion-policy", default=DEFAULT_PROMOTION_POLICY_PATH)
    parser.add_argument("--paper-champion", default="execution/champion.paper.json")
    args = parser.parse_args()

    out_path = Path(args.out or f"execution/champion.{(args.stage or 'paper')}.json")
    stage = args.stage or infer_stage_from_path(out_path)
    promotion_policy = load_promotion_policy(args.promotion_policy)
    stage_policy = promotion_policy.paper if stage == "paper" else promotion_policy.live
    selector_policy = args.policy or stage_policy.selector_policy
    stage_policy = StagePromotionPolicy(
        selector_policy=selector_policy,
        require_accepted=stage_policy.require_accepted,
        require_result_artifact=stage_policy.require_result_artifact,
        require_matching_paper_candidate=stage_policy.require_matching_paper_candidate,
    )
    rows = load_rows(args.leaderboard)
    record = select_stage_record(rows, stage, stage_policy)
    if record is None:
        raise SystemExit(f"No leaderboard entry matched promotion stage `{stage}` with selector `{stage_policy.selector_policy}`.")
    try:
        validate_stage_candidate_match(
            record=record,
            stage_policy=stage_policy,
            paper_candidate_id=load_current_candidate_id(args.paper_champion),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result_path = record.get("result_path")
    if not result_path:
        raise SystemExit("Selected leaderboard row has no result_path; cannot build champion file.")
    artifact = Path(result_path)
    if stage_policy.require_result_artifact and not artifact.exists():
        raise SystemExit(f"Selected leaderboard result_path does not exist: {artifact}")
    result = json.loads(artifact.read_text())
    try:
        validate_result_for_stage(result=result, stage=stage, stage_policy=stage_policy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    existing = json.loads(out_path.read_text()) if out_path.exists() else None
    promotion = build_promotion_record(
        stage=stage,
        policy_path=args.promotion_policy,
        policy_id=promotion_policy.policy_id,
        selector_policy=stage_policy.selector_policy,
        promoted_from=args.leaderboard,
    )
    payload = build_champion_payload(result, promotion=promotion, existing=existing)
    validate_champion_payload(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(out_path),
                "stage": stage,
                "candidate_id": payload["source"]["candidate_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
