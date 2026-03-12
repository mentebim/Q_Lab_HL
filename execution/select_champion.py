from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def build_champion_payload(result: dict, existing: dict | None = None) -> dict:
    existing = existing or {}
    live = dict(existing.get("live", {}))
    spec = result.get("spec", {})
    return {
        "source": {
            "policy": existing.get("source", {}).get("policy", "manual"),
            "experiment_id": result.get("experiment_id"),
            "candidate_id": result.get("candidate_id"),
            "result_path": result.get("result_path"),
        },
        "strategy_path": spec.get("strategy_path", result.get("strategy_path", "strategy.py")),
        "strategy_spec": spec.get("strategy_spec"),
        "execution_overrides": spec.get("execution_overrides"),
        "data_dir": spec.get("data_dir", "data/market_cache_1h"),
        "live": {
            "mode": live.get("mode", "paper"),
            "network": live.get("network", "mainnet"),
            "account_address": live.get("account_address"),
            "vault_address": live.get("vault_address"),
            "secret_key_env": live.get("secret_key_env", "HL_SECRET_KEY"),
            "paper_account_value": live.get("paper_account_value", 10000.0),
            "min_trade_notional_usd": live.get("min_trade_notional_usd", 25.0),
            "max_single_order_notional_usd": live.get("max_single_order_notional_usd", 500.0),
            "slippage": live.get("slippage", 0.01),
            "state_path": live.get("state_path", "execution/state.json"),
            "log_dir": live.get("log_dir", "execution/logs"),
            "kill_switch_path": live.get("kill_switch_path", "execution/STOP"),
            "max_data_lag_hours": live.get("max_data_lag_hours", 3.0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin a live champion from autoresearch results.")
    parser.add_argument("--leaderboard", default="autoresearch/leaderboard.jsonl")
    parser.add_argument("--policy", choices=["latest-accepted", "best-accepted", "best-any"], default="best-accepted")
    parser.add_argument("--out", default="execution/champion.json")
    args = parser.parse_args()

    rows = load_rows(args.leaderboard)
    record = select_record(rows, args.policy)
    if record is None:
        raise SystemExit(f"No leaderboard entry matched policy `{args.policy}`.")
    result_path = record.get("result_path")
    if not result_path:
        raise SystemExit("Selected leaderboard row has no result_path; cannot build champion file.")
    result = json.loads(Path(result_path).read_text())
    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else None
    payload = build_champion_payload(result, existing=existing)
    payload["source"]["policy"] = args.policy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"status": "ok", "out": str(out_path), "candidate_id": payload["source"]["candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
