from __future__ import annotations


def _to_float(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate_for_paper(inner_metrics: dict, audit_metrics: dict) -> dict:
    failed = []
    if _to_float(inner_metrics, "score_inner") <= 0.0:
        failed.append("score_inner")
    if _to_float(inner_metrics, "active_sharpe_annualized") <= 0.5:
        failed.append("inner_active_sharpe")
    if abs(_to_float(inner_metrics, "beta_to_market")) > 0.10:
        failed.append("inner_beta")
    if _to_float(audit_metrics, "DSR") <= 0.0:
        failed.append("audit_dsr")
    ci = audit_metrics.get("bootstrap_sharpe_ci", (0.0, 0.0))
    try:
        ci_low = float(ci[0])
    except (TypeError, ValueError, IndexError):
        ci_low = _to_float(audit_metrics, "bootstrap_sharpe_ci_low")
    if ci_low <= 0.0:
        failed.append("audit_ci")
    approved = not failed
    return {
        "approved": approved,
        "status": "approved_for_paper" if approved else "rejected_for_paper",
        "failed_checks": failed,
    }

