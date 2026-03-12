from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from q_lab_hl.backtest import load_strategy
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore
from q_lab_hl.evaluate import evaluate


DEFAULT_LEADERBOARD_PATH = "autoresearch/leaderboard.jsonl"
DEFAULT_RESULTS_DIR = "autoresearch/results"


@dataclass(frozen=True)
class AcceptancePolicy:
    primary_metric: str = "periods.outer.active_sharpe_annualized"
    primary_min: float = 0.0
    min_active_sharpe: float = 0.0
    max_beta_abs: float = 0.15
    max_turnover: float = 0.75
    compare_to_best: bool = False
    compare_to_candidate_id: str | None = None
    min_primary_lift: float = 0.0


@dataclass(frozen=True)
class RecordingConfig:
    leaderboard_path: str = DEFAULT_LEADERBOARD_PATH
    results_dir: str = DEFAULT_RESULTS_DIR
    append_leaderboard: bool = True
    write_result: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str = "unnamed_experiment"
    candidate_id: str = "unnamed_candidate"
    hypothesis: str = ""
    strategy_path: str = "strategy.py"
    strategy_spec: dict[str, Any] | None = None
    execution_overrides: dict[str, Any] | None = None
    data_dir: str = "data/market_cache_1h"
    synthetic: bool = False
    evaluation_periods: tuple[str, ...] = ("inner", "outer")
    notes: str = ""
    acceptance: AcceptancePolicy = AcceptancePolicy()
    recording: RecordingConfig = RecordingConfig()


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    payload = json.loads(Path(path).read_text())
    return ExperimentSpec(
        experiment_id=str(payload.get("experiment_id") or ExperimentSpec.experiment_id),
        candidate_id=str(payload.get("candidate_id") or ExperimentSpec.candidate_id),
        hypothesis=str(payload.get("hypothesis") or ""),
        strategy_path=str(payload.get("strategy_path") or ExperimentSpec.strategy_path),
        strategy_spec=payload.get("strategy_spec"),
        execution_overrides=payload.get("execution_overrides"),
        data_dir=str(payload.get("data_dir") or ExperimentSpec.data_dir),
        synthetic=bool(payload.get("synthetic", False)),
        evaluation_periods=tuple(payload.get("evaluation_periods", ExperimentSpec.evaluation_periods)),
        notes=str(payload.get("notes") or ""),
        acceptance=AcceptancePolicy(**payload.get("acceptance", {})),
        recording=RecordingConfig(**payload.get("recording", {})),
    )


def override_spec(
    spec: ExperimentSpec,
    *,
    experiment_id: str | None = None,
    candidate_id: str | None = None,
    hypothesis: str | None = None,
    strategy_path: str | None = None,
    strategy_spec: dict[str, Any] | None = None,
    execution_overrides: dict[str, Any] | None = None,
    data_dir: str | None = None,
    synthetic: bool | None = None,
) -> ExperimentSpec:
    updates = {}
    if experiment_id is not None:
        updates["experiment_id"] = experiment_id
    if candidate_id is not None:
        updates["candidate_id"] = candidate_id
    if hypothesis is not None:
        updates["hypothesis"] = hypothesis
    if strategy_path is not None:
        updates["strategy_path"] = strategy_path
    if strategy_spec is not None:
        updates["strategy_spec"] = strategy_spec
    if execution_overrides is not None:
        updates["execution_overrides"] = execution_overrides
    if data_dir is not None:
        updates["data_dir"] = data_dir
    if synthetic is not None:
        updates["synthetic"] = synthetic
    return replace(spec, **updates)


def run_experiment(
    spec: ExperimentSpec,
    data_store: DataStore | None = None,
    write_result: bool | None = None,
    append_leaderboard: bool | None = None,
) -> dict[str, Any]:
    write_result = spec.recording.write_result if write_result is None else write_result
    append_leaderboard = spec.recording.append_leaderboard if append_leaderboard is None else append_leaderboard
    strategy = load_strategy(spec.strategy_path)
    if hasattr(strategy, "apply_runtime_overrides"):
        strategy.apply_runtime_overrides(strategy_spec=spec.strategy_spec, execution_overrides=spec.execution_overrides)
    execution = getattr(strategy, "EXECUTION", ExecutionConfig())
    data_store = data_store or _load_data_store(spec)
    leaderboard = load_leaderboard(spec.recording.leaderboard_path)
    result: dict[str, Any] = {
        "timestamp": now_utc_iso(),
        "experiment_id": spec.experiment_id,
        "candidate_id": spec.candidate_id,
        "hypothesis": spec.hypothesis,
        "notes": spec.notes,
        "strategy_path": spec.strategy_path,
        "strategy_hash": strategy_hash(spec.strategy_path),
        "git_commit": git_commit(),
        "data": {
            "mode": "synthetic" if spec.synthetic else "parquet",
            "path": None if spec.synthetic else spec.data_dir,
            "bars": len(data_store.index),
            "assets": len(data_store.assets),
        },
        "execution": _jsonable(asdict(execution)),
        "spec": experiment_spec_to_dict(spec),
        "periods": {},
    }
    for period in spec.evaluation_periods:
        metrics = evaluate(strategy, data_store, period=period, execution=execution)
        result["periods"][period] = compact_metrics(metrics)
    if hasattr(strategy, "last_fit_summary"):
        result["model_fit"] = _jsonable(strategy.last_fit_summary())
    result["acceptance"] = evaluate_acceptance(result, spec.acceptance, leaderboard)
    if write_result:
        output_path = write_experiment_result(result, spec.recording.results_dir)
        result["result_path"] = str(output_path)
    if append_leaderboard:
        append_leaderboard_entry(leaderboard_record(result), spec.recording.leaderboard_path)
    return result


def experiment_spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["evaluation_periods"] = list(spec.evaluation_periods)
    return _jsonable(payload)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in metrics.items()
        if not isinstance(value, (pd.Series, pd.DataFrame))
    }


def evaluate_acceptance(result: dict[str, Any], policy: AcceptancePolicy, leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
    judged_period = preferred_period(result.get("periods", {}))
    period_metrics = result.get("periods", {}).get(judged_period, {})
    primary_value = as_float(resolve_metric_path(result, policy.primary_metric))
    failed_checks: list[str] = []
    review_reasons: list[str] = []
    if primary_value is None or primary_value < policy.primary_min:
        failed_checks.append("primary_metric")
    if as_float(period_metrics.get("active_sharpe_annualized"), default=-float("inf")) < policy.min_active_sharpe:
        failed_checks.append(f"{judged_period}_active_sharpe")
    if abs(as_float(period_metrics.get("beta_to_market"), default=float("inf"))) > policy.max_beta_abs:
        failed_checks.append(f"{judged_period}_beta")
    if as_float(period_metrics.get("turnover"), default=float("inf")) > policy.max_turnover:
        failed_checks.append(f"{judged_period}_turnover")
    reference = select_reference_record(leaderboard, policy, policy.primary_metric)
    reference_value = None
    if policy.compare_to_best or policy.compare_to_candidate_id:
        if reference is None:
            review_reasons.append("reference_missing")
        else:
            reference_value = as_float(resolve_metric_path(reference, policy.primary_metric))
            if primary_value is None or reference_value is None or primary_value < reference_value + policy.min_primary_lift:
                failed_checks.append("reference_comparison")
    status = "accepted"
    if failed_checks:
        status = "rejected"
    elif review_reasons:
        status = "needs_review"
    return {
        "status": status,
        "accepted": status == "accepted",
        "judged_period": judged_period,
        "primary_metric": policy.primary_metric,
        "primary_metric_value": primary_value,
        "failed_checks": failed_checks,
        "review_reasons": review_reasons,
        "reference_experiment_id": None if reference is None else reference.get("experiment_id"),
        "reference_candidate_id": None if reference is None else reference.get("candidate_id"),
        "reference_primary_value": reference_value,
    }


def preferred_period(periods: dict[str, Any]) -> str:
    for name in ("outer", "test", "inner", "train"):
        if name in periods:
            return name
    return next(iter(periods), "inner")


def select_reference_record(
    leaderboard: list[dict[str, Any]],
    policy: AcceptancePolicy,
    metric_path: str,
) -> dict[str, Any] | None:
    if policy.compare_to_candidate_id:
        matches = [row for row in leaderboard if row.get("candidate_id") == policy.compare_to_candidate_id]
        return matches[-1] if matches else None
    if not policy.compare_to_best:
        return None
    best_row = None
    best_value = None
    for row in leaderboard:
        value = as_float(resolve_metric_path(row, metric_path))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_row = row
            best_value = value
    return best_row


def leaderboard_record(result: dict[str, Any]) -> dict[str, Any]:
    judged_period = result["acceptance"]["judged_period"]
    return {
        "timestamp": result["timestamp"],
        "experiment_id": result["experiment_id"],
        "candidate_id": result["candidate_id"],
        "hypothesis": result["hypothesis"],
        "strategy_path": result["strategy_path"],
        "strategy_hash": result["strategy_hash"],
        "git_commit": result["git_commit"],
        "status": result["acceptance"]["status"],
        "accepted": result["acceptance"]["accepted"],
        "primary_metric": result["acceptance"]["primary_metric"],
        "primary_metric_value": result["acceptance"]["primary_metric_value"],
        "periods": result["periods"],
        "model_fit": result.get("model_fit"),
        "summary": {
            "judged_period": judged_period,
            "active_sharpe_annualized": result["periods"].get(judged_period, {}).get("active_sharpe_annualized"),
            "beta_to_market": result["periods"].get(judged_period, {}).get("beta_to_market"),
            "turnover": result["periods"].get(judged_period, {}).get("turnover"),
        },
        "acceptance": result["acceptance"],
        "result_path": result.get("result_path"),
    }


def load_leaderboard(path: str | Path) -> list[dict[str, Any]]:
    leaderboard_path = Path(path)
    if not leaderboard_path.exists():
        return []
    rows = []
    for line in leaderboard_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_leaderboard_entry(record: dict[str, Any], path: str | Path) -> None:
    leaderboard_path = Path(path)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with leaderboard_path.open("a") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True))
        handle.write("\n")


def write_experiment_result(result: dict[str, Any], results_dir: str | Path) -> Path:
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    filename = slugify(f"{result['timestamp']}_{result['experiment_id']}_{result['acceptance']['status']}") + ".json"
    output_path = results_root / filename
    payload = dict(result)
    payload["result_path"] = str(output_path)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return output_path


def resolve_metric_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def strategy_hash(path: str | Path) -> str:
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _load_data_store(spec: ExperimentSpec) -> DataStore:
    if spec.synthetic:
        return DataStore.synthetic(n_assets=16, periods=24 * 25, seed=7)
    return DataStore.from_parquet_dir(spec.data_dir)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
