from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LEADERBOARD_PATH = "autoresearch/leaderboard.jsonl"
DEFAULT_RESULTS_DIR = "autoresearch/results"
DEFAULT_RESEARCH_POLICY_PATH = "autoresearch/research_policy.json"


@dataclass(frozen=True)
class ResearchPolicy:
    policy_id: str = "quant_autoresearch_v1"
    version: int = 1
    mission: str = (
        "Constrained autoresearch for quant strategies with a fixed judge, "
        "gated promotion, and automatic execution only for promoted champions."
    )
    mutable_paths: tuple[str, ...] = (
        "strategy.py",
        "strategy_model.py",
        "autoresearch/",
    )
    fixed_paths: tuple[str, ...] = (
        "q_lab_hl/data.py",
        "q_lab_hl/backtest.py",
        "q_lab_hl/evaluate.py",
        "q_lab_hl/config.py",
        "execution/",
        "run.py",
    )
    selection_contract: str = (
        "Candidates must pass a fast express filter and then the full fixed judge before promotion."
    )

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyFamily:
    family_id: str = "linear_cross_section_v1"
    description: str = (
        "Approved family of trainable cross-sectional linear models on lagged Hyperliquid perp features."
    )
    mutable_parameters: tuple[str, ...] = (
        "strategy_spec.features",
        "strategy_spec.target",
        "strategy_spec.model",
        "strategy_spec.train_window_bars",
        "strategy_spec.min_train_rows",
        "strategy_spec.position_bucket",
        "execution_overrides.rebalance_every_bars",
    )

    def summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptancePolicy:
    primary_metric: str = "periods.test.sharpe_annualized"
    primary_min: float = 0.3
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
class ExpressFilterConfig:
    enabled: bool = True
    period: str = "outer"
    trailing_bars: int = 24 * 120
    max_assets: int = 20
    bootstrap_samples: int = 50
    primary_metric: str = "sharpe_annualized"
    primary_min: float = -0.5
    max_beta_abs: float = 0.25
    max_turnover: float = 1.0


@dataclass(frozen=True)
class CandidateSpec:
    experiment_id: str = "unnamed_experiment"
    candidate_id: str = "unnamed_candidate"
    hypothesis: str = ""
    strategy_path: str = "strategy.py"
    strategy_family: str = StrategyFamily().family_id
    research_policy_path: str = DEFAULT_RESEARCH_POLICY_PATH
    strategy_spec: dict[str, Any] | None = None
    execution_overrides: dict[str, Any] | None = None
    data_dir: str = "data/active_1h/machine"
    synthetic: bool = False
    evaluation_periods: tuple[str, ...] = ("inner", "outer", "test")
    notes: str = ""
    enable_walk_forward: bool = True
    walk_forward_runway_bars: int = 504
    walk_forward_eval_bars: int = 500
    walk_forward_step_bars: int = 250
    express_filter: ExpressFilterConfig = field(default_factory=ExpressFilterConfig)
    acceptance: AcceptancePolicy = field(default_factory=AcceptancePolicy)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation_periods"] = list(self.evaluation_periods)
        return payload


# Compatibility alias during migration of the autoresearch pipeline.
ExperimentSpec = CandidateSpec


def load_research_policy(path: str | Path = DEFAULT_RESEARCH_POLICY_PATH) -> ResearchPolicy:
    payload = _read_json_if_exists(path)
    if payload is None:
        return ResearchPolicy()
    defaults = ResearchPolicy()
    return ResearchPolicy(
        policy_id=str(payload.get("policy_id") or defaults.policy_id),
        version=int(payload.get("version", defaults.version)),
        mission=str(payload.get("mission") or defaults.mission),
        mutable_paths=tuple(payload.get("mutable_paths", defaults.mutable_paths)),
        fixed_paths=tuple(payload.get("fixed_paths", defaults.fixed_paths)),
        selection_contract=str(payload.get("selection_contract") or defaults.selection_contract),
    )


def load_strategy_family(payload: dict[str, Any] | None = None) -> StrategyFamily:
    payload = payload or {}
    defaults = StrategyFamily()
    return StrategyFamily(
        family_id=str(payload.get("family_id") or defaults.family_id),
        description=str(payload.get("description") or defaults.description),
        mutable_parameters=tuple(payload.get("mutable_parameters", defaults.mutable_parameters)),
    )


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    import json

    return json.loads(target.read_text())
