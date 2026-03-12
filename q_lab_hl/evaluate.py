from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from q_lab_hl.backtest import BacktestResult, run_backtest, strategy_warmup_timestamps
from q_lab_hl.config import ExecutionConfig, SplitConfig
from q_lab_hl.data import DataStore


@dataclass(frozen=True)
class TimeSlices:
    train: pd.DatetimeIndex
    inner: pd.DatetimeIndex
    outer: pd.DatetimeIndex
    test: pd.DatetimeIndex


def build_time_slices(index: pd.DatetimeIndex, split: SplitConfig | None = None) -> TimeSlices:
    split = split or SplitConfig()
    n = len(index)
    train_end = int(n * split.train_fraction)
    inner_end = train_end + int(n * split.inner_fraction)
    outer_end = inner_end + int(n * split.outer_fraction)
    return TimeSlices(
        train=pd.DatetimeIndex(index[:train_end]),
        inner=pd.DatetimeIndex(index[train_end:inner_end]),
        outer=pd.DatetimeIndex(index[inner_end:outer_end]),
        test=pd.DatetimeIndex(index[outer_end:]),
    )


def sharpe_annualized(returns: pd.Series, bars_per_year: int) -> float:
    rets = pd.Series(returns).dropna()
    if len(rets) < 2:
        return 0.0
    std = rets.std(ddof=1)
    if pd.isna(std) or std == 0:
        return 0.0
    return float(rets.mean() / std * np.sqrt(bars_per_year))


def annualized_return(returns: pd.Series, bars_per_year: int) -> float:
    rets = pd.Series(returns).dropna()
    if rets.empty:
        return 0.0
    total = float((1.0 + rets).prod())
    years = len(rets) / bars_per_year
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    rets = pd.Series(returns).fillna(0.0)
    curve = (1.0 + rets).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def beta_to_market(returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([pd.Series(returns), pd.Series(market_returns)], axis=1).dropna()
    if len(df) < 2:
        return 0.0
    market_var = df.iloc[:, 1].var(ddof=1)
    if pd.isna(market_var) or market_var == 0:
        return 0.0
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / market_var)


def rolling_median_sharpe(active_returns: pd.Series, bars_per_year: int, window: int = 24 * 7, step: int = 24) -> tuple[float, float]:
    active = pd.Series(active_returns).dropna()
    if len(active) < window:
        return 0.0, 0.0
    scores = []
    for start in range(0, len(active) - window + 1, step):
        scores.append(sharpe_annualized(active.iloc[start:start + window], bars_per_year))
    if not scores:
        return 0.0, 0.0
    return float(np.median(scores)), float(np.subtract(*np.percentile(scores, [75, 25])))


def inner_objective(result: BacktestResult, bars_per_year: int) -> tuple[float, dict]:
    median_active_sharpe, instability = rolling_median_sharpe(result.active_returns, bars_per_year)
    turnover = float(result.turnover.mean()) if not result.turnover.empty else 0.0
    beta = beta_to_market(result.returns, result.returns - result.active_returns)
    max_abs = float(result.diagnostics.get("final_weight_diagnostics", {}).get("max_abs_weight", 0.0))
    turnover_penalty = max(0.0, turnover - 0.35) * 1.25
    beta_penalty = max(0.0, abs(beta) - 0.10) * 3.0
    concentration_penalty = max(0.0, max_abs - 0.12) * 2.0
    instability_penalty = max(0.0, instability - 1.25) * 0.35
    score = median_active_sharpe - turnover_penalty - beta_penalty - concentration_penalty - instability_penalty
    return score, {
        "rolling_median_active_sharpe": median_active_sharpe,
        "rolling_active_sharpe_iqr": instability,
        "turnover_penalty": turnover_penalty,
        "beta_penalty": beta_penalty,
        "concentration_penalty": concentration_penalty,
        "instability_penalty": instability_penalty,
    }


def evaluate(
    strategy_module,
    data_store: DataStore,
    period: str,
    execution: ExecutionConfig | None = None,
) -> dict:
    execution = execution or ExecutionConfig()
    usable_index = strategy_warmup_timestamps(data_store, execution)
    slices = build_time_slices(usable_index)
    period_index = getattr(slices, period)
    if len(period_index) == 0:
        raise ValueError(f"No timestamps available for period '{period}'")
    return evaluate_timestamps(
        strategy_module,
        data_store,
        timestamps=period_index,
        execution=execution,
        period_label=period,
    )


def evaluate_timestamps(
    strategy_module,
    data_store: DataStore,
    timestamps: pd.DatetimeIndex,
    execution: ExecutionConfig | None = None,
    period_label: str = "custom",
) -> dict:
    execution = execution or ExecutionConfig()
    if len(timestamps) == 0:
        raise ValueError("No timestamps provided for evaluation")
    result = run_backtest(strategy_module, data_store, timestamps=pd.DatetimeIndex(timestamps), execution=execution)
    market_returns = result.returns - result.active_returns
    ci_low, ci_high = bootstrap_sharpe_ci(result.active_returns, execution.bars_per_year)
    metrics = {
        "period": period_label,
        "bars": len(timestamps),
        "annualized_return": annualized_return(result.returns, execution.bars_per_year),
        "active_annualized_return": annualized_return(result.active_returns, execution.bars_per_year),
        "sharpe_annualized": sharpe_annualized(result.returns, execution.bars_per_year),
        "active_sharpe_annualized": sharpe_annualized(result.active_returns, execution.bars_per_year),
        "max_drawdown": max_drawdown(result.returns),
        "turnover": float(result.turnover.mean()) if not result.turnover.empty else 0.0,
        "avg_gross_exposure": float(result.gross.mean()) if not result.gross.empty else 0.0,
        "avg_net_exposure": float(result.net.mean()) if not result.net.empty else 0.0,
        "beta_to_market": beta_to_market(result.returns, market_returns),
        "funding_pnl_share": _share(result.funding_pnl.sum(), result.returns.sum()),
        "fee_and_slippage_share": _share(result.cost_pnl.sum(), result.returns.sum()),
        "active_sharpe_ci": (ci_low, ci_high),
        "trade_count": int(result.diagnostics.get("trade_count", 0)),
        "filtered_assets_total": int(result.diagnostics.get("filtered_assets_total", 0)),
        "equity_curve": result.equity_curve,
        "active_returns_series": result.active_returns,
        "returns_series": result.returns,
    }
    score, parts = inner_objective(result, execution.bars_per_year)
    metrics["score_inner"] = score
    metrics.update(parts)
    return metrics


def format_metrics(metrics: dict) -> str:
    ordered = [
        "period",
        "bars",
        "score_inner",
        "sharpe_annualized",
        "active_sharpe_annualized",
        "annualized_return",
        "active_annualized_return",
        "max_drawdown",
        "turnover",
        "avg_gross_exposure",
        "avg_net_exposure",
        "beta_to_market",
        "funding_pnl_share",
        "fee_and_slippage_share",
        "active_sharpe_ci",
        "rolling_median_active_sharpe",
        "rolling_active_sharpe_iqr",
        "turnover_penalty",
        "beta_penalty",
        "concentration_penalty",
        "instability_penalty",
        "candidate_id",
    ]
    lines = []
    for key in ordered:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _share(part: float, total: float) -> float:
    if total == 0:
        return 0.0
    return float(part / total)


def bootstrap_sharpe_ci(returns: pd.Series, bars_per_year: int, n_boot: int = 200, seed: int = 7) -> tuple[float, float]:
    series = pd.Series(returns, dtype=float).dropna()
    if len(series) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    values = series.to_numpy(dtype=float)
    draws = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        std = float(np.std(sample, ddof=1))
        draws.append(0.0 if std == 0.0 else float(np.mean(sample) / std * np.sqrt(bars_per_year)))
    low, high = np.percentile(draws, [5, 95])
    return float(low), float(high)
