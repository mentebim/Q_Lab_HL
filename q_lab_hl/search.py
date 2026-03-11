from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from q_lab_hl.config import ExecutionConfig, SplitConfig
from q_lab_hl.data import default_universe_kwargs
from q_lab_hl.evaluate import TimeSlices, build_time_slices, evaluate_timestamps


@dataclass(frozen=True)
class TimeSeriesCVConfig:
    mode: str = "walk_forward"
    n_folds: int = 3
    train_size: int | None = None
    validation_size: int | None = None
    step_size: int | None = None
    gap_size: int = 0
    purge_size: int = 0
    embargo_size: int = 0


def enumerate_param_grid(param_grid: dict[str, list]) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    values = [list(param_grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def build_walk_forward_folds(
    index: pd.DatetimeIndex,
    split: SplitConfig | None = None,
    n_folds: int = 3,
) -> list[dict]:
    split = split or SplitConfig()
    slices: TimeSlices = build_time_slices(index, split)
    search_index = pd.DatetimeIndex(slices.train.append(slices.inner))
    if len(search_index) < n_folds + 1:
        return []
    blocks = [chunk for chunk in np.array_split(search_index, n_folds + 1) if len(chunk) > 0]
    folds = []
    for i in range(1, len(blocks)):
        train = pd.DatetimeIndex(np.concatenate(blocks[:i]))
        validation = pd.DatetimeIndex(blocks[i])
        if len(train) == 0 or len(validation) == 0:
            continue
        folds.append({"fold_id": i, "train": train, "validation": validation})
    return folds


def build_time_series_cv_folds(
    index: pd.DatetimeIndex,
    cv: TimeSeriesCVConfig | None = None,
    split: SplitConfig | None = None,
) -> list[dict]:
    cv = cv or TimeSeriesCVConfig()
    if cv.mode == "walk_forward":
        folds = build_walk_forward_folds(index, split=split, n_folds=cv.n_folds)
        return _apply_purge_to_folds(folds, purge_size=cv.purge_size + cv.gap_size)
    usable = pd.DatetimeIndex(index)
    if len(usable) == 0:
        return []
    validation_size = int(cv.validation_size or max(1, len(usable) // (cv.n_folds + 1)))
    if cv.mode == "expanding":
        train_size = int(cv.train_size or max(1, len(usable) // (cv.n_folds + 1)))
        step_size = int(cv.step_size or (validation_size + cv.embargo_size))
        return _build_expanding_folds(usable, train_size, validation_size, step_size, int(cv.gap_size + cv.purge_size), int(cv.n_folds))
    if cv.mode == "rolling":
        if cv.train_size is None:
            raise ValueError("rolling CV requires train_size")
        step_size = int(cv.step_size or (validation_size + cv.embargo_size))
        return _build_rolling_folds(usable, int(cv.train_size), validation_size, step_size, int(cv.gap_size + cv.purge_size), int(cv.n_folds))
    raise ValueError(f"Unknown CV mode '{cv.mode}'")


def run_walk_forward_grid(
    strategy_module,
    data_store,
    execution: ExecutionConfig | None = None,
    split: SplitConfig | None = None,
    n_folds: int = 3,
    param_grid: dict[str, list] | None = None,
    cv: TimeSeriesCVConfig | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    execution = execution or getattr(strategy_module, "EXECUTION", ExecutionConfig())
    cv = cv or TimeSeriesCVConfig(mode="walk_forward", n_folds=n_folds)
    param_grid = param_grid if param_grid is not None else getattr(strategy_module, "PARAM_GRID", {})
    configs = enumerate_param_grid(param_grid)
    rows = []
    last_folds = []
    for config_id, params in enumerate(configs, start=1):
        usable_index = usable_strategy_index(strategy_module, data_store, params, execution=execution)
        folds = build_time_series_cv_folds(usable_index, cv=cv, split=split)
        last_folds = folds
        fold_metrics = []
        with temporary_module_params(strategy_module, params):
            for fold in folds:
                train_metrics = evaluate_timestamps(strategy_module, data_store, timestamps=fold["train"], execution=execution, period_label=f"wf_train_{fold['fold_id']}")
                validation_metrics = evaluate_timestamps(strategy_module, data_store, timestamps=fold["validation"], execution=execution, period_label=f"wf_val_{fold['fold_id']}")
                fold_metrics.append(
                    {
                        "fold_id": fold["fold_id"],
                        "train_score": train_metrics.get("score_inner", 0.0),
                        "train_active_sharpe": train_metrics.get("active_sharpe_annualized", 0.0),
                        "validation_score": validation_metrics.get("score_inner", 0.0),
                        "validation_active_sharpe": validation_metrics.get("active_sharpe_annualized", 0.0),
                        "validation_turnover": validation_metrics.get("turnover", 0.0),
                        "validation_beta": validation_metrics.get("beta_to_market", 0.0),
                    }
                )
        frame = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "config_id": config_id,
                "params": params,
                "cv_mode": cv.mode,
                "folds": len(frame),
                "train_score_median": float(frame["train_score"].median()) if not frame.empty else 0.0,
                "validation_score_median": float(frame["validation_score"].median()) if not frame.empty else 0.0,
                "validation_score_mean": float(frame["validation_score"].mean()) if not frame.empty else 0.0,
                "validation_active_sharpe_median": float(frame["validation_active_sharpe"].median()) if not frame.empty else 0.0,
                "validation_turnover_mean": float(frame["validation_turnover"].mean()) if not frame.empty else 0.0,
                "validation_beta_abs_mean": float(frame["validation_beta"].abs().mean()) if not frame.empty else 0.0,
                "is_oos_gap": float(frame["train_score"].median() - frame["validation_score"].median()) if not frame.empty else 0.0,
                "fold_metrics": fold_metrics,
            }
        )
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["validation_score_median", "validation_active_sharpe_median", "validation_beta_abs_mean"], ascending=[False, False, True]).reset_index(drop=True)
    return results, last_folds


def format_grid_results(results: pd.DataFrame, top_k: int = 5) -> str:
    if results.empty:
        return "No grid results."
    lines = []
    for _, row in results.head(top_k).iterrows():
        lines.append(
            " | ".join(
                [
                    f"config_id={int(row['config_id'])}",
                    f"cv_mode={row['cv_mode']}",
                    f"validation_score_median={row['validation_score_median']:.6f}",
                    f"validation_active_sharpe_median={row['validation_active_sharpe_median']:.6f}",
                    f"validation_turnover_mean={row['validation_turnover_mean']:.6f}",
                    f"validation_beta_abs_mean={row['validation_beta_abs_mean']:.6f}",
                    f"is_oos_gap={row['is_oos_gap']:.6f}",
                    f"params={row['params']}",
                ]
            )
        )
    return "\n".join(lines)


@contextmanager
def temporary_module_params(module, params: dict):
    previous = {name: getattr(module, name) for name in params if hasattr(module, name)}
    for name, value in params.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


def usable_strategy_index(strategy_module, data_store, params: dict, execution: ExecutionConfig | None = None) -> pd.DatetimeIndex:
    execution = execution or getattr(strategy_module, "EXECUTION", ExecutionConfig())
    bucket = int(params.get("POSITION_BUCKET", getattr(strategy_module, "POSITION_BUCKET", 4)))
    min_assets = bucket * 2 + 4
    universe_kwargs = default_universe_kwargs(execution)
    usable = []
    for ts in data_store.index:
        if len(data_store.tradable_universe(ts, **universe_kwargs)) >= min_assets:
            usable.append(ts)
    return pd.DatetimeIndex(usable)


def _build_expanding_folds(index: pd.DatetimeIndex, train_size: int, validation_size: int, step_size: int, gap_size: int, n_folds: int) -> list[dict]:
    folds = []
    train_end = train_size
    fold_id = 1
    while fold_id <= n_folds:
        val_start = train_end + gap_size
        val_end = val_start + validation_size
        if val_end > len(index):
            break
        train = pd.DatetimeIndex(index[:train_end])
        validation = pd.DatetimeIndex(index[val_start:val_end])
        if len(train) and len(validation):
            folds.append({"fold_id": fold_id, "train": train, "validation": validation})
        train_end += step_size
        fold_id += 1
    return folds


def _build_rolling_folds(index: pd.DatetimeIndex, train_size: int, validation_size: int, step_size: int, gap_size: int, n_folds: int) -> list[dict]:
    folds = []
    train_start = 0
    fold_id = 1
    while fold_id <= n_folds:
        train_end = train_start + train_size
        val_start = train_end + gap_size
        val_end = val_start + validation_size
        if val_end > len(index):
            break
        train = pd.DatetimeIndex(index[train_start:train_end])
        validation = pd.DatetimeIndex(index[val_start:val_end])
        if len(train) and len(validation):
            folds.append({"fold_id": fold_id, "train": train, "validation": validation})
        train_start += step_size
        fold_id += 1
    return folds


def _apply_purge_to_folds(folds: list[dict], purge_size: int) -> list[dict]:
    if purge_size <= 0:
        return folds
    adjusted = []
    for fold in folds:
        train = pd.DatetimeIndex(fold["train"][:-purge_size]) if len(fold["train"]) > purge_size else pd.DatetimeIndex([])
        validation = pd.DatetimeIndex(fold["validation"])
        if len(train) and len(validation):
            adjusted.append({"fold_id": fold["fold_id"], "train": train, "validation": validation})
    return adjusted

