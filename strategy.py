"""Winner deployment candidate: OLS rank 3-feature Hyperliquid model."""

from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd

from q_lab_hl.config import ExecutionConfig
from strategy_model import (
    FeatureSpec,
    ModelSpec,
    StrategySpec,
    TargetSpec,
    get_strategy_family,
    build_training_dataset,
    construct_portfolio,
    fit_linear_model,
    latest_snapshot,
    predict_scores,
    strategy_spec_from_dict,
    validate_strategy_spec,
)


DEFAULT_EXECUTION = ExecutionConfig(
    rebalance_every_bars=24,
    min_history_bars=24 * 14,
    min_dollar_volume=500_000.0,
    min_price=0.10,
    listing_cooldown_bars=24 * 3,
)
MODEL_FAMILY = "ols"
DEFAULT_SPEC = StrategySpec(
    train_window_bars=24 * 21,
    min_train_rows=200,
    position_bucket=5,
    features=(
        FeatureSpec(name="ret_1h", kind="return", lookback=1, transform="rank", clip=None),
        FeatureSpec(name="ma_gap_24h", kind="ma_gap", lookback=24, transform="rank", clip=None),
        FeatureSpec(name="funding_8h", kind="funding_mean", lookback=8, transform="rank", clip=None),
    ),
    target=TargetSpec(name="next_bar_open_to_close", kind="next_open_to_close_return"),
    model=ModelSpec(family=MODEL_FAMILY, l2_reg=0.0, prediction_clip=3.0),
)
EXECUTION = DEFAULT_EXECUTION
SPEC = DEFAULT_SPEC
_STATE: dict = {}
STRATEGY_FAMILY = "linear_cross_section_v1"


def reset_state():
    _STATE.clear()


def signals(data, ts):
    dataset = build_training_dataset(
        data,
        ts,
        execution=EXECUTION,
        strategy_spec=SPEC,
    )
    if len(dataset["y_train"]) < SPEC.min_train_rows or len(dataset["current_assets"]) < SPEC.position_bucket * 2:
        return pd.Series(dtype=float)
    model = fit_linear_model(
        dataset["X_train"],
        dataset["y_train"],
        feature_names=dataset["feature_names"],
        family=SPEC.model.family,
        l2_reg=SPEC.model.l2_reg,
        train_start=dataset["train_start"],
        train_end=dataset["train_end"],
        train_timestamps=dataset.get("train_timestamps"),
    )
    if model is None:
        return pd.Series(dtype=float)
    _STATE["last_fit"] = {
        "strategy_spec": SPEC.summary(),
        "model_fit": model.summary(),
    }
    scores = predict_scores(
        model,
        dataset["X_now"],
        dataset["current_assets"],
        clip_predictions=SPEC.model.prediction_clip,
    )
    return scores.sort_values(ascending=False)


def construct(scores, data, ts):
    return construct_portfolio(pd.Series(scores, dtype=float), SPEC.position_bucket)


def risk(weights, data, ts):
    return pd.Series(weights, dtype=float)


def last_fit_summary():
    return dict(_STATE.get("last_fit", {}))


def paper_trade_snapshot(data, ts):
    return latest_snapshot(data, ts, execution=EXECUTION, strategy_spec=SPEC)


def apply_runtime_overrides(strategy_spec: dict | None = None, execution_overrides: dict | None = None):
    global EXECUTION, SPEC
    EXECUTION = DEFAULT_EXECUTION
    SPEC = DEFAULT_SPEC
    family = get_strategy_family(STRATEGY_FAMILY)
    validate_strategy_spec(SPEC, family)
    if strategy_spec:
        SPEC = strategy_spec_from_dict(strategy_spec, base=DEFAULT_SPEC, strategy_family=STRATEGY_FAMILY)
    if execution_overrides:
        EXECUTION = replace(DEFAULT_EXECUTION, **execution_overrides)
    _STATE["runtime_overrides"] = {
        "strategy_family": family.summary(),
        "strategy_spec": SPEC.summary(),
        "execution": asdict(EXECUTION),
    }
