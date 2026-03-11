from __future__ import annotations

import numpy as np
import pandas as pd


def gross_exposure(weights: pd.Series) -> float:
    return float(pd.Series(weights, dtype=float).abs().sum())


def net_exposure(weights: pd.Series) -> float:
    return float(pd.Series(weights, dtype=float).sum())


def normalize_long_short_weights(
    raw: pd.Series,
    gross_target: float,
    net_target: float,
    max_abs_weight: float,
    groups: pd.Series | None = None,
    max_group_gross: float | None = None,
) -> pd.Series:
    weights = pd.Series(raw, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if weights.empty:
        return weights
    longs = weights[weights > 0.0]
    shorts = weights[weights < 0.0]
    if longs.empty or shorts.empty:
        return pd.Series(dtype=float)
    long_target = 0.5 * (gross_target + net_target)
    short_target = 0.5 * (gross_target - net_target)
    long_weights = longs / float(longs.sum()) * long_target
    short_weights = shorts.abs() / float(shorts.abs().sum()) * short_target
    normalized = pd.concat([long_weights, -short_weights])
    normalized = normalized.clip(lower=-max_abs_weight, upper=max_abs_weight)
    aligned_groups = groups.reindex(normalized.index) if groups is not None else None
    for _ in range(20):
        normalized = normalized.clip(lower=-max_abs_weight, upper=max_abs_weight)
        if aligned_groups is not None and max_group_gross is not None:
            normalized = _enforce_group_caps(normalized, aligned_groups, max_group_gross)
        normalized = _fill_side_to_target(
            normalized,
            side=1,
            target=long_target,
            max_abs_weight=max_abs_weight,
            groups=aligned_groups,
            max_group_gross=max_group_gross,
        )
        normalized = _fill_side_to_target(
            normalized,
            side=-1,
            target=short_target,
            max_abs_weight=max_abs_weight,
            groups=aligned_groups,
            max_group_gross=max_group_gross,
        )
        diag = exposure_diagnostics(normalized, groups=aligned_groups)
        if (
            abs(float(normalized[normalized > 0.0].sum()) - long_target) <= 1e-9
            and abs(float(normalized[normalized < 0.0].abs().sum()) - short_target) <= 1e-9
            and (max_group_gross is None or diag["max_group_gross"] <= max_group_gross + 1e-9)
        ):
            break
    return normalized.sort_index()


def validate_exposures(weights: pd.Series, max_gross: float, target_net: float, net_tolerance: float = 2e-2) -> None:
    gross = gross_exposure(weights)
    net = net_exposure(weights)
    if gross > max_gross + 1e-9:
        raise ValueError(f"gross exposure {gross:.6f} exceeds {max_gross:.6f}")
    if abs(net - target_net) > net_tolerance:
        raise ValueError(f"net exposure {net:.6f} differs from target {target_net:.6f}")


def exposure_diagnostics(weights: pd.Series, groups: pd.Series | None = None) -> dict:
    weights = pd.Series(weights, dtype=float)
    diag = {
        "gross": gross_exposure(weights),
        "net": net_exposure(weights),
        "max_abs_weight": float(weights.abs().max()) if not weights.empty else 0.0,
    }
    if groups is not None and not weights.empty:
        grouped = weights.abs().groupby(groups.reindex(weights.index)).sum()
        diag["max_group_gross"] = float(grouped.max()) if not grouped.empty else 0.0
    else:
        diag["max_group_gross"] = 0.0
    return diag


def _renormalize_sides(weights: pd.Series, gross_target: float, net_target: float) -> pd.Series:
    weights = pd.Series(weights, dtype=float)
    longs = weights[weights > 0.0]
    shorts = weights[weights < 0.0]
    if longs.empty or shorts.empty:
        return weights
    long_target = 0.5 * (gross_target + net_target)
    short_target = 0.5 * (gross_target - net_target)
    if float(longs.sum()) > 0.0:
        weights.loc[longs.index] = longs / float(longs.sum()) * long_target
    if float(shorts.abs().sum()) > 0.0:
        weights.loc[shorts.index] = -shorts.abs() / float(shorts.abs().sum()) * short_target
    return weights


def _enforce_group_caps(weights: pd.Series, groups: pd.Series, max_group_gross: float) -> pd.Series:
    adjusted = pd.Series(weights, dtype=float).copy()
    for _ in range(20):
        gross_by_group = adjusted.abs().groupby(groups).sum()
        offenders = gross_by_group[gross_by_group > max_group_gross + 1e-12]
        if offenders.empty:
            break
        for group_name, gross in offenders.items():
            members = groups[groups == group_name].index
            scale = max_group_gross / float(gross)
            adjusted.loc[members] *= scale
    return adjusted


def _fill_side_to_target(
    weights: pd.Series,
    side: int,
    target: float,
    max_abs_weight: float,
    groups: pd.Series | None,
    max_group_gross: float | None,
) -> pd.Series:
    adjusted = pd.Series(weights, dtype=float).copy()
    mask = adjusted > 0.0 if side > 0 else adjusted < 0.0
    names = adjusted.index[mask]
    if len(names) == 0:
        return adjusted
    current = adjusted.loc[names].abs()
    current_total = float(current.sum())
    if current_total > target + 1e-12:
        scale = target / current_total if current_total > 0.0 else 0.0
        adjusted.loc[names] = np.sign(adjusted.loc[names]) * current * scale
        return adjusted
    deficit = target - current_total
    for _ in range(20):
        if deficit <= 1e-12:
            break
        capacities = pd.Series(max_abs_weight, index=names, dtype=float) - adjusted.loc[names].abs()
        capacities = capacities.clip(lower=0.0)
        if groups is not None and max_group_gross is not None:
            group_gross = adjusted.abs().groupby(groups.reindex(adjusted.index)).sum()
            for name in names:
                group_name = groups.get(name)
                if pd.isna(group_name):
                    continue
                spare = max_group_gross - float(group_gross.get(group_name, 0.0))
                capacities.loc[name] = max(0.0, min(float(capacities.loc[name]), spare))
        capacities = capacities[capacities > 1e-12]
        if capacities.empty:
            break
        base = adjusted.loc[capacities.index].abs()
        if float(base.sum()) <= 1e-12:
            props = pd.Series(1.0 / len(capacities), index=capacities.index)
        else:
            props = base / float(base.sum())
        increment = pd.concat([capacities, props * deficit], axis=1).min(axis=1)
        if float(increment.sum()) <= 1e-12:
            break
        adjusted.loc[increment.index] += side * increment
        deficit -= float(increment.sum())
    return adjusted
