from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_state() -> dict[str, Any]:
    return {
        "last_signal_bar": None,
        "paper_positions": {},
        "paper_fills": [],
        "reconciliation": {
            "last_fill_time_ms": None,
            "open_orders": [],
            "recent_fills": [],
            "snapshot": None,
        },
        "runs": [],
    }


def load_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return default_state()
    return _merge_state(default_state(), json.loads(state_path.read_text()))


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def seen_signal_bar(state: dict[str, Any], signal_bar) -> bool:
    return state.get("last_signal_bar") == str(signal_bar)


def record_run(state: dict[str, Any], signal_bar, report: dict[str, Any], keep_last: int = 100) -> dict[str, Any]:
    next_state = dict(state)
    next_state["last_signal_bar"] = str(signal_bar)
    runs = list(next_state.get("runs", []))
    runs.append(report)
    next_state["runs"] = runs[-keep_last:]
    return next_state


def record_reconciliation(state: dict[str, Any], snapshot: dict[str, Any], keep_recent_fills: int = 200) -> dict[str, Any]:
    next_state = _merge_state(default_state(), state)
    reconciliation = dict(next_state.get("reconciliation", {}))
    fills = list(snapshot.get("recent_fills", []))
    last_fill_time_ms = reconciliation.get("last_fill_time_ms")
    if fills:
        fill_times = [int(fill.get("time")) for fill in fills if fill.get("time") is not None]
        if fill_times:
            candidate_times = list(fill_times)
            if last_fill_time_ms is not None:
                candidate_times.append(int(last_fill_time_ms))
            last_fill_time_ms = max(candidate_times)
    reconciliation.update(
        {
            "last_fill_time_ms": last_fill_time_ms,
            "open_orders": list(snapshot.get("open_orders", [])),
            "recent_fills": fills[-keep_recent_fills:],
            "snapshot": {
                key: value
                for key, value in snapshot.items()
                if key not in {"recent_fills", "open_orders"}
            },
        }
    )
    next_state["reconciliation"] = reconciliation
    return next_state


def last_fill_time_ms(state: dict[str, Any]) -> int | None:
    reconciliation = dict(state.get("reconciliation", {}))
    value = reconciliation.get("last_fill_time_ms")
    return None if value is None else int(value)


def _merge_state(base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_state(merged[key], value)
        else:
            merged[key] = value
    return merged
