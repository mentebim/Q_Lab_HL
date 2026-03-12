from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {
            "last_signal_bar": None,
            "paper_positions": {},
            "runs": [],
        }
    return json.loads(state_path.read_text())


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
