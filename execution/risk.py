from __future__ import annotations

from pathlib import Path

import pandas as pd

from q_lab_hl.portfolio import validate_exposures


def validate_target_weights(weights: pd.Series, max_gross_exposure: float, target_net_exposure: float) -> None:
    if pd.Series(weights, dtype=float).empty:
        return
    validate_exposures(pd.Series(weights, dtype=float), max_gross_exposure, target_net_exposure)


def kill_switch_active(path: str | Path) -> bool:
    return Path(path).exists()
