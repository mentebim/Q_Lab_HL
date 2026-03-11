from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def bootstrap_sharpe_ci(returns: pd.Series, bars_per_year: int, n_boot: int = 200, seed: int = 7) -> tuple[float, float]:
    series = pd.Series(returns, dtype=float).dropna()
    if len(series) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    samples = []
    values = series.to_numpy(dtype=float)
    for _ in range(n_boot):
        draw = rng.choice(values, size=len(values), replace=True)
        std = float(np.std(draw, ddof=1))
        samples.append(0.0 if std == 0.0 else float(np.mean(draw) / std * np.sqrt(bars_per_year)))
    low, high = np.percentile(samples, [5, 95])
    return float(low), float(high)


def summarize_trial_family(
    family_matrix: pd.DataFrame,
    active_returns: pd.Series,
    bars_per_year: int,
    candidate_id: str | None = None,
) -> dict:
    active = pd.Series(active_returns, dtype=float).dropna()
    mean = float(active.mean()) if not active.empty else 0.0
    std = float(active.std(ddof=1)) if len(active) > 1 else 0.0
    sharpe = 0.0 if std == 0.0 else float(mean / std * np.sqrt(bars_per_year))
    ci_low, ci_high = bootstrap_sharpe_ci(active, bars_per_year)
    n_eff = int(family_matrix.shape[1] + (1 if candidate_id else 0))
    dsr = 0.0 if sharpe <= 0.0 else float(max(0.0, sharpe - np.sqrt(max(np.log(max(n_eff, 1)), 0.0))))
    return {
        "DSR": dsr,
        "N_eff": n_eff,
        "bootstrap_sharpe_ci": (ci_low, ci_high),
    }


@dataclass
class AuditFamilyStore:
    path: str

    def matrix(self) -> pd.DataFrame:
        try:
            return pd.read_parquet(self.path)
        except FileNotFoundError:
            return pd.DataFrame()

    def upsert(self, candidate_id: str, active_returns: pd.Series) -> None:
        matrix = self.matrix()
        column = pd.Series(active_returns, dtype=float).rename(candidate_id)
        matrix = matrix.join(column, how="outer") if not matrix.empty else column.to_frame()
        matrix.to_parquet(self.path)

