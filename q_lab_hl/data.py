from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from q_lab_hl.config import DEFAULT_WARMUP_BARS, ExecutionConfig


@dataclass
class MarketPanels:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    funding: pd.DataFrame | None = None
    open_interest: pd.DataFrame | None = None
    tradable: pd.DataFrame | None = None
    metadata: dict | None = None
    trades: pd.DataFrame | None = None


class DataStore:
    def __init__(self, panels: MarketPanels):
        self.panels = panels
        self.close = _clean_matrix(panels.close)
        self.open = _clean_matrix(panels.open).reindex_like(self.close).ffill()
        self.high = _clean_matrix(panels.high).reindex_like(self.close).ffill()
        self.low = _clean_matrix(panels.low).reindex_like(self.close).ffill()
        self.volume = _clean_matrix(panels.volume).reindex_like(self.close).fillna(0.0)
        self.funding_panel = _optional_matrix(panels.funding, self.close)
        self.oi_panel = _optional_matrix(panels.open_interest, self.close)
        self.tradable_panel = _tradable_matrix(panels.tradable, self.close)
        self.trades_panel = _optional_matrix(panels.trades, self.close).fillna(0.0)
        self.metadata = panels.metadata or {asset: {} for asset in self.close.columns}
        self.index = self.close.index
        self.assets = list(self.close.columns)
        self._listing_start = {
            asset: self._infer_listing_start(asset) for asset in self.assets
        }

    @classmethod
    def from_parquet_dir(cls, path: str | Path) -> "DataStore":
        path = Path(path)
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        panels = MarketPanels(
            open=pd.read_parquet(path / "open.parquet"),
            high=pd.read_parquet(path / "high.parquet"),
            low=pd.read_parquet(path / "low.parquet"),
            close=pd.read_parquet(path / "close.parquet"),
            volume=pd.read_parquet(path / "volume.parquet"),
            funding=pd.read_parquet(path / "funding.parquet") if (path / "funding.parquet").exists() else None,
            open_interest=pd.read_parquet(path / "open_interest.parquet") if (path / "open_interest.parquet").exists() else None,
            tradable=pd.read_parquet(path / "tradable.parquet") if (path / "tradable.parquet").exists() else None,
            metadata=metadata.get("assets", metadata),
            trades=pd.read_parquet(path / "trades.parquet") if (path / "trades.parquet").exists() else None,
        )
        return cls(panels)

    @classmethod
    def synthetic(cls, n_assets: int = 20, periods: int = 24 * 120, seed: int = 1) -> "DataStore":
        rng = np.random.default_rng(seed)
        index = pd.date_range("2025-01-01", periods=periods, freq="h")
        assets = [f"A{i:02d}" for i in range(n_assets)]
        shocks = rng.normal(0.0, 0.01, size=(periods, n_assets))
        market = rng.normal(0.0, 0.004, size=(periods, 1))
        returns = market + shocks
        close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=assets)
        open_ = close.shift(1).fillna(close.iloc[0])
        high = pd.DataFrame(np.maximum(open_.to_numpy(), close.to_numpy()), index=index, columns=assets)
        low = pd.DataFrame(np.minimum(open_.to_numpy(), close.to_numpy()), index=index, columns=assets)
        volume = pd.DataFrame(rng.uniform(5e5, 5e6, size=(periods, n_assets)), index=index, columns=assets)
        funding = pd.DataFrame(rng.normal(0.0, 0.0002, size=(periods, n_assets)), index=index, columns=assets)
        oi = pd.DataFrame(rng.uniform(1e7, 5e7, size=(periods, n_assets)), index=index, columns=assets)
        tradable = pd.DataFrame(True, index=index, columns=assets)
        if n_assets > 0:
            late_asset = assets[-1]
            tradable.loc[index[: 24 * 5], late_asset] = False
            volume.loc[index[: 24 * 5], late_asset] = 0.0
        metadata = {asset: {"sector": f"S{i % 4}"} for i, asset in enumerate(assets)}
        return cls(
            MarketPanels(
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                funding=funding,
                open_interest=oi,
                tradable=tradable,
                metadata=metadata,
            )
        )

    def prices(
        self,
        assets: list[str] | pd.Index | None = None,
        start=None,
        end=None,
        field: str = "close",
    ) -> pd.DataFrame:
        frame = getattr(self, field)
        return _slice_frame(frame, assets, start, end)

    def funding(self, assets: list[str] | pd.Index | None = None, start=None, end=None) -> pd.DataFrame:
        return _slice_frame(self.funding_panel, assets, start, end)

    def open_interest(self, assets: list[str] | pd.Index | None = None, start=None, end=None) -> pd.DataFrame:
        return _slice_frame(self.oi_panel, assets, start, end)

    def dollar_volume(self, window: int, ts) -> pd.Series:
        close = self.prices(end=ts, field="close").iloc[-window:]
        volume = self.prices(end=ts, field="volume").iloc[-window:]
        if close.empty or volume.empty:
            return pd.Series(dtype=float)
        return (close * volume).mean()

    def tradable_universe(
        self,
        ts,
        min_history_bars: int,
        min_dollar_volume: float,
        min_price: float,
        listing_cooldown_bars: int,
    ) -> list[str]:
        if ts not in self.index:
            return []
        pos = int(self.index.get_loc(ts))
        universe = []
        for asset in self.assets:
            if not self.can_trade(asset, ts):
                continue
            if pos + 1 < min_history_bars:
                continue
            hist = self.close[asset].iloc[: pos + 1].dropna()
            if len(hist) < min_history_bars:
                continue
            start_pos = int(self.index.get_loc(self._listing_start[asset]))
            if pos - start_pos < listing_cooldown_bars:
                continue
            price = float(self.close.at[ts, asset])
            if not np.isfinite(price) or price < min_price:
                continue
            dv = self.dollar_volume(min(min_history_bars if min_history_bars > 0 else 1, pos + 1), ts).get(asset, 0.0)
            if float(dv) < min_dollar_volume:
                continue
            universe.append(asset)
        return universe

    def can_trade(self, asset: str, ts) -> bool:
        if asset not in self.assets or ts not in self.index:
            return False
        tradable = bool(self.tradable_panel.at[ts, asset]) if asset in self.tradable_panel.columns else True
        price = self.close.at[ts, asset]
        return tradable and pd.notna(price) and float(price) > 0.0

    def sector(self, asset: str) -> str:
        return str(self.metadata.get(asset, {}).get("sector", "unknown"))

    def market_return_series(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        close = self.close.reindex(timestamps)
        market = close.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
        return market.reindex(timestamps).fillna(0.0)

    def zscore_cross_section(self, values: pd.Series) -> pd.Series:
        series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 2:
            return series * 0.0
        std = float(series.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            return series * 0.0
        return (series - float(series.mean())) / std

    def winsorize_cross_section(self, values: pd.Series, lower_pct: float, upper_pct: float) -> pd.Series:
        series = pd.Series(values, dtype=float).dropna()
        if series.empty:
            return series
        lower = float(series.quantile(lower_pct))
        upper = float(series.quantile(upper_pct))
        return series.clip(lower=lower, upper=upper)

    def neutralize_cross_section(self, values: pd.Series, by: list[pd.Series]) -> pd.Series:
        residual = pd.Series(values, dtype=float).dropna().copy()
        if residual.empty:
            return residual
        for feature in by:
            aligned = pd.Series(feature).reindex(residual.index)
            if aligned.dropna().empty:
                continue
            if aligned.dtype == object or str(aligned.dtype).startswith("category"):
                group_means = residual.groupby(aligned).transform("mean")
                residual = residual - group_means.fillna(0.0)
                continue
            x = pd.to_numeric(aligned, errors="coerce")
            valid = residual.index.intersection(x.dropna().index)
            if len(valid) < 3:
                continue
            xv = x.loc[valid].to_numpy(dtype=float)
            yv = residual.loc[valid].to_numpy(dtype=float)
            xv = xv - xv.mean()
            denom = float(np.dot(xv, xv))
            if denom == 0.0:
                continue
            beta = float(np.dot(xv, yv) / denom)
            residual.loc[valid] = yv - beta * xv
        return residual

    def _infer_listing_start(self, asset: str):
        tradable = self.tradable_panel[asset] & self.close[asset].notna() & (self.volume[asset] > 0.0)
        if tradable.any():
            return tradable[tradable].index[0]
        return self.index[0]


class DateLimitedStore:
    def __init__(self, base: DataStore, ts):
        self.base = base
        self.ts = pd.Timestamp(ts)
        self.index = base.index[base.index <= self.ts]
        self.assets = base.assets

    def __getattr__(self, name):
        return getattr(self.base, name)

    def prices(self, assets=None, start=None, end=None, field: str = "close") -> pd.DataFrame:
        end = self.ts if end is None or pd.Timestamp(end) > self.ts else end
        return self.base.prices(assets=assets, start=start, end=end, field=field)

    def funding(self, assets=None, start=None, end=None) -> pd.DataFrame:
        end = self.ts if end is None or pd.Timestamp(end) > self.ts else end
        return self.base.funding(assets=assets, start=start, end=end)

    def open_interest(self, assets=None, start=None, end=None) -> pd.DataFrame:
        end = self.ts if end is None or pd.Timestamp(end) > self.ts else end
        return self.base.open_interest(assets=assets, start=start, end=end)


def default_universe_kwargs(execution: ExecutionConfig) -> dict:
    return {
        "min_history_bars": execution.min_history_bars,
        "min_dollar_volume": execution.min_dollar_volume,
        "min_price": execution.min_price,
        "listing_cooldown_bars": execution.listing_cooldown_bars,
    }


def recommended_warmup_bars(execution: ExecutionConfig) -> int:
    return max(DEFAULT_WARMUP_BARS, execution.min_history_bars)


def _clean_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = pd.DataFrame(frame).copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned.sort_index(axis=1)
    return cleaned.apply(pd.to_numeric, errors="coerce")


def _optional_matrix(frame: pd.DataFrame | None, template: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(0.0, index=template.index, columns=template.columns)
    return _clean_matrix(frame).reindex(index=template.index, columns=template.columns).fillna(0.0)


def _tradable_matrix(frame: pd.DataFrame | None, template: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(True, index=template.index, columns=template.columns)
    matrix = pd.DataFrame(frame).reindex(index=template.index, columns=template.columns).fillna(False)
    return matrix.astype(bool)


def _slice_frame(frame: pd.DataFrame, assets, start, end) -> pd.DataFrame:
    out = frame
    if assets is not None:
        out = out.reindex(columns=list(assets))
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out
