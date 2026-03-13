from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from q_lab_hl.ingest import _interval_to_timedelta, fetch_candles_chunked, fetch_funding_chunked


@dataclass(frozen=True)
class CacheBuildConfig:
    start: str
    end: str
    interval: str
    include_delisted: bool = False
    top_n: int | None = None
    min_current_day_ntl_vlm: float | None = None
    max_bars_per_call: int = 5000
    max_hours_per_funding_call: int = 24 * 30


def build_hyperliquid_cache(
    output_dir: str | Path,
    config: CacheBuildConfig,
    client,
    coins: list[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_meta, asset_ctxs = _meta_and_ctxs(client)
    selected = coins or _select_coins(asset_meta, asset_ctxs, config)
    if not selected:
        raise ValueError("No Hyperliquid markets matched the cache build selection.")
    candle_frames = {}
    funding_frames = {}
    metadata = {}
    for coin in selected:
        candles = fetch_candles_chunked(
            client,
            coin,
            config.interval,
            config.start,
            config.end,
            max_bars_per_call=config.max_bars_per_call,
        )
        funding = fetch_funding_chunked(
            client,
            coin,
            config.start,
            config.end,
            max_hours_per_call=config.max_hours_per_funding_call,
        )
        candle_frames[coin] = candles.set_index("date") if not candles.empty else pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades"])
        funding_frames[coin] = funding.set_index("date") if not funding.empty else pd.DataFrame(columns=["funding_rate", "premium"])
        meta = asset_meta.get(coin, {})
        ctx = asset_ctxs.get(coin, {})
        metadata[coin] = _build_asset_metadata(coin, meta, ctx, candle_frames[coin])
    index = _union_index([frame.index for frame in candle_frames.values()])
    matrices = {}
    for field in ["open", "high", "low", "close", "volume", "trades"]:
        matrices[field] = pd.DataFrame(index=index, columns=selected, dtype=float)
        for coin in selected:
            frame = candle_frames[coin]
            if field in frame:
                matrices[field][coin] = frame[field].reindex(index)
    funding_matrix = pd.DataFrame(index=index, columns=selected, dtype=float)
    for coin in selected:
        frame = funding_frames[coin]
        if "funding_rate" in frame:
            funding_matrix[coin] = frame["funding_rate"].reindex(index).fillna(0.0)
    tradable = (matrices["close"].notna() & matrices["volume"].fillna(0.0).ge(0.0)).astype(bool)
    for field in ["open", "high", "low", "close", "volume", "trades"]:
        matrices[field].to_parquet(output_dir / f"{field}.parquet")
    funding_matrix.fillna(0.0).to_parquet(output_dir / "funding.parquet")
    tradable.to_parquet(output_dir / "tradable.parquet")
    validate_cache_frames(matrices=matrices, funding=funding_matrix, tradable=tradable)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (output_dir / "schema.json").write_text(
        json.dumps(
            {
                "provider": "hyperliquid",
                "schema_version": 1,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "bars": int(len(index)),
                "coins": selected,
                "interval": config.interval,
                "start": config.start,
                "end": config.end,
                "config": asdict(config),
            },
            indent=2,
            default=str,
        )
    )
    return {
        "coins": len(selected),
        "bars": int(len(index)),
        "start": str(index.min()) if len(index) else None,
        "end": str(index.max()) if len(index) else None,
        "interval": config.interval,
    }


def validate_market_cache_dir(
    path: str | Path,
    *,
    interval: str | None = None,
    max_data_lag_hours: float | None = None,
) -> dict:
    root = Path(path)
    required = ["open", "high", "low", "close", "volume", "funding", "tradable"]
    frames = {}
    for name in required:
        file_path = root / f"{name}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file is missing: {file_path}")
        frames[name] = pd.read_parquet(file_path)
    validate_cache_frames(
        matrices={name: frames[name] for name in ["open", "high", "low", "close", "volume"]},
        funding=frames["funding"],
        tradable=frames["tradable"],
    )
    close = pd.DataFrame(frames["close"]).copy()
    close.index = pd.to_datetime(close.index)
    latest_ts = pd.Timestamp(close.index.max())
    lag_hours = float((pd.Timestamp.now("UTC").tz_localize(None) - latest_ts) / pd.Timedelta(hours=1))
    if max_data_lag_hours is not None and lag_hours > float(max_data_lag_hours):
        raise ValueError(
            f"Cache is stale by {lag_hours:.2f} hours; max allowed is {float(max_data_lag_hours):.2f} hours."
        )
    if interval is not None and len(close.index) >= 2:
        step = _interval_to_timedelta(interval)
        diffs = close.index.to_series().diff().dropna()
        if (diffs <= pd.Timedelta(0)).any():
            raise ValueError("Cache index must be strictly increasing.")
        if (diffs < step).any():
            raise ValueError("Cache index contains bars closer together than the configured interval.")
    latest_row = close.loc[latest_ts]
    active_assets = int(latest_row.notna().sum())
    if active_assets == 0:
        raise ValueError("Cache latest bar has no active assets.")
    return {
        "bars": int(len(close)),
        "assets": int(len(close.columns)),
        "latest_bar": latest_ts.isoformat(),
        "lag_hours": lag_hours,
        "active_assets_latest_bar": active_assets,
    }


def validate_cache_frames(*, matrices: dict[str, pd.DataFrame], funding: pd.DataFrame, tradable: pd.DataFrame) -> None:
    close = _normalize_frame(matrices["close"])
    if close.empty:
        raise ValueError("Cache close matrix is empty.")
    base_index = close.index
    base_columns = close.columns
    for name, frame in matrices.items():
        normalized = _normalize_frame(frame)
        if not normalized.index.equals(base_index):
            raise ValueError(f"Cache matrix '{name}' index does not match close matrix.")
        if not normalized.columns.equals(base_columns):
            raise ValueError(f"Cache matrix '{name}' columns do not match close matrix.")
        if not normalized.index.is_monotonic_increasing:
            raise ValueError(f"Cache matrix '{name}' index is not sorted.")
        if normalized.index.has_duplicates:
            raise ValueError(f"Cache matrix '{name}' index has duplicate timestamps.")
    funding_frame = _normalize_frame(funding)
    tradable_frame = _normalize_frame(tradable)
    if not funding_frame.index.equals(base_index) or not funding_frame.columns.equals(base_columns):
        raise ValueError("Funding matrix does not align with price matrices.")
    if not tradable_frame.index.equals(base_index) or not tradable_frame.columns.equals(base_columns):
        raise ValueError("Tradable matrix does not align with price matrices.")


def _meta_and_ctxs(client) -> tuple[dict, dict]:
    payload = client.meta_and_asset_ctxs()
    if isinstance(payload, list) and len(payload) == 2:
        meta_payload, ctx_payload = payload
    else:
        meta_payload = client.meta()
        ctx_payload = []
    universe_rows = list(meta_payload.get("universe", []))
    universe = {row["name"]: row for row in universe_rows}
    ctxs = {}
    for idx, row in enumerate(ctx_payload):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if name is None and idx < len(universe_rows):
            name = universe_rows[idx].get("name")
        if name is None:
            continue
        ctxs[str(name)] = row
    return universe, ctxs


def _select_coins(asset_meta: dict, asset_ctxs: dict, config: CacheBuildConfig) -> list[str]:
    rows = []
    for coin, meta in asset_meta.items():
        ctx = asset_ctxs.get(coin, {})
        ntl = float(ctx.get("dayNtlVlm", 0.0) or 0.0)
        if config.min_current_day_ntl_vlm is not None and ntl < config.min_current_day_ntl_vlm:
            continue
        rows.append((coin, ntl))
    rows.sort(key=lambda item: item[1], reverse=True)
    coins = [coin for coin, _ in rows]
    return coins[: config.top_n] if config.top_n is not None else coins


def _union_index(indexes: list[pd.Index]) -> pd.DatetimeIndex:
    valid = [pd.DatetimeIndex(index) for index in indexes if len(index) > 0]
    if not valid:
        return pd.DatetimeIndex([])
    union = valid[0]
    for idx in valid[1:]:
        union = union.union(idx)
    return union.sort_values()


def _build_asset_metadata(coin: str, meta: dict, ctx: dict, candles: pd.DataFrame) -> dict:
    payload = dict(ctx)
    payload.update(meta)
    payload["name"] = coin
    payload["sector"] = "perps"
    if candles.empty:
        payload["listing_start"] = None
        payload["listing_end"] = None
    else:
        payload["listing_start"] = pd.Timestamp(candles.index.min()).isoformat(sep=" ")
        payload["listing_end"] = pd.Timestamp(candles.index.max()).isoformat(sep=" ")
    return payload


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(frame).copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    normalized = normalized.sort_index(axis=1)
    return normalized
