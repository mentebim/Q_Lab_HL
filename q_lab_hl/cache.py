from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from q_lab_hl.ingest import fetch_candles_chunked, fetch_funding_chunked


@dataclass(frozen=True)
class CacheBuildConfig:
    start: str
    end: str
    interval: str
    include_delisted: bool = False
    top_n: int | None = None
    min_current_day_ntl_vlm: float | None = None


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
    candle_frames = {}
    funding_frames = {}
    metadata = {}
    for coin in selected:
        candles = fetch_candles_chunked(client, coin, config.interval, config.start, config.end)
        funding = fetch_funding_chunked(client, coin, config.start, config.end)
        candle_frames[coin] = candles.set_index("date") if not candles.empty else pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trades"])
        funding_frames[coin] = funding.set_index("date") if not funding.empty else pd.DataFrame(columns=["funding_rate", "premium"])
        meta = asset_meta.get(coin, {})
        ctx = asset_ctxs.get(coin, {})
        metadata[coin] = {
            "sector": "perps",
            "szDecimals": meta.get("szDecimals"),
            "maxLeverage": meta.get("maxLeverage"),
            "dayNtlVlm": ctx.get("dayNtlVlm"),
            "openInterest": ctx.get("openInterest"),
        }
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
    (output_dir / "metadata.json").write_text(json.dumps({"assets": metadata}, indent=2, default=str))
    (output_dir / "schema.json").write_text(
        json.dumps(
            {
                "provider": "hyperliquid",
                "interval": config.interval,
                "start": config.start,
                "end": config.end,
                "coins": selected,
                "bars": int(len(index)),
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

