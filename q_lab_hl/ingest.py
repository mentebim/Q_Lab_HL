from __future__ import annotations

import json
import ssl
import time
import urllib.request

import pandas as pd


BASE_URL = "https://api.hyperliquid.xyz/info"


class HyperliquidInfoClient:
    def __init__(self, verify_ssl: bool = True, timeout: float = 30.0, retries: int = 3):
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.retries = retries

    def meta(self):
        return self._post({"type": "meta"})

    def meta_and_asset_ctxs(self):
        return self._post({"type": "metaAndAssetCtxs"})

    def candle_snapshot(self, coin: str, interval: str, start_ms: int, end_ms: int):
        return self._post({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms}})

    def funding_history(self, coin: str, start_ms: int, end_ms: int):
        return self._post({"type": "fundingHistory", "coin": coin, "startTime": start_ms, "endTime": end_ms})

    def _post(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(BASE_URL, data=body, headers={"Content-Type": "application/json"})
        context = None if self.verify_ssl else ssl._create_unverified_context()
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=context) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception:
                if attempt + 1 == self.retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")


def fetch_candles_chunked(client, coin: str, interval: str, start, end, max_bars_per_call: int = 5000) -> pd.DataFrame:
    start_ts = _parse_range_bound(start, is_end=False)
    end_ts = _parse_range_bound(end, is_end=True)
    if start_ts > end_ts:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "trades"])
    step = _interval_to_timedelta(interval)
    bars_per_call = max(int(max_bars_per_call), 1)
    chunk_span = step * bars_per_call
    rows = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(end_ts, cursor + chunk_span - pd.Timedelta(milliseconds=1))
        rows.extend(client.candle_snapshot(coin, interval, _to_ms(cursor), _to_ms(chunk_end)) or [])
        cursor = chunk_end + pd.Timedelta(milliseconds=1)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "trades"])
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trades"})
    cols = ["date", "open", "high", "low", "close", "volume", "trades"]
    out = frame[cols].copy()
    for col in cols[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_funding_chunked(client, coin: str, start, end, max_hours_per_call: int = 24 * 30) -> pd.DataFrame:
    start_ts = _parse_range_bound(start, is_end=False)
    end_ts = _parse_range_bound(end, is_end=True)
    if start_ts > end_ts:
        return pd.DataFrame(columns=["date", "funding_rate", "premium"])
    chunk_span = pd.Timedelta(hours=max(int(max_hours_per_call), 1))
    rows = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(end_ts, cursor + chunk_span - pd.Timedelta(milliseconds=1))
        rows.extend(client.funding_history(coin, _to_ms(cursor), _to_ms(chunk_end)) or [])
        cursor = chunk_end + pd.Timedelta(milliseconds=1)
    if not rows:
        return pd.DataFrame(columns=["date", "funding_rate", "premium"])
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["time"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.rename(columns={"fundingRate": "funding_rate"})
    out = frame[["date", "funding_rate", "premium"]].copy()
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    out["premium"] = pd.to_numeric(out["premium"], errors="coerce")
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported interval: {interval}")


def _to_ms(ts: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _parse_range_bound(value, is_end: bool) -> pd.Timestamp:
    raw = str(value)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if len(raw) == 10 and raw.count("-") == 2:
        if is_end:
            timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return timestamp
