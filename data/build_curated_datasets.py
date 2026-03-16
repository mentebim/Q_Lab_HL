from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError
from requests.exceptions import RequestException


DATA_DIR = Path(__file__).resolve().parent
ACTIVE_DIR = DATA_DIR / "active_1h"
ARCHIVE_DIR = DATA_DIR / "archive_1h"
ACTIVE_ASSET_COUNT = 20
ACTIVE_HOURS = 5000
FUNDING_CHUNK_HOURS = 500
CLIENT_TIMEOUT_SECONDS = 20.0
MANIFEST_NAME = "manifest.json"
CORE_PANELS = ("open", "high", "low", "close", "volume", "trades", "funding", "tradable")


def _client() -> Info:
    return Info(base_url=constants.MAINNET_API_URL, skip_ws=True, timeout=CLIENT_TIMEOUT_SECONDS)


def _request(func, *args, **kwargs):
    delay = 1.0
    for _ in range(6):
        try:
            return func(*args, **kwargs)
        except ClientError as exc:
            if exc.status_code != 429:
                raise
            time.sleep(delay)
            delay *= 2.0
        except RequestException:
            time.sleep(delay)
            delay *= 2.0
    raise RuntimeError("Exceeded retry budget for Hyperliquid request")


def _latest_complete_hour() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)


def _active_window(end: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    window_end = end or _latest_complete_hour()
    window_start = window_end - pd.Timedelta(hours=ACTIVE_HOURS - 1)
    return window_start, window_end


def _machine_dir(base: Path) -> Path:
    return base / "machine"


def _reset_dataset_dir(base: Path) -> None:
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    _machine_dir(base).mkdir(parents=True, exist_ok=True)


def _write_panel(base: Path, name: str, frame: pd.DataFrame) -> None:
    frame.to_parquet(_machine_dir(base) / f"{name}.parquet")


def _write_machine_json(base: Path, name: str, payload: dict) -> None:
    (_machine_dir(base) / name).write_text(json.dumps(payload, indent=2))


def _write_long_csv(base: Path, frames: dict[str, pd.DataFrame]) -> None:
    fields = [field for field in CORE_PANELS if field in frames]
    parts = [frames[field].stack(dropna=False).rename(field) for field in fields]
    table = pd.concat(parts, axis=1).reset_index()
    table.columns = ["date", "asset", *fields]
    table.to_csv(base / "market_data.csv", index=False)


def _read_manifest(base: Path) -> dict:
    path = base / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _read_machine_frames(base: Path, names: tuple[str, ...] = CORE_PANELS) -> dict[str, pd.DataFrame]:
    machine = _machine_dir(base)
    frames: dict[str, pd.DataFrame] = {}
    if not machine.exists():
        return frames
    for name in names:
        path = machine / f"{name}.parquet"
        if path.exists():
            frame = pd.read_parquet(path).sort_index().sort_index(axis=1)
            frame.index = pd.to_datetime(frame.index)
            frame.index.name = "date"
            frames[name] = frame
    return frames


def _load_live_ranking(info: Info) -> tuple[pd.DataFrame, dict[str, dict], dict[str, dict]]:
    meta, ctxs = _request(info.meta_and_asset_ctxs)
    universe = {item["name"]: item for item in meta["universe"]}
    context = {item["name"]: row for item, row in zip(meta["universe"], ctxs)}
    rows = []
    for item, row in zip(meta["universe"], ctxs):
        if item.get("isDelisted"):
            continue
        rows.append({"name": item["name"], "dayNtlVlm": float(row["dayNtlVlm"])})
    ranking = pd.DataFrame(rows).sort_values("dayNtlVlm", ascending=False).reset_index(drop=True)
    return ranking, universe, context


def _fetch_hourly_candles(info: Info, coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = _request(
        info.candles_snapshot,
        coin,
        "1h",
        int(start.timestamp() * 1000),
        int(end.timestamp() * 1000),
    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v", "n"])
    frame["date"] = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
    expected = pd.date_range(start.tz_localize(None), end.tz_localize(None), freq="1h", name="date")
    frame = frame.reindex(expected)
    return frame


def _has_full_active_history(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    if len(frame.index) != ACTIVE_HOURS:
        return False
    return not frame[["o", "h", "l", "c", "v", "n"]].isna().any().any()


def _fetch_hourly_funding(info: Info, coin: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    expected = pd.date_range(start.tz_localize(None), end.tz_localize(None), freq="1h", name="date")
    values: dict[pd.Timestamp, float] = {}
    chunk_start = start
    end_inclusive = end + pd.Timedelta(hours=1)
    try:
        while chunk_start < end_inclusive:
            chunk_end = min(chunk_start + pd.Timedelta(hours=FUNDING_CHUNK_HOURS), end_inclusive)
            rows = _request(
                info.funding_history,
                coin,
                int(chunk_start.timestamp() * 1000),
                int(chunk_end.timestamp() * 1000),
            )
            for row in rows:
                ts = pd.to_datetime(row["time"], unit="ms", utc=True).floor("h").tz_localize(None)
                values[ts] = float(row["fundingRate"])
            chunk_start = chunk_end
    except RuntimeError:
        return pd.Series(index=expected, dtype=float)
    return pd.Series(values, dtype=float).reindex(expected)


def _context_row(
    coin: str,
    universe: dict[str, dict],
    context: dict[str, dict],
    first_included_at: str,
    window_start: str,
    window_end: str,
) -> dict:
    row = {**universe.get(coin, {}), **context.get(coin, {})}
    row["first_included_at"] = first_included_at
    row["window_start"] = window_start
    row["window_end"] = window_end
    return row


def _select_active_assets(
    info: Info,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], dict[str, dict], dict[str, dict]]:
    ranking, universe, context = _load_live_ranking(info)
    accepted_candles: dict[str, pd.DataFrame] = {}
    accepted_volume: dict[str, float] = {}
    for coin, day_ntl_vlm in ranking[["name", "dayNtlVlm"]].itertuples(index=False):
        candles = _fetch_hourly_candles(info, coin, start, end)
        if not _has_full_active_history(candles):
            continue
        accepted_candles[coin] = candles
        accepted_volume[coin] = float(day_ntl_vlm)
        if len(accepted_candles) >= ACTIVE_ASSET_COUNT:
            break
        time.sleep(0.2)
    if len(accepted_candles) != ACTIVE_ASSET_COUNT:
        raise RuntimeError(f"Expected {ACTIVE_ASSET_COUNT} active assets, found {len(accepted_candles)}")
    return accepted_candles, accepted_volume, universe, context


def _merge_series(left: pd.Series | None, right: pd.Series | None, *, dtype=None) -> pd.Series:
    parts = []
    if left is not None and not left.empty:
        parts.append(left)
    if right is not None and not right.empty:
        parts.append(right)
    if not parts:
        return pd.Series(dtype=dtype or float)
    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if dtype is not None:
        merged = merged.astype(dtype)
    return merged


def _update_archive_dataset(
    info: Info,
    selected_candles: dict[str, pd.DataFrame],
    active_coins: list[str],
    first_included_at: dict[str, str],
    active_start: pd.Timestamp,
    active_end: pd.Timestamp,
    universe: dict[str, dict],
    context: dict[str, dict],
) -> tuple[dict[str, pd.DataFrame], dict]:
    previous_manifest = _read_manifest(ARCHIVE_DIR)
    previous_frames = _read_machine_frames(ARCHIVE_DIR)
    archive_first_included = dict(previous_manifest.get("first_included_at", {}))
    for coin in active_coins:
        archive_first_included.setdefault(coin, first_included_at[coin])
    archive_coins = sorted(archive_first_included)
    latest_hour = active_end.tz_localize(None)

    panel_series: dict[str, dict[str, pd.Series]] = {name: {} for name in CORE_PANELS}
    metadata: dict[str, dict] = {}
    for coin in archive_coins:
        existing_close = previous_frames.get("close", pd.DataFrame()).get(coin)
        last_archived = None if existing_close is None or existing_close.dropna().empty else pd.to_datetime(existing_close.dropna().index.max())
        fetched_candles = None
        fetched_funding = None
        fetch_start = pd.Timestamp(archive_first_included[coin]).tz_localize(None) if last_archived is None else last_archived + pd.Timedelta(hours=1)
        if coin in active_coins and last_archived is None:
            fetched_candles = selected_candles[coin]
            fetched_funding = _fetch_hourly_funding(info, coin, active_start, active_end)
        elif fetch_start <= latest_hour:
            fetch_start_utc = fetch_start.tz_localize("UTC")
            end_utc = latest_hour.tz_localize("UTC")
            fetched_candles = _fetch_hourly_candles(info, coin, fetch_start_utc, end_utc)
            fetched_funding = _fetch_hourly_funding(info, coin, fetch_start_utc, end_utc)

        for field in CORE_PANELS:
            existing_series = previous_frames.get(field, pd.DataFrame()).get(coin)
            base_series = existing_series.copy() if existing_series is not None else None

            fetched_series = None
            if fetched_candles is not None and not fetched_candles.empty:
                if field == "open":
                    fetched_series = pd.to_numeric(fetched_candles["o"], errors="coerce")
                elif field == "high":
                    fetched_series = pd.to_numeric(fetched_candles["h"], errors="coerce")
                elif field == "low":
                    fetched_series = pd.to_numeric(fetched_candles["l"], errors="coerce")
                elif field == "close":
                    fetched_series = pd.to_numeric(fetched_candles["c"], errors="coerce")
                elif field == "volume":
                    fetched_series = pd.to_numeric(fetched_candles["v"], errors="coerce")
                elif field == "trades":
                    fetched_series = pd.to_numeric(fetched_candles["n"], errors="coerce")
                elif field == "tradable":
                    fetched_series = pd.Series(True, index=fetched_candles.index, dtype=bool)
            if field == "funding":
                fetched_series = fetched_funding
            if field == "tradable":
                merged = _merge_series(base_series, fetched_series, dtype=bool)
            else:
                merged = _merge_series(base_series, fetched_series)
            panel_series[field][coin] = merged

        first_hour = pd.Timestamp(archive_first_included[coin]).tz_localize(None)
        metadata[coin] = _context_row(
            coin,
            universe,
            context,
            first_included_at=archive_first_included[coin],
            window_start=str(first_hour),
            window_end=str(latest_hour),
        )
        time.sleep(0.2)

    archive_frames: dict[str, pd.DataFrame] = {}
    for name, mapping in panel_series.items():
        frame = pd.DataFrame(mapping).sort_index().sort_index(axis=1)
        frame.index.name = "date"
        if name == "tradable":
            frame = frame.fillna(False).astype(bool)
        archive_frames[name] = frame

    _reset_dataset_dir(ARCHIVE_DIR)
    for name, frame in archive_frames.items():
        _write_panel(ARCHIVE_DIR, name, frame)

    has_ohlcv = (
        archive_frames["open"].notna()
        & archive_frames["high"].notna()
        & archive_frames["low"].notna()
        & archive_frames["close"].notna()
        & archive_frames["volume"].notna()
        & archive_frames["trades"].notna()
    )
    _write_panel(ARCHIVE_DIR, "has_ohlcv", has_ohlcv)
    _write_panel(ARCHIVE_DIR, "has_funding", archive_frames["funding"].notna())
    _write_machine_json(ARCHIVE_DIR, "metadata.json", metadata)

    current_asset_ctxs = pd.DataFrame.from_dict(metadata, orient="index").sort_index()
    current_asset_ctxs.index.name = "name"
    _write_panel(ARCHIVE_DIR, "current_asset_ctxs", current_asset_ctxs)

    manifest = {
        "dataset": "archive_1h",
        "coins": archive_coins,
        "first_included_at": archive_first_included,
        "start": None if archive_frames["close"].empty else str(archive_frames["close"].dropna(how="all").index.min()),
        "end": None if archive_frames["close"].empty else str(archive_frames["close"].dropna(how="all").index.max()),
        "bars": int(len(archive_frames["close"])),
    }
    (ARCHIVE_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (ARCHIVE_DIR / "schema.json").write_text(
        json.dumps(
            {
                "provider": "hyperliquid",
                "schema_version": 1,
                "dataset": "archive_1h",
                "written_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "bars": int(len(archive_frames["close"])),
                "coins": archive_coins,
                "interval": "1h",
                "start": manifest["start"],
                "end": manifest["end"],
            },
            indent=2,
        )
    )
    _write_long_csv(ARCHIVE_DIR, archive_frames)
    return archive_frames, manifest


def _build_active_from_archive(
    archive_frames: dict[str, pd.DataFrame],
    active_coins: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    volume_ranking: dict[str, float],
    first_included_at: dict[str, str],
    universe: dict[str, dict],
    context: dict[str, dict],
    *,
    membership_unchanged: bool,
) -> None:
    _reset_dataset_dir(ACTIVE_DIR)
    index = pd.date_range(start.tz_localize(None), end.tz_localize(None), freq="1h", name="date")

    active_frames = {}
    for name in CORE_PANELS:
        frame = archive_frames[name].reindex(index=index, columns=active_coins)
        if name == "tradable":
            frame = frame.fillna(False).astype(bool)
        active_frames[name] = frame
        _write_panel(ACTIVE_DIR, name, frame)

    has_ohlcv = (
        active_frames["open"].notna()
        & active_frames["high"].notna()
        & active_frames["low"].notna()
        & active_frames["close"].notna()
        & active_frames["volume"].notna()
        & active_frames["trades"].notna()
    )
    _write_panel(ACTIVE_DIR, "has_ohlcv", has_ohlcv)
    _write_panel(ACTIVE_DIR, "has_funding", active_frames["funding"].notna())
    _write_panel(ACTIVE_DIR, "is_research_eligible", has_ohlcv)

    metadata = {
        coin: _context_row(
            coin,
            universe,
            context,
            first_included_at=first_included_at[coin],
            window_start=str(start.tz_localize(None)),
            window_end=str(end.tz_localize(None)),
        )
        for coin in active_coins
    }
    _write_machine_json(ACTIVE_DIR, "metadata.json", metadata)

    current_asset_ctxs = pd.DataFrame.from_dict(metadata, orient="index").sort_index()
    current_asset_ctxs.index.name = "name"
    _write_panel(ACTIVE_DIR, "current_asset_ctxs", current_asset_ctxs)

    manifest = {
        "dataset": "active_1h",
        "window_start": str(start.tz_localize(None)),
        "window_end": str(end.tz_localize(None)),
        "window_hours": ACTIVE_HOURS,
        "coins": active_coins,
        "selection_rule": "descending_current_dayNtlVlm_then_full_5000h_ohlcv_history",
        "volume_ranking": volume_ranking,
        "source": "archive_1h",
        "membership_unchanged": membership_unchanged,
    }
    (ACTIVE_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (ACTIVE_DIR / "schema.json").write_text(
        json.dumps(
            {
                "provider": "hyperliquid",
                "schema_version": 1,
                "dataset": "active_1h",
                "written_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "bars": len(index),
                "coins": active_coins,
                "interval": "1h",
                "start": str(start.tz_localize(None)),
                "end": str(end.tz_localize(None)),
            },
            indent=2,
        )
    )
    _write_long_csv(ACTIVE_DIR, active_frames)


def main() -> None:
    info = _client()
    active_start, active_end = _active_window()
    selected_candles, volume_ranking, universe, context = _select_active_assets(info, active_start, active_end)
    active_coins = list(selected_candles.keys())
    first_included_at = {coin: str(active_start.tz_localize(None)) for coin in active_coins}

    previous_active = _read_manifest(ACTIVE_DIR)
    previous_coins = previous_active.get("coins", [])
    membership_unchanged = len(previous_coins) == len(active_coins) and set(previous_coins) == set(active_coins)

    archive_frames, archive_manifest = _update_archive_dataset(
        info,
        selected_candles,
        active_coins,
        first_included_at,
        active_start,
        active_end,
        universe,
        context,
    )
    archive_first_included = archive_manifest["first_included_at"]
    active_first_included = {coin: archive_first_included[coin] for coin in active_coins}
    _build_active_from_archive(
        archive_frames,
        active_coins,
        active_start,
        active_end,
        volume_ranking,
        active_first_included,
        universe,
        context,
        membership_unchanged=membership_unchanged,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "membership_unchanged": membership_unchanged,
                "active_1h": {
                    "coins": active_coins,
                    "start": str(active_start.tz_localize(None)),
                    "end": str(active_end.tz_localize(None)),
                    "path": str(ACTIVE_DIR),
                },
                "archive_1h": {
                    "coins": archive_manifest["coins"],
                    "start": archive_manifest["start"],
                    "end": archive_manifest["end"],
                    "path": str(ARCHIVE_DIR),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
