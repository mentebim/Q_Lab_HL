from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q_lab_hl.cache import CacheBuildConfig, build_hyperliquid_cache, validate_market_cache_dir
from q_lab_hl.ingest import HyperliquidInfoClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the trailing window of the local Hyperliquid cache.")
    parser.add_argument("--data-dir", default="data/market_cache_1h")
    parser.add_argument("--refresh-days", type=int, default=45)
    parser.add_argument("--timeframe", default=None, help="Override timeframe instead of reading schema.json")
    parser.add_argument("--no-ssl-verify", action="store_true")
    parser.add_argument("--max-data-lag-hours", type=float, default=3.0, help="Fail if the refreshed cache is older than this.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    close = pd.read_parquet(data_dir / "close.parquet")
    last_ts = pd.Timestamp(close.index.max())
    coins = list(close.columns)
    schema_path = data_dir / "schema.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}
    interval = args.timeframe or schema.get("interval") or schema.get("config", {}).get("interval", "1h")
    start = (last_ts - pd.Timedelta(days=args.refresh_days)).isoformat()
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("h").isoformat()
    client = HyperliquidInfoClient(verify_ssl=not args.no_ssl_verify)
    with tempfile.TemporaryDirectory(prefix="q_lab_hl_refresh_") as tmpdir:
        build_hyperliquid_cache(
            tmpdir,
            config=CacheBuildConfig(
                start=start,
                end=end,
                interval=interval,
                max_bars_per_call=int(schema.get("config", {}).get("max_bars_per_call", 5000)),
                max_hours_per_funding_call=int(schema.get("config", {}).get("max_hours_per_funding_call", 24 * 30)),
            ),
            client=client,
            coins=coins,
        )
        for name in ("open", "high", "low", "close", "volume", "trades", "funding", "tradable"):
            src = Path(tmpdir) / f"{name}.parquet"
            if not src.exists():
                continue
            current = pd.read_parquet(data_dir / f"{name}.parquet") if (data_dir / f"{name}.parquet").exists() else pd.DataFrame()
            fresh = pd.read_parquet(src)
            merged = merge_matrix(current, fresh, pd.Timestamp(start))
            merged.to_parquet(data_dir / f"{name}.parquet")
        metadata_src = Path(tmpdir) / "metadata.json"
        schema_src = Path(tmpdir) / "schema.json"
        if metadata_src.exists():
            (data_dir / "metadata.json").write_text(metadata_src.read_text())
        if schema_src.exists():
            schema_payload = json.loads(schema_src.read_text())
            preserved_start = schema.get("config", {}).get("start", schema.get("start", schema_payload["config"].get("start")))
            schema_payload["config"]["start"] = preserved_start
            schema_payload["start"] = preserved_start
            close_merged = pd.read_parquet(data_dir / "close.parquet")
            schema_payload["bars"] = int(len(close_merged))
            schema_payload["coins"] = list(close_merged.columns)
            if len(close_merged.index):
                schema_payload["end"] = str(pd.Timestamp(close_merged.index.max()))
            (data_dir / "schema.json").write_text(json.dumps(schema_payload, indent=2))
    validation = validate_market_cache_dir(
        data_dir,
        interval=interval,
        max_data_lag_hours=args.max_data_lag_hours,
    )
    print(json.dumps({"status": "ok", "data_dir": str(data_dir), "start": start, "end": end, "coins": len(coins), "validation": validation}, indent=2))


def merge_matrix(current: pd.DataFrame, fresh: pd.DataFrame, refresh_start: pd.Timestamp) -> pd.DataFrame:
    current_frame = pd.DataFrame(current).copy()
    fresh_frame = pd.DataFrame(fresh).copy()
    if not current_frame.empty:
        current_frame.index = pd.to_datetime(current_frame.index)
    if not fresh_frame.empty:
        fresh_frame.index = pd.to_datetime(fresh_frame.index)
    history = current_frame[current_frame.index < refresh_start] if not current_frame.empty else current_frame
    merged = pd.concat([history, fresh_frame], axis=0).sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index(axis=1)
    return merged


if __name__ == "__main__":
    main()
