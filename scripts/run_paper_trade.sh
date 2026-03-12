#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 run.py --build-cache --cache-dir data/market_cache_1h --start 2025-01-01 --timeframe 1h --top-n 20 --no-ssl-verify
python3 paper_trade.py --data-dir data/market_cache_1h --output paper/live_signal_latest.json
