#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JOB="5 * * * * cd '$ROOT' && ./scripts/run_paper_trade.sh >> '$ROOT/paper/paper_trade.log' 2>&1"

mkdir -p "$ROOT/paper"
(
  crontab -l 2>/dev/null | grep -Fv "./scripts/run_paper_trade.sh" || true
  echo "$JOB"
) | crontab -

echo "Installed cron job:"
echo "$JOB"
