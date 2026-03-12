#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/scripts/com.stackcapital.q_lab_hl.papertrade.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.stackcapital.q_lab_hl.papertrade.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/paper"
cp "$PLIST_SRC" "$PLIST_DST"
launchctl bootout "gui/$(id -u)"/com.stackcapital.q_lab_hl.papertrade >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"
launchctl enable "gui/$(id -u)"/com.stackcapital.q_lab_hl.papertrade
launchctl kickstart -k "gui/$(id -u)"/com.stackcapital.q_lab_hl.papertrade
launchctl print "gui/$(id -u)"/com.stackcapital.q_lab_hl.papertrade
