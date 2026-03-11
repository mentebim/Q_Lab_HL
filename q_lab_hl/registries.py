from __future__ import annotations

import csv
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strategy_hash(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def default_candidate_id(strategy_path: str | Path) -> str:
    return Path(strategy_path).stem + "_" + now_utc_iso().replace(":", "").replace("-", "")


@dataclass
class TSVRegistry:
    path: Path

    def upsert(self, row: dict) -> None:
        rows = self.rows()
        key = row["candidate_id"]
        replaced = False
        for i, existing in enumerate(rows):
            if existing.get("candidate_id") == key:
                rows[i] = {**existing, **row}
                replaced = True
                break
        if not replaced:
            rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for item in rows for k in item.keys()})
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        return [{key: _coerce_value(value) for key, value in row.items()} for row in rows]

    def latest_for_candidate(self, candidate_id: str) -> dict | None:
        matches = [row for row in self.rows() if row.get("candidate_id") == candidate_id]
        return matches[-1] if matches else None


def research_registry(root: str | Path) -> TSVRegistry:
    return TSVRegistry(Path(root) / "research.tsv")


def audit_registry(root: str | Path) -> TSVRegistry:
    return TSVRegistry(Path(root) / "audit.tsv")


def _coerce_value(value):
    if value in ("", None):
        return value
    if value in ("True", "False"):
        return value == "True"
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return value
