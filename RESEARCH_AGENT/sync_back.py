from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "context_manifest.json"


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def normalize_paths(paths: list[str]) -> list[str]:
    return [path.rstrip("/") for path in paths]


def syncable_paths(manifest: dict[str, Any]) -> list[str]:
    base = set(normalize_paths(list(manifest.get("editable_paths", []))))
    base.update(normalize_paths(list(manifest.get("outputs", []))))
    ordered = sorted(base)
    collapsed: list[str] = []
    for path in ordered:
        if any(path == existing or path.startswith(existing + "/") for existing in collapsed):
            continue
        collapsed.append(path)
    return collapsed


def files_differ(source: Path, target: Path) -> bool:
    if not target.exists():
        return True
    if source.is_dir() != target.is_dir():
        return True
    if source.is_dir():
        source_entries = sorted(path.relative_to(source).as_posix() for path in source.rglob("*"))
        target_entries = sorted(path.relative_to(target).as_posix() for path in target.rglob("*"))
        return source_entries != target_entries
    return not filecmp.cmp(source, target, shallow=False)


def sync_entry(source_root: Path, target_root: Path, relative_path: str, *, apply: bool) -> dict[str, Any]:
    source = source_root / relative_path
    target = target_root / relative_path
    if not source.exists():
        return {"path": relative_path, "status": "missing_source"}
    changed = files_differ(source, target)
    if not changed:
        return {"path": relative_path, "status": "unchanged"}
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return {"path": relative_path, "status": "updated" if apply else "pending"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync allowed research workspace outputs back into the main repo.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--workspace", default=None, help="Workspace root path. Defaults to manifest.workspace_dir under source root.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--apply", action="store_true", help="Apply the sync. Without this flag, only print the plan.")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    manifest = load_manifest(args.manifest)
    workspace = Path(args.workspace or source_root / manifest.get("workspace_dir", ".research_agent_workspace")).resolve()
    if not workspace.exists():
        raise SystemExit(f"Workspace does not exist: {workspace}")
    results = [
        sync_entry(workspace, source_root, relative_path, apply=args.apply)
        for relative_path in syncable_paths(manifest)
    ]
    print(json.dumps({"status": "ok", "apply": bool(args.apply), "results": results}, indent=2))


if __name__ == "__main__":
    main()
