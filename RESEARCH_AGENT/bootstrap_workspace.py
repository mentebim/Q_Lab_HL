from __future__ import annotations

import argparse
import json
import os
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


def planned_workspace_entries(manifest: dict[str, Any]) -> dict[str, list[str]]:
    editable = normalize_paths(list(manifest.get("editable_paths", [])))
    fixed = normalize_paths(list(manifest.get("fixed_paths", [])))
    excluded = normalize_paths(list(manifest.get("excluded_paths", [])))
    return {
        "editable": sorted(set(editable)),
        "fixed": sorted(set(fixed)),
        "excluded": sorted(set(excluded)),
    }


def copy_entry(source_root: Path, output_root: Path, relative_path: str) -> None:
    source = source_root / relative_path
    target = output_root / relative_path
    if not source.exists():
        raise FileNotFoundError(f"Workspace source path does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def set_permissions_recursive(target: Path, *, writable: bool) -> None:
    if not target.exists():
        return
    if target.is_dir():
        os.chmod(target, 0o755 if writable else 0o555)
        for child in target.iterdir():
            set_permissions_recursive(child, writable=writable)
        return
    mode = 0o644 if writable else 0o444
    os.chmod(target, mode)


def remove_readonly(func, path, _excinfo) -> None:
    parent = Path(path).parent
    if parent.exists():
        os.chmod(parent, 0o755)
    os.chmod(path, 0o755)
    func(path)


def build_workspace(
    *,
    source_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    data_mode: str = "copy",
    force: bool = False,
) -> dict[str, Any]:
    plan = planned_workspace_entries(manifest)
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Workspace already exists: {output_root}")
        shutil.rmtree(output_root, onerror=remove_readonly)
    output_root.mkdir(parents=True, exist_ok=True)

    for relative_path in plan["editable"]:
        copy_entry(source_root, output_root, relative_path)
        set_permissions_recursive(output_root / relative_path, writable=True)

    for relative_path in plan["fixed"]:
        source = source_root / relative_path
        target = output_root / relative_path
        if relative_path == "data/market_cache_1h" and data_mode == "link":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            continue
        copy_entry(source_root, output_root, relative_path)
        set_permissions_recursive(target, writable=False)

    for relative_path in manifest.get("load_order", []):
        path = relative_path["path"] if isinstance(relative_path, dict) else relative_path
        normalized = str(path).rstrip("/")
        if normalized in plan["editable"] or normalized in plan["fixed"]:
            continue
        copy_entry(source_root, output_root, normalized)
        set_permissions_recursive(output_root / normalized, writable=False)

    policy = {
        "source_root": str(source_root),
        "workspace_root": str(output_root),
        "data_mode": data_mode,
        "editable_paths": plan["editable"],
        "fixed_paths": plan["fixed"],
        "excluded_paths": plan["excluded"],
        "run_command": manifest.get("run_command"),
    }
    (output_root / "WORKSPACE_POLICY.json").write_text(json.dumps(policy, indent=2, sort_keys=True))
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an isolated research workspace for the Q_Lab_HL research agent.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output", default=None, help="Workspace output path. Defaults to manifest.workspace_dir under source root.")
    parser.add_argument("--data-mode", choices=["copy", "link"], default="copy")
    parser.add_argument("--force", action="store_true", help="Overwrite the existing workspace if it already exists.")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    manifest = load_manifest(args.manifest)
    output = args.output or str(source_root / manifest.get("workspace_dir", ".research_agent_workspace"))
    policy = build_workspace(
        source_root=source_root,
        output_root=Path(output).resolve(),
        manifest=manifest,
        data_mode=args.data_mode,
        force=args.force,
    )
    print(json.dumps({"status": "ok", "workspace_root": policy["workspace_root"], "data_mode": policy["data_mode"]}, indent=2))


if __name__ == "__main__":
    main()
