from __future__ import annotations

import json
import shutil
from pathlib import Path

from q_lab_hl.config import ExecutionConfig


def package_candidate_artifact(
    candidate_id: str,
    strategy_path: str | Path,
    inner_metrics: dict,
    audit_metrics: dict,
    execution: ExecutionConfig,
    artifact_root: str | Path,
    promotion_status: str = "approved_for_paper",
) -> Path:
    root = Path(artifact_root)
    artifact_dir = root / candidate_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = Path(strategy_path)
    shutil.copy2(strategy_path, artifact_dir / "strategy.py")
    manifest = {
        "artifact_id": candidate_id,
        "promotion_status": promotion_status,
        "strategy_file": "strategy.py",
        "execution": execution.__dict__,
    }
    metrics = {"inner": inner_metrics, "audit": audit_metrics}
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (artifact_dir / "config.json").write_text(json.dumps({"execution": execution.__dict__}, indent=2, default=str))
    return artifact_dir

