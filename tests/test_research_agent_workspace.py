from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module(path: str | Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


BOOTSTRAP = load_module(
    Path(__file__).resolve().parents[1] / "RESEARCH_AGENT" / "bootstrap_workspace.py",
    "research_agent_bootstrap",
)
SYNC_BACK = load_module(
    Path(__file__).resolve().parents[1] / "RESEARCH_AGENT" / "sync_back.py",
    "research_agent_sync_back",
)


class ResearchAgentWorkspaceTests(unittest.TestCase):
    def test_build_workspace_copies_editable_and_locks_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "autoresearch").mkdir()
            (root / "autoresearch" / "config.agent.json").write_text("{}")
            (root / "q_lab_hl").mkdir()
            (root / "q_lab_hl" / "judge.py").write_text("x = 1\n")
            (root / "data").mkdir()
            (root / "data" / "market_cache_1h").mkdir(parents=True)
            (root / "data" / "market_cache_1h" / "schema.json").write_text("{}")
            (root / "strategy.py").write_text("VALUE = 1\n")
            (root / "strategy_model.py").write_text("MODEL = 1\n")
            (root / "README.md").write_text("# repo\n")
            (root / "RESEARCH_PROMPT.md").write_text("# prompt\n")
            (root / "autoresearch.py").write_text("print('ok')\n")
            (root / "run.py").write_text("print('run')\n")
            (root / "pyproject.toml").write_text("[project]\nname='x'\n")

            manifest = {
                "workspace_dir": ".research_agent_workspace",
                "load_order": [{"path": "README.md"}, {"path": "RESEARCH_PROMPT.md"}],
                "editable_paths": ["autoresearch/", "strategy.py", "strategy_model.py"],
                "fixed_paths": ["q_lab_hl/", "autoresearch.py", "run.py", "pyproject.toml", "data/market_cache_1h/"],
                "excluded_paths": ["execution/"],
                "run_command": "python3 autoresearch.py --config autoresearch/config.agent.json",
            }
            workspace = root / ".research_agent_workspace"
            policy = BOOTSTRAP.build_workspace(
                source_root=root,
                output_root=workspace,
                manifest=manifest,
                data_mode="copy",
                force=False,
            )
            self.assertEqual(policy["workspace_root"], str(workspace))
            self.assertTrue((workspace / "strategy.py").exists())
            self.assertTrue((workspace / "q_lab_hl" / "judge.py").exists())
            self.assertTrue((workspace / "WORKSPACE_POLICY.json").exists())
            self.assertFalse((workspace / "execution").exists())

    def test_syncable_paths_include_editable_and_outputs(self):
        manifest = {
            "editable_paths": ["autoresearch/", "strategy.py"],
            "outputs": ["autoresearch/results/", "autoresearch/leaderboard.jsonl"],
        }
        paths = SYNC_BACK.syncable_paths(manifest)
        self.assertEqual(paths, ["autoresearch", "strategy.py"])

    def test_sync_entry_reports_pending_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            repo = Path(tmp) / "repo"
            workspace.mkdir()
            repo.mkdir()
            (workspace / "strategy.py").write_text("A = 2\n")
            result = SYNC_BACK.sync_entry(workspace, repo, "strategy.py", apply=False)
            self.assertEqual(result["status"], "pending")
            self.assertFalse((repo / "strategy.py").exists())


if __name__ == "__main__":
    unittest.main()
