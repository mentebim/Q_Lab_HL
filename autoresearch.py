from __future__ import annotations

import argparse
import json
from pathlib import Path

from q_lab_hl.autoresearch import load_experiment_spec, override_spec, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic bounded autoresearch runner for Q_Lab_HL.")
    parser.add_argument("--config", type=str, required=True, help="Path to an autoresearch experiment spec JSON file.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Override experiment_id from the config.")
    parser.add_argument("--candidate-id", type=str, default=None, help="Override candidate_id from the config.")
    parser.add_argument("--hypothesis", type=str, default=None, help="Override hypothesis from the config.")
    parser.add_argument("--strategy-path", type=str, default=None, help="Override strategy path from the config.")
    parser.add_argument("--strategy-spec", type=str, default=None, help="Path to a JSON file with strategy spec overrides.")
    parser.add_argument("--execution-overrides", type=str, default=None, help="Path to a JSON file with execution overrides.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override parquet data directory from the config.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data mode.")
    parser.add_argument("--no-write-result", action="store_true", help="Print JSON only and skip result file output.")
    parser.add_argument("--no-append-leaderboard", action="store_true", help="Skip leaderboard append and family-matrix updates.")
    parser.add_argument("--skip-journal-check", action="store_true", help="Skip enforcement that the previous result has been logged to research_journal.jsonl.")
    args = parser.parse_args()

    raw_payload = json.loads(Path(args.config).read_text())
    _validate_research_metadata(raw_payload, args.config)
    spec = load_experiment_spec(args.config)
    spec = override_spec(
        spec,
        experiment_id=args.experiment_id,
        candidate_id=args.candidate_id,
        hypothesis=args.hypothesis,
        strategy_path=args.strategy_path,
        strategy_spec=_load_optional_json(args.strategy_spec),
        execution_overrides=_load_optional_json(args.execution_overrides),
        data_dir=args.data_dir,
        synthetic=True if args.synthetic else None,
    )
    if not args.skip_journal_check:
        _ensure_previous_result_logged(spec)
    result = run_experiment(
        spec,
        write_result=not args.no_write_result,
        append_leaderboard=not args.no_append_leaderboard,
    )
    print(json.dumps(result, indent=2, sort_keys=True))

def _load_optional_json(path: str | None):
    if path is None:
        return None
    with open(path) as handle:
        return json.load(handle)


def _validate_research_metadata(payload: dict, config_path: str) -> None:
    metadata = payload.get("research_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"{config_path} is missing top-level research_metadata. "
            "Copy the current candidate.template.json and fill in its metadata block."
        )
    required = (
        "phase",
        "family_tag",
        "parent_candidate_id",
        "one_change",
        "expected_effect",
        "previous_failure_type",
        "previous_failed_checks",
        "pivot_reason",
        "anomaly_audit",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"{config_path} research_metadata is missing required keys: {', '.join(missing)}")
    anomaly = metadata.get("anomaly_audit")
    if not isinstance(anomaly, dict):
        raise ValueError(f"{config_path} research_metadata.anomaly_audit must be an object")
    anomaly_required = ("required", "trigger_source_candidate_id", "reason")
    missing_anomaly = [key for key in anomaly_required if key not in anomaly]
    if missing_anomaly:
        raise ValueError(
            f"{config_path} research_metadata.anomaly_audit is missing required keys: {', '.join(missing_anomaly)}"
        )


def _ensure_previous_result_logged(spec) -> None:
    leaderboard_path = Path(spec.recording.leaderboard_path)
    if not leaderboard_path.exists():
        return
    rows = [json.loads(line) for line in leaderboard_path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    latest = rows[-1]
    latest_candidate_id = latest.get("candidate_id")
    if not latest_candidate_id:
        return
    journal_path = leaderboard_path.with_name("research_journal.jsonl")
    if not journal_path.exists():
        raise ValueError(
            f"Previous result '{latest_candidate_id}' exists but {journal_path} does not. "
            "Log the last experiment before launching a new one."
        )
    entries = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(
            f"Previous result '{latest_candidate_id}' exists but the journal is empty. "
            "Log the last experiment before launching a new one."
        )
    last = entries[-1]
    logged_id = (
        last.get("after_experiment")
        or last.get("candidate_id")
        or last.get("experiment_id")
    )
    if logged_id != latest_candidate_id:
        raise ValueError(
            "The latest leaderboard result has not been journaled. "
            f"Latest result: '{latest_candidate_id}', latest journal entry: '{logged_id}'. "
            "Append a journal entry for the latest result before starting a new run."
        )


if __name__ == "__main__":
    main()
