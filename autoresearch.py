from __future__ import annotations

import argparse
import json

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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
