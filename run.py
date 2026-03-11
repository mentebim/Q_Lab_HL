from __future__ import annotations

import argparse
import pandas as pd

from q_lab_hl.audit import AuditFamilyStore
from q_lab_hl.artifacts import package_candidate_artifact
from q_lab_hl.backtest import load_strategy
from q_lab_hl.cache import CacheBuildConfig, build_hyperliquid_cache
from q_lab_hl.config import CACHE_DIR, ExecutionConfig
from q_lab_hl.data import DataStore
from q_lab_hl.evaluate import evaluate, format_metrics
from q_lab_hl.ingest import HyperliquidInfoClient
from q_lab_hl.promotion import evaluate_for_paper
from q_lab_hl.registries import audit_registry, default_candidate_id, research_registry, strategy_hash, git_commit, now_utc_iso
from q_lab_hl.search import TimeSeriesCVConfig, format_grid_results, run_walk_forward_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Q_Lab_HL long-short stat-arb harness")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with matrix parquet market panels")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic market data")
    parser.add_argument("--build-cache", action="store_true", help="Build a real Hyperliquid parquet cache")
    parser.add_argument("--backtest", action="store_true", help="Run a period evaluation")
    parser.add_argument("--audit", action="store_true", help="Run audit metrics against the family store")
    parser.add_argument("--promote", action="store_true", help="Evaluate deterministic promotion and package an artifact")
    parser.add_argument("--grid-search", action="store_true", help="Run walk-forward grid search over strategy.PARAM_GRID")
    parser.add_argument("--cv-mode", type=str, default="walk_forward", choices=["walk_forward", "expanding", "rolling"], help="Time-series cross-validation mode")
    parser.add_argument("--period", type=str, default="inner", choices=["train", "inner", "outer", "test"])
    parser.add_argument("--candidate-id", type=str, default=None)
    parser.add_argument("--strategy-path", type=str, default="strategy.py")
    parser.add_argument("--artifact-root", type=str, default="artifacts")
    parser.add_argument("--registry-dir", type=str, default="registries")
    parser.add_argument("--cache-dir", type=str, default="data/market_cache_1h", help="Directory for built market cache or backtest input")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Cache build start timestamp")
    parser.add_argument("--end", type=str, default=None, help="Cache build end timestamp")
    parser.add_argument("--timeframe", type=str, default="1h", help="Hyperliquid candle interval, e.g. 15m, 1h, 4h")
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated explicit coin list")
    parser.add_argument("--top-n", type=int, default=25, help="Top current-liquidity coins to fetch when --coins is omitted")
    parser.add_argument("--min-current-day-ntl-vlm", type=float, default=None, help="Current day notional volume filter")
    parser.add_argument("--include-delisted", action="store_true", help="Include delisted markets in the cache")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification for API fetches")
    parser.add_argument("--folds", type=int, default=3, help="Number of walk-forward validation folds")
    parser.add_argument("--top-k", type=int, default=5, help="Number of grid-search results to print")
    parser.add_argument("--cv-train-bars", type=int, default=None, help="Training bars per fold for expanding/rolling CV")
    parser.add_argument("--cv-validation-bars", type=int, default=None, help="Validation bars per fold for expanding/rolling CV")
    parser.add_argument("--cv-step-bars", type=int, default=None, help="Step size between folds for expanding/rolling CV")
    parser.add_argument("--cv-gap-bars", type=int, default=0, help="Gap bars between train and validation windows")
    parser.add_argument("--cv-purge-bars", type=int, default=0, help="Bars removed from the end of train before validation")
    parser.add_argument("--cv-embargo-bars", type=int, default=0, help="Bars skipped between successive validation windows")
    args = parser.parse_args()

    if not any([args.build_cache, args.backtest, args.audit, args.promote, args.grid_search]):
        parser.print_help()
        return
    if not args.synthetic and not args.data_dir and not args.build_cache:
        raise SystemExit("Provide --data-dir for real data or use --synthetic.")

    if args.build_cache:
        coins = [coin.strip() for coin in args.coins.split(",") if coin.strip()] if args.coins else None
        client = HyperliquidInfoClient(verify_ssl=not args.no_ssl_verify)
        summary = build_hyperliquid_cache(
            output_dir=args.cache_dir,
            config=CacheBuildConfig(
                start=args.start,
                end=args.end or pd.Timestamp.now("UTC").tz_localize(None).isoformat(),
                interval=args.timeframe,
                include_delisted=args.include_delisted,
                top_n=args.top_n if coins is None else None,
                min_current_day_ntl_vlm=args.min_current_day_ntl_vlm,
            ),
            client=client,
            coins=coins,
        )
        for key, value in summary.items():
            print(f"{key}: {value}")

    if not any([args.backtest, args.audit, args.promote, args.grid_search]):
        return

    data_root = args.data_dir or args.cache_dir
    data_store = DataStore.synthetic() if args.synthetic else DataStore.from_parquet_dir(data_root)
    strategy = load_strategy(args.strategy_path)
    execution = getattr(strategy, "EXECUTION", ExecutionConfig())
    candidate_id = args.candidate_id or default_candidate_id(args.strategy_path)
    strategy_digest = strategy_hash(args.strategy_path)
    commit = git_commit()
    research_log = research_registry(args.registry_dir)
    audit_log = audit_registry(args.registry_dir)
    inner_metrics = None
    outer_metrics = None

    if args.grid_search:
        cv = TimeSeriesCVConfig(
            mode=args.cv_mode,
            n_folds=args.folds,
            train_size=args.cv_train_bars,
            validation_size=args.cv_validation_bars,
            step_size=args.cv_step_bars,
            gap_size=args.cv_gap_bars,
            purge_size=args.cv_purge_bars,
            embargo_size=args.cv_embargo_bars,
        )
        results, folds = run_walk_forward_grid(
            strategy,
            data_store,
            execution=execution,
            cv=cv,
        )
        print(f"model_family: {getattr(strategy, 'MODEL_FAMILY', 'unknown')}")
        print(f"cv_mode: {cv.mode}")
        print(f"grid_size: {len(results)}")
        print(f"folds: {len(folds)}")
        print(format_grid_results(results, top_k=args.top_k))

    if args.backtest:
        metrics = evaluate(strategy, data_store, period=args.period, execution=execution)
        metrics["candidate_id"] = candidate_id
        if args.period == "inner":
            inner_metrics = metrics
            research_log.upsert(
                {
                    "candidate_id": candidate_id,
                    "created_at": now_utc_iso(),
                    "strategy_hash": strategy_digest,
                    "git_commit": commit,
                    "period": args.period,
                    "score_inner": metrics.get("score_inner"),
                    "active_sharpe_annualized": metrics.get("active_sharpe_annualized"),
                    "turnover": metrics.get("turnover"),
                    "beta_to_market": metrics.get("beta_to_market"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "status": "backtested",
                    "notes": "cli_backtest",
                }
            )
        print(format_metrics(metrics))

    if args.audit:
        family_store = AuditFamilyStore(CACHE_DIR / "audit_family.parquet")
        family_matrix = family_store.matrix()
        metrics = evaluate(
            strategy,
            data_store,
            period=args.period,
            execution=execution,
            family_matrix=family_matrix,
            candidate_id=candidate_id,
        )
        if args.period != "inner":
            family_store.upsert(candidate_id, metrics["active_returns_series"])
            outer_metrics = metrics
            ci_low, ci_high = metrics.get("bootstrap_sharpe_ci", (None, None))
            audit_log.upsert(
                {
                    "candidate_id": candidate_id,
                    "audited_at": now_utc_iso(),
                    "strategy_hash": strategy_digest,
                    "git_commit": commit,
                    "period": args.period,
                    "DSR": metrics.get("DSR"),
                    "bootstrap_sharpe_ci_low": ci_low,
                    "bootstrap_sharpe_ci_high": ci_high,
                    "active_sharpe_annualized": metrics.get("active_sharpe_annualized"),
                    "turnover": metrics.get("turnover"),
                    "beta_to_market": metrics.get("beta_to_market"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "promotable": False,
                    "status": "audited",
                }
            )
        metrics["candidate_id"] = candidate_id
        print(format_metrics(metrics))

    if args.promote:
        if inner_metrics is None:
            saved_inner = research_log.latest_for_candidate(candidate_id)
            if saved_inner is None:
                raise SystemExit(f"No inner backtest record found for {candidate_id}. Run --backtest --period inner first.")
            inner_metrics = saved_inner
        if outer_metrics is None:
            saved_outer = audit_log.latest_for_candidate(candidate_id)
            if saved_outer is None:
                raise SystemExit(f"No audit record found for {candidate_id}. Run --audit --period outer first.")
            outer_metrics = saved_outer
        decision = evaluate_for_paper(inner_metrics, outer_metrics)
        output = {"candidate_id": candidate_id, "promotion_status": decision["status"], "failed_checks": decision["failed_checks"]}
        print(format_metrics(output))
        if decision["approved"]:
            artifact_dir = package_candidate_artifact(
                candidate_id=candidate_id,
                strategy_path=args.strategy_path,
                inner_metrics=inner_metrics,
                audit_metrics=outer_metrics,
                execution=execution,
                artifact_root=args.artifact_root,
                promotion_status=decision["status"],
            )
            audit_row = audit_log.latest_for_candidate(candidate_id) or {}
            audit_row.update(
                {
                    "candidate_id": candidate_id,
                    "audited_at": audit_row.get("audited_at", now_utc_iso()),
                    "strategy_hash": strategy_digest,
                    "git_commit": commit,
                    "period": audit_row.get("period", "outer"),
                    "DSR": audit_row.get("DSR"),
                    "bootstrap_sharpe_ci_low": audit_row.get("bootstrap_sharpe_ci_low"),
                    "bootstrap_sharpe_ci_high": audit_row.get("bootstrap_sharpe_ci_high"),
                    "active_sharpe_annualized": audit_row.get("active_sharpe_annualized"),
                    "turnover": audit_row.get("turnover"),
                    "beta_to_market": audit_row.get("beta_to_market"),
                    "max_drawdown": audit_row.get("max_drawdown"),
                    "promotable": True,
                    "status": decision["status"],
                }
            )
            audit_log.upsert(audit_row)
            print(f"artifact_dir: {artifact_dir}")


if __name__ == "__main__":
    main()
