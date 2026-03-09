"""
Q_Lab: hardened quantitative research harness.

This file owns the fixed data pipeline, point-in-time data access, backtest
engine, and evaluation workflow. Normal autonomous research edits `strategy.py`
only; this file changes only during explicit infrastructure hardening passes.

Usage:
    uv run prepare.py --download
    uv run prepare.py --backtest
    uv run prepare.py --audit
    uv run prepare.py --test
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import integrate
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import norm
from scipy.stats import skew as scipy_skew

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()


# ---------------------------------------------------------------------------
# Section 1: Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "qlab"
COMPANIES_JSON = (
    Path(__file__).resolve().parent.parent
    / "stack-intelligence-engine"
    / "registry"
    / "companies.json"
)

BACKTEST_START = "2021-03-01"
TRAIN_END = "2023-03-01"
INNER_END = "2024-03-01"
OUTER_END = "2025-03-01"
BACKTEST_END = "2026-03-01"

INITIAL_CAPITAL = 1_000_000.0
COMMISSION_BPS = 5.0
SLIPPAGE_BPS = 5.0
TIME_BUDGET = 120

DEFAULT_BENCHMARK = "SPY"
DEFAULT_COUNTRIES = ("US",)
DEFAULT_MIN_HISTORY_DAYS = 252
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_DOLLAR_VOLUME = 5_000_000.0
CACHE_SCHEMA_VERSION = 2

GROSS_EXPOSURE_LIMIT = 1.00
NET_EXPOSURE_TARGET = 1.00
EXECUTION_MODE = "next_close"

AUDIT_DSR_THRESHOLD = 0.95
AUDIT_SPA_ALPHA = 0.05
PBO_MIN_FAMILY_SIZE = 8
PBO_SLICE_COUNT = 8

FMP_BASE = "https://financialmodelingprep.com/stable"
FRED_BASE = "https://api.stlouisfed.org/fred"
FMP_RATE_LIMIT = int(os.environ.get("FMP_RATE_LIMIT_PER_MIN", "300"))
FRED_RATE_LIMIT = int(os.environ.get("FRED_RATE_LIMIT_PER_MIN", "120"))
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "8"))
US_PRIMARY_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSE_ARCA", "CBOE", "OTC", "PNK"}
NON_EQUITY_NAME_RE = re.compile(
    r"\b(ETF|ETN|FUND|MONEY MARKET|MUNICIPAL|TREASURY BOND|TREASURY ETF|ADRHEDGED)\b",
    re.IGNORECASE,
)

RAW_STATEMENT_FIELDS = {
    "revenue": "revenue",
    "gross_profit": "grossProfit",
    "operating_income": "operatingIncome",
    "net_income": "netIncome",
    "ebitda": "ebitda",
    "assets": "totalAssets",
    "current_assets": "totalCurrentAssets",
    "current_liabilities": "totalCurrentLiabilities",
    "cash": "cashAndCashEquivalents",
    "book_equity": "totalStockholdersEquity",
    "inventory": "inventory",
    "receivables": "netReceivables",
    "shares_out": "weightedAverageShsOut",
    "cfo": "netCashProvidedByOperatingActivities",
    "capex": "capitalExpenditure",
    "free_cash_flow": "freeCashFlow",
}

STRICT_PIT_FILING_DATES = True

DERIVED_FUNDAMENTAL_FIELDS = {
    "book_to_price",
    "earnings_yield",
    "free_cash_flow_yield",
    "gross_profitability",
    "asset_growth",
    "leverage",
    "current_ratio",
    "roe",
}

LEGACY_RATIO_MAP = {
    "priceToBookRatio": "pb",
    "priceToBookRatioTTM": "pb",
    "earningsYield": "earnings_yield",
    "earningsYieldTTM": "earnings_yield",
    "freeCashFlowYield": "fcf_yield",
    "freeCashFlowYieldTTM": "fcf_yield",
    "returnOnEquity": "roe",
    "returnOnEquityTTM": "roe",
    "debtEquityRatio": "debt_to_equity",
    "debtEquityRatioTTM": "debt_to_equity",
    "currentRatio": "current_ratio",
    "currentRatioTTM": "current_ratio",
    "grossProfitMargin": "gross_margin",
    "grossProfitMarginTTM": "gross_margin",
    "revenueGrowth": "revenue_growth",
    "revenueGrowthTTM": "revenue_growth",
    "piotroskiIScore": "piotroski",
    "piotroskiIScoreTTM": "piotroski",
}

MACRO_SERIES = {
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "consumer_sentiment": "UMCSENT",
    "fed_funds": "FEDFUNDS",
}

AUDIT_RETURNS_FILE = CACHE_DIR / "audit_outer_returns.parquet"
AUDIT_REGISTRY_FILE = CACHE_DIR / "audit_registry.tsv"


# ---------------------------------------------------------------------------
# Section 2: Generic helpers
# ---------------------------------------------------------------------------


def ensure_cache_dir(cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def clear_cache_parquets(cache_dir: Path = CACHE_DIR) -> None:
    ensure_cache_dir(cache_dir)
    for path in cache_dir.glob("*.parquet"):
        path.unlink(missing_ok=True)
    for path in (AUDIT_RETURNS_FILE, AUDIT_REGISTRY_FILE):
        path.unlink(missing_ok=True)


def sha1_file(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]


def safe_float(value, default=np.nan):
    try:
        if value in (None, "", "None"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_date(value) -> pd.Timestamp | None:
    if value in (None, "", "NaT"):
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception:
        return None


def normalize_exchange(exchange: str | None) -> str:
    return str(exchange or "").strip().upper().replace(" ", "_")


def is_supported_us_equity(symbol: str | None, name: str | None, exchange: str | None, country: str | None = "US") -> bool:
    if not symbol or str(country or "").upper() != "US":
        return False
    norm_exchange = normalize_exchange(exchange)
    if norm_exchange and norm_exchange not in US_PRIMARY_EXCHANGES:
        return False
    if name and NON_EQUITY_NAME_RE.search(name):
        return False
    return True


def next_trading_day(index: pd.DatetimeIndex, dt: pd.Timestamp | None) -> pd.Timestamp | None:
    if dt is None or len(index) == 0:
        return None
    pos = index.searchsorted(pd.Timestamp(dt), side="right")
    if pos >= len(index):
        return None
    return index[pos]


def strip_timezone(value: pd.Series) -> pd.Series:
    value = pd.to_datetime(value, errors="coerce")
    if getattr(value.dt, "tz", None) is not None:
        return value.dt.tz_convert(None)
    return value


def sample_skewness(returns: pd.Series) -> float:
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(arr) < 3:
        return 0.0
    value = float(scipy_skew(arr, bias=False))
    return 0.0 if not np.isfinite(value) else value


def sample_kurtosis(returns: pd.Series) -> float:
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(arr) < 4:
        return 3.0
    value = float(scipy_kurtosis(arr, fisher=False, bias=False))
    return 3.0 if not np.isfinite(value) else value


def newey_west_daily_vol(returns: pd.Series, max_lag: int | None = None) -> float:
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(arr)
    if n < 2:
        return 0.0
    arr = arr - arr.mean()
    max_lag = max_lag if max_lag is not None else min(5, n - 1)
    gamma0 = np.mean(arr * arr)
    var = gamma0
    for lag in range(1, max_lag + 1):
        cov = np.mean(arr[lag:] * arr[:-lag])
        weight = 1 - lag / (max_lag + 1)
        var += 2 * weight * cov
    return float(max(var, 0.0) ** 0.5)


def sharpe_daily(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    excess = pd.Series(returns).dropna() - risk_free_daily
    if len(excess) < 2:
        return 0.0
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std)


def sharpe_standard_error(sr_hat: float, sample_len: int, skewness: float, kurt: float) -> float:
    if sample_len <= 1:
        return 0.0
    numer = 1 - skewness * sr_hat + ((kurt - 1) / 4) * (sr_hat ** 2)
    numer = max(numer, 1e-12)
    return float(math.sqrt(numer / (sample_len - 1)))


def sharpe_annualized_lo(returns: pd.Series, periods_per_year: int = 252) -> float:
    rets = pd.Series(returns).dropna()
    if len(rets) < 2:
        return 0.0
    mean = rets.mean()
    nw_vol = newey_west_daily_vol(rets)
    if nw_vol == 0:
        return 0.0
    return float((mean / nw_vol) * math.sqrt(periods_per_year))


def annual_return(returns: pd.Series) -> float:
    rets = pd.Series(returns).dropna()
    if len(rets) == 0:
        return 0.0
    total = float((1 + rets).prod())
    years = len(rets) / 252
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def max_drawdown(returns: pd.Series) -> float:
    rets = pd.Series(returns).fillna(0.0)
    if len(rets) == 0:
        return 0.0
    curve = (1 + rets).cumprod()
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return float(drawdown.min())


def sortino_ratio(returns: pd.Series) -> float:
    rets = pd.Series(returns).dropna()
    if len(rets) == 0:
        return 0.0
    downside = rets[rets < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return 0.0
    return float(rets.mean() / downside.std(ddof=1) * math.sqrt(252))


def calmar_ratio(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annual_return(returns) / mdd)


def expected_max_standard_normal_exact(n_trials: float) -> float:
    n_trials = float(max(1.0, n_trials))
    if n_trials <= 1.0:
        return 0.0

    def integrand(z):
        return n_trials * z * norm.pdf(z) * (norm.cdf(z) ** (n_trials - 1))

    value, _ = integrate.quad(integrand, -10, 10, limit=200)
    return float(value)


def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_threshold: float,
    sample_len: int,
    skewness: float,
    kurt: float,
) -> float:
    if sample_len <= 1:
        return 0.0
    denom = 1 - skewness * sr_hat + ((kurt - 1) / 4) * (sr_hat ** 2)
    denom = max(denom, 1e-12)
    stat = (sr_hat - sr_threshold) * math.sqrt(sample_len - 1) / math.sqrt(denom)
    return float(norm.cdf(stat))


def deflated_sharpe_ratio(
    sr_hat: float,
    sr_variance_across_trials: float,
    n_trials: float,
    sample_len: int,
    skewness: float,
    kurt: float,
) -> tuple[float, float]:
    n_trials = float(max(1.0, min(float(n_trials), 1_000_000.0)))
    sr_star = math.sqrt(max(sr_variance_across_trials, 0.0)) * expected_max_standard_normal_exact(
        n_trials
    )
    return probabilistic_sharpe_ratio(sr_hat, sr_star, sample_len, skewness, kurt), sr_star


def estimate_effective_independent_trials(matrix: pd.DataFrame) -> float:
    if matrix.shape[1] <= 1:
        return 1.0
    corr = matrix.corr().fillna(0.0).to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    vals = np.linalg.eigvalsh(corr)
    vals = np.clip(vals, 0.0, None)
    denom = float(np.square(vals).sum())
    if denom <= 0:
        return 1.0
    eff = float((vals.sum() ** 2) / denom)
    return float(min(max(eff, 1.0), matrix.shape[1]))


def estimate_stationary_block_length(returns: pd.Series) -> int:
    arr = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(arr)
    if n < 12:
        return max(2, n // 2)
    arr = arr - arr.mean()
    max_lag = min(int(n ** 0.5), n - 1)
    acf = []
    var = np.var(arr)
    if var == 0:
        return max(2, min(5, n // 2))
    for lag in range(1, max_lag + 1):
        acf.append(np.corrcoef(arr[lag:], arr[:-lag])[0, 1])
    acf = np.nan_to_num(np.asarray(acf), nan=0.0)
    tail = float(np.abs(acf).sum())
    block = int(round((n ** (1 / 3)) * (1 + tail)))
    return int(min(max(block, 2), max(2, n // 2)))


def stationary_bootstrap_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    p = 1.0 / max(block_length, 1)
    idx = np.empty(length, dtype=int)
    idx[0] = rng.integers(0, length)
    for i in range(1, length):
        if rng.random() < p:
            idx[i] = rng.integers(0, length)
        else:
            idx[i] = (idx[i - 1] + 1) % length
    return idx


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, int, float]:
    rets = pd.Series(returns).dropna()
    if len(rets) < 12:
        return (0.0, 0.0, 0, 0.0)
    rng = np.random.default_rng(seed)
    block_length = estimate_stationary_block_length(rets)
    sr_hat = sharpe_daily(rets)
    se_hat = sharpe_standard_error(sr_hat, len(rets), sample_skewness(rets), sample_kurtosis(rets))
    if se_hat == 0:
        return (sr_hat, sr_hat, block_length, 0.0)
    t_stats = []
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(len(rets), block_length, rng)
        sample = rets.iloc[idx].reset_index(drop=True)
        sr_star = sharpe_daily(sample)
        se_star = sharpe_standard_error(
            sr_star, len(sample), sample_skewness(sample), sample_kurtosis(sample)
        )
        if se_star == 0:
            continue
        t_stat = (sr_star - sr_hat) / se_star
        if np.isfinite(t_stat):
            t_stats.append(float(t_stat))
    if not t_stats:
        return (sr_hat, sr_hat, block_length, se_hat)
    alpha = (1 - ci) / 2
    q_low = float(np.quantile(t_stats, 1 - alpha))
    q_high = float(np.quantile(t_stats, alpha))
    if not np.isfinite(q_low) or not np.isfinite(q_high):
        return (sr_hat, sr_hat, block_length, se_hat)
    return (sr_hat - q_low * se_hat, sr_hat - q_high * se_hat, block_length, se_hat)


def spa_pvalue(active_return_matrix: pd.DataFrame, seed: int = 17, n_bootstrap: int = 500) -> float:
    matrix = active_return_matrix.dropna(how="all")
    if matrix.empty:
        return 1.0
    matrix = matrix.fillna(0.0)
    t_count = len(matrix)
    if t_count < 12 or matrix.shape[1] == 0:
        return 1.0

    means = matrix.mean()
    stds = matrix.std(ddof=1).replace(0, np.nan)
    observed = float(np.nanmax(np.sqrt(t_count) * means / stds))
    if not np.isfinite(observed):
        return 1.0

    centered = matrix - np.maximum(means, 0.0)
    block = estimate_stationary_block_length(centered.mean(axis=1))
    rng = np.random.default_rng(seed)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        idx = stationary_bootstrap_indices(t_count, block, rng)
        sample = centered.iloc[idx].reset_index(drop=True)
        sample_means = sample.mean()
        sample_stds = sample.std(ddof=1).replace(0, np.nan)
        sample_stat = np.sqrt(t_count) * sample_means / sample_stds
        if not np.isfinite(sample_stat).any():
            continue
        stat = np.nanmax(sample_stat)
        if np.isfinite(stat):
            bootstrap_stats.append(float(stat))
    if not bootstrap_stats:
        return 1.0
    return float(np.mean(np.asarray(bootstrap_stats) >= observed))


def compute_pbo(active_return_matrix: pd.DataFrame, n_slices: int = PBO_SLICE_COUNT) -> float | None:
    matrix = active_return_matrix.dropna(how="all").fillna(0.0)
    t_count, n_models = matrix.shape
    if n_models < 2 or t_count < n_slices * 4:
        return None
    slices = [idx for idx in np.array_split(np.arange(t_count), n_slices) if len(idx) > 0]
    split_ids = range(len(slices))
    train_size = len(slices) // 2
    combos = list(combinations(split_ids, train_size))
    if len(combos) > 70:
        rng = np.random.default_rng(123)
        picks = rng.choice(len(combos), size=70, replace=False)
        combos = [combos[i] for i in picks]

    logits = []
    for combo in combos:
        train_idx = np.concatenate([slices[i] for i in combo])
        test_idx = np.concatenate([slices[i] for i in split_ids if i not in combo])
        train = matrix.iloc[train_idx]
        test = matrix.iloc[test_idx]
        is_scores = train.mean() / train.std(ddof=1).replace(0, np.nan)
        best = is_scores.idxmax()
        oos_scores = test.mean() / test.std(ddof=1).replace(0, np.nan)
        rank = float(oos_scores.rank(pct=True, method="average").get(best, np.nan))
        if not np.isfinite(rank):
            continue
        clipped = float(np.clip(rank, 1e-6, 1 - 1e-6))
        logits.append(math.log(clipped / (1 - clipped)))
    if not logits:
        return None
    return float(np.mean(np.asarray(logits) <= 0))


def count_loc(filepath: str) -> int:
    try:
        with open(filepath) as handle:
            return sum(1 for line in handle if line.strip() and not line.strip().startswith("#"))
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Section 3: API clients and download pipeline
# ---------------------------------------------------------------------------


class FMPClient:
    def __init__(self, api_key: str | None = None, cache_dir: Path = CACHE_DIR):
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            raise ValueError("FMP_API_KEY not set. Add it to .env or environment.")
        self.cache_dir = Path(cache_dir)
        self.base_url = FMP_BASE
        self._request_times: list[float] = []
        self._rate_limit_lock = threading.Lock()

    def _rate_limit(self) -> None:
        while True:
            sleep_time = 0.0
            with self._rate_limit_lock:
                now = time.time()
                self._request_times = [t for t in self._request_times if now - t < 60]
                if len(self._request_times) < FMP_RATE_LIMIT:
                    self._request_times.append(now)
                    return
                sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get(self, endpoint: str, params: dict | None = None, max_retries: int = 3):
        params = dict(params or {})
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("Error Message"):
                    return None
                return data
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** (attempt + 1))
        return None

    def save_parquet(self, df: pd.DataFrame, name: str) -> None:
        ensure_cache_dir(self.cache_dir)
        df.to_parquet(self.cache_dir / f"{name}.parquet", index=False)

    def load_parquet(self, name: str) -> pd.DataFrame | None:
        path = self.cache_dir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None

    def cache_fresh(self, name: str, max_age_hours: int = 24) -> bool:
        path = self.cache_dir / f"{name}.parquet"
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < max_age_hours * 3600


class FREDClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._request_times: list[float] = []
        self._rate_limit_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _rate_limit(self) -> None:
        while True:
            sleep_time = 0.0
            with self._rate_limit_lock:
                now = time.time()
                self._request_times = [t for t in self._request_times if now - t < 60]
                if len(self._request_times) < FRED_RATE_LIMIT:
                    self._request_times.append(now)
                    return
                sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get(self, endpoint: str, params: dict | None = None, max_retries: int = 3):
        if not self.enabled:
            return None
        payload = dict(params or {})
        payload["api_key"] = self.api_key
        payload["file_type"] = "json"
        url = f"{FRED_BASE}/{endpoint}"
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, params=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** (attempt + 1))
        return None


def load_companies_json() -> dict:
    if not COMPANIES_JSON.exists():
        return {}
    with open(COMPANIES_JSON) as handle:
        data = json.load(handle)
    return data.get("companies", {})


def _coalesce_metadata_rows(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for key, value in incoming.items():
        if key == "ticker":
            merged[key] = value
            continue
        if merged.get(key) in (None, "", pd.NaT) and value not in (None, "", pd.NaT):
            merged[key] = value
    return merged


def cache_available_tickers(cache_dir: Path, prefix: str) -> set[str]:
    return {path.stem[len(prefix) + 1 :].replace("_", ".") for path in Path(cache_dir).glob(f"{prefix}_*.parquet")}


def load_reference_catalogs(client: "FMPClient") -> tuple[set[str], set[str], pd.DataFrame]:
    def load_or_fetch(name: str, endpoint: str, max_age_hours: int = 24) -> pd.DataFrame:
        cached = client.load_parquet(name)
        if cached is not None and client.cache_fresh(name, max_age_hours=max_age_hours):
            return cached
        data = client.get(endpoint)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
        if not df.empty:
            client.save_parquet(df, name)
        return df

    stock_list = load_or_fetch("ref_stock_list", "stock-list", max_age_hours=24)
    financial_symbols = load_or_fetch("ref_financial_statement_symbol_list", "financial-statement-symbol-list", max_age_hours=24)
    symbol_change = load_or_fetch("ref_symbol_change", "symbol-change", max_age_hours=48)
    stock_symbols = set(stock_list.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
    financial_stmt_symbols = set(financial_symbols.get("symbol", pd.Series(dtype=str)).dropna().astype(str))
    return stock_symbols, financial_stmt_symbols, symbol_change


def build_coverage_audit(meta_df: pd.DataFrame, cache_dir: Path, client: "FMPClient" | None = None) -> pd.DataFrame:
    coverage = meta_df.copy()
    stock_symbols: set[str] = set()
    statement_symbols: set[str] = set()
    symbol_change = pd.DataFrame()
    if client is not None:
        stock_symbols, statement_symbols, symbol_change = load_reference_catalogs(client)
    coverage["direct_stock_list_support"] = coverage["ticker"].isin(stock_symbols) if stock_symbols else False
    coverage["direct_financial_statement_symbol_support"] = coverage["ticker"].isin(statement_symbols) if statement_symbols else False
    coverage["symbol_change_new_symbol"] = None
    coverage["symbol_change_date"] = pd.NaT
    coverage["symbol_change_new_symbol_direct_support"] = False
    if not symbol_change.empty and {"oldSymbol", "newSymbol"}.issubset(symbol_change.columns):
        working = symbol_change.copy()
        working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
        working = working.sort_values("date").drop_duplicates(subset=["oldSymbol"], keep="last")
        successor = working.set_index("oldSymbol")["newSymbol"]
        successor_dates = working.set_index("oldSymbol")["date"]
        coverage["symbol_change_new_symbol"] = coverage["ticker"].map(successor)
        coverage["symbol_change_date"] = coverage["ticker"].map(successor_dates)
        if stock_symbols:
            coverage["symbol_change_new_symbol_direct_support"] = coverage["symbol_change_new_symbol"].isin(stock_symbols)
    for prefix in ["prices", "marketcap", "statements", "earnings", "sec"]:
        coverage[f"has_{prefix}"] = coverage["ticker"].isin(cache_available_tickers(cache_dir, prefix))
    coverage["vendor_backtest_supported"] = coverage["has_prices"]
    coverage["vendor_identity_weak"] = coverage["name"].fillna("").eq("") | coverage["exchange"].fillna("").eq("")
    coverage["vendor_support_status"] = np.select(
        [
            coverage["has_prices"],
            coverage["symbol_change_new_symbol_direct_support"] & ~coverage["has_prices"],
            coverage[["has_marketcap", "has_statements", "has_earnings", "has_sec"]].any(axis=1),
        ],
        [
            "backtest_supported",
            "resolved_symbol_candidate",
            "partial_vendor_support",
        ],
        default="unsupported_by_vendor",
    )
    return coverage


def build_us_universe_metadata(client: FMPClient) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    current = client.get("sp500-constituent") or []
    historical = client.get("historical-sp500-constituent") or []
    delisted = []
    page = 0
    while True:
        chunk = client.get("delisted-companies", params={"page": page, "limit": 1000}) or []
        if not chunk or not isinstance(chunk, list):
            break
        delisted.extend(chunk)
        if len(chunk) < 1000:
            break
        page += 1

    meta_by_ticker: dict[str, dict] = {}
    membership_rows = []
    tickers = set()

    for row in current:
        symbol = row.get("symbol")
        if not symbol:
            continue
        tickers.add(symbol)
        incoming = {
            "ticker": symbol,
            "name": row.get("name", ""),
            "sector": row.get("sector", ""),
            "industry": row.get("subSector", ""),
            "country": "US",
            "exchange": row.get("exchange", ""),
            "listing_start_date": None,
            "listing_end_date": None,
        }
        meta_by_ticker[symbol] = _coalesce_metadata_rows(meta_by_ticker.get(symbol, {}), incoming)

    for row in historical:
        symbol = row.get("symbol") or row.get("ticker")
        if not symbol:
            continue
        tickers.add(symbol)
        historical_start = normalize_date(row.get("dateAdded") or row.get("date"))
        historical_end = normalize_date(row.get("dateRemoved"))
        incoming = {
            "ticker": symbol,
            "name": row.get("name", ""),
            "sector": "",
            "industry": "",
            "country": "US",
            "exchange": row.get("exchange", ""),
            "listing_start_date": historical_start,
            "listing_end_date": historical_end,
        }
        meta_by_ticker[symbol] = _coalesce_metadata_rows(meta_by_ticker.get(symbol, {}), incoming)
        membership_rows.append(
            {
                "ticker": symbol,
                "start_date": historical_start,
                "end_date": historical_end,
            }
        )

    for row in delisted:
        symbol = row.get("symbol")
        name = row.get("companyName", row.get("name", ""))
        exchange = row.get("exchange", "")
        if not is_supported_us_equity(symbol, name, exchange):
            continue
        tickers.add(symbol)
        incoming = {
            "ticker": symbol,
            "name": name,
            "sector": "",
            "industry": "",
            "country": "US",
            "exchange": exchange,
            "listing_start_date": normalize_date(row.get("ipoDate")),
            "listing_end_date": normalize_date(row.get("delistedDate")),
        }
        meta_by_ticker[symbol] = _coalesce_metadata_rows(meta_by_ticker.get(symbol, {}), incoming)

    registry = load_companies_json()
    for ticker, info in registry.items():
        if not is_supported_us_equity(
            ticker,
            info.get("name", ""),
            info.get("exchange", ""),
            info.get("country"),
        ):
            continue
        tickers.add(ticker)
        incoming = {
            "ticker": ticker,
            "name": info.get("name", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country"),
            "exchange": info.get("exchange", ""),
            "listing_start_date": None,
            "listing_end_date": None,
        }
        meta_by_ticker[ticker] = _coalesce_metadata_rows(meta_by_ticker.get(ticker, {}), incoming)

    meta_df = pd.DataFrame(meta_by_ticker.values()).drop_duplicates(subset=["ticker"], keep="first")
    meta_df["schema_version"] = CACHE_SCHEMA_VERSION
    meta_df["schema_written_at"] = pd.Timestamp.utcnow().tz_localize(None)
    membership_df = pd.DataFrame(membership_rows)
    if membership_df.empty:
        membership_df = pd.DataFrame(columns=["ticker", "start_date", "end_date"])
    return sorted(tickers), meta_df, membership_df


def download_prices(client: FMPClient, ticker: str) -> tuple[str, bool]:
    cache_name = f"prices_{ticker.replace('.', '_')}"
    if client.cache_fresh(cache_name, max_age_hours=12):
        return ticker, True
    data = client.get("historical-price-eod/full", params={"symbol": ticker, "from": "2020-01-01"})
    if not data or not isinstance(data, list):
        return ticker, False
    df = pd.DataFrame(data)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return ticker, False
    if "adjClose" not in df.columns:
        df["adjClose"] = df["close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    client.save_parquet(df, cache_name)
    return ticker, True


def download_market_cap(client: FMPClient, ticker: str) -> tuple[str, bool]:
    cache_name = f"marketcap_{ticker.replace('.', '_')}"
    if client.cache_fresh(cache_name, max_age_hours=24):
        return ticker, True
    data = client.get("historical-market-capitalization", params={"symbol": ticker})
    if not data or not isinstance(data, list):
        return ticker, False
    df = pd.DataFrame(data)
    if "date" not in df.columns:
        return ticker, False
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    client.save_parquet(df, cache_name)
    return ticker, True


def download_statements(client: FMPClient, ticker: str) -> tuple[str, bool]:
    cache_name = f"statements_{ticker.replace('.', '_')}"
    if client.cache_fresh(cache_name, max_age_hours=48):
        return ticker, True
    frames = []
    for endpoint, statement_type in [
        ("income-statement", "income"),
        ("balance-sheet-statement", "balance"),
        ("cash-flow-statement", "cashflow"),
    ]:
        data = client.get(endpoint, params={"symbol": ticker, "period": "quarter", "limit": 40})
        if not data or not isinstance(data, list):
            continue
        df = pd.DataFrame(data)
        if "date" not in df.columns:
            continue
        df["statement_type"] = statement_type
        frames.append(df)
    if not frames:
        return ticker, False
    merged = None
    for frame in frames:
        shared = [c for c in ["date", "symbol", "calendarYear", "period", "filingDate", "acceptedDate"] if c in frame.columns]
        non_shared = [c for c in frame.columns if c not in shared + ["statement_type"]]
        payload = frame[shared + non_shared].copy()
        if merged is None:
            merged = payload
        else:
            merged = merged.merge(payload, on=shared, how="outer")
    merged["ticker"] = ticker
    merged["date"] = pd.to_datetime(merged["date"])
    if "filingDate" in merged.columns:
        merged["filingDate"] = strip_timezone(merged["filingDate"])
    if "acceptedDate" in merged.columns:
        merged["acceptedDate"] = strip_timezone(merged["acceptedDate"])
    client.save_parquet(merged.sort_values("date").reset_index(drop=True), cache_name)
    return ticker, True


def download_earnings(client: FMPClient, ticker: str) -> tuple[str, bool]:
    cache_name = f"earnings_{ticker.replace('.', '_')}"
    if client.cache_fresh(cache_name, max_age_hours=48):
        return ticker, True
    data = client.get(
        "earnings-calendar",
        params={"symbol": ticker, "from": "2020-01-01", "to": BACKTEST_END},
    )
    if not data or not isinstance(data, list):
        return ticker, False
    df = pd.DataFrame(data)
    if "date" not in df.columns:
        return ticker, False
    df["date"] = strip_timezone(df["date"])
    df["ticker"] = ticker
    client.save_parquet(df.sort_values("date").reset_index(drop=True), cache_name)
    return ticker, True


def download_sec_filings(client: FMPClient, ticker: str) -> tuple[str, bool]:
    cache_name = f"sec_{ticker.replace('.', '_')}"
    if client.cache_fresh(cache_name, max_age_hours=48):
        return ticker, True
    data = client.get(
        "sec-filings-search/symbol",
        params={"symbol": ticker, "from": "2020-01-01", "to": BACKTEST_END, "page": 0, "limit": 100},
    )
    if not data or not isinstance(data, list):
        return ticker, False
    df = pd.DataFrame(data)
    if "acceptedDate" in df.columns:
        df["acceptedDate"] = strip_timezone(df["acceptedDate"])
    if "filingDate" in df.columns:
        df["filingDate"] = strip_timezone(df["filingDate"])
    df["ticker"] = ticker
    client.save_parquet(df.reset_index(drop=True), cache_name)
    return ticker, True


def download_macro(client: FMPClient, fred: FREDClient) -> None:
    if not client.cache_fresh("macro_treasury", max_age_hours=12):
        data = client.get("treasury-rates", params={"from": "2020-01-01"})
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                client.save_parquet(df.sort_values("date"), "macro_treasury")

    if not client.cache_fresh("macro_vix", max_age_hours=12):
        data = client.get("historical-price-eod/full", params={"symbol": "^VIX", "from": "2020-01-01"})
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                client.save_parquet(df.sort_values("date"), "macro_vix")

    if fred.enabled and not client.cache_fresh("macro_vintages", max_age_hours=48):
        rows = []
        for field, series_id in MACRO_SERIES.items():
            vintage_payload = fred.get(
                "series/vintagedates",
                {
                    "series_id": series_id,
                    "realtime_start": "1776-07-04",
                    "realtime_end": "9999-12-31",
                },
            )
            if not vintage_payload:
                continue
            vintage_dates = vintage_payload.get("vintage_dates", [])
            if not vintage_dates:
                continue
            release_id = None
            release_payload = fred.get("series/release", {"series_id": series_id})
            if release_payload and release_payload.get("releases"):
                release_id = release_payload["releases"][0].get("id")
            for start in range(0, len(vintage_dates), 100):
                chunk = vintage_dates[start : start + 100]
                obs_payload = fred.get(
                    "series/observations",
                    {
                        "series_id": series_id,
                        "vintage_dates": ",".join(chunk),
                        "output_type": 2,
                    },
                )
                observations = (obs_payload or {}).get("observations", [])
                for obs in observations:
                    obs_date = normalize_date(obs.get("date"))
                    if obs_date is None:
                        continue
                    for vintage in chunk:
                        vintage_date = normalize_date(vintage)
                        if vintage_date is None:
                            continue
                        value_key = f"{series_id}_{vintage.replace('-', '')}"
                        if value_key not in obs:
                            continue
                        rows.append(
                            {
                                "field": field,
                                "series_id": series_id,
                                "observation_date": obs_date,
                                "value": safe_float(obs.get(value_key)),
                                "vintage_date": vintage_date,
                                "first_trade_date": None,
                                "release_id": release_id,
                                "frequency": (obs_payload or {}).get("frequency_short"),
                                "units": (obs_payload or {}).get("units_short"),
                                "retrieved_at": pd.Timestamp.utcnow().tz_localize(None),
                            }
                        )
        if rows:
            df = pd.DataFrame(rows)
            client.save_parquet(df.sort_values(["field", "observation_date", "vintage_date"]), "macro_vintages")


def download_all(client: FMPClient, fred: FREDClient | None = None) -> None:
    fred = fred or FREDClient()
    ensure_cache_dir(client.cache_dir)
    tickers, meta_df, membership_df = build_us_universe_metadata(client)
    client.save_parquet(build_coverage_audit(meta_df, client.cache_dir, client=client), "metadata")
    client.save_parquet(membership_df, "sp500_membership")

    def run_parallel(items: Iterable[str], func, label: str) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ok = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(func, client, ticker): ticker for ticker in items}
            for i, future in enumerate(as_completed(futures), start=1):
                _, success = future.result()
                ok += int(success)
                fail += int(not success)
                if i % 100 == 0:
                    print(f"{label}: {i}/{len(tickers)} (ok={ok}, fail={fail})")
        print(f"{label}: {ok} ok, {fail} failed")

    run_parallel(tickers, download_prices, "prices")
    price_supported = sorted(cache_available_tickers(client.cache_dir, "prices"))
    meta_df = build_coverage_audit(meta_df, client.cache_dir, client=client)
    client.save_parquet(meta_df, "metadata")
    print(f"price-supported universe: {len(price_supported)}/{len(tickers)}")

    run_parallel(price_supported, download_market_cap, "market cap")
    run_parallel(price_supported, download_statements, "statements")
    run_parallel(price_supported, download_earnings, "earnings")
    run_parallel(price_supported, download_sec_filings, "sec filings")
    meta_df = build_coverage_audit(meta_df, client.cache_dir, client=client)
    client.save_parquet(meta_df, "metadata")
    client.save_parquet(meta_df, "coverage_audit")
    download_macro(client, fred)


# ---------------------------------------------------------------------------
# Section 4: Data loading and point-in-time access
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    daily_returns: pd.Series
    equity_curve: pd.Series
    weights_history: list[tuple[pd.Timestamp, pd.Series]]
    turnover_history: pd.Series
    dates: pd.DatetimeIndex
    trade_dates: list[pd.Timestamp]
    diagnostics: dict


class AuditFamilyStore:
    def __init__(self, path: Path = AUDIT_RETURNS_FILE):
        self.path = path

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=["candidate_id", "date", "active_return"])

    def upsert(self, candidate_id: str, returns: pd.Series) -> None:
        ensure_cache_dir(self.path.parent)
        existing = self.load()
        existing = existing[existing["candidate_id"] != candidate_id]
        append = pd.DataFrame(
            {
                "candidate_id": candidate_id,
                "date": pd.to_datetime(returns.index),
                "active_return": returns.values,
            }
        )
        combined = pd.concat([existing, append], ignore_index=True)
        combined.to_parquet(self.path, index=False)

    def matrix(self) -> pd.DataFrame:
        data = self.load()
        if data.empty:
            return pd.DataFrame()
        data["date"] = pd.to_datetime(data["date"])
        return (
            data.pivot_table(index="date", columns="candidate_id", values="active_return", aggfunc="last")
            .sort_index()
            .fillna(0.0)
        )


class AuditRegistryStore:
    columns = [
        "candidate_id",
        "audited_at",
        "audit_state",
        "registry_state",
        "ci_low",
        "ci_high",
        "DSR_eff",
        "DSR_raw",
        "spa_pvalue",
        "N_eff",
        "N_raw",
        "inner_mutations_total",
    ]

    def __init__(self, path: Path = AUDIT_REGISTRY_FILE):
        self.path = path

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_csv(self.path, sep="\t")
        return pd.DataFrame(columns=self.columns)

    def upsert(self, row: dict) -> None:
        ensure_cache_dir(self.path.parent)
        existing = self.load()
        existing = existing[existing["candidate_id"] != row["candidate_id"]]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        combined = combined[self.columns]
        combined.to_csv(self.path, sep="\t", index=False)


class DataStore:
    def __init__(
        self,
        signal_prices: pd.DataFrame,
        total_return_prices: pd.DataFrame,
        open_prices: pd.DataFrame,
        volumes: pd.DataFrame,
        market_caps: pd.DataFrame,
        raw_fundamental_panels: dict[str, pd.DataFrame],
        legacy_fundamentals: dict[str, pd.DataFrame],
        macro_vintage_table: pd.DataFrame,
        market_macro: dict[str, pd.Series],
        metadata: dict[str, dict],
        sp500_membership: dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None]]],
        cache_dir: Path = CACHE_DIR,
    ):
        self._signal_prices = signal_prices.sort_index()
        self._total_return_prices = total_return_prices.sort_index()
        self._open_prices = open_prices.sort_index()
        self._volumes = volumes.sort_index()
        self._market_caps = market_caps.sort_index()
        self._raw_fundamental_panels = raw_fundamental_panels
        self._legacy_fundamentals = legacy_fundamentals
        self._macro_vintage_table = macro_vintage_table
        self._market_macro = market_macro
        self._metadata = metadata
        self._sp500_membership = sp500_membership
        self._cache_dir = Path(cache_dir)
        self._fundamental_cache: dict[str, pd.DataFrame] = {}
        self._macro_cache: dict[str, pd.Series] = {}

    @classmethod
    def from_cache(cls, cache_dir: Path = CACHE_DIR):
        cache = Path(cache_dir)
        ensure_cache_dir(cache)

        meta_path = cache / "metadata.parquet"
        meta_df = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame(columns=["ticker"])
        if meta_df.empty or "schema_version" not in meta_df.columns:
            raise RuntimeError(
                "Cache schema mismatch: metadata.schema_version is missing. "
                "Rebuild the PIT cache with `python3 prepare.py --download --rebuild`."
            )
        schema_versions = set(pd.Series(meta_df["schema_version"]).dropna().astype(int).tolist())
        if schema_versions != {CACHE_SCHEMA_VERSION}:
            raise RuntimeError(
                f"Cache schema mismatch: expected {CACHE_SCHEMA_VERSION}, found {sorted(schema_versions)}. "
                "Rebuild the PIT cache with `python3 prepare.py --download --rebuild`."
            )
        metadata = {}
        for _, row in meta_df.iterrows():
            entry = row.to_dict()
            ticker = entry.pop("ticker")
            metadata[ticker] = entry
        allowed_tickers = set(metadata)

        signal_frames = {}
        total_return_frames = {}
        open_frames = {}
        volume_frames = {}
        for pf in sorted(cache.glob("prices_*.parquet")):
            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue
            if "date" not in df.columns:
                continue
            ticker = pf.stem[len("prices_") :].replace("_", ".")
            if ticker not in allowed_tickers:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            if "open" in df.columns:
                open_frames[ticker] = df["open"].astype(float)
            if "close" in df.columns:
                signal_frames[ticker] = df["close"].astype(float)
            if "adjClose" in df.columns:
                total_return_frames[ticker] = df["adjClose"].astype(float)
            elif "close" in df.columns:
                total_return_frames[ticker] = df["close"].astype(float)
            if "volume" in df.columns:
                volume_frames[ticker] = df["volume"].astype(float)

        signal_prices = pd.DataFrame(signal_frames).sort_index()
        total_return_prices = pd.DataFrame(total_return_frames).sort_index()
        open_prices = pd.DataFrame(open_frames).sort_index()
        volumes = pd.DataFrame(volume_frames).sort_index()

        market_caps = cls._load_market_caps(cache, allowed_tickers)
        trading_dates = signal_prices.index
        raw_fundamentals = cls._load_raw_fundamentals(cache, trading_dates, volumes.index, allowed_tickers)
        legacy_fundamentals = cls._load_legacy_fundamentals(cache, trading_dates, allowed_tickers)
        macro_vintages = cls._load_macro_vintages(cache, trading_dates)
        market_macro = cls._load_market_macro(cache)
        sp500_membership = cls._load_sp500_membership(cache)
        cls._inject_listing_bounds(metadata, signal_prices, total_return_prices)

        print(
            f"DataStore loaded: {len(signal_prices.columns)} tickers, "
            f"{len(signal_prices)} trading days, "
            f"{len(raw_fundamentals) + len(legacy_fundamentals)} fundamental fields, "
            f"{len(macro_vintages['field'].unique()) if not macro_vintages.empty else len(market_macro)} macro series"
        )

        return cls(
            signal_prices=signal_prices,
            total_return_prices=total_return_prices,
            open_prices=open_prices,
            volumes=volumes,
            market_caps=market_caps,
            raw_fundamental_panels=raw_fundamentals,
            legacy_fundamentals=legacy_fundamentals,
            macro_vintage_table=macro_vintages,
            market_macro=market_macro,
            metadata=metadata,
            sp500_membership=sp500_membership,
            cache_dir=cache,
        )

    @staticmethod
    def _load_market_caps(cache: Path, allowed_tickers: set[str] | None = None) -> pd.DataFrame:
        frames = {}
        for path in sorted(cache.glob("marketcap_*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if "date" not in df.columns:
                continue
            ticker = path.stem[len("marketcap_") :].replace("_", ".")
            if allowed_tickers is not None and ticker not in allowed_tickers:
                continue
            value_col = "marketCap" if "marketCap" in df.columns else "marketCapitalization"
            if value_col not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            frames[ticker] = df.set_index("date")[value_col].astype(float)
        return pd.DataFrame(frames).sort_index()

    @classmethod
    def _load_raw_fundamentals(
        cls,
        cache: Path,
        trading_dates: pd.DatetimeIndex,
        volume_dates: pd.DatetimeIndex,
        allowed_tickers: set[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        panels = {field: {} for field in RAW_STATEMENT_FIELDS}
        for path in sorted(cache.glob("statements_*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if len(df) == 0 or "ticker" not in df.columns or "date" not in df.columns:
                continue
            ticker = df["ticker"].iloc[0]
            if allowed_tickers is not None and ticker not in allowed_tickers:
                continue
            df["date"] = pd.to_datetime(df["date"])
            if "filingDate" in df.columns:
                df["filingDate"] = strip_timezone(df["filingDate"])
            if "acceptedDate" in df.columns:
                df["acceptedDate"] = strip_timezone(df["acceptedDate"])
            df = df.sort_values("date")
            missing_examples = []
            events = []
            for _, row in df.iterrows():
                accepted = normalize_date(row.get("acceptedDate"))
                filed = normalize_date(row.get("filingDate"))
                available = accepted or filed
                if available is None:
                    missing_examples.append(str(pd.Timestamp(row.get("date")).date()))
                    continue
                first_trade = next_trading_day(trading_dates, available)
                if first_trade is None:
                    continue
                record = {"first_trade_date": first_trade}
                record.update({local: safe_float(row.get(source)) for local, source in RAW_STATEMENT_FIELDS.items()})
                debt = safe_float(row.get("totalDebt"))
                if np.isnan(debt):
                    debt = safe_float(row.get("longTermDebt")) + safe_float(row.get("shortTermDebt"))
                record["debt"] = debt
                events.append(record)
            if missing_examples:
                examples = ", ".join(missing_examples[:5])
                raise RuntimeError(
                    f"Statement PIT load for {ticker} is missing acceptedDate/filingDate. "
                    f"Examples: {examples}"
                )
            if not events:
                continue
            ticker_df = pd.DataFrame(events).drop_duplicates("first_trade_date", keep="last")
            ticker_df = ticker_df.set_index("first_trade_date").sort_index()
            ticker_df = ticker_df.reindex(trading_dates).ffill()
            for field in list(RAW_STATEMENT_FIELDS) + ["debt"]:
                if field in ticker_df.columns:
                    panels.setdefault(field, {})[ticker] = ticker_df[field].astype(float)
        out = {}
        for field, mapping in panels.items():
            if mapping:
                out[field] = pd.DataFrame(mapping).sort_index()
        return out

    @staticmethod
    def _load_legacy_fundamentals(
        cache: Path,
        trading_dates: pd.DatetimeIndex,
        allowed_tickers: set[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        panels: dict[str, dict[str, list[tuple[pd.Timestamp, float]]]] = {}
        for path in sorted(cache.glob("fundamentals_*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if len(df) == 0 or "date" not in df.columns:
                continue
            ticker = path.stem[len("fundamentals_") :].replace("_", ".")
            if allowed_tickers is not None and ticker not in allowed_tickers:
                continue
            df["date"] = pd.to_datetime(df["date"])
            missing_examples = []
            for _, row in df.sort_values("date").iterrows():
                avail = normalize_date(row.get("acceptedDate") or row.get("filingDate"))
                if avail is None:
                    missing_examples.append(str(pd.Timestamp(row.get("date")).date()))
                    continue
                for source, target in LEGACY_RATIO_MAP.items():
                    if source not in row.index:
                        continue
                    value = safe_float(row.get(source))
                    if np.isnan(value):
                        continue
                    panels.setdefault(target, {}).setdefault(ticker, []).append((avail, value))
            if missing_examples:
                examples = ", ".join(missing_examples[:5])
                raise RuntimeError(
                    f"Legacy fundamental PIT load for {ticker} is missing acceptedDate/filingDate. "
                    f"Examples: {examples}"
                )
        out = {}
        for field, ticker_obs in panels.items():
            sparse_rows = []
            for ticker, obs in ticker_obs.items():
                for dt, val in obs:
                    sparse_rows.append({"date": dt, "ticker": ticker, "value": val})
            if not sparse_rows:
                continue
            sparse = pd.DataFrame(sparse_rows)
            pivot = sparse.pivot_table(index="date", columns="ticker", values="value", aggfunc="last")
            out[field] = pivot.reindex(trading_dates).sort_index().ffill()
        return out

    @staticmethod
    def _load_macro_vintages(cache: Path, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        path = cache / "macro_vintages.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if df.empty:
            return df
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df["vintage_date"] = pd.to_datetime(df["vintage_date"])
        if "first_trade_date" in df.columns:
            df["first_trade_date"] = pd.to_datetime(df["first_trade_date"], errors="coerce")
        else:
            df["first_trade_date"] = pd.NaT
        missing = df["first_trade_date"].isna()
        if missing.any():
            df.loc[missing, "first_trade_date"] = [
                next_trading_day(trading_dates, d + pd.Timedelta(days=1))
                for d in df.loc[missing, "vintage_date"]
            ]
        return df.sort_values(["field", "first_trade_date", "observation_date"])

    @staticmethod
    def _load_market_macro(cache: Path) -> dict[str, pd.Series]:
        macro = {}
        treasury_path = cache / "macro_treasury.parquet"
        if treasury_path.exists():
            df = pd.read_parquet(treasury_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for raw, name in {"year10": "t10y", "year2": "t2y", "month3": "t3m"}.items():
                    if raw in df.columns:
                        macro[name] = df[raw].astype(float)
                if "t10y" in macro and "t2y" in macro:
                    macro["t10y_2y_spread"] = macro["t10y"] - macro["t2y"]
        vix_path = cache / "macro_vix.parquet"
        if vix_path.exists():
            df = pd.read_parquet(vix_path)
            if "date" in df.columns and "close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                macro["vix"] = df.set_index("date")["close"].astype(float).sort_index()
        return macro

    @staticmethod
    def _load_sp500_membership(cache: Path) -> dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None]]]:
        path = cache / "sp500_membership.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        if df.empty:
            return {}
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        membership = {}
        for ticker, group in df.groupby("ticker"):
            membership[ticker] = [
                (
                    normalize_date(row.get("start_date")),
                    normalize_date(row.get("end_date")),
                )
                for _, row in group.iterrows()
            ]
        return membership

    @staticmethod
    def _inject_listing_bounds(metadata: dict, signal_prices: pd.DataFrame, total_return_prices: pd.DataFrame) -> None:
        combined = signal_prices.combine_first(total_return_prices)
        for ticker in combined.columns:
            series = combined[ticker].dropna()
            entry = metadata.setdefault(ticker, {})
            if "listing_start_date" not in entry or pd.isna(entry.get("listing_start_date")):
                entry["listing_start_date"] = series.index.min() if len(series) else None
            if "listing_end_date" not in entry or pd.isna(entry.get("listing_end_date")):
                entry["listing_end_date"] = series.index.max() if len(series) else None

    # --- public API ---

    def prices_signal(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        return self._slice_matrix(self._signal_prices, tickers, start, end)

    def prices_total_return(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        return self._slice_matrix(self._total_return_prices, tickers, start, end)

    def open_prices(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        return self._slice_matrix(self._open_prices, tickers, start, end)

    def prices(self, tickers=None, start=None, end=None) -> pd.DataFrame:
        return self.prices_signal(tickers=tickers, start=start, end=end)

    def returns(self, tickers=None, period: int = 1, total_return: bool = False) -> pd.DataFrame:
        prices = self.prices_total_return(tickers) if total_return else self.prices_signal(tickers)
        return prices.pct_change(period)

    def volume(self, tickers=None) -> pd.DataFrame:
        return self._slice_matrix(self._volumes, tickers, None, None)

    def market_cap(self, date) -> pd.Series:
        ts = pd.Timestamp(date)
        if not self._market_caps.empty and ts in self._market_caps.index:
            row = self._market_caps.loc[ts]
            if row.notna().any():
                return row.dropna()
        try:
            shares = self.latest_fundamental("shares_out", ts)
        except KeyError:
            return pd.Series(dtype=float)
        prices = self.prices_signal(start=str(ts.date()), end=str(ts.date()))
        if prices.empty:
            return pd.Series(dtype=float)
        return (shares * prices.iloc[-1]).dropna()

    def latest_fundamental(self, field: str, date) -> pd.Series:
        ts = pd.Timestamp(date)
        if field in self._raw_fundamental_panels:
            panel = self._raw_fundamental_panels[field]
            if ts not in panel.index:
                return pd.Series(dtype=float)
            return panel.loc[ts].dropna()
        if field in self._legacy_fundamentals:
            panel = self._legacy_fundamentals[field]
            if ts not in panel.index:
                return pd.Series(dtype=float)
            return panel.loc[ts].dropna()
        if field in DERIVED_FUNDAMENTAL_FIELDS:
            return self._compute_derived_cross_section(field, ts)
        raise KeyError(f"Unknown fundamental field '{field}'")

    def fundamental(self, field: str) -> pd.DataFrame:
        if field in self._raw_fundamental_panels:
            return self._raw_fundamental_panels[field]
        if field in self._legacy_fundamentals:
            return self._legacy_fundamentals[field]
        if field not in self._fundamental_cache:
            data = {}
            for date in self._signal_prices.index:
                series = self._compute_derived_cross_section(field, date)
                if len(series):
                    data[pd.Timestamp(date)] = series
            self._fundamental_cache[field] = pd.DataFrame.from_dict(data, orient="index").sort_index()
        return self._fundamental_cache[field]

    def latest_macro(self, field: str, date):
        ts = pd.Timestamp(date)
        if field in self._market_macro:
            series = self._market_macro[field].loc[:ts].dropna()
            return float(series.iloc[-1]) if len(series) else np.nan
        if self._macro_vintage_table.empty:
            return np.nan
        subset = self._macro_vintage_table[
            (self._macro_vintage_table["field"] == field)
            & (self._macro_vintage_table["first_trade_date"] <= ts)
        ]
        if subset.empty:
            return np.nan
        latest = subset.sort_values(["observation_date", "first_trade_date"]).iloc[-1]
        return float(latest["value"])

    def macro(self, field: str) -> pd.Series:
        if field in self._market_macro:
            return self._market_macro[field]
        if field in self._macro_cache:
            return self._macro_cache[field]
        if self._macro_vintage_table.empty:
            raise KeyError(f"Unknown macro field '{field}'")
        subset = self._macro_vintage_table[self._macro_vintage_table["field"] == field]
        if subset.empty:
            raise KeyError(f"Unknown macro field '{field}'")
        values = {}
        current = np.nan
        grouped = subset.sort_values("first_trade_date").groupby("first_trade_date")
        for dt in self._signal_prices.index:
            if dt in grouped.groups:
                current = grouped.get_group(dt).sort_values("observation_date").iloc[-1]["value"]
            values[dt] = current
        series = pd.Series(values, name=field, dtype=float)
        self._macro_cache[field] = series
        return series

    def dollar_volume(self, window: int, date) -> pd.Series:
        ts = pd.Timestamp(date)
        prices = self.prices_signal(end=str(ts.date()))
        vols = self.volume()
        if prices.empty or vols.empty:
            return pd.Series(dtype=float)
        aligned = prices * vols.reindex(prices.index)
        windowed = aligned.iloc[-window:]
        return windowed.mean().dropna()

    def universe(self, date=None) -> list[str]:
        if date is None:
            return list(self._signal_prices.columns)
        return self.tradable_universe(date=date, min_history_days=60, min_price=0.0, min_dollar_volume=0.0)

    def tradable_universe(
        self,
        date,
        min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
        min_price: float = DEFAULT_MIN_PRICE,
        min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
        countries: tuple[str, ...] | None = DEFAULT_COUNTRIES,
        exchanges: tuple[str, ...] | None = None,
        sp500_only: bool = False,
    ) -> list[str]:
        ts = pd.Timestamp(date)
        prices = self.prices_signal(end=str(ts.date()))
        if prices.empty or ts not in prices.index:
            return []
        row = prices.loc[ts]
        if min_history_days > 0:
            enough_history = prices.loc[:ts].notna().sum() >= min_history_days
        else:
            enough_history = row.notna()
        dv = self.dollar_volume(20, ts) if min_dollar_volume > 0 else pd.Series(index=row.index, data=np.inf)
        candidates = []
        for ticker in row.dropna().index:
            meta = self._metadata.get(ticker, {})
            listing_start = normalize_date(meta.get("listing_start_date"))
            listing_end = normalize_date(meta.get("listing_end_date"))
            if listing_start and ts < listing_start:
                continue
            if listing_end and ts > listing_end:
                continue
            if countries and meta.get("country", "US") not in countries:
                continue
            if exchanges and meta.get("exchange") not in exchanges:
                continue
            if sp500_only and not self.is_sp500_member(ticker, ts):
                continue
            if not bool(enough_history.get(ticker, False)):
                continue
            if safe_float(row.get(ticker)) < min_price:
                continue
            if safe_float(dv.get(ticker, np.inf)) < min_dollar_volume:
                continue
            candidates.append(ticker)
        return candidates

    def can_trade(self, ticker: str, date) -> bool:
        ts = pd.Timestamp(date)
        if ticker not in self._signal_prices.columns or ts not in self._signal_prices.index:
            return False
        price = self._signal_prices.at[ts, ticker]
        volume = self._volumes.at[ts, ticker] if ticker in self._volumes.columns and ts in self._volumes.index else np.nan
        return bool(np.isfinite(price) and np.isfinite(volume) and price > 0 and volume > 0)

    def sector(self, ticker: str) -> str:
        return self._metadata.get(ticker, {}).get("sector", "Unknown")

    def country(self, ticker: str) -> str:
        return self._metadata.get(ticker, {}).get("country", "Unknown")

    def metadata_for(self, ticker: str) -> dict:
        return self._metadata.get(ticker, {})

    def is_sp500_member(self, ticker: str, date) -> bool:
        ts = pd.Timestamp(date)
        ranges = self._sp500_membership.get(ticker, [])
        if not ranges:
            return ticker == DEFAULT_BENCHMARK
        for start, end in ranges:
            if (start is None or ts >= start) and (end is None or ts <= end):
                return True
        return False

    def correlation(self, tickers, window: int = 60) -> pd.DataFrame:
        rets = self.returns(tickers=tickers, total_return=True)
        if len(rets) < window:
            return rets.corr()
        return rets.iloc[-window:].corr()

    def factor_rank(self, series: pd.Series, method: str = "pct") -> pd.Series:
        series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
        if method == "pct":
            return series.rank(pct=True)
        return series.rank()

    def winsorize_cross_section(self, series: pd.Series, lower_pct: float = 0.02, upper_pct: float = 0.98) -> pd.Series:
        series = pd.Series(series).replace([np.inf, -np.inf], np.nan)
        if series.dropna().empty:
            return series
        lo = series.quantile(lower_pct)
        hi = series.quantile(upper_pct)
        return series.clip(lower=lo, upper=hi)

    def neutralize_cross_section(self, series: pd.Series, by: list[pd.Series]) -> pd.Series:
        y = pd.Series(series).dropna()
        if y.empty:
            return y
        for exposure in by:
            exp = pd.Series(exposure).reindex(y.index)
            if exp.dtype == object:
                groups = exp.fillna("Unknown")
                y = y - y.groupby(groups).transform("mean")
                continue
            exp = exp.replace([np.inf, -np.inf], np.nan)
            mask = exp.notna()
            if mask.sum() < 5:
                continue
            x = exp.loc[mask]
            std = x.std(ddof=1)
            std = 1.0 if pd.isna(std) or std == 0 else std
            x = (x - x.mean()) / std
            y_masked = y.loc[mask]
            beta = float((x * y_masked).sum() / max((x * x).sum(), 1e-8))
            y.loc[mask] = y_masked - beta * x
        return y - y.mean()

    def _slice_matrix(self, frame: pd.DataFrame, tickers=None, start=None, end=None) -> pd.DataFrame:
        df = frame
        if tickers is not None:
            keep = [ticker for ticker in tickers if ticker in df.columns]
            df = df[keep]
        if start is not None:
            df = df.loc[pd.Timestamp(start) :]
        if end is not None:
            df = df.loc[: pd.Timestamp(end)]
        return df

    def _compute_derived_cross_section(self, field: str, ts: pd.Timestamp) -> pd.Series:
        def latest_or_empty(name: str) -> pd.Series:
            try:
                return self.latest_fundamental(name, ts)
            except KeyError:
                return pd.Series(dtype=float)

        def legacy_or_empty(name: str) -> pd.Series:
            panel = self._legacy_fundamentals.get(name)
            if panel is None or ts not in panel.index:
                return pd.Series(dtype=float)
            return panel.loc[ts].dropna()

        if field == "book_to_price":
            book = latest_or_empty("book_equity")
            if len(book):
                mcap = self.market_cap(ts)
                return (book / mcap.replace(0, np.nan)).dropna()
            pb = legacy_or_empty("pb").replace(0, np.nan)
            if len(pb):
                return (1.0 / pb).dropna()
            return pd.Series(dtype=float)
        if field == "earnings_yield":
            income = latest_or_empty("net_income")
            if len(income):
                mcap = self.market_cap(ts)
                return ((income * 4) / mcap.replace(0, np.nan)).dropna()
            return legacy_or_empty("earnings_yield")
        if field == "free_cash_flow_yield":
            fcf = latest_or_empty("free_cash_flow")
            if len(fcf):
                mcap = self.market_cap(ts)
                return ((fcf * 4) / mcap.replace(0, np.nan)).dropna()
            return legacy_or_empty("fcf_yield")
        if field == "gross_profitability":
            gross = latest_or_empty("gross_profit")
            assets = latest_or_empty("assets")
            if len(gross) and len(assets):
                return (gross / assets.replace(0, np.nan)).dropna()
            return legacy_or_empty("gross_margin")
        if field == "asset_growth":
            panel = self._raw_fundamental_panels.get("assets")
            if panel is not None and ts in panel.index:
                now = panel.loc[ts]
                shifted = panel.shift(252)
                if ts in shifted.index:
                    prev = shifted.loc[ts].replace(0, np.nan)
                    return (now / prev - 1).dropna()
            return legacy_or_empty("revenue_growth")
        if field == "leverage":
            debt = latest_or_empty("debt")
            assets = latest_or_empty("assets")
            if len(debt) and len(assets):
                return (debt / assets.replace(0, np.nan)).dropna()
            return legacy_or_empty("debt_to_equity")
        if field == "current_ratio":
            assets = latest_or_empty("current_assets")
            liabilities = latest_or_empty("current_liabilities")
            if len(assets) and len(liabilities):
                return (assets / liabilities.replace(0, np.nan)).dropna()
            return legacy_or_empty("current_ratio")
        if field == "roe":
            income = latest_or_empty("net_income")
            equity = latest_or_empty("book_equity")
            if len(income) and len(equity):
                return ((income * 4) / equity.replace(0, np.nan)).dropna()
            return legacy_or_empty("roe")
        raise KeyError(f"Unknown derived field '{field}'")


class DateLimitedStore:
    def __init__(self, store: DataStore, cutoff):
        self._store = store
        self._cutoff = pd.Timestamp(cutoff)

    def _limit(self, end=None):
        if end is None:
            return self._cutoff
        return min(pd.Timestamp(end), self._cutoff)

    def prices_signal(self, tickers=None, start=None, end=None):
        return self._store.prices_signal(tickers=tickers, start=start, end=self._limit(end))

    def prices_total_return(self, tickers=None, start=None, end=None):
        return self._store.prices_total_return(tickers=tickers, start=start, end=self._limit(end))

    def open_prices(self, tickers=None, start=None, end=None):
        return self._store.open_prices(tickers=tickers, start=start, end=self._limit(end))

    def prices(self, tickers=None, start=None, end=None):
        return self.prices_signal(tickers=tickers, start=start, end=end)

    def returns(self, tickers=None, period=1, total_return=False):
        prices = self.prices_total_return(tickers=tickers) if total_return else self.prices_signal(tickers=tickers)
        return prices.pct_change(period)

    def volume(self, tickers=None):
        return self._store.volume(tickers=tickers).loc[: self._cutoff]

    def market_cap(self, date):
        return self._store.market_cap(min(pd.Timestamp(date), self._cutoff))

    def latest_fundamental(self, field: str, date):
        return self._store.latest_fundamental(field, min(pd.Timestamp(date), self._cutoff))

    def fundamental(self, field: str):
        return self._store.fundamental(field).loc[: self._cutoff]

    def latest_macro(self, field: str, date):
        return self._store.latest_macro(field, min(pd.Timestamp(date), self._cutoff))

    def macro(self, field: str):
        return self._store.macro(field).loc[: self._cutoff]

    def dollar_volume(self, window: int, date):
        return self._store.dollar_volume(window, min(pd.Timestamp(date), self._cutoff))

    def tradable_universe(self, date, **kwargs):
        return self._store.tradable_universe(min(pd.Timestamp(date), self._cutoff), **kwargs)

    def universe(self, date=None):
        return self._store.universe(date or self._cutoff)

    def can_trade(self, ticker: str, date):
        return self._store.can_trade(ticker, min(pd.Timestamp(date), self._cutoff))

    def factor_rank(self, series: pd.Series, method: str = "pct"):
        return self._store.factor_rank(series, method=method)

    def winsorize_cross_section(self, series: pd.Series, lower_pct: float = 0.02, upper_pct: float = 0.98):
        return self._store.winsorize_cross_section(series, lower_pct=lower_pct, upper_pct=upper_pct)

    def neutralize_cross_section(self, series: pd.Series, by: list[pd.Series]):
        return self._store.neutralize_cross_section(series, by=by)

    def sector(self, ticker: str):
        return self._store.sector(ticker)

    def country(self, ticker: str):
        return self._store.country(ticker)

    def metadata_for(self, ticker: str):
        return self._store.metadata_for(ticker)

    def is_sp500_member(self, ticker: str, date):
        return self._store.is_sp500_member(ticker, min(pd.Timestamp(date), self._cutoff))

    def correlation(self, tickers, window=60):
        rets = self.returns(tickers=tickers, total_return=True)
        if len(rets) < window:
            return rets.corr()
        return rets.iloc[-window:].corr()


# ---------------------------------------------------------------------------
# Section 5: Backtest engine
# ---------------------------------------------------------------------------


def normalize_target_weights(weights: pd.Series) -> pd.Series:
    weights = pd.Series(weights, dtype=float).dropna()
    if weights.empty:
        return weights
    weights = weights.groupby(level=0).sum()
    asset_book = weights.drop("__CASH__", errors="ignore")
    if (asset_book < -1e-9).any():
        offenders = ", ".join(sorted(asset_book[asset_book < -1e-9].index.tolist())[:5])
        raise ValueError(f"Negative weights are not supported in V1. Offenders: {offenders}")
    if "__CASH__" not in weights.index:
        cash_weight = NET_EXPOSURE_TARGET - weights.sum()
        weights.loc["__CASH__"] = cash_weight
    if weights.loc["__CASH__"] < -1e-9:
        raise ValueError("Negative cash is not allowed")
    gross = weights.drop("__CASH__", errors="ignore").abs().sum()
    if gross > GROSS_EXPOSURE_LIMIT + 1e-9:
        raise ValueError(f"Gross exposure {gross:.3f} exceeds limit {GROSS_EXPOSURE_LIMIT:.2f}")
    net = weights.sum()
    if abs(net - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0 including cash")
    return weights.sort_index()


def select_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    period_map = {"M": "M", "W": "W", "Q": "Q"}
    period_code = period_map.get(freq, "M")
    groups = index.to_period(period_code)
    stamps = pd.Series(index, index=index)
    return pd.DatetimeIndex(stamps.groupby(groups).head(1).tolist())


def compute_weight_diagnostics(weights_history: list[tuple[pd.Timestamp, pd.Series]], store: DataStore) -> dict:
    if not weights_history:
        return {"avg_top10_weight": 0.0, "avg_max_sector_weight": 0.0}
    top10 = []
    sector_max = []
    for _, weights in weights_history:
        book = weights.drop("__CASH__", errors="ignore")
        if book.empty:
            top10.append(0.0)
            sector_max.append(0.0)
            continue
        top10.append(float(book.abs().nlargest(min(10, len(book))).sum()))
        sector_map = {}
        for ticker, weight in book.items():
            sector = store.sector(ticker)
            sector_map[sector] = sector_map.get(sector, 0.0) + abs(weight)
        sector_max.append(max(sector_map.values()) if sector_map else 0.0)
    return {
        "avg_top10_weight": float(np.mean(top10)),
        "avg_max_sector_weight": float(np.mean(sector_max)),
    }


def choose_benchmark_returns(store: DataStore, dates: pd.DatetimeIndex) -> tuple[str, pd.Series]:
    start = str(dates.min().date())
    end = str(dates.max().date())
    bench_prices = store.prices_total_return([DEFAULT_BENCHMARK], start=start, end=end)
    if DEFAULT_BENCHMARK in bench_prices.columns and bench_prices[DEFAULT_BENCHMARK].notna().sum() > 5:
        bench_rets = bench_prices[DEFAULT_BENCHMARK].pct_change().reindex(dates).fillna(0.0)
        return DEFAULT_BENCHMARK, bench_rets
    eq_prices = store.prices_total_return(start=start, end=end)
    bench_rets = eq_prices.pct_change().mean(axis=1).reindex(dates).fillna(0.0)
    return "equal_weight_universe", bench_rets


def run_backtest(strategy_module, data_store: DataStore, start: str, end: str, rebalance_freq: str = "M") -> BacktestResult:
    freq = getattr(strategy_module, "REBALANCE_FREQ", rebalance_freq)
    lead_start = pd.Timestamp(start) - pd.DateOffset(months=14)
    total_return_prices = data_store.prices_total_return(start=str(lead_start.date()), end=end)
    if total_return_prices.empty:
        raise ValueError("No total-return price data available")
    bt_prices = total_return_prices.loc[start:end]
    if bt_prices.empty:
        raise ValueError(f"No prices in [{start}, {end}]")
    trading_dates = bt_prices.index
    rebal_dates = select_rebalance_dates(trading_dates, freq)
    signal_dates = set(rebal_dates)

    capital = INITIAL_CAPITAL
    current_weights = pd.Series({"__CASH__": 1.0})
    equity_curve = []
    daily_rets = []
    weights_history = []
    turnover_rows = []
    trade_dates = []
    pending_target = None
    pending_trade_date = None
    prev_date = None
    cost_rate = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000
    missing_return_count = 0
    filtered_trade_target_count = 0

    for i, date in enumerate(trading_dates):
        if prev_date is None:
            daily_rets.append(0.0)
            equity_curve.append(capital)
        else:
            day_ret = bt_prices.loc[date] / bt_prices.loc[prev_date] - 1
            port_ret = 0.0
            values = {}
            for ticker, weight in current_weights.items():
                if ticker == "__CASH__":
                    values[ticker] = weight * capital
                    continue
                asset_ret = safe_float(day_ret.get(ticker), default=0.0)
                if not np.isfinite(asset_ret):
                    asset_ret = 0.0
                    missing_return_count += 1
                values[ticker] = weight * capital * (1 + asset_ret)
                port_ret += weight * asset_ret
            capital *= 1 + port_ret
            weight_series = pd.Series(values, dtype=float) / capital if capital != 0 else pd.Series(dtype=float)
            current_weights = weight_series
            daily_rets.append(port_ret)
            equity_curve.append(capital)

        if pending_trade_date is not None and date == pending_trade_date and pending_target is not None:
            old = current_weights.copy()
            tradeable_target = {}
            cash_buffer = 0.0
            for ticker, weight in pd.Series(pending_target, dtype=float).items():
                if ticker == "__CASH__":
                    cash_buffer += float(weight)
                    continue
                prev_px = safe_float(bt_prices.at[prev_date, ticker], default=np.nan) if prev_date is not None and ticker in bt_prices.columns else np.nan
                trade_px = safe_float(bt_prices.at[date, ticker], default=np.nan) if ticker in bt_prices.columns else np.nan
                if data_store.can_trade(ticker, date) and np.isfinite(prev_px) and np.isfinite(trade_px):
                    tradeable_target[ticker] = float(weight)
                else:
                    cash_buffer += float(weight)
                    filtered_trade_target_count += 1
            if cash_buffer:
                tradeable_target["__CASH__"] = tradeable_target.get("__CASH__", 0.0) + cash_buffer
            new = normalize_target_weights(pd.Series(tradeable_target, dtype=float))
            turnover = 0.5 * sum(abs(new.get(name, 0.0) - old.get(name, 0.0)) for name in set(old.index) | set(new.index))
            trade_cost = turnover * cost_rate * 2
            capital *= 1 - trade_cost
            if daily_rets:
                daily_rets[-1] = (1 + daily_rets[-1]) * (1 - trade_cost) - 1
                equity_curve[-1] = capital
            current_weights = new
            weights_history.append((date, new.copy()))
            turnover_rows.append((date, turnover))
            trade_dates.append(date)
            pending_target = None
            pending_trade_date = None

        if date in signal_dates:
            limited = DateLimitedStore(data_store, date)
            try:
                scores = strategy_module.signals(limited, date)
                if scores is not None and len(scores) > 0:
                    target = strategy_module.construct(scores, limited, date)
                    target = strategy_module.risk(target, limited, date)
                    if target is not None and len(target) > 0:
                        pending_target = target
                        if i + 1 < len(trading_dates):
                            pending_trade_date = trading_dates[i + 1]
            except Exception as exc:
                print(f"  Warning: strategy error on {date.date()}: {exc}")
        prev_date = date

    turnover_series = pd.Series(dict(turnover_rows), name="turnover", dtype=float)
    diagnostics = compute_weight_diagnostics(weights_history, data_store)
    diagnostics["execution_mode"] = EXECUTION_MODE
    diagnostics["missing_asset_return_count"] = missing_return_count
    diagnostics["filtered_trade_target_count"] = filtered_trade_target_count
    return BacktestResult(
        daily_returns=pd.Series(daily_rets, index=trading_dates, name="portfolio_returns"),
        equity_curve=pd.Series(equity_curve, index=trading_dates, name="equity"),
        weights_history=weights_history,
        turnover_history=turnover_series,
        dates=trading_dates,
        trade_dates=trade_dates,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Section 6: Evaluation workflow
# ---------------------------------------------------------------------------


def get_period_bounds(period: str) -> tuple[str, str]:
    if period == "train":
        return BACKTEST_START, TRAIN_END
    if period == "inner":
        return TRAIN_END, INNER_END
    if period == "outer":
        return INNER_END, OUTER_END
    if period == "test":
        return OUTER_END, BACKTEST_END
    raise ValueError(f"Unknown period '{period}'")


def active_slice_score(active_returns: pd.Series, slices: int = 4) -> tuple[float, float]:
    if len(active_returns) < slices * 5:
        return 0.0, 0.0
    segments = [seg for seg in np.array_split(active_returns.dropna(), slices) if len(seg) > 1]
    if not segments:
        return 0.0, 0.0
    sharpes = [sharpe_annualized_lo(pd.Series(segment)) for segment in segments]
    median = float(np.median(sharpes))
    instability = float(np.subtract(*np.percentile(sharpes, [75, 25]))) if len(sharpes) > 1 else 0.0
    return median, instability


def inner_objective(active_returns: pd.Series, result: BacktestResult) -> tuple[float, dict]:
    median_sharpe, instability = active_slice_score(active_returns)
    turnover = float(result.turnover_history.mean()) if len(result.turnover_history) else 0.0
    concentration = float(result.diagnostics.get("avg_max_sector_weight", 0.0))
    top10 = float(result.diagnostics.get("avg_top10_weight", 0.0))
    turnover_penalty = max(0.0, turnover - 0.25) * 1.5
    concentration_penalty = max(0.0, concentration - 0.30) * 2.0 + max(0.0, top10 - 0.45) * 1.5
    instability_penalty = max(0.0, instability - 0.75) * 0.5
    score = median_sharpe - turnover_penalty - concentration_penalty - instability_penalty
    return score, {
        "slice_median_active_sharpe": median_sharpe,
        "slice_instability_iqr": instability,
        "turnover_penalty": turnover_penalty,
        "concentration_penalty": concentration_penalty,
        "instability_penalty": instability_penalty,
    }


def summarize_trial_family(
    family_matrix: pd.DataFrame,
    current_returns: pd.Series,
) -> dict:
    current = current_returns.dropna()
    if family_matrix.empty or family_matrix.shape[1] == 0:
        sr_hat = sharpe_daily(current)
        skew = sample_skewness(current)
        kurt = sample_kurtosis(current)
        return {
            "trial_var_sr": 0.0,
            "N_raw": 1,
            "N_eff": 1.0,
            "sr_hat_daily": sr_hat,
            "skew": skew,
            "kurtosis": kurt,
            "DSR_eff": 0.0,
            "DSR_raw": 0.0,
            "sr_star_eff": 0.0,
            "sr_star_raw": 0.0,
            "spa_pvalue": 1.0,
            "pbo": None,
        }
    trial_sharpes = family_matrix.apply(sharpe_daily, axis=0)
    trial_var = float(trial_sharpes.var(ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr_hat = sharpe_daily(current)
    skew = sample_skewness(current)
    kurt = sample_kurtosis(current)
    n_raw = int(family_matrix.shape[1])
    n_eff = estimate_effective_independent_trials(family_matrix)
    dsr_eff, sr_star_eff = deflated_sharpe_ratio(sr_hat, trial_var, n_eff, len(current), skew, kurt)
    dsr_raw, sr_star_raw = deflated_sharpe_ratio(sr_hat, trial_var, n_raw, len(current), skew, kurt)
    return {
        "trial_var_sr": trial_var,
        "N_raw": n_raw,
        "N_eff": n_eff,
        "sr_hat_daily": sr_hat,
        "skew": skew,
        "kurtosis": kurt,
        "DSR_eff": dsr_eff,
        "DSR_raw": dsr_raw,
        "sr_star_eff": sr_star_eff,
        "sr_star_raw": sr_star_raw,
        "spa_pvalue": spa_pvalue(family_matrix),
        "pbo": compute_pbo(family_matrix) if n_raw >= PBO_MIN_FAMILY_SIZE else None,
    }


def evaluate(
    strategy_module,
    data_store: DataStore,
    period: str,
    inner_mutations_total: int = 0,
    candidate_id: str | None = None,
    persist_outer_audit: bool = False,
) -> dict:
    start, end = get_period_bounds(period)
    result = run_backtest(strategy_module, data_store, start, end)
    benchmark_name, benchmark_returns = choose_benchmark_returns(data_store, result.dates)
    active_returns = (result.daily_returns - benchmark_returns).fillna(0.0)
    portfolio_sharpe = sharpe_daily(result.daily_returns)
    portfolio_sharpe_lo = sharpe_annualized_lo(result.daily_returns)
    active_sharpe = sharpe_daily(active_returns)
    active_sharpe_lo = sharpe_annualized_lo(active_returns)
    portfolio_annual = annual_return(result.daily_returns)
    active_annual = annual_return(active_returns)
    portfolio_mdd = max_drawdown(result.daily_returns)
    portfolio_sortino = sortino_ratio(result.daily_returns)
    portfolio_calmar = calmar_ratio(result.daily_returns)

    ci_low, ci_high, block_length, bootstrap_se = bootstrap_sharpe_ci(active_returns)
    metrics = {
        "period": period,
        "benchmark": benchmark_name,
        "active_sharpe_daily": active_sharpe,
        "active_sharpe_annualized_lo": active_sharpe_lo,
        "portfolio_sharpe_daily": portfolio_sharpe,
        "portfolio_sharpe_annualized_lo": portfolio_sharpe_lo,
        "active_annual_return": active_annual,
        "portfolio_annual_return": portfolio_annual,
        "portfolio_max_drawdown": portfolio_mdd,
        "portfolio_sortino": portfolio_sortino,
        "portfolio_calmar": portfolio_calmar,
        "sharpe_daily": active_sharpe,
        "sharpe_annualized_lo": active_sharpe_lo,
        "annual_return": portfolio_annual,
        "max_drawdown": portfolio_mdd,
        "sortino": portfolio_sortino,
        "calmar": portfolio_calmar,
        "turnover": float(result.turnover_history.mean()) if len(result.turnover_history) else 0.0,
        "n_positions": int(np.mean([len(w.drop("__CASH__", errors="ignore")) for _, w in result.weights_history])) if result.weights_history else 0,
        "active_sharpe_ci_95_daily": (ci_low, ci_high),
        "bootstrap_method": "stationary_bootstrap_studentized",
        "bootstrap_block_length": block_length,
        "bootstrap_se": bootstrap_se,
        "complexity_loc": count_loc("strategy.py"),
        "inner_mutations_total": inner_mutations_total,
        "outer_promotions_total": 0,
        "outer_family_size_current": 0,
        "execution_mode": result.diagnostics.get("execution_mode", EXECUTION_MODE),
        "missing_asset_return_count": int(result.diagnostics.get("missing_asset_return_count", 0)),
        "filtered_trade_target_count": int(result.diagnostics.get("filtered_trade_target_count", 0)),
        "avg_top10_weight": result.diagnostics.get("avg_top10_weight", 0.0),
        "avg_max_sector_weight": result.diagnostics.get("avg_max_sector_weight", 0.0),
        "equity_curve": result.equity_curve,
    }

    if period == "inner":
        score_inner, score_parts = inner_objective(active_returns, result)
        metrics["score_inner"] = score_inner
        metrics.update(score_parts)
        metrics["baseline_update_scope"] = "inner_only"
        family_matrix = AuditFamilyStore().matrix()
        metrics["outer_promotions_total"] = int(family_matrix.shape[1]) if not family_matrix.empty else 0
        metrics["outer_family_size_current"] = metrics["outer_promotions_total"]
        return metrics

    family_store = AuditFamilyStore()
    if period == "outer" and persist_outer_audit and candidate_id:
        family_store.upsert(candidate_id, active_returns)
    family_matrix = family_store.matrix()
    family_stats = summarize_trial_family(family_matrix, active_returns)
    metrics.update(family_stats)
    metrics["outer_promotions_total"] = int(family_matrix.shape[1]) if not family_matrix.empty else 0
    metrics["outer_family_size_current"] = metrics["outer_promotions_total"]
    audit_pass = bool(
        family_stats["DSR_raw"] >= AUDIT_DSR_THRESHOLD
        and ci_low > 0
        and family_stats["spa_pvalue"] < AUDIT_SPA_ALPHA
    )
    metrics["audit_state"] = "audit_pass" if audit_pass else "audit_fail"
    metrics["registry_state"] = "pending_human_review" if audit_pass else "not_registered"
    metrics["baseline_update_scope"] = "human_only"
    if period == "outer" and candidate_id:
        AuditRegistryStore().upsert(
            {
                "candidate_id": candidate_id,
                "audited_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
                "audit_state": metrics["audit_state"],
                "registry_state": metrics["registry_state"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "DSR_eff": family_stats["DSR_eff"],
                "DSR_raw": family_stats["DSR_raw"],
                "spa_pvalue": family_stats["spa_pvalue"],
                "N_eff": family_stats["N_eff"],
                "N_raw": family_stats["N_raw"],
                "inner_mutations_total": inner_mutations_total,
            }
        )
    return metrics


def print_metrics(metrics: dict, label: str = "") -> None:
    if label:
        print(f"\n=== {label} ===")
    order = [
        "period",
        "benchmark",
        "score_inner",
        "active_sharpe_daily",
        "active_sharpe_annualized_lo",
        "portfolio_sharpe_daily",
        "portfolio_sharpe_annualized_lo",
        "active_annual_return",
        "portfolio_annual_return",
        "portfolio_max_drawdown",
        "portfolio_sortino",
        "portfolio_calmar",
        "turnover",
        "n_positions",
        "missing_asset_return_count",
        "filtered_trade_target_count",
        "active_sharpe_ci_95_daily",
        "bootstrap_method",
        "bootstrap_block_length",
        "bootstrap_se",
        "DSR_eff",
        "DSR_raw",
        "sr_hat_daily",
        "sr_star_eff",
        "sr_star_raw",
        "trial_var_sr",
        "N_eff",
        "N_raw",
        "spa_pvalue",
        "pbo",
        "slice_median_active_sharpe",
        "slice_instability_iqr",
        "turnover_penalty",
        "concentration_penalty",
        "instability_penalty",
        "avg_top10_weight",
        "avg_max_sector_weight",
        "inner_mutations_total",
        "outer_promotions_total",
        "outer_family_size_current",
        "baseline_update_scope",
        "audit_state",
        "registry_state",
        "execution_mode",
        "complexity_loc",
    ]
    print("---")
    for key in order:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, tuple):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


# ---------------------------------------------------------------------------
# Section 7: CLI
# ---------------------------------------------------------------------------


def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
    if spec is None or spec.loader is None:
        print("Error: strategy.py not found in current directory.")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_candidate_id() -> str:
    return f"strategy-{sha1_file('strategy.py')}"


def main():
    parser = argparse.ArgumentParser(description="Q_Lab: hardened quantitative research")
    parser.add_argument("--download", action="store_true", help="Download/refresh PIT data from FMP and FRED")
    parser.add_argument("--backtest", action="store_true", help="Run inner visible search evaluation")
    parser.add_argument("--audit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test", action="store_true", help="Run final holdout test")
    parser.add_argument("--rebuild", action="store_true", help="Clear cached parquet artifacts before download")
    parser.add_argument("--candidate-id", type=str, default=None, help="Candidate id for audit-family persistence")
    parser.add_argument("--n-trials", type=int, default=0, help="Total inner-search mutations so far")
    args = parser.parse_args()

    if not any([args.download, args.backtest, args.audit, args.test]):
        parser.print_help()
        sys.exit(0)

    if args.download:
        if args.rebuild:
            clear_cache_parquets()
        fmp = FMPClient()
        fred = FREDClient()
        download_all(fmp, fred)

    if args.backtest or args.audit or args.test:
        t0 = time.time()
        data_store = DataStore.from_cache()
        print(f"Data loaded in {time.time() - t0:.1f}s")
        strategy = load_strategy()

        if args.backtest:
            metrics = evaluate(
                strategy,
                data_store,
                period="inner",
                inner_mutations_total=args.n_trials,
            )
            print_metrics(metrics, "Inner Validation")

        if args.audit:
            if os.environ.get("QLAB_AUDITOR_MODE") != "1":
                raise SystemExit(
                    "--audit is reserved for auditor-only execution. "
                    "Run it from a separate auditor process with QLAB_AUDITOR_MODE=1."
                )
            metrics = evaluate(
                strategy,
                data_store,
                period="outer",
                inner_mutations_total=args.n_trials,
                candidate_id=args.candidate_id or default_candidate_id(),
                persist_outer_audit=True,
            )
            print_metrics(metrics, "Outer Audit")

        if args.test:
            metrics = evaluate(
                strategy,
                data_store,
                period="test",
                inner_mutations_total=args.n_trials,
            )
            print_metrics(metrics, "Final Holdout Test")


if __name__ == "__main__":
    main()
