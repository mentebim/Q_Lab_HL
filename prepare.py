"""
Q_Lab: Autonomous quantitative research infrastructure.

Fixed evaluation harness, data pipeline, and backtest engine.
The agent modifies strategy.py only — this file is read-only.

Usage:
    uv run prepare.py --download     # fetch/refresh all data from FMP
    uv run prepare.py --backtest     # run strategy.py, evaluate, print metrics
    uv run prepare.py --test         # run on holdout test period (human only)
"""

import os
import sys
import time
import json
import math
import argparse
import importlib
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# ---------------------------------------------------------------------------
# Section 1: Constants (FIXED — agent cannot touch)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "qlab")
COMPANIES_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "stack-intelligence-engine", "registry", "companies.json",
)

BACKTEST_START = "2021-03-01"
BACKTEST_END = "2026-03-01"
TRAIN_END = "2024-03-01"       # 60% in-sample
VAL_END = "2025-03-01"         # 20% validation (agent sees this)
# Test: 2025-03-01 to 2026-03-01 (20%, NEVER shown to agent)

INITIAL_CAPITAL = 1_000_000
COMMISSION_BPS = 5
SLIPPAGE_BPS = 5
TIME_BUDGET = 120              # max seconds for a backtest run

REPORTING_LAG_DAYS = 45        # conservative fundamental data lag

FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_RATE_LIMIT = 300           # requests per minute
DOWNLOAD_WORKERS = 8

# ---------------------------------------------------------------------------
# Section 2: FMP API Client + Data Download
# ---------------------------------------------------------------------------

class FMPClient:
    """Thin FMP API wrapper with rate limiting, retries, and parquet caching."""

    def __init__(self, api_key=None, cache_dir=CACHE_DIR):
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            raise ValueError("FMP_API_KEY not set. Add it to .env or environment.")
        self.cache_dir = cache_dir
        self.base_url = FMP_BASE
        self._request_times = []

    def _rate_limit(self):
        """Enforce 300 requests/minute."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= FMP_RATE_LIMIT:
            sleep_time = 60 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._request_times.append(time.time())

    def get(self, endpoint, params=None, max_retries=3):
        """GET with rate limiting and exponential backoff."""
        params = params or {}
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "Error Message" in data:
                    print(f"  FMP error for {endpoint}: {data['Error Message']}")
                    return None
                return data
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    print(f"  Failed {endpoint} after {max_retries} attempts: {e}")
                    return None

    def _cache_path(self, name):
        return os.path.join(self.cache_dir, f"{name}.parquet")

    def _cache_fresh(self, name, max_age_hours=24):
        path = self._cache_path(name)
        if not os.path.exists(path):
            return False
        age = time.time() - os.path.getmtime(path)
        return age < max_age_hours * 3600

    def save_parquet(self, df, name):
        os.makedirs(self.cache_dir, exist_ok=True)
        df.to_parquet(self._cache_path(name))

    def load_parquet(self, name):
        path = self._cache_path(name)
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None


def load_companies_json():
    path = os.path.normpath(COMPANIES_JSON)
    if not os.path.exists(path):
        print(f"  companies.json not found at {path}, skipping international tickers.")
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("companies", {})


def get_master_ticker_list(client):
    sp500_data = client.get("sp500-constituent")
    sp500_tickers = set()
    sp500_meta = {}
    if sp500_data:
        for item in sp500_data:
            ticker = item.get("symbol", "")
            if ticker:
                sp500_tickers.add(ticker)
                sp500_meta[ticker] = {
                    "name": item.get("name", ""),
                    "sector": item.get("sector", ""),
                    "industry": item.get("subSector", ""),
                    "country": "US",
                    "exchange": item.get("exchange", ""),
                }

    companies = load_companies_json()
    intl_meta = {}
    for ticker, info in companies.items():
        intl_meta[ticker] = {
            "name": info.get("name", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country", "US"),
            "exchange": info.get("exchange", ""),
            "nodes": info.get("nodes", []),
            "tags": info.get("tags", []),
        }

    all_meta = {**sp500_meta}
    for ticker, meta in intl_meta.items():
        if ticker in all_meta:
            all_meta[ticker].update(meta)
        else:
            all_meta[ticker] = meta

    all_tickers = sorted(set(sp500_tickers) | set(intl_meta.keys()))
    print(f"  Master universe: {len(all_tickers)} tickers "
          f"({len(sp500_tickers)} S&P 500, {len(intl_meta)} from companies.json, "
          f"{len(all_tickers)} deduplicated)")
    return all_tickers, all_meta


def download_prices(client, ticker):
    cache_name = f"prices_{ticker.replace('.', '_')}"
    if client._cache_fresh(cache_name, max_age_hours=12):
        return ticker, True

    data = client.get("historical-price-eod/full", params={
        "symbol": ticker, "from": "2020-01-01",
    })
    if not data or not isinstance(data, list) or len(data) == 0:
        return ticker, False

    df = pd.DataFrame(data)
    for col in ["date", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            return ticker, False
    if "adjClose" not in df.columns:
        df["adjClose"] = df["close"]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    client.save_parquet(df, cache_name)
    return ticker, True


def download_fundamentals_quarterly(client, ticker):
    """Download HISTORICAL quarterly key-metrics and ratios.

    This gives us point-in-time fundamental data — what was actually known
    at each quarter end, not just today's snapshot. Critical for avoiding
    lookahead bias in backtests.
    """
    cache_name = f"fundamentals_q_{ticker.replace('.', '_')}"
    if client._cache_fresh(cache_name, max_age_hours=48):
        return ticker, True

    metrics = client.get("key-metrics", params={
        "symbol": ticker, "period": "quarter", "limit": 40,
    })
    ratios = client.get("ratios", params={
        "symbol": ticker, "period": "quarter", "limit": 40,
    })

    if not metrics and not ratios:
        return ticker, False

    metrics_by_date = {}
    if metrics and isinstance(metrics, list):
        for row in metrics:
            d = row.get("date", "")
            if d:
                metrics_by_date[d] = row

    ratios_by_date = {}
    if ratios and isinstance(ratios, list):
        for row in ratios:
            d = row.get("date", "")
            if d:
                ratios_by_date[d] = row

    all_dates = sorted(set(list(metrics_by_date.keys()) + list(ratios_by_date.keys())))
    records = []
    for d in all_dates:
        record = {"date": d, "ticker": ticker}
        if d in metrics_by_date:
            record.update(metrics_by_date[d])
        if d in ratios_by_date:
            for k, v in ratios_by_date[d].items():
                if k not in record or record[k] is None:
                    record[k] = v
        records.append(record)

    if not records:
        return ticker, False

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    client.save_parquet(df, cache_name)
    return ticker, True


def download_macro(client):
    print("  Downloading macro data...")

    if not client._cache_fresh("macro_treasury", max_age_hours=12):
        data = client.get("treasury", params={"from": "2020-01-01"})
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                client.save_parquet(df, "macro_treasury")
                print("    Treasury rates: OK")

    indicators = {
        "federalFunds": "fed_funds",
        "CPI": "cpi",
        "unemploymentRate": "unemployment",
        "consumerSentiment": "consumer_sentiment",
    }
    for fmp_name, local_name in indicators.items():
        cache_name = f"macro_{local_name}"
        if client._cache_fresh(cache_name, max_age_hours=48):
            continue
        data = client.get("economic", params={"name": fmp_name, "from": "2020-01-01"})
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                client.save_parquet(df, cache_name)
                print(f"    {local_name}: OK")

    if not client._cache_fresh("macro_vix", max_age_hours=12):
        data = client.get("historical-price-eod/full", params={
            "symbol": "^VIX", "from": "2020-01-01",
        })
        if data and isinstance(data, list):
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                client.save_parquet(df, "macro_vix")
                print("    VIX: OK")


def _parallel_download(client, tickers, download_fn, label):
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_fn, client, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            ticker, success = future.result()
            if success:
                ok += 1
            else:
                fail += 1
            if (i + 1) % 100 == 0:
                print(f"    {label}: {i+1}/{len(tickers)} (ok={ok}, fail={fail})")
    return ok, fail


def download_all(client):
    os.makedirs(client.cache_dir, exist_ok=True)

    print("Step 1: Building master ticker list...")
    all_tickers, all_meta = get_master_ticker_list(client)

    meta_df = pd.DataFrame.from_dict(all_meta, orient="index")
    meta_df.index.name = "ticker"
    client.save_parquet(meta_df.reset_index(), "metadata")

    print(f"\nStep 2: Downloading prices ({len(all_tickers)} tickers)...")
    ok, fail = _parallel_download(client, all_tickers, download_prices, "prices")
    print(f"  Prices done: {ok} ok, {fail} failed")

    print(f"\nStep 3: Downloading quarterly fundamentals ({len(all_tickers)} tickers)...")
    ok, fail = _parallel_download(client, all_tickers, download_fundamentals_quarterly, "fundamentals")
    print(f"  Quarterly fundamentals done: {ok} ok, {fail} failed")

    print(f"\nStep 4: Downloading macro data...")
    download_macro(client)

    print(f"\nDownload complete. Cache: {client.cache_dir}")


# ---------------------------------------------------------------------------
# Section 3: DataStore Class
# ---------------------------------------------------------------------------

class DataStore:
    """
    Read-only data interface for strategies. Prevents lookahead bias.

    Fundamental data is point-in-time: built from historical quarterly
    reports, each lagged by REPORTING_LAG_DAYS after fiscal period end,
    then forward-filled. The backtest sees only what was actually known.
    """

    def __init__(self, price_data, volume_data, fundamental_data,
                 macro_data, metadata, cache_dir=CACHE_DIR):
        self._prices = price_data
        self._volume = volume_data
        self._fundamentals = fundamental_data
        self._macro = macro_data
        self._metadata = metadata
        self._cache_dir = cache_dir

    @classmethod
    def from_cache(cls, cache_dir=CACHE_DIR):
        cache = Path(cache_dir)

        # Metadata
        meta_df = pd.read_parquet(cache / "metadata.parquet")
        metadata = {}
        for _, row in meta_df.iterrows():
            metadata[row["ticker"]] = row.to_dict()

        # Prices + volumes
        price_frames = {}
        volume_frames = {}
        for pf in sorted(cache.glob("prices_*.parquet")):
            try:
                df = pd.read_parquet(pf)
            except Exception:
                continue
            ticker_raw = pf.stem[len("prices_"):]
            matched = next((t for t in metadata if t.replace(".", "_") == ticker_raw), ticker_raw)

            if "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            price_col = "adjClose" if "adjClose" in df.columns else "close"
            if price_col in df.columns:
                price_frames[matched] = df[price_col].astype(float)
            if "volume" in df.columns:
                volume_frames[matched] = df["volume"].astype(float)

        prices = pd.DataFrame(price_frames).sort_index()
        volumes = pd.DataFrame(volume_frames).sort_index()

        # Quarterly fundamentals — point-in-time aligned
        fundamentals = cls._load_fundamentals_quarterly(cache, prices.index)

        # Macro
        macro = cls._load_macro(cache)

        print(f"DataStore loaded: {len(prices.columns)} tickers, "
              f"{len(prices)} trading days, "
              f"{len(fundamentals)} fundamental fields, "
              f"{len(macro)} macro series")

        return cls(prices, volumes, fundamentals, macro, metadata, cache_dir)

    @staticmethod
    def _load_fundamentals_quarterly(cache, trading_dates):
        """
        Build point-in-time fundamental DataFrames from historical quarterly data.

        Each quarterly report becomes visible REPORTING_LAG_DAYS after the
        fiscal period end date. Between reports, the last known value is
        forward-filled. Result: a date x ticker DataFrame for each field
        where every cell is the value that was actually knowable on that date.
        """
        field_map = {
            # Non-TTM names (quarterly endpoints)
            "peRatio": "pe",
            "priceToBookRatio": "pb",
            "priceToSalesRatio": "ps",
            "enterpriseValueOverEBITDA": "ev_ebitda",
            "freeCashFlowYield": "fcf_yield",
            "earningsYield": "earnings_yield",
            "returnOnEquity": "roe",
            "returnOnAssets": "roa",
            "returnOnCapitalEmployed": "roic",
            "debtEquityRatio": "debt_to_equity",
            "currentRatio": "current_ratio",
            "grossProfitMargin": "gross_margin",
            "netProfitMargin": "net_margin",
            "revenueGrowth": "revenue_growth",
            "piotroskiIScore": "piotroski",
            "altmanZScore": "altman_z",
            # TTM-suffixed variants (some endpoints use these)
            "peRatioTTM": "pe",
            "priceToBookRatioTTM": "pb",
            "priceToSalesRatioTTM": "ps",
            "enterpriseValueOverEBITDATTM": "ev_ebitda",
            "freeCashFlowYieldTTM": "fcf_yield",
            "earningsYieldTTM": "earnings_yield",
            "returnOnEquityTTM": "roe",
            "returnOnAssetsTTM": "roa",
            "returnOnCapitalEmployedTTM": "roic",
            "debtEquityRatioTTM": "debt_to_equity",
            "currentRatioTTM": "current_ratio",
            "grossProfitMarginTTM": "gross_margin",
            "netProfitMarginTTM": "net_margin",
            "revenueGrowthTTM": "revenue_growth",
            "piotroskiIScoreTTM": "piotroski",
            "altmanZScoreTTM": "altman_z",
        }

        # Collect observations: field -> {ticker: [(available_date, value), ...]}
        our_fields = set(field_map.values())
        field_obs = {f: {} for f in our_fields}

        for ff in sorted(cache.glob("fundamentals_q_*.parquet")):
            try:
                df = pd.read_parquet(ff)
            except Exception:
                continue
            if "ticker" not in df.columns or "date" not in df.columns or len(df) == 0:
                continue

            ticker = df["ticker"].iloc[0]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            for _, row in df.iterrows():
                # FMP "date" = fiscal period end. Data becomes available after lag.
                available_date = row["date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)

                for fmp_field, our_field in field_map.items():
                    if fmp_field not in row.index:
                        continue
                    val = row[fmp_field]
                    try:
                        val = float(val) if val is not None else np.nan
                    except (ValueError, TypeError):
                        continue
                    if np.isnan(val):
                        continue
                    if ticker not in field_obs[our_field]:
                        field_obs[our_field][ticker] = []
                    field_obs[our_field][ticker].append((available_date, val))

        # Build date x ticker DataFrames with forward-fill
        fundamentals = {}
        for field, ticker_data in field_obs.items():
            if not ticker_data:
                continue

            records = []
            for ticker, obs in ticker_data.items():
                for avail_date, value in obs:
                    records.append({"date": avail_date, "ticker": ticker, "value": value})
            if not records:
                continue

            sparse = pd.DataFrame(records)
            pivoted = sparse.pivot_table(
                index="date", columns="ticker", values="value", aggfunc="last"
            )
            # Reindex to trading dates, forward-fill between quarterly reports
            pivoted = pivoted.reindex(trading_dates).sort_index().ffill()
            fundamentals[field] = pivoted

        return fundamentals

    @staticmethod
    def _load_macro(cache):
        macro = {}

        treasury_path = cache / "macro_treasury.parquet"
        if treasury_path.exists():
            df = pd.read_parquet(treasury_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                for fmp_col, name in {"year10": "t10y", "year2": "t2y", "month3": "t3m"}.items():
                    if fmp_col in df.columns:
                        macro[name] = df[fmp_col].astype(float)
                if "t10y" in macro and "t2y" in macro:
                    macro["t10y_2y_spread"] = macro["t10y"] - macro["t2y"]

        for local_name in ["fed_funds", "cpi", "unemployment", "consumer_sentiment"]:
            path = cache / f"macro_{local_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if "date" in df.columns and "value" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    macro[local_name] = df["value"].astype(float)

        vix_path = cache / "macro_vix.parquet"
        if vix_path.exists():
            df = pd.read_parquet(vix_path)
            if "date" in df.columns and "close" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                macro["vix"] = df["close"].astype(float)

        return macro

    # --- Public API ---

    def prices(self, tickers=None, start=None, end=None):
        df = self._prices
        if tickers is not None:
            df = df[[t for t in tickers if t in df.columns]]
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        return df

    def returns(self, tickers=None, period=1):
        return self.prices(tickers).pct_change(period)

    def volume(self, tickers=None):
        df = self._volume
        if tickers is not None:
            df = df[[t for t in tickers if t in df.columns]]
        return df

    def fundamental(self, field):
        """
        Point-in-time fundamental data: date x ticker DataFrame.

        Built from historical quarterly reports. Each value is what was
        actually knowable on that date (fiscal period end + 45-day lag,
        forward-filled between quarters).

        Fields: pe, pb, ps, ev_ebitda, fcf_yield, earnings_yield, roe, roa,
                roic, debt_to_equity, current_ratio, gross_margin, net_margin,
                revenue_growth, piotroski, altman_z
        """
        if field not in self._fundamentals:
            available = list(self._fundamentals.keys())
            raise KeyError(f"Unknown fundamental field '{field}'. Available: {available}")
        return self._fundamentals[field]

    def macro(self, field):
        if field not in self._macro:
            available = list(self._macro.keys())
            raise KeyError(f"Unknown macro field '{field}'. Available: {available}")
        return self._macro[field]

    def universe(self, date=None):
        if date is None:
            return list(self._prices.columns)
        prices_to_date = self._prices.loc[:date]
        valid = prices_to_date.columns[prices_to_date.notna().sum() >= 60]
        return list(valid)

    def sector(self, ticker):
        meta = self._metadata.get(ticker, {})
        return meta.get("sector", meta.get("profile_sector", "Unknown"))

    def country(self, ticker):
        meta = self._metadata.get(ticker, {})
        return meta.get("country", meta.get("profile_country", "Unknown"))

    def metadata_for(self, ticker):
        return self._metadata.get(ticker, {})

    def correlation(self, tickers, window=60):
        rets = self.returns(tickers)
        if len(rets) < window:
            return rets.corr()
        return rets.iloc[-window:].corr()


# ---------------------------------------------------------------------------
# Section 4: Backtest Engine
# ---------------------------------------------------------------------------

class BacktestResult:
    def __init__(self, daily_returns, equity_curve, weights_history,
                 turnover_history, dates):
        self.daily_returns = daily_returns
        self.equity_curve = equity_curve
        self.weights_history = weights_history
        self.turnover_history = turnover_history
        self.dates = dates


class DateLimitedStore:
    """DataStore wrapper that prevents lookahead past a cutoff date."""

    def __init__(self, store, cutoff):
        self._store = store
        self._cutoff = pd.Timestamp(cutoff)

    def prices(self, tickers=None, start=None, end=None):
        end_ts = self._cutoff
        if end:
            end_ts = min(pd.Timestamp(end), self._cutoff)
        return self._store.prices(tickers, start, str(end_ts.date()))

    def returns(self, tickers=None, period=1):
        return self.prices(tickers).pct_change(period)

    def volume(self, tickers=None):
        return self._store.volume(tickers).loc[:self._cutoff]

    def fundamental(self, field):
        return self._store.fundamental(field).loc[:self._cutoff]

    def macro(self, field):
        return self._store.macro(field).loc[:self._cutoff]

    def universe(self, date=None):
        return self._store.universe(date or self._cutoff)

    def sector(self, ticker):
        return self._store.sector(ticker)

    def country(self, ticker):
        return self._store.country(ticker)

    def metadata_for(self, ticker):
        return self._store.metadata_for(ticker)

    def correlation(self, tickers, window=60):
        rets = self.returns(tickers)
        if len(rets) < window:
            return rets.corr()
        return rets.iloc[-window:].corr()


def run_backtest(strategy_module, data_store, start, end, rebalance_freq="M"):
    """
    Portfolio backtest with transaction costs.

    On each rebalance date:
      1. Create DateLimitedStore (no lookahead)
      2. strategy.signals(data, date) -> scores
      3. strategy.construct(scores, data, date) -> target weights
      4. strategy.risk(weights, data, date) -> final weights
    Between rebalances: mark-to-market from daily returns.
    """
    freq = getattr(strategy_module, "REBALANCE_FREQ", rebalance_freq)

    lead_start = pd.Timestamp(start) - pd.DateOffset(months=14)
    prices = data_store.prices(start=str(lead_start.date()), end=end)

    if prices.empty:
        raise ValueError("No price data available for backtest period")

    bt_prices = prices.loc[start:end]
    if bt_prices.empty:
        raise ValueError(f"No prices in [{start}, {end}]")

    trading_dates = bt_prices.index

    period_map = {"M": "M", "W": "W", "Q": "Q"}
    period_code = period_map.get(freq, "M")
    rebal_dates = bt_prices.groupby(bt_prices.index.to_period(period_code)).apply(
        lambda x: x.index[0]
    ).values
    rebal_dates = pd.DatetimeIndex(rebal_dates)

    capital = float(INITIAL_CAPITAL)
    current_weights = pd.Series(dtype=float)
    equity = []
    daily_rets = []
    weights_history = []
    turnover_list = []
    prev_date = None
    cost_rate = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000

    for date in trading_dates:
        if prev_date is not None:
            day_ret = bt_prices.loc[date] / bt_prices.loc[prev_date] - 1
            if len(current_weights) > 0:
                valid = current_weights.index.intersection(day_ret.dropna().index)
                if len(valid) > 0:
                    port_ret = (current_weights[valid] * day_ret[valid]).sum()
                    drifted = current_weights.copy()
                    for t in valid:
                        drifted[t] = current_weights[t] * (1 + day_ret[t])
                    if drifted.sum() > 0:
                        drifted = drifted / drifted.sum()
                    current_weights = drifted
                else:
                    port_ret = 0.0
            else:
                port_ret = 0.0
            capital *= (1 + port_ret)
            daily_rets.append(port_ret)
        else:
            daily_rets.append(0.0)

        equity.append(capital)

        if date in rebal_dates:
            try:
                limited = DateLimitedStore(data_store, date)

                scores = strategy_module.signals(limited, date)
                if scores is None or len(scores) == 0:
                    continue

                target_weights = strategy_module.construct(scores, limited, date)
                if target_weights is None or len(target_weights) == 0:
                    continue

                final_weights = strategy_module.risk(target_weights, limited, date)
                if final_weights is None or len(final_weights) == 0:
                    continue

                final_weights = final_weights[final_weights > 0]
                if final_weights.sum() > 0:
                    final_weights = final_weights / final_weights.sum()

                old_set = set(current_weights.index) if len(current_weights) > 0 else set()
                new_set = set(final_weights.index)
                turnover = sum(
                    abs(final_weights.get(t, 0.0) - current_weights.get(t, 0.0))
                    for t in old_set | new_set
                ) / 2

                capital *= (1 - turnover * cost_rate * 2)
                current_weights = final_weights
                weights_history.append((date, final_weights.copy()))
                turnover_list.append((date, turnover))

            except Exception as e:
                print(f"  Warning: strategy error on {date}: {e}")
                continue

        prev_date = date

    daily_returns = pd.Series(daily_rets, index=trading_dates, name="returns")
    equity_curve = pd.Series(equity, index=trading_dates, name="equity")
    turnover_series = pd.Series(
        dict(turnover_list), name="turnover"
    ) if turnover_list else pd.Series(dtype=float)

    return BacktestResult(daily_returns, equity_curve, weights_history,
                          turnover_series, trading_dates)


# ---------------------------------------------------------------------------
# Section 5: Evaluation (FIXED metric)
# ---------------------------------------------------------------------------

def evaluate(strategy_module, data_store, n_trials_so_far=0, period="val"):
    if period == "val":
        start, end = TRAIN_END, VAL_END
    elif period == "test":
        start, end = VAL_END, BACKTEST_END
    elif period == "train":
        start, end = BACKTEST_START, TRAIN_END
    else:
        raise ValueError(f"Unknown period: {period}")

    result = run_backtest(strategy_module, data_store, start, end)
    rets = result.daily_returns.dropna()

    if len(rets) < 10:
        return {"dsr": 0.0, "sharpe": 0.0, "annual_return": 0.0,
                "max_drawdown": 0.0, "sortino": 0.0, "calmar": 0.0,
                "turnover": 0.0, "n_positions": 0,
                "sharpe_ci_95": (0.0, 0.0), "complexity_loc": count_loc("strategy.py"),
                "equity_curve": result.equity_curve}

    sharpe = annualized_sharpe(rets)
    dsr = deflated_sharpe(sharpe, max(n_trials_so_far, 1), len(rets))
    ci_low, ci_high = bootstrap_sharpe_ci(rets, n_bootstrap=1000)

    n_pos = np.mean([len(w) for _, w in result.weights_history]) if result.weights_history else 0
    avg_turnover = result.turnover_history.mean() if len(result.turnover_history) > 0 else 0.0

    return {
        "dsr": dsr,
        "sharpe": sharpe,
        "annual_return": annual_return(rets),
        "max_drawdown": max_drawdown(rets),
        "sortino": sortino_ratio(rets),
        "calmar": calmar_ratio(rets),
        "turnover": avg_turnover,
        "n_positions": int(n_pos),
        "sharpe_ci_95": (ci_low, ci_high),
        "complexity_loc": count_loc("strategy.py"),
        "equity_curve": result.equity_curve,
    }


def count_loc(filepath):
    try:
        with open(filepath) as f:
            lines = f.readlines()
        return sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Section 6: Statistical Utilities
# ---------------------------------------------------------------------------

def annualized_sharpe(returns, risk_free=0.0):
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def annual_return(returns):
    total = (1 + returns).prod()
    n_years = len(returns) / 252
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def sortino_ratio(returns, risk_free=0.0):
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def calmar_ratio(returns):
    mdd = max_drawdown(returns)
    ann = annual_return(returns)
    if mdd == 0:
        return 0.0
    return float(ann / abs(mdd))


def deflated_sharpe(sharpe, n_trials, T, skew=0.0, kurtosis=3.0):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""
    from scipy.stats import norm

    if T <= 1 or n_trials <= 0:
        return 0.0

    if n_trials == 1:
        e_max_z = 0.0
    else:
        gamma = 0.5772156649
        e_max_z = (norm.ppf(1 - 1 / n_trials) * (1 - gamma) +
                   gamma * norm.ppf(1 - 1 / (n_trials * np.e)))

    sr_daily = sharpe / np.sqrt(252)
    var_sr = (1 + 0.5 * sr_daily**2 - skew * sr_daily +
              ((kurtosis - 3) / 4) * sr_daily**2) / T

    if var_sr <= 0:
        return 0.0

    test_stat = (sr_daily * np.sqrt(T) - e_max_z) / np.sqrt(max(var_sr * T, 1e-10))
    return float(norm.cdf(test_stat))


def bootstrap_sharpe_ci(returns, n_bootstrap=1000, ci=0.95, block_size=21):
    rng = np.random.default_rng(42)
    returns_arr = np.asarray(returns)
    T = len(returns_arr)

    if T < block_size * 2:
        return (0.0, 0.0)

    sharpes = []
    n_blocks = int(np.ceil(T / block_size))

    for _ in range(n_bootstrap):
        starts = rng.integers(0, T - block_size + 1, size=n_blocks)
        sample = np.concatenate([returns_arr[s:s + block_size] for s in starts])[:T]
        if sample.std() > 0:
            sharpes.append(sample.mean() / sample.std() * np.sqrt(252))

    if len(sharpes) == 0:
        return (0.0, 0.0)

    alpha = (1 - ci) / 2
    return (float(np.percentile(sharpes, alpha * 100)),
            float(np.percentile(sharpes, (1 - alpha) * 100)))


# ---------------------------------------------------------------------------
# Section 7: CLI Entry Point
# ---------------------------------------------------------------------------

def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
    if spec is None:
        print("Error: strategy.py not found in current directory.")
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_metrics(metrics, label=""):
    if label:
        print(f"\n=== {label} ===")
    print("---")
    print(f"dsr:            {metrics['dsr']:.6f}")
    print(f"sharpe:         {metrics['sharpe']:.6f}")
    print(f"annual_return:  {metrics['annual_return']:.6f}")
    print(f"max_drawdown:   {metrics['max_drawdown']:.6f}")
    print(f"sortino:        {metrics['sortino']:.6f}")
    print(f"calmar:         {metrics['calmar']:.6f}")
    print(f"turnover:       {metrics['turnover']:.6f}")
    print(f"n_positions:    {metrics['n_positions']}")
    ci = metrics['sharpe_ci_95']
    print(f"sharpe_ci_95:   ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"complexity_loc: {metrics['complexity_loc']}")


def main():
    parser = argparse.ArgumentParser(description="Q_Lab: Autonomous quantitative research")
    parser.add_argument("--download", action="store_true", help="Download/refresh all data from FMP")
    parser.add_argument("--backtest", action="store_true", help="Run strategy.py on validation period")
    parser.add_argument("--test", action="store_true", help="Run on holdout test period (human only)")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of trials so far (for DSR)")
    args = parser.parse_args()

    if not any([args.download, args.backtest, args.test]):
        parser.print_help()
        sys.exit(0)

    if args.download:
        print("Downloading data from FMP API...")
        client = FMPClient()
        download_all(client)

    if args.backtest or args.test:
        print("Loading data from cache...")
        t0 = time.time()
        data_store = DataStore.from_cache()
        print(f"Data loaded in {time.time() - t0:.1f}s")

        strategy = load_strategy()

        if args.backtest:
            print("\nRunning backtest on VALIDATION period...")
            t0 = time.time()
            metrics = evaluate(strategy, data_store, n_trials_so_far=args.n_trials,
                              period="val")
            elapsed = time.time() - t0
            print(f"Backtest completed in {elapsed:.1f}s")
            print_metrics(metrics, "Validation Results")

        if args.test:
            print("\nRunning backtest on TEST period (holdout)...")
            t0 = time.time()
            metrics = evaluate(strategy, data_store, n_trials_so_far=args.n_trials,
                              period="test")
            elapsed = time.time() - t0
            print(f"Backtest completed in {elapsed:.1f}s")
            print_metrics(metrics, "Test Results (Holdout)")


if __name__ == "__main__":
    main()
