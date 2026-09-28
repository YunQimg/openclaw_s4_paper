# -*- coding: utf-8 -*-
"""市场数据下载器（Roadmap V2 §4.2）。

**所有网络请求必须走代理**（`DEFAULT_PROXY`，可用 `PAPER_PROXY` 环境变量覆盖）。
未显式配置代理时，本模块**拒绝发起请求**（硬失败），避免在生产环境静默直连。

下载内容：
  * BTC/ETH/SOL/BNB 现货 4H OHLCV  -> data/paper_trading/market/4h.csv
  * BTC/ETH/SOL/BNB 现货 1D OHLCV  -> data/paper_trading/market/1d.csv
  * FRED DFF 有效联邦基金利率      -> data/paper_trading/market/cash_rate_dff.csv

下载器只写入市场数据目录，不持有任何交易权限（§4.2 末段）。
"""
from __future__ import annotations

import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .config import ASSETS, DEFAULT_PROXY

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO / "data" / "paper_trading" / "market"

SYMBOLS: Dict[str, str] = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}

KLINE_URLS: List[str] = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"

START = "2017-01-01"
COLS = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
_PAGE_SLEEP = 0.12


class FetcherUnavailable(RuntimeError):
    """无可用代理或网络被显式禁用。"""


def resolve_proxies(proxy: Optional[str] = None) -> Dict[str, str]:
    """解析代理配置。**没有代理就不允许下载**（Roadmap 要求）。

    代理来源优先级：显式参数 > `PAPER_PROXY` 环境变量 > `DEFAULT_PROXY`。
    显式传入空字符串视为「明确要求无代理」，直接硬失败。
    """
    if os.environ.get("PAPER_DISABLE_NETWORK_FETCHES") == "1":
        raise FetcherUnavailable(
            "network fetches disabled by PAPER_DISABLE_NETWORK_FETCHES=1")
    if proxy is not None and not proxy.strip():
        raise FetcherUnavailable(
            "empty proxy disables downloads; all downloads must go through a proxy")
    url = proxy or os.environ.get("PAPER_PROXY") or DEFAULT_PROXY
    if not url or not url.strip():
        raise FetcherUnavailable(
            "no proxy configured; all downloads must go through a proxy. "
            "Set PAPER_PROXY or pass proxy= explicitly.")
    url = url.strip()
    return {"http": url, "https": url}


def _date_to_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(url: str, *, params: Optional[dict], proxies: Dict[str, str],
         timeout: int = 30) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout, proxies=proxies)
    r.raise_for_status()
    return r


def fetch_klines(symbol: str, interval: str, proxies: Dict[str, str],
                 start: str = START) -> pd.DataFrame:
    """分页下载 Binance 现货 K 线（走代理），返回 timestamp/open/high/low/close/volume。"""
    start_ms = _date_to_ms(start)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows: list = []
    base = KLINE_URLS[0]

    while start_ms < end_ms:
        params = {"symbol": symbol, "interval": interval, "limit": 1000,
                  "startTime": start_ms, "endTime": end_ms}
        data = None
        last_err: Optional[Exception] = None
        for url in [base] + [u for u in KLINE_URLS if u != base]:
            try:
                data = _get(url, params=params, proxies=proxies).json()
                base = url
                break
            except Exception as e:                              # noqa: BLE001
                last_err = e
        if data is None:
            raise FetcherUnavailable(f"{symbol} {interval} download failed: {last_err}")
        if not data:
            break
        rows.extend(data)
        new_start = data[-1][0] + 1
        if new_start <= start_ms:
            break
        start_ms = new_start
        if len(data) < 1000:
            break
        time.sleep(_PAGE_SLEEP)

    if not rows:
        raise FetcherUnavailable(f"{symbol} {interval} returned no data")

    df = pd.DataFrame(rows)[list(range(6))]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_cash_rate(proxies: Dict[str, str]) -> pd.DataFrame:
    """FRED DFF 有效联邦基金利率（年化 %，日频），走代理。"""
    r = _get(FRED_URL, params=None, proxies=proxies, timeout=40)
    raw = pd.read_csv(io.StringIO(r.text))
    date_col, val_col = raw.columns[0], raw.columns[1]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
    df = raw.dropna().rename(columns={date_col: "timestamp", val_col: "rate_pct"})
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df.sort_values("timestamp").reset_index(drop=True)


def validate_market_frame(df: pd.DataFrame, interval: str) -> List[str]:
    """§4.2 数据契约校验，返回错误信息列表（空列表 = 通过）。"""
    errors: List[str] = []
    required = {"timestamp", "asset", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["market frame is empty"]
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("timestamp must be UTC datetime")
    if not str(df["timestamp"].dt.tz).startswith("UTC"):
        errors.append("timestamp must be tz-aware UTC")
    dup = df.duplicated(subset=["asset", "timestamp"])
    if bool(dup.any()):
        errors.append(f"{int(dup.sum())} duplicate asset/timestamp rows")
    if bool((df["close"] <= 0).any()):
        errors.append("close must be > 0")
    bad_hi = df["high"] < df[["open", "close"]].max(axis=1) - 1e-12
    if bool(bad_hi.any()):
        errors.append(f"high < max(open, close) in {int(bad_hi.sum())} rows")
    bad_lo = df["low"] > df[["open", "close"]].min(axis=1) + 1e-12
    if bool(bad_lo.any()):
        errors.append(f"low > min(open, close) in {int(bad_lo.sum())} rows")
    unknown = sorted(set(df["asset"]) - set(ASSETS))
    if unknown:
        errors.append(f"unknown assets: {unknown}")
    for a, g in df.groupby("asset", sort=False):
        if not g["timestamp"].is_monotonic_increasing:
            errors.append(f"{a} timestamps not ascending")
    return errors


def download_market_data(out_dir: Path = DEFAULT_OUT_DIR,
                         proxy: Optional[str] = None,
                         assets: Optional[List[str]] = None,
                         start: str = START) -> Dict[str, Path]:
    """下载 4H + 1D + 现金利率并落盘，返回 {name: path}。

    所有请求走代理；代理缺失时抛 `FetcherUnavailable`，不静默直连。
    """
    proxies = resolve_proxies(proxy)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(assets or ASSETS)

    frames_4h: List[pd.DataFrame] = []
    frames_1d: List[pd.DataFrame] = []
    for a in wanted:
        sym = SYMBOLS[a]
        for interval, bucket in (("4h", frames_4h), ("1d", frames_1d)):
            df = fetch_klines(sym, interval, proxies, start=start)
            df.insert(1, "asset", a)
            bucket.append(df[COLS])
            time.sleep(0.3)

    paths: Dict[str, Path] = {}
    for name, bucket in (("4h", frames_4h), ("1d", frames_1d)):
        merged = (pd.concat(bucket, ignore_index=True)
                  .sort_values(["timestamp", "asset"]).reset_index(drop=True))
        errors = validate_market_frame(merged, name)
        if errors:
            raise ConfigErrorLike(errors)
        p = out_dir / f"{name}.csv"
        merged.to_csv(p, index=False)
        paths[name] = p

    cash = fetch_cash_rate(proxies)
    p_cash = out_dir / "cash_rate_dff.csv"
    cash.to_csv(p_cash, index=False)
    paths["cash_rate_dff"] = p_cash
    return paths


class ConfigErrorLike(ValueError):
    """下载数据未通过 §4.2 契约校验。"""

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        super().__init__("downloaded market data failed validation: " + "; ".join(errors))


def main() -> int:                                              # pragma: no cover
    """CLI 入口：python -m openclaw_s4_paper.download_data"""
    import argparse

    ap = argparse.ArgumentParser(description="Download S4+R9 paper-trading market data (proxy required)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--proxy", default=None, help=f"proxy URL (default {DEFAULT_PROXY})")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    proxies = resolve_proxies(args.proxy)
    print(f"proxy: {proxies['https']}")
    print(f"out  : {args.out_dir}")
    paths = download_market_data(args.out_dir, args.proxy, args.assets, args.start)
    for k, v in paths.items():
        print(f"  [ok] {k}: {v}  ({v.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())