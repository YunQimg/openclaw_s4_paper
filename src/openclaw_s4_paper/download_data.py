# -*- coding: utf-8 -*-
"""市场数据下载器（Roadmap V2 §4.2）。

**代理策略（§13.1 修订）**：先直连探测 Google 连通性——
  * 可直连   -> 不使用代理；
  * 不可直连 -> 必须走代理（`PAPER_PROXY` > `DEFAULT_PROXY`），无可用代理则硬失败。
显式传入 `--proxy URL` 时直接采用该代理；显式传空串视为非法并硬失败。

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
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from .config import ASSETS, DEFAULT_PROXY

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / "data" / "paper_trading" / "market"

SYMBOLS: Dict[str, str] = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}

# K 线源：Binance.US 为主（美国境内可访问），data-api.binance.vision 为同 REST 形状的备用。
KLINE_URLS: List[str] = [
    "https://api.binance.us/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]

# 直连连通性探针：Google 可达即视为不需要代理。
# 不要改成 www.gstatic.com 之类的 CDN 域：Google 主站被阻断时 gstatic 常仍可直连，
# 会造成「判定可直连 -> 不走代理 -> 数据源全部超时」的假阳性。
GOOGLE_PROBE_URLS: List[str] = ["https://www.google.com/generate_204"]
PROBE_TIMEOUT = 5
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"

START = "2017-01-01"
COLS = ["timestamp", "asset", "open", "high", "low", "close", "volume"]
_PAGE_SLEEP = 0.12


class FetcherUnavailable(RuntimeError):
    """网络被显式禁用，或需要代理却没有可用代理。"""


def has_direct_internet(timeout: int = PROBE_TIMEOUT,
                        urls: Optional[List[str]] = None) -> bool:
    """不走代理直连探测 Google，任一探针成功即视为可直连。"""
    for url in urls or GOOGLE_PROBE_URLS:
        try:
            r = requests.get(url, timeout=timeout, proxies={})
        except requests.RequestException:
            continue
        if 200 <= r.status_code < 400:
            return True
    return False


def resolve_proxies(proxy: Optional[str] = None,
                    direct_check: Optional[Callable[[], bool]] = None) -> Dict[str, str]:
    """解析代理配置（§13.1 修订）。

    顺序：
      1. `PAPER_DISABLE_NETWORK_FETCHES=1` -> 硬失败。
      2. 显式传空字符串 -> 硬失败（空代理非法）。
      3. 显式传非空代理 -> 直接采用，覆盖自动探测。
      4. 直连探测 Google：可直连 -> 返回空 dict（不走代理）；
         不可直连 -> 采用 `PAPER_PROXY` > `DEFAULT_PROXY`，都无则硬失败。

    `direct_check` 仅供测试注入，默认执行真实探测。
    """
    if os.environ.get("PAPER_DISABLE_NETWORK_FETCHES") == "1":
        raise FetcherUnavailable(
            "network fetches disabled by PAPER_DISABLE_NETWORK_FETCHES=1")
    if proxy is not None:
        url = proxy.strip()
        if not url:
            raise FetcherUnavailable(
                "empty proxy disables downloads; pass a proxy URL or omit it "
                "to auto-detect direct connectivity")
        return {"http": url, "https": url}

    if (direct_check or has_direct_internet)():
        return {}

    url = (os.environ.get("PAPER_PROXY") or DEFAULT_PROXY or "").strip()
    if not url:
        raise FetcherUnavailable(
            "no direct internet (Google unreachable) and no proxy configured; "
            "set PAPER_PROXY or pass proxy= explicitly.")
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
    """分页下载 Binance.US 现货 K 线，返回 timestamp/open/high/low/close/volume。"""
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
    """FRED DFF 有效联邦基金利率（年化 %，日频）。"""
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

    按 `resolve_proxies` 判定是否走代理；需要代理却无可用代理时抛
    `FetcherUnavailable`，不静默直连。
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

    ap = argparse.ArgumentParser(description="Download S4+R9 paper-trading market data (auto proxy)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--proxy", default=None,
                    help=f"force this proxy (default: auto-detect; fallback {DEFAULT_PROXY})")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    proxies = resolve_proxies(args.proxy)
    print(f"proxy: {proxies.get('https') or 'direct (no proxy)'}")
    print(f"out  : {args.out_dir}")
    paths = download_market_data(args.out_dir, args.proxy, args.assets, args.start)
    for k, v in paths.items():
        print(f"  [ok] {k}: {v}  ({v.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())