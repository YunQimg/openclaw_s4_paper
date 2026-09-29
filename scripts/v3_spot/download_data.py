# -*- coding: utf-8 -*-
"""V3 现货多策略系统 —— 数据下载器

下载 BTC / ETH / SOL / BNB 现货 OHLCV：
  - 4H（信号主时间轴：MA200 / ATR / ADX / Persistence / MTF-4H）
  - 1D（TSMOM 6M/12M、MTF-1D、日频再平衡）
  - 1W（MTF-1W MA40，由 1D 重采样得到）
另下载现金收益率（FRED DFF，实际有效联邦基金利率日频）用于 Cash Return Version B。

全部网络请求走本地代理 PROXIES。
输出目录：data/v3_spot/
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "v3_spot"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}

ASSETS = [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL"), ("BNBUSDT", "BNB")]

KLINE_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"

START = "2017-01-01"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _date_to_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def download_klines(symbol: str, interval: str, out: Path, force: bool = False) -> pd.DataFrame:
    """分页下载 Binance 现货 K 线（走代理）。"""
    if out.exists() and not force:
        df = pd.read_csv(out, parse_dates=["timestamp"])
        print(f"  [skip] {out.name}: {len(df)} rows "
              f"{df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()}")
        return df

    start_ms = _date_to_ms(START)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows: list = []
    base = KLINE_URLS[0]
    while start_ms < end_ms:
        params = {"symbol": symbol, "interval": interval, "limit": 1000,
                  "startTime": start_ms, "endTime": end_ms}
        data = None
        last_err = None
        for url in ([base] + [u for u in KLINE_URLS if u != base]):
            try:
                r = requests.get(url, params=params, timeout=30, proxies=PROXIES)
                r.raise_for_status()
                data = r.json()
                base = url
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if data is None:
            raise RuntimeError(f"{symbol} {interval} 下载失败: {last_err}")
        if not data:
            break
        rows.extend(data)
        new_start = data[-1][0] + 1
        if new_start <= start_ms:
            break
        start_ms = new_start
        print(f"  {symbol} {interval}: {len(rows)} candles", end="\r", flush=True)
        if len(data) < 1000:
            break
        time.sleep(0.12)

    if not rows:
        raise RuntimeError(f"{symbol} {interval} 无数据返回")

    df = pd.DataFrame(rows)[list(range(6))]
    df.columns = COLS
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[COLS].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\n  [ok] {out.name}: {len(df)} rows "
          f"{df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()}")
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """1D -> 1W（周日起始 W-SUN，UTC），close 为周内最后收盘。"""
    d = df.set_index("timestamp").sort_index()
    o = d["open"].resample("W-SUN").first()
    h = d["high"].resample("W-SUN").max()
    lo = d["low"].resample("W-SUN").min()
    c = d["close"].resample("W-SUN").last()
    v = d["volume"].resample("W-SUN").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}).dropna()
    return out.reset_index()


def download_cash_rate(out: Path, force: bool = False) -> pd.DataFrame:
    """FRED DFF：有效联邦基金利率（年化 %，日频）。作为 Cash 的无风险收益率。"""
    if out.exists() and not force:
        df = pd.read_csv(out, parse_dates=["timestamp"])
        print(f"  [skip] {out.name}: {len(df)} rows")
        return df
    try:
        r = requests.get(FRED_URL, timeout=40, proxies=PROXIES)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
        date_col = raw.columns[0]
        val_col = raw.columns[1]
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
        df = raw.dropna().rename(columns={date_col: "timestamp", val_col: "rate_pct"})
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"  [ok] {out.name}: {len(df)} rows "
              f"{df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()}")
        return df
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] FRED DFF 下载失败({e})；Cash Return Version B 将退化为 0%")
        return pd.DataFrame(columns=["timestamp", "rate_pct"])


def main() -> None:
    print("=== V3 Spot Data Downloader ===")
    print(f"Proxy: {PROXIES['https']}")
    print(f"Output: {DATA_DIR}\n")

    daily = {}
    for symbol, name in ASSETS:
        print(f"[4H] {name}")
        download_klines(symbol, "4h", DATA_DIR / f"{name}_4h.csv")
        time.sleep(0.3)
        print(f"[1D] {name}")
        daily[name] = download_klines(symbol, "1d", DATA_DIR / f"{name}_1d.csv")
        time.sleep(0.3)
        wk = resample_weekly(daily[name])
        wk.to_csv(DATA_DIR / f"{name}_1w.csv", index=False)
        print(f"  [ok] {name}_1w.csv: {len(wk)} rows")

    print("[CASH] FRED DFF")
    download_cash_rate(DATA_DIR / "cash_rate_dff.csv")

    print("\n=== Done ===")
    for f in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()