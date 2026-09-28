# -*- coding: utf-8 -*-
"""市场数据装载与校验（Roadmap V2 §4 / §13.1）。

职责：
  * 从本地 CSV 装载 4H 面板（不联网、不读账户）；
  * 校验 §4.2 数据契约；
  * 判定数据新鲜度（§4.3，超过 12 小时 -> STALE_DATA）；
  * 定位最后一个完整 4H bar。

禁止：submit_order / load_exchange_account / 任何交易所客户端依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import ASSETS, BARS_PER_DAY, FRESHNESS_MAX_HOURS, ConfigError
from .models import ValidationResult

REQUIRED_COLUMNS = ["timestamp", "asset", "open", "high", "low", "close", "volume"]


@dataclass
class MarketPanel:
    """4H 面板：index（UTC 4H 轴）+ 每资产 OHLCV 与 close 矩阵。"""

    index: pd.DatetimeIndex
    frames: Dict[str, pd.DataFrame]
    close: pd.DataFrame

    @property
    def n_bars(self) -> int:
        return len(self.index)

    def listed_assets(self, at: datetime) -> List[str]:
        """在 `at` 时刻已上市（有有效收盘价）的资产。"""
        if at not in self.close.index:
            return []
        row = self.close.loc[at]
        return [a for a in ASSETS if not pd.isna(row.get(a))]

    def price(self, asset: str, at: datetime) -> Optional[float]:
        if at not in self.close.index:
            return None
        v = self.close.at[at, asset]
        return None if pd.isna(v) else float(v)

    def prices_at(self, at: datetime) -> Dict[str, float]:
        return {a: (self.price(a, at) or 0.0) for a in ASSETS}

    def open_price(self, asset: str, at: datetime) -> Optional[float]:
        """执行价基准：该 bar 的开盘价（§6.3）。"""
        f = self.frames.get(asset)
        if f is None or at not in f.index:
            return None
        v = f.at[at, "open"]
        return None if pd.isna(v) else float(v)

    def opens_at(self, at: datetime) -> Dict[str, float]:
        return {a: (self.open_price(a, at) or 0.0) for a in ASSETS}


def load_market_data(path: Path, assets: Optional[List[str]] = None) -> MarketPanel:
    """装载 4H 市场数据 CSV（§4.2 格式）。"""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"market data file not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ConfigError(f"market data missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if bool(df["timestamp"].isna().any()):
        raise ConfigError("market data has unparseable timestamps")

    wanted = list(assets or ASSETS)
    unknown = sorted(set(df["asset"].unique()) - set(wanted))
    if unknown:
        raise ConfigError(f"market data contains unexpected assets: {unknown}")

    idx = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
    frames: Dict[str, pd.DataFrame] = {}
    closes: Dict[str, pd.Series] = {}
    for a in wanted:
        sub = (df[df["asset"] == a]
               .drop_duplicates("timestamp")
               .sort_values("timestamp")
               .set_index("timestamp")
               .reindex(idx))
        sub = sub[["open", "high", "low", "close", "volume"]].astype(float)
        frames[a] = sub
        closes[a] = sub["close"]
    return MarketPanel(index=idx, frames=frames, close=pd.DataFrame(closes).reindex(idx))


def validate_market_data(panel: MarketPanel) -> ValidationResult:
    """§4.2 / §14.5 数据契约校验。"""
    errors: List[str] = []
    warnings: List[str] = []

    if panel.n_bars == 0:
        return ValidationResult(False, 0, 0, ["market panel is empty"])

    if panel.index.tz is None or str(panel.index.tz) != "UTC":
        errors.append("index must be tz-aware UTC")
    if not panel.index.is_monotonic_increasing:
        errors.append("index must be ascending")
    if bool(panel.index.duplicated().any()):
        errors.append("index has duplicate timestamps")

    for a, f in panel.frames.items():
        listed = f["close"].dropna()
        if listed.empty:
            warnings.append(f"{a} has no listed bars")
            continue
        if bool((listed <= 0).any()):
            errors.append(f"{a} has non-positive close")
        hi = f["high"].dropna()
        lo = f["low"].dropna()
        op = f["open"].dropna()
        cl = f["close"].dropna()
        if bool((hi < pd.concat([op, cl], axis=1).max(axis=1) - 1e-12).any()):
            errors.append(f"{a} violates high >= max(open, close)")
        if bool((lo > pd.concat([op, cl], axis=1).min(axis=1) + 1e-12).any()):
            errors.append(f"{a} violates low <= min(open, close)")

    return ValidationResult(not errors, panel.n_bars, len(panel.frames), errors, warnings)


def latest_complete_bar(panel: MarketPanel, now: Optional[datetime] = None) -> datetime:
    """最后一个**完整** 4H bar 的时间戳（§4.3）。

    完整 bar 的时间戳必须严格早于当前时刻（避免使用未收盘 bar）。
    """
    if panel.n_bars == 0:
        raise ConfigError("market panel is empty; no complete bar available")
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    eligible = panel.index[panel.index < ref - timedelta(seconds=1)]
    if len(eligible) == 0:
        raise ConfigError("no complete 4H bar strictly before now")
    return eligible[-1]


def freshness(panel: MarketPanel, now: Optional[datetime] = None,
              max_hours: int = FRESHNESS_MAX_HOURS) -> dict:
    """§4.3 数据新鲜度判定。"""
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    last = latest_complete_bar(panel, ref)
    age_hours = (ref - last.to_pydatetime()).total_seconds() / 3600.0
    return {
        "data_last_time": last,
        "as_of": ref,
        "age_hours": age_hours,
        "max_hours": max_hours,
        "is_fresh": bool(age_hours <= max_hours),
        "status": "OK" if age_hours <= max_hours else "STALE_DATA",
    }


def next_execution_bar(panel: MarketPanel, signal_time: datetime,
                       lag_bars: int = 1) -> datetime:
    """信号 bar -> 执行 bar（§6.4 `execution_lag_bars`）。"""
    pos = panel.index.get_indexer([signal_time])[0]
    if pos < 0:
        raise ConfigError(f"signal_time {signal_time} not in market index")
    target = pos + int(lag_bars)
    if target >= panel.n_bars:
        raise ConfigError(
            f"no execution bar available: signal at {signal_time} + {lag_bars} bars "
            f"exceeds panel end {panel.index[-1]}")
    return panel.index[target]


def daily_decision_bars(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """日频决策 bar = 每日首根 4H bar（§6.4 `decision_cadence=daily`）。

    与 V3/V4/V5 的 `decision_bars(cadence="daily")` 逐位一致：
    对 `index.normalize()` 取唯一值并保留首次出现的下标。
    """
    key = index.normalize().to_numpy()
    first_pos: Dict[object, int] = {}
    for i, k in enumerate(key):
        first_pos.setdefault(k, i)
    return index[sorted(first_pos.values())]


def load_daily_close(path: Path, assets: Optional[List[str]] = None) -> pd.DataFrame:
    """装载 1D 收盘价矩阵（TSMOM 6M 使用，§5.2）。"""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"daily market data not found: {path}")
    df = pd.read_csv(path)
    missing = sorted({"timestamp", "asset", "close"} - set(df.columns))
    if missing:
        raise ConfigError(f"daily data missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    wanted = list(assets or ASSETS)
    wide = (df[df["asset"].isin(wanted)]
            .pivot_table(index="timestamp", columns="asset", values="close", aggfunc="last")
            .sort_index())
    return wide.reindex(columns=wanted)


__all__ = [
    "MarketPanel", "load_market_data", "validate_market_data", "latest_complete_bar",
    "freshness", "next_execution_bar", "daily_decision_bars", "load_daily_close",
    "BARS_PER_DAY",
]