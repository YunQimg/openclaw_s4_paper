# -*- coding: utf-8 -*-
"""共享测试夹具（合成数据构造）。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from openclaw_s4_paper.config import ASSETS, StrategyConfig

BARS_PER_DAY = 6
UTC = timezone.utc


def make_bars(start: str = "2022-01-01", days: int = 400,
              prices: Optional[Dict[str, float]] = None,
              drift: float = 0.0) -> pd.DatetimeIndex:
    t0 = pd.Timestamp(start, tz="UTC")
    return pd.DatetimeIndex([t0 + timedelta(hours=4 * i)
                             for i in range(days * BARS_PER_DAY)])


def synth_ohlcv(index: pd.DatetimeIndex, start_price: float, daily_drift: float = 0.0,
                start_offset_days: int = 0) -> pd.DataFrame:
    """构造几何随机游走 OHLCV（确定性，无 random 依赖）。"""
    n = len(index)
    step = daily_drift / BARS_PER_DAY
    # 用确定性伪随机（基于 sin）避免依赖 random 模块
    noise = np.sin(np.arange(n) * 0.7) * 0.004
    close = start_price * np.cumprod(1.0 + step + noise)
    high = close * (1.0 + 0.003)
    low = close * (1.0 - 0.003)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close]) * 1.0
    low = np.minimum.reduce([low, open_, close]) * 1.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": 1000.0,
    }, index=index)
    if start_offset_days:
        df.iloc[:start_offset_days * BARS_PER_DAY] = np.nan
    return df


def write_market_csv(dirpath: Path, start: str = "2022-01-01", days: int = 400,
                     price_map: Optional[Dict[str, float]] = None,
                     drifts: Optional[Dict[str, float]] = None,
                     offsets: Optional[Dict[str, int]] = None) -> Dict[str, Path]:
    """写出 4h.csv / 1d.csv / cash_rate_dff.csv 三个文件。"""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    price_map = price_map or {"BTC": 40000.0, "ETH": 2500.0, "SOL": 100.0, "BNB": 400.0}
    drifts = drifts or {a: 0.001 for a in ASSETS}
    offsets = offsets or {}

    index = make_bars(start, days)
    rows4h: List[pd.DataFrame] = []
    rows1d: List[pd.DataFrame] = []
    for a in ASSETS:
        df = synth_ohlcv(index, price_map[a], drifts.get(a, 0.0),
                         offsets.get(a, 0))
        d = df.reset_index().rename(columns={"index": "timestamp"})
        d["asset"] = a
        rows4h.append(d[["timestamp", "asset", "open", "high", "low", "close", "volume"]])

        daily = df["close"].resample("1D").last().dropna()
        dd = daily.reset_index().rename(columns={"index": "timestamp", "close": "close"})
        dd["asset"] = a
        for c in ("open", "high", "low"):
            dd[c] = dd["close"]
        dd["volume"] = 1000.0
        rows1d.append(dd[["timestamp", "asset", "open", "high", "low", "close", "volume"]])

    p4h = dirpath / "4h.csv"
    pd.concat(rows4h, ignore_index=True).sort_values(
        ["timestamp", "asset"]).to_csv(p4h, index=False)
    p1d = dirpath / "1d.csv"
    pd.concat(rows1d, ignore_index=True).sort_values(
        ["timestamp", "asset"]).to_csv(p1d, index=False)

    pcash = dirpath / "cash_rate_dff.csv"
    days_idx = pd.date_range(start, periods=days + 5, freq="1D", tz="UTC")
    pd.DataFrame({"timestamp": days_idx, "rate_pct": 5.0}).to_csv(pcash, index=False)
    return {"4h": p4h, "1d": p1d, "cash": pcash}


@pytest.fixture
def synth_market(tmp_path: Path) -> Dict[str, Path]:
    """默认合成市场数据：全部资产稳步上行。"""
    return write_market_csv(tmp_path / "market")


@pytest.fixture
def strategy_cfg() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.validate()
    return cfg


def run_paper(mkt: Dict[str, Path], state_dir: Path, as_of,
              *, account_cfg=None, notification_cfg=None, **kw):
    """便捷调用 `run_once`（合成数据版，默认关闭邮件发送）。"""
    from openclaw_s4_paper.config import NotificationConfig, PaperAccountConfig
    from openclaw_s4_paper.run_once import run_once

    cfg = StrategyConfig()
    cfg.validate()
    return run_once(
        market_path=mkt["4h"], daily_path=mkt["1d"], cash_path=mkt["cash"],
        state_dir=state_dir, strategy_cfg=cfg,
        account_cfg=account_cfg or PaperAccountConfig(account_id="paper-s4-r9-001"),
        notification_cfg=(notification_cfg
                          if notification_cfg is not None
                          else NotificationConfig(enabled=False)),
        as_of=as_of, **kw)