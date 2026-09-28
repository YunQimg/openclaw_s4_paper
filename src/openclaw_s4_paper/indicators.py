# -*- coding: utf-8 -*-
"""指标层：MA200（4H）与 TSMOM 6M（日线，point-in-time）（Roadmap V2 §5.1 / §5.2）。

**逐位复现 V3/V5 冻结口径**，用于 Milestone 2 的 oracle 比对：
  * MA200      = rolling_mean(close_4h, 200)，min_periods = 200
  * ma_sign    = +1 / -1 / 0
  * TSMOM 6M   = tanh(close_1d / close_1d.shift(126) - 1, scale=0.50)，
                 日线先 shift(1)（只用已收盘日线），再 ffill 到 4H 轴

时间轴纪律（§5.2）：4H bar t 上可用的 tsmom6m 只依赖 <= t-1 的日线收盘价，
因此不存在未来数据泄漏。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import ASSETS

TSMOM_SCALE_DEFAULT = 0.5


def ma(close: pd.Series, length: int = 200) -> pd.Series:
    """简单移动平均。不足 length 根 -> NaN（不用 min_periods < length 的近似）。"""
    return close.rolling(length, min_periods=length).mean()


def ma_sign(close: pd.Series, length: int = 200) -> pd.Series:
    """MA 方向符号：+1（close > ma）/ -1（close < ma）/ 0（相等或 MA 不可用）。"""
    m = ma(close, length)
    return pd.Series(
        np.where(close > m, 1.0, np.where(close < m, -1.0, 0.0)),
        index=close.index, dtype=float,
    ).where(m.notna(), 0.0)


def _to_daily(close_4h: pd.Series) -> pd.Series:
    """4H 收盘序列 -> 日线收盘序列（每日最后一根 4H 收盘）。"""
    s = close_4h.dropna()
    if s.empty:
        return s
    return s.resample("1D").last().dropna()


def tsmom_daily(close_daily: pd.Series, days: int = 126,
                scale: float = TSMOM_SCALE_DEFAULT) -> pd.Series:
    """日线动量 -> tanh 归一化分数。使用已收盘日线，不含 shift（由调用方处理）。"""
    mom = close_daily / close_daily.shift(days) - 1.0
    return np.tanh(mom / scale)


def map_daily_to_4h(series_daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """日线指标 shift(1) 后前向填充到 4H 轴（严格 point-in-time，§5.2）。

    shift(1) 保证 4H bar t 上读到的是 <= t-1 的日线值。
    """
    return series_daily.shift(1).reindex(index, method="ffill")


def build_asset_indicators(close_4h: pd.Series, close_daily: pd.Series,
                           ma_length: int = 200, tsmom_days: int = 126,
                           tanh_scale: float = TSMOM_SCALE_DEFAULT) -> pd.DataFrame:
    """单资产指标表：close / ma200 / ma_sign / tsmom6m。索引 = 4H 轴。"""
    idx = close_4h.index
    out = pd.DataFrame(index=idx)
    out["close"] = close_4h
    m = ma(close_4h, ma_length)
    out["ma200"] = m
    out["ma_sign"] = ma_sign(close_4h, ma_length)

    if close_daily is not None and not close_daily.empty:
        ts = tsmom_daily(close_daily, tsmom_days, tanh_scale)
        out["tsmom6m"] = map_daily_to_4h(ts, idx)
    else:
        out["tsmom6m"] = np.nan
    return out


def build_indicator_panel(close_4h: pd.DataFrame, close_daily: pd.DataFrame,
                          ma_length: int = 200, tsmom_days: int = 126,
                          tanh_scale: float = TSMOM_SCALE_DEFAULT,
                          ) -> Dict[str, pd.DataFrame]:
    """对四个资产构造指标表，统一到同一 4H 轴。"""
    out: Dict[str, pd.DataFrame] = {}
    for a in ASSETS:
        cd = close_daily[a] if (close_daily is not None and a in close_daily.columns) else None
        out[a] = build_asset_indicators(close_4h[a], cd, ma_length, tsmom_days, tanh_scale)
    return out


# ---------------------------------------------------------------------------
# §13.2 单指标入口（细粒度 API，供外部按需复用；与 build_* 逐位一致）
# ---------------------------------------------------------------------------
def compute_ma200_sign(close: pd.Series, length: int = 200) -> pd.DataFrame:
    """§13.2 `compute_ma200_sign`：单资产 MA200 与 ma_sign（列 `ma200` / `ma_sign`）。

    纯委托 `ma` / `ma_sign`，数值与 `build_asset_indicators` 完全一致。
    索引 = 传入 4H 收盘价序列的索引。
    """
    return pd.DataFrame({"ma200": ma(close, length),
                         "ma_sign": ma_sign(close, length)}, index=close.index)


def compute_tsmom6m(close_daily: pd.Series, days: int = 126,
                    tanh_scale: float = TSMOM_SCALE_DEFAULT) -> pd.DataFrame:
    """§13.2 `compute_tsmom6m`：单资产 TSMOM 6M（列 `tsmom6m`），日线轴。

    纯委托 `tsmom_daily`。日线 `shift(1)` + 4H 前向填充由调用方经
    `map_daily_to_4h` 处理（§5.2 point-in-time）。
    """
    return pd.DataFrame({"tsmom6m": tsmom_daily(close_daily, days, tanh_scale)},
                        index=close_daily.index)


__all__ = [
    "ma", "ma_sign", "tsmom_daily", "map_daily_to_4h",
    "build_asset_indicators", "build_indicator_panel", "TSMOM_SCALE_DEFAULT",
    "compute_ma200_sign", "compute_tsmom6m",
]