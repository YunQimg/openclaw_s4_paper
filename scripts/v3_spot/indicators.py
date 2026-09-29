# -*- coding: utf-8 -*-
"""V3 现货多策略系统 —— 指标库（§9-§16）

时间轴约定
----------
主时间轴 = 4H K 线（UTC 00/04/08/12/16/20），决策在该 bar 收盘时做出并以该收盘价成交。
所有跨周期指标严格 point-in-time：
  * 4H 指标：仅使用截至当前 4H bar 收盘的数据；
  * 1D / 1W 指标：先聚合到日/周，再 shift(1)（只用"已收盘"的日/周），
    然后前向填充回 4H 时间轴 —— 不存在未来数据泄漏。

模块清单（§8）：
  A. MA200（4H / 1D）/ MA150 / MA250
  B. TSMOM 6M（126 交易日）
  C. TSMOM 12M（252 交易日）
  D. ATR Trend（ATR14 -> 目标波动率缩放）
  E. ADX（趋势强度）
  F. Adaptive Trend（趋势强度 -> 相对基准配置）
  G. Multi-Timeframe Trend（4H MA200 / 1D MA200 / 1W MA40）
  H. Regime Persistence（连续 N 根 4H 反向确认）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

ASSETS = ["BTC", "ETH", "SOL", "BNB"]

BARS_PER_DAY = 6          # 4H bars per day
DAYS_PER_YEAR = 365

# ---- 先验常数（不属于可调参数，写入报告便于复现）----
TSMOM_SCALE = {63: 0.30, 126: 0.50, 189: 0.65, 252: 0.80}   # 3M/6M/9M/12M 归一化尺度
ADX_LO, ADX_HI = 15.0, 35.0                                  # ADX 连续得分映射区间
ATR_ROC_BARS = 42                                            # ATR Trend 参照窗口（7 天 4H）


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def wilder_rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder 平滑（RMA）。"""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder_rma(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ADX。"""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder_rma(true_range(df), n)
    plus_di = 100.0 * wilder_rma(plus_dm, n) / tr.replace(0.0, np.nan)
    minus_di = 100.0 * wilder_rma(minus_dm, n) / tr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder_rma(dx, n)


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "open": d["open"].resample("1D").first(),
        "high": d["high"].resample("1D").max(),
        "low": d["low"].resample("1D").min(),
        "close": d["close"].resample("1D").last(),
        "volume": d["volume"].resample("1D").sum(),
    }).dropna()
    return out


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "open": d["open"].resample("W-SUN").first(),
        "high": d["high"].resample("W-SUN").max(),
        "low": d["low"].resample("W-SUN").min(),
        "close": d["close"].resample("W-SUN").last(),
        "volume": d["volume"].resample("W-SUN").sum(),
    }).dropna()
    return out


def _map_shifted_to_4h(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """低频指标 shift(1) 后前向填充到 4H 轴（严格无未来信息）。"""
    return series.shift(1).reindex(idx, method="ffill")


def tanh_norm(x: pd.Series, scale: float) -> pd.Series:
    return np.tanh(x / scale)


# ---------------------------------------------------------------------------
# 单资产指标集合
# ---------------------------------------------------------------------------
def build_asset_features(df4h: pd.DataFrame, ma_lens=(150, 200, 250),
                         tsmom_lens=(63, 126, 189, 252),
                         atr_lens=(10, 14, 20), adx_lens=(14, 20, 30),
                         adx_lo: float = ADX_LO, adx_hi: float = ADX_HI) -> pd.DataFrame:
    """为单个资产构造全部原始因子（索引 = 4H 时间轴）。

    一次性计算所有参数变体（MA150/200/250、TSMOM 3/6/9/12M、ATR 10/14/20、ADX 14/20/30），
    供 §50 参数稳定性测试直接复用，避免重复构造。
    """
    df = df4h.set_index("timestamp").sort_index()
    idx = df.index
    out = pd.DataFrame(index=idx)
    close = df["close"]

    # --- A. MA（4H）---
    for n in ma_lens:
        ma = close.rolling(n, min_periods=n).mean()
        out[f"ma{'_200' if n == 200 else '_' + str(n)}"] = ma
        out[f"ma_sign_{n}"] = np.where(close > ma, 1.0, np.where(close < ma, -1.0, 0.0))
    out["ma_4h"] = out["ma_200"] if "ma_200" in out else out[f"ma_{ma_lens[0]}"]
    out["ma200_sign"] = np.where(close > out["ma_4h"], 1.0,
                                 np.where(close < out["ma_4h"], -1.0, 0.0))
    out["ma_slope_4h"] = out["ma_4h"].pct_change(BARS_PER_DAY * 7)

    d = _to_daily(df4h)
    w = _to_weekly(df4h)

    # --- A'. MA200（1D）---
    ma_d = d["close"].rolling(200, min_periods=200).mean()
    out["ma200d_sign"] = _map_shifted_to_4h(
        pd.Series(np.where(d["close"] > ma_d, 1.0, np.where(d["close"] < ma_d, -1.0, 0.0)),
                  index=d.index), idx).fillna(0.0)

    # --- B/C. TSMOM（交易日尺度）---
    for days in tsmom_lens:
        mom = d["close"] / d["close"].shift(days) - 1.0
        score = tanh_norm(mom, TSMOM_SCALE.get(days, 0.5))
        out[f"tsmom_{days}"] = _map_shifted_to_4h(score, idx)
    out["tsmom_short"] = out["tsmom_126"]
    out["tsmom_long"] = out["tsmom_252"]

    # --- D. ATR Trend ---
    for n in atr_lens:
        a = atr(df, n)
        pct = a / close
        roc = (close - close.shift(ATR_ROC_BARS)) / (a * np.sqrt(ATR_ROC_BARS))
        out[f"atr_{n}"] = a
        out[f"atr_pct_{n}"] = pct
        out[f"realized_vol_{n}"] = pct * np.sqrt(BARS_PER_DAY * DAYS_PER_YEAR)
        out[f"atr_trend_{n}"] = (0.5 + 0.5 * np.tanh(roc)).clip(0.0, 1.0)
    out["atr"] = out["atr_14"]
    out["realized_vol"] = out["realized_vol_14"]
    out["atr_trend"] = out["atr_trend_14"]
    # 波动率的中枢（point-in-time 滚动中位数，用于相对化 ATR 缩放）
    for n in atr_lens:
        out[f"vol_median_{n}"] = out[f"realized_vol_{n}"].rolling(
            BARS_PER_DAY * 730, min_periods=BARS_PER_DAY * 180).median()
    out["vol_median"] = out["vol_median_14"]

    # --- E. ADX ---
    for n in adx_lens:
        ad = adx(df, n)
        out[f"adx_{n}"] = ad
    out["adx"] = out["adx_14"]
    out["adx_score"] = ((out["adx_14"] - adx_lo) / (adx_hi - adx_lo)).clip(0.0, 1.0)
    for n in adx_lens:
        out[f"adx_score_{n}"] = ((out[f"adx_{n}"] - adx_lo) / (adx_hi - adx_lo)).clip(0.0, 1.0)

    # --- F. Adaptive Trend（有符号趋势分 + 0~1 质量分）---
    dist = close / out["ma_4h"] - 1.0
    dist_scale = dist.rolling(BARS_PER_DAY * 180, min_periods=BARS_PER_DAY * 30).std()
    slope_scale = out["ma_slope_4h"].abs().rolling(
        BARS_PER_DAY * 180, min_periods=BARS_PER_DAY * 30).mean()
    z_dist = (dist / dist_scale.replace(0.0, np.nan)).clip(-4, 4)
    z_slope = (out["ma_slope_4h"] / slope_scale.replace(0.0, np.nan)).clip(-4, 4)
    out["adaptive"] = (0.6 * np.tanh(z_dist / 1.5) + 0.4 * np.tanh(z_slope / 1.5)).clip(-1, 1)
    out["adaptive_quality"] = ((out["adaptive"] + 1.0) / 2.0).clip(0.0, 1.0)

    # --- G. Multi-Timeframe Trend（4H MA200 / 1D MA200 / 1W MA40）---
    ma_w = w["close"].rolling(40, min_periods=40).mean()
    out["mtf_4h"] = pd.Series(np.where(close > out["ma_4h"], 1.0, -1.0), index=idx)
    out["mtf_1d"] = out["ma200d_sign"]
    out["mtf_1w"] = _map_shifted_to_4h(
        pd.Series(np.where(w["close"] > ma_w, 1.0, np.where(w["close"] < ma_w, -1.0, 0.0)),
                  index=w.index), idx).fillna(0.0)
    out["mtf"] = (out["mtf_4h"] + out["mtf_1d"] + out["mtf_1w"]) / 3.0

    out["close"] = close
    out["high"] = df["high"]
    out["low"] = df["low"]
    return out


def regime_persistence(sign: pd.Series, bars: int) -> pd.Series:
    """H. Regime Persistence（§16）

    原始 MA200 方向 sign（+1/-1）翻转后，必须连续 `bars` 根 4H 反向才确认。
    返回确认后的方向序列（确认前保持上一状态）。bars<=1 时等价于原始信号。
    """
    s = sign.fillna(0.0).to_numpy(dtype=float)
    n = len(s)
    out = np.zeros(n)
    if n == 0:
        return pd.Series(out, index=sign.index)
    state = s[0] if s[0] != 0 else -1.0
    run = 0
    bars = max(int(bars), 1)
    for i in range(n):
        v = s[i]
        if v == 0:
            out[i] = state
            continue
        if v == state:
            run = 0
        else:
            run += 1
            if run >= bars:
                state = v
                run = 0
        out[i] = state
    return pd.Series(out, index=sign.index)


def persistence_state(sign: pd.Series, bars: int) -> pd.DataFrame:
    """返回确认方向 + 当前连续反向根数（用于 Risk Modifier）。"""
    confirmed = regime_persistence(sign, bars)
    raw = sign.fillna(0.0)
    run = np.zeros(len(raw))
    cnt = 0
    prev = raw.iloc[0] if len(raw) else 0.0
    for i, v in enumerate(raw.to_numpy(dtype=float)):
        if v != prev:
            cnt = 1
            prev = v
        elif v != 0:
            cnt += 1
        run[i] = cnt
    return pd.DataFrame({"persist_confirm": confirmed,
                         "persist_run": pd.Series(run, index=raw.index)})


# ---------------------------------------------------------------------------
# 面板构建
# ---------------------------------------------------------------------------
def load_panel(data_dir, assets=ASSETS) -> Dict[str, pd.DataFrame]:
    """读取 data/v3_spot 下的 4H OHLCV，返回 {asset: df(index=timestamp)}。"""
    panel = {}
    for a in assets:
        df = pd.read_csv(data_dir / f"{a}_4h.csv", parse_dates=["timestamp"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        panel[a] = df
    return panel


def build_feature_panel(panel: Dict[str, pd.DataFrame]) -> dict:
    """对所有资产构造因子（含全部参数变体），统一到公共 4H 时间轴。"""
    idx = pd.DatetimeIndex(panel[ASSETS[0]]["timestamp"])
    for a in ASSETS[1:]:
        idx = idx.union(pd.DatetimeIndex(panel[a]["timestamp"]))
    idx = pd.DatetimeIndex(idx).sort_values()

    feats, prices = {}, {}
    for a in ASSETS:
        f = build_asset_features(panel[a]).reindex(idx)
        p = panel[a].set_index("timestamp")["close"].reindex(idx)
        prices[a] = p
        feats[a] = f

    px = pd.DataFrame(prices).reindex(idx)
    return {"index": idx, "px": px, "feats": feats, "panel": panel}


def slice_panel(fp: dict, start=None, end=None) -> dict:
    idx = fp["index"]
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
    return {"index": idx, "px": fp["px"].reindex(idx),
            "feats": {a: f.reindex(idx) for a, f in fp["feats"].items()},
            "panel": fp["panel"]}


def daily_close(feats_asset: pd.DataFrame) -> pd.Series:
    return feats_asset["close"].resample("1D").last().dropna()