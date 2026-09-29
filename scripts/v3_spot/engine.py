# -*- coding: utf-8 -*-
"""V3 现货多策略系统 —— 组合回测引擎（§22-§29, §34-§38）"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.v3_spot.indicators import ASSETS, BARS_PER_DAY, DAYS_PER_YEAR

BARS_PER_YEAR = BARS_PER_DAY * DAYS_PER_YEAR


def load_cash_rate(idx: pd.DatetimeIndex, data_dir) -> pd.Series:
    """DFF 年化利率 -> 每根 4H bar 的现金收益率（shift(1)，无未来信息）。"""
    f = data_dir / "cash_rate_dff.csv"
    if not f.exists():
        return pd.Series(0.0, index=idx)
    df = pd.read_csv(f, parse_dates=["timestamp"])
    if df.empty:
        return pd.Series(0.0, index=idx)
    s = df.set_index("timestamp")["rate_pct"].sort_index() / 100.0
    daily = s.reindex(idx.normalize().unique()).ffill().shift(1)
    daily = daily.reindex(idx.normalize())
    daily.index = idx
    r_bar = (1.0 + daily.clip(lower=0.0).fillna(0.0)) ** (1.0 / DAYS_PER_YEAR) - 1.0
    r_bar = r_bar / BARS_PER_DAY          # 日利率按 6 根 bar 均摊
    return r_bar.fillna(0.0)


def run_backtest(fp: dict, targets: pd.DataFrame, cfg, cash_rate: Optional[pd.Series] = None
                 ) -> dict:
    """事件驱动回测。

    targets : 决策 bar -> 目标权重（含 risky_total / cash）
    再平衡门槛（§25）：任一资产偏离目标 > rebal_threshold，或风险资产总权重偏离 > 阈值。
    手续费与滑点在每条交易腿上计提（§28）。
    """
    idx = fp["index"]
    px = fp["px"]
    n = len(idx)
    vals = px.to_numpy(dtype=float)
    col = {a: i for i, a in enumerate(ASSETS)}
    tgt = targets.reindex(idx).ffill()
    tgt_vals = tgt[[*ASSETS, "risky_total"]].to_numpy(dtype=float)

    if cash_rate is None:
        cash_rate = pd.Series(0.0, index=idx)
    cr = cash_rate.reindex(idx).fillna(0.0).to_numpy(dtype=float)

    c = cfg.cost_per_leg
    shares = {a: 0.0 for a in ASSETS}
    cash = float(cfg.initial)
    trades: List[dict] = []
    eq = np.zeros(n)
    w_hist = np.zeros((n, len(ASSETS)))
    cash_w = np.zeros(n)
    turnover = 0.0
    guard_scale = 1.0
    peak = float(cfg.initial)
    cur_target = np.zeros(len(ASSETS))
    dd_guard_active = np.zeros(n, dtype=bool)

    def portfolio_value(i: int) -> float:
        v = cash
        row = vals[i]
        for a in ASSETS:
            q = shares[a]
            if q:
                p = row[col[a]]
                if not np.isnan(p):
                    v += q * p
        return v

    def apply_guard(i: int, tv: np.ndarray) -> np.ndarray:
        nonlocal guard_scale
        if cfg.dd_guard <= 0:
            return tv
        pv = portfolio_value(i)
        peak_local = max(peak, pv)
        dd = pv / peak_local - 1.0 if peak_local > 0 else 0.0
        if dd <= -cfg.dd_guard:
            guard_scale = cfg.dd_guard_scale
        elif dd > -cfg.dd_guard * 0.5:
            guard_scale = 1.0
        dd_guard_active[i] = guard_scale < 1.0
        out = tv.copy()
        out[:len(ASSETS)] *= guard_scale
        out[len(ASSETS)] = float(np.nansum(out[:len(ASSETS)]))
        return out

    def rebalance(i: int, tv: np.ndarray, first: bool = False) -> None:
        nonlocal cash, turnover
        row = vals[i]
        pv = portfolio_value(i)
        if pv <= 0:
            return
        tw = {}
        for a in ASSETS:
            p = row[col[a]]
            w = tv[col[a]]
            tw[a] = 0.0 if (np.isnan(p) or np.isnan(w)) else float(w)
        # 先卖后买
        for a in ASSETS:
            p = row[col[a]]
            if np.isnan(p) or p <= 0:
                continue
            cur_val = shares[a] * p
            tgt_val = tw[a] * pv
            d = tgt_val - cur_val
            if d < -1e-9:
                q = -d / p
                shares[a] -= q
                proceeds = -d * (1.0 - c)
                cash += proceeds
                turnover += (-d) / pv
                trades.append({"date": idx[i], "asset": a,
                               "action": "buy_initial" if first else "sell",
                               "qty": q, "price": p, "value": -d, "cost": (-d) * c})
        for a in ASSETS:
            p = row[col[a]]
            if np.isnan(p) or p <= 0:
                continue
            cur_val = shares[a] * p
            tgt_val = tw[a] * pv
            d = tgt_val - cur_val
            if d > 1e-9:
                need = d / (1.0 - c)
                if need > cash:
                    need = max(cash, 0.0)
                if need <= 0:
                    continue
                buy_val = need * (1.0 - c)
                q = buy_val / p
                shares[a] += q
                cash -= need
                turnover += buy_val / pv
                trades.append({"date": idx[i], "asset": a, "action": "buy",
                               "qty": q, "price": p, "value": buy_val, "cost": need * c})

    # 初始建仓
    tv0 = tgt_vals[0].copy()
    if np.isnan(tv0).any():
        tv0 = np.nan_to_num(tv0, nan=0.0)
    rebalance(0, tv0, first=True)
    cur_target = tv0

    for i in range(n):
        if i > 0:
            if cr[i]:
                cash *= (1.0 + cr[i]) if cfg.cash_return == "B" else 1.0
            tv = tgt_vals[i]
            if np.isnan(tv).any():
                tv = np.where(np.isnan(tv), cur_target, tv)
            pv = portfolio_value(i)
            row = vals[i]
            cur_w = np.zeros(len(ASSETS))
            for a in ASSETS:
                p = row[col[a]]
                if not np.isnan(p) and pv > 0:
                    cur_w[col[a]] = shares[a] * p / pv
            drift = np.nanmax(np.abs(cur_w - tv[:len(ASSETS)])) if len(ASSETS) else 0.0
            risky_cur = float(np.nansum(cur_w))
            risky_tgt = float(np.nansum(tv[:len(ASSETS)]))
            need = (drift > cfg.rebal_threshold) or \
                   (abs(risky_cur - risky_tgt) > cfg.rebal_threshold)
            if need:
                tv_g = apply_guard(i, tv)
                rebalance(i, tv_g)
                cur_target = tv_g
        pv = portfolio_value(i)
        peak = max(peak, pv)
        eq[i] = pv
        row = vals[i]
        for a in ASSETS:
            p = row[col[a]]
            if not np.isnan(p) and pv > 0:
                w_hist[i, col[a]] = shares[a] * p / pv
        cash_w[i] = cash / pv if pv > 0 else 0.0

    eqs = pd.Series(eq, index=idx)
    years = max((idx[-1] - idx[0]).days, 1) / 365.25
    exp = pd.DataFrame(w_hist, index=idx, columns=ASSETS)
    exp["cash"] = cash_w
    exp["risky_total"] = exp[ASSETS].sum(axis=1)
    return {
        "equity": eqs,
        "weights": exp,
        "trades": pd.DataFrame(trades),
        "turnover_annual": float(turnover / years) if years > 0 else 0.0,
        "turnover_total": float(turnover),
        "dd_guard_active": pd.Series(dd_guard_active, index=idx),
    }


def benchmark_buy_hold(fp: dict, weights: Dict[str, float], cfg,
                       cash_rate: Optional[pd.Series] = None) -> dict:
    """四资产静态组合 Buy & Hold（同一成本模型，§29）。"""
    bars = fp["index"]
    tgt = pd.DataFrame(
        [{**{a: (0.0 if np.isnan(fp["px"].at[t, a]) else weights.get(a, 0.0)) for a in ASSETS}}
         for t in bars], index=bars)
    tgt["risky_total"] = tgt[ASSETS].sum(axis=1)
    eq = _static_equity(fp, weights, cfg, cash_rate)
    return {"equity": eq, "weights": tgt,
            "trades": pd.DataFrame(), "turnover_annual": 0.0, "turnover_total": 0.0}


def _period_labels(idx: pd.DatetimeIndex, freq: str) -> list:
    """日历周期标签（用于判定再平衡边界）。

    freq: "M"=每月 / "Q"=每季 / "2Q"=每半年 / "Y"=每年。
    显式按日历年月切分，避免 pandas to_period 对倍数频率（如 2Q）的静默归一化。
    """
    span = {"M": 1, "Q": 3, "2Q": 6, "Y": 12}[freq]
    return [(t.year, (t.month - 1) // span) for t in idx]


def benchmark_calendar(fp: dict, weights: Dict[str, float], cfg, freq: str = "M",
                       cash_rate: Optional[pd.Series] = None) -> dict:
    """四资产静态权重 Buy & Hold + 日历再平衡（§4 补充基准）。

    freq: "M"=每月 / "Q"=每季 / "2Q"=每半年。
    每个周期的首根 bar 把组合拉回目标权重（先卖后买，逐腿计提成本）；
    未上市资产价格缺失时跳过该腿，其权重以 Cash 形式等待上市（§4）。
    """
    idx = fp["index"]
    vals = fp["px"].to_numpy(dtype=float)
    n = len(idx)
    c = cfg.cost_per_leg
    col = {a: i for i, a in enumerate(ASSETS)}
    shares = {a: 0.0 for a in ASSETS}
    cash = float(cfg.initial)
    cr = (cash_rate.reindex(idx).fillna(0.0).to_numpy(dtype=float)
          if cash_rate is not None else np.zeros(n))
    eq = np.zeros(n)
    trades: List[dict] = []
    turnover = 0.0

    def pv_at(i: int) -> float:
        v = cash
        row = vals[i]
        for a in ASSETS:
            p = row[col[a]]
            if not np.isnan(p):
                v += shares[a] * p
        return v

    labels = _period_labels(idx, freq)
    prev = None
    for i in range(n):
        if i > 0 and cfg.cash_return == "B":
            cash *= (1.0 + cr[i])
        if prev is None or labels[i] != prev:
            pv = pv_at(i)
            row = vals[i]
            for a in ASSETS:                      # 先卖
                p = row[col[a]]
                if np.isnan(p) or p <= 0:
                    continue
                d = weights.get(a, 0.0) * pv - shares[a] * p
                if d < -1e-9:
                    q = -d / p
                    shares[a] -= q
                    cash += (-d) * (1.0 - c)
                    turnover += (-d) / pv
                    trades.append({"date": idx[i], "asset": a, "action": "sell",
                                   "qty": q, "price": p, "value": -d, "cost": (-d) * c})
            for a in ASSETS:                      # 后买
                p = row[col[a]]
                if np.isnan(p) or p <= 0:
                    continue
                d = weights.get(a, 0.0) * pv - shares[a] * p
                if d > 1e-9:
                    need = min(d / (1.0 - c), max(cash, 0.0))
                    if need <= 0:
                        continue
                    buy_val = need * (1.0 - c)
                    shares[a] += buy_val / p
                    cash -= need
                    turnover += buy_val / pv
                    trades.append({"date": idx[i], "asset": a, "action": "buy",
                                   "qty": buy_val / p, "price": p,
                                   "value": buy_val, "cost": need * c})
        prev = labels[i]
        eq[i] = pv_at(i)

    tgt = pd.DataFrame(
        [{**{a: (0.0 if np.isnan(fp["px"].at[t, a]) else weights.get(a, 0.0))
            for a in ASSETS}} for t in idx], index=idx)
    tgt["risky_total"] = tgt[ASSETS].sum(axis=1)
    years = max((idx[-1] - idx[0]).days, 1) / 365.25
    return {"equity": pd.Series(eq, index=idx), "weights": tgt,
            "trades": pd.DataFrame(trades), "rebal_freq": freq,
            "turnover_annual": float(turnover / years), "turnover_total": float(turnover)}


def _static_equity(fp: dict, weights: Dict[str, float], cfg,
                   cash_rate: Optional[pd.Series] = None) -> pd.Series:
    """静态权重 Buy & Hold：目标权重在资产可交易的首根 bar 一次性建仓（含成本），
    此后永不再平衡（未上市资产的权重以 Cash 形式等待其上市）。"""
    px = fp["px"]
    vals = px.to_numpy(dtype=float)
    n = len(vals)
    c = cfg.cost_per_leg
    shares = {a: 0.0 for a in ASSETS}
    cash = float(cfg.initial)
    bond = {a: False for a in ASSETS}
    cr = (cash_rate.reindex(fp["index"]).fillna(0.0).to_numpy(dtype=float)
          if cash_rate is not None else np.zeros(len(vals)))
    eq = np.zeros(n)
    for i in range(n):
        if i > 0 and cfg.cash_return == "B":
            cash *= (1.0 + cr[i])
        for a in ASSETS:
            j = ASSETS.index(a)
            p = vals[i][j]
            if not bond[a] and not np.isnan(p) and p > 0:
                cur_val = cash + sum(shares[k] * vals[i][ASSETS.index(k)]
                                     for k in ASSETS
                                     if not np.isnan(vals[i][ASSETS.index(k)]))
                spend = min(weights.get(a, 0.0) * cur_val, cash)
                if spend > 0:
                    shares[a] = spend * (1 - c) / p
                    cash -= spend
                bond[a] = True
        v = cash
        for a in ASSETS:
            p = vals[i][ASSETS.index(a)]
            if not np.isnan(p):
                v += shares[a] * p
        eq[i] = v
    return pd.Series(eq, index=fp["index"])