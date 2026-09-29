# -*- coding: utf-8 -*-
"""V3 现货多策略系统 —— 绩效与行为指标（§41-§54）"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.v3_spot.indicators import ASSETS

DAYS_PER_YEAR = 365
ANN = np.sqrt(DAYS_PER_YEAR)


def to_daily(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().dropna()


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return 0.0
    return float(drawdown(eq).min())


def summarize(eq: pd.Series) -> dict:
    """日频口径的收益/风险指标。"""
    e = to_daily(eq)
    if len(e) < 3:
        return {}
    years = max((e.index[-1] - e.index[0]).days, 1) / 365.25
    r = e.pct_change().dropna()
    total = float(e.iloc[-1] / e.iloc[0] - 1.0)
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1.0 if e.iloc[0] > 0 else 0.0
    vol = float(r.std() * ANN) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std() * ANN) if r.std() > 1e-12 else 0.0
    down = r[r < 0]
    ddev = float(down.std() * ANN) if len(down) > 1 else 0.0
    sortino = float(r.mean() * DAYS_PER_YEAR / ddev) if ddev > 1e-12 else 0.0
    mdd = max_drawdown(e)
    calmar = float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0
    yearly = {str(y): float(g.iloc[-1] / g.iloc[0] - 1.0)
              for y, g in e.groupby(e.index.year)}
    return {
        "total_return": total,
        "cagr": float(cagr),
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": mdd,
        "final_wealth": float(e.iloc[-1]),
        "best_year": max(yearly.values()) if yearly else 0.0,
        "worst_year": min(yearly.values()) if yearly else 0.0,
        "years": years,
        "yearly": yearly,
    }


def time_in_market(weights: pd.DataFrame) -> dict:
    """§41 Time in Market：风险资产总权重超过 25/50/75% 的时间占比。"""
    rt = weights["risky_total"].dropna()
    if rt.empty:
        return {}
    return {
        "tim_gt_75": float((rt > 0.75).mean()),
        "tim_gt_50": float((rt > 0.50).mean()),
        "tim_gt_25": float((rt > 0.25).mean()),
        "avg_risky": float(rt.mean()),
        "min_risky": float(rt.min()),
        "max_risky": float(rt.max()),
    }


def classify_phases(bench_eq: pd.Series, dd_bear: float = -0.20) -> pd.Series:
    """§47 市场阶段：基于基准组合自身的 1D MA200 与回撤状态。

    Bull      : 基准在 MA200 上方
    Recovery  : 基准在 MA200 上方但自峰值回撤 <= -20%（深度回撤后的修复期）
    Bear      : 基准在 MA200 下方且回撤 <= -20%
    Sideways  : 基准在 MA200 下方但回撤 > -20%
    """
    e = to_daily(bench_eq)
    ma = e.rolling(200, min_periods=60).mean()
    dd = drawdown(e)
    phase = pd.Series("Sideways", index=e.index)
    phase[(e <= ma) & (dd <= dd_bear)] = "Bear"
    phase[(e > ma)] = "Bull"
    phase[(e > ma) & (dd <= dd_bear)] = "Recovery"
    return phase


def regime_table(phase: pd.Series, bench_eq: pd.Series, strat_eq: pd.Series,
                 weights: pd.DataFrame) -> pd.DataFrame:
    b = to_daily(bench_eq)
    s = to_daily(strat_eq)
    rt = weights["risky_total"].resample("1D").last().reindex(b.index).ffill()
    rows = []
    for p in ["Bull", "Bear", "Sideways", "Recovery"]:
        m = phase == p
        if m.sum() < 2:
            rows.append({"phase": p, "days": int(m.sum())})
            continue
        bi, si = b[m], s[m]
        rows.append({
            "phase": p,
            "days": int(m.sum()),
            "bench_return": float(bi.iloc[-1] / bi.iloc[0] - 1.0),
            "strategy_return": float(si.iloc[-1] / si.iloc[0] - 1.0),
            "bench_maxdd": max_drawdown(bi),
            "strategy_maxdd": max_drawdown(si),
            "avg_risky": float(rt[m].mean()),
        })
    return pd.DataFrame(rows)


def capture_ratios(phase: pd.Series, bench_eq: pd.Series, strat_eq: pd.Series) -> dict:
    """§42/§43 Upside / Downside Capture。

    Upside   = 策略在 Bull 阶段累计收益 / 基准在 Bull 阶段累计收益
    Downside = 策略在 Bear 阶段累计收益 / 基准在 Bear 阶段累计收益
    同时给出日频"上涨日/下跌日"口径的捕获率。
    """
    b = to_daily(bench_eq)
    s = to_daily(strat_eq)
    out = {}
    for key, ph in [("upside_capture_bull", "Bull"), ("downside_capture_bear", "Bear")]:
        m = phase == ph
        if m.sum() > 2:
            bret = float(b[m].iloc[-1] / b[m].iloc[0] - 1.0)
            sret = float(s[m].iloc[-1] / s[m].iloc[0] - 1.0)
            out[key] = float(sret / bret) if abs(bret) > 1e-9 else np.nan
        else:
            out[key] = np.nan
    rb, rs = b.pct_change(), s.pct_change()
    up, dn = rb > 0, rb < 0
    out["upside_capture_daily"] = (float(rs[up].mean() / rb[up].mean())
                                   if up.sum() > 2 else np.nan)
    out["downside_capture_daily"] = (float(rs[dn].mean() / rb[dn].mean())
                                     if dn.sum() > 2 else np.nan)
    return out


def drawdown_episodes(eq: pd.Series, top: int = 5) -> List[dict]:
    """§44 回撤事件：深度、谷底、恢复时间。"""
    e = to_daily(eq)
    dd = drawdown(e)
    eps = []
    in_dd = False
    start = None
    for t, v in dd.items():
        if v < -1e-9 and not in_dd:
            in_dd = True
            start = t
            trough_t, trough_v = t, v
        elif in_dd:
            if v < trough_v:
                trough_t, trough_v = t, v
            if v >= -1e-9:
                rec = (t - trough_t).days
                eps.append({"start": start, "trough": trough_t, "recovered": t,
                            "depth": float(trough_v),
                            "days_to_recovery": rec,
                            "days_total": (t - start).days})
                in_dd = False
    if in_dd:
        eps.append({"start": start, "trough": trough_t, "recovered": None,
                    "depth": float(trough_v), "days_to_recovery": None,
                    "days_total": (dd.index[-1] - start).days})
    eps = sorted(eps, key=lambda d: d["depth"])
    return eps[:top]


def rolling_metrics(eq: pd.Series, window_days: int) -> pd.DataFrame:
    """§46 滚动窗口：Rolling CAGR / Sharpe / MaxDD。"""
    e = to_daily(eq)
    r = e.pct_change()
    n = int(window_days)
    if len(e) <= n + 2:
        return pd.DataFrame()
    yrs = n / 365.25
    cagr = (e / e.shift(n)) ** (1 / yrs) - 1.0
    vol = r.rolling(n).std() * ANN
    sharpe = (r.rolling(n).mean() / r.rolling(n).std()) * ANN
    mdd = e.rolling(n).apply(lambda x: float((x / np.maximum.accumulate(x) - 1).min()), raw=True)
    return pd.DataFrame({"cagr": cagr, "sharpe": sharpe, "max_dd": mdd}).dropna()


def contribution_episodes(weights: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """§52 Bootstrap：把组合收益拆成"每资产每段持有期"的贡献。

    对每个资产，连续 risky 权重 > 0 的 bar 视为一个持有段（episode），
    其贡献 = Σ (w_asset × asset_return)；段间按贡献排序后可做 Top-N 剔除测试。
    """
    w = weights[ASSETS].reindex(px.index).fillna(0.0)
    rets = px[ASSETS].pct_change().fillna(0.0)
    contrib = w * rets
    rows = []
    for a in ASSETS:
        active = (w[a] > 0).to_numpy()
        c = contrib[a].to_numpy()
        i = 0
        n = len(active)
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            seg = c[i:j + 1]
            rows.append({"asset": a, "start": px.index[i], "end": px.index[j],
                         "bars": j - i + 1, "contribution": float(np.nansum(seg))})
            i = j + 1
    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)


def daily_contributions(weights: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    w = weights[ASSETS].reindex(px.index).fillna(0.0)
    rets = px[ASSETS].pct_change().fillna(0.0)
    return (w * rets)


def score_rank(s: dict) -> float:
    """§54 综合评分：25% Sharpe + 20% Sortino + 20% Calmar + 15% CAGR + 10% MaxDD
    + 5% Upside Capture + 5% Robustness（Robustness 由调用方注入，默认中性 0.5）。"""
    def nz(x, cap):
        return float(np.clip(x / cap, -1.0, 1.0))
    sharpe = nz(s.get("sharpe", 0.0), 2.0)
    sortino = nz(s.get("sortino", 0.0), 3.0)
    calmar = nz(s.get("calmar", 0.0), 1.5)
    cagr = nz(s.get("cagr", 0.0), 0.5)
    mdd = float(np.clip(1.0 - abs(s.get("max_dd", 0.0)) / 0.8, -1.0, 1.0))
    up = float(np.clip((s.get("upside_capture_bull", 0.0) or 0.0) / 1.2, 0.0, 1.0))
    rob = float(np.clip(s.get("robustness", 0.5), 0.0, 1.0))
    return float(0.25 * sharpe + 0.20 * sortino + 0.20 * calmar + 0.15 * cagr
                 + 0.10 * mdd + 0.05 * up + 0.05 * rob)