# -*- coding: utf-8 -*-
"""V4 —— 绩效、交易、暴露与相对基准指标（Roadmap §10/§11）"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.v4_spot.config import ASSETS, SEED

DAYS_PER_YEAR = 365
ANN = np.sqrt(DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def to_daily(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().dropna()


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return 0.0
    return float(drawdown(eq).min())


def ulcer_index(eq: pd.Series) -> float:
    dd = drawdown(to_daily(eq))
    return float(np.sqrt((dd ** 2).mean())) if len(dd) else 0.0


def expected_shortfall(eq: pd.Series, q: float = 0.05) -> float:
    r = to_daily(eq).pct_change().dropna()
    if r.empty:
        return 0.0
    var = np.quantile(r, q)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def drawdown_episodes(eq: pd.Series, top: int = 5) -> List[dict]:
    e = to_daily(eq)
    dd = drawdown(e)
    eps, in_dd, start, trough_t, trough_v = [], False, None, None, 0.0
    for t, v in dd.items():
        if v < -1e-9 and not in_dd:
            in_dd, start, trough_t, trough_v = True, t, t, v
        elif in_dd:
            if v < trough_v:
                trough_t, trough_v = t, v
            if v >= -1e-9:
                eps.append({"start": start, "trough": trough_t, "recovered": t,
                            "depth": float(trough_v),
                            "days_to_recovery": (t - trough_t).days,
                            "days_total": (t - start).days})
                in_dd = False
    if in_dd:
        eps.append({"start": start, "trough": trough_t, "recovered": None,
                    "depth": float(trough_v), "days_to_recovery": None,
                    "days_total": (dd.index[-1] - start).days})
    return sorted(eps, key=lambda d: d["depth"])[:top]


def longest_dd_duration(eq: pd.Series) -> int:
    eps = drawdown_episodes(eq, top=999)
    return int(max((e["days_total"] for e in eps), default=0))


def summarize_core(eq: pd.Series) -> dict:
    """§10 收益 + 风险指标（日频口径）。"""
    e = to_daily(eq)
    if len(e) < 3:
        return {}
    years = max((e.index[-1] - e.index[0]).days, 1) / 365.25
    r = e.pct_change().dropna()
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1.0 if e.iloc[0] > 0 else 0.0
    vol = float(r.std() * ANN) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std() * ANN) if r.std() > 1e-12 else 0.0
    down = r[r < 0]
    ddev = float(down.std() * ANN) if len(down) > 1 else 0.0
    sortino = float(r.mean() * DAYS_PER_YEAR / ddev) if ddev > 1e-12 else 0.0
    mdd = max_drawdown(e)
    monthly = e.resample("ME").last().pct_change().dropna()
    weekly = e.resample("W").last().pct_change().dropna()
    yearly = {str(y): float(g.iloc[-1] / g.iloc[0] - 1.0) for y, g in e.groupby(e.index.year)}
    eps = drawdown_episodes(e, top=1)
    return {
        "years": years,
        "total_return": float(e.iloc[-1] / e.iloc[0] - 1.0),
        "cagr": float(cagr),
        "annual_return_arithmetic": float(r.mean() * DAYS_PER_YEAR),
        "final_wealth": float(e.iloc[-1]),
        "best_year": max(yearly.values()) if yearly else 0.0,
        "worst_year": min(yearly.values()) if yearly else 0.0,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0,
        "max_dd": mdd,
        "ulcer_index": ulcer_index(e),
        "expected_shortfall_95": expected_shortfall(e, 0.05),
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "worst_week": float(weekly.min()) if len(weekly) else 0.0,
        "longest_dd_days": longest_dd_duration(e),
        "dd_recovery_days": (eps[0]["days_to_recovery"] if eps and
                             eps[0]["days_to_recovery"] is not None else np.nan),
    }


def exposure_stats(weights: pd.DataFrame) -> dict:
    """§10 暴露指标。"""
    rt = weights["risky_total"].dropna()
    if rt.empty:
        return {}
    return {
        "avg_risky_exposure": float(rt.mean()),
        "median_risky_exposure": float(rt.median()),
        "tim_gt_25": float((rt > 0.25).mean()),
        "tim_gt_50": float((rt > 0.50).mean()),
        "tim_gt_75": float((rt > 0.75).mean()),
        "avg_cash": float(weights["cash"].mean()),
        "max_cash": float(weights["cash"].max()),
    }


def trade_stats(res: dict) -> dict:
    """§10 交易/成本指标。"""
    tr = res.get("trades")
    n = int(res.get("n_trades", 0))
    notional = float(tr["notional"].sum()) if n and "notional" in tr else 0.0
    cost = float(tr["cost_total"].sum()) if n and "cost_total" in tr else 0.0
    return {
        "turnover_total": float(res.get("turnover_total", 0.0)),
        "turnover_annualized": float(res.get("turnover_annual", 0.0)),
        "trade_legs": n,
        "rebalance_count": int(res.get("n_rebalances", 0)),
        "avg_trade_size": float(notional / n) if n else 0.0,
        "cost_total": cost,
        "cost_per_leg_avg": float(cost / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# 相对基准
# ---------------------------------------------------------------------------
def classify_phases(bench_eq: pd.Series, dd_bear: float = -0.20) -> pd.Series:
    """§11.1 市场阶段（基于 B0 自身的 1D MA200 与回撤）。"""
    e = to_daily(bench_eq)
    ma = e.rolling(200, min_periods=60).mean()
    dd = drawdown(e)
    phase = pd.Series("Sideways", index=e.index)
    phase[(e <= ma) & (dd <= dd_bear)] = "Bear"
    phase[(e > ma)] = "Bull"
    phase[(e > ma) & (dd <= dd_bear)] = "Recovery"
    return phase


def capture_ratios(phase: pd.Series, bench_eq: pd.Series, strat_eq: pd.Series) -> dict:
    """§10/§11 Upside / Downside Capture（阶段口径 + 日频口径）。"""
    b, s = to_daily(bench_eq), to_daily(strat_eq)
    out: Dict[str, float] = {}
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
    out["upside_capture_daily"] = float(rs[up].mean() / rb[up].mean()) if up.sum() > 2 else np.nan
    out["downside_capture_daily"] = float(rs[dn].mean() / rb[dn].mean()) if dn.sum() > 2 else np.nan
    return out


def active_stats(bench_eq: pd.Series, strat_eq: pd.Series) -> dict:
    """§10 相对基准：主动收益 / 跟踪误差 / 信息比率 / 跑输概率 / 机会成本。"""
    b, s = to_daily(bench_eq), to_daily(strat_eq)
    b, s = b.align(s, join="inner")
    if len(b) < 3:
        return {}
    rb, rs = b.pct_change().dropna(), s.pct_change().dropna()
    rb, rs = rb.align(rs, join="inner")
    active = rs - rb
    te = float(active.std() * ANN)
    cum_active = float((1 + active).prod() - 1.0)
    # 机会成本：基准上涨而策略未同步跟涨的累计差额（仅统计基准为正的时段）
    up = rb > 0
    opp_cost = float((rb[up] - rs[up]).clip(lower=0).sum()) if up.any() else 0.0
    return {
        "active_return_total": cum_active,
        "active_return_annual": float(active.mean() * DAYS_PER_YEAR),
        "tracking_error": te,
        "information_ratio": float(active.mean() * DAYS_PER_YEAR / te) if te > 1e-12 else 0.0,
        "prob_underperformance": float((active < 0).mean()),
        "opportunity_cost": opp_cost,
    }


def behavior_stats(res: dict, bench_eq: pd.Series) -> dict:
    """§10 行为指标：退出到底部距离 / 再入场延迟 / 错过上涨 / 集中度。"""
    w = res["weights"]
    tr = res.get("trades", pd.DataFrame())
    eps = drawdown_episodes(bench_eq, top=3)
    out = {
        "exit_to_bottom_distance": np.nan,
        "re_entry_delay_days": np.nan,
        "asset_concentration_hhi": float(((w[ASSETS].mean()) ** 2).sum()),
        "max_asset_weight": float(w[ASSETS].max().max()),
    }
    if not tr.empty:
        sells = tr[tr["side"] == "sell"]
        buys = tr[tr["side"] == "buy"]
        if not sells.empty and eps:
            t0 = pd.Timestamp(eps[0]["start"]).tz_localize(sells["execution_time"].dt.tz) \
                if pd.Timestamp(eps[0]["start"]).tz is None else pd.Timestamp(eps[0]["start"])
            after = sells[sells["execution_time"] >= t0]
            if not after.empty:
                out["exit_to_bottom_distance"] = int(
                    (pd.Timestamp(eps[0]["trough"]) - after["execution_time"].min()).days)
        if out.get("exit_to_bottom_distance") == out.get("exit_to_bottom_distance") and eps:
            tr_t = pd.Timestamp(eps[0]["trough"])
            if tr_t.tz is None:
                tr_t = tr_t.tz_localize("UTC")
            post = buys[buys["execution_time"] >= tr_t]
            if not post.empty:
                out["re_entry_delay_days"] = int((post["execution_time"].min() - tr_t).days)
    return out


def cost_analysis(res: dict) -> dict:
    """§6 必须额外报告的：gross return / transaction cost / net return
    / cost as percentage of profit。"""
    eq = to_daily(res["equity"])
    tr = res.get("trades", pd.DataFrame())
    cost = float(tr["cost_total"].sum()) if len(tr) else 0.0
    init = float(eq.iloc[0]) if len(eq) else 0.0
    final = float(eq.iloc[-1]) if len(eq) else 0.0
    net_profit = final - init
    gross_profit = net_profit + cost
    return {
        "cost_currency_total": cost,
        "cost_fee_total": float(tr["fee"].sum()) if len(tr) else 0.0,
        "cost_slippage_total": float(tr["slippage"].sum()) if len(tr) else 0.0,
        "gross_return": float(gross_profit / init) if init > 0 else 0.0,
        "net_return": float(net_profit / init) if init > 0 else 0.0,
        "cost_pct_of_gross_profit": float(cost / gross_profit) if gross_profit > 0 else 0.0,
    }


def metrics_row(label: str, kind: str, res: dict,
                bench_eq: Optional[pd.Series] = None,
                phase: Optional[pd.Series] = None) -> dict:
    """把单次运行汇总为 §10 的一整行。"""
    row = {"label": label, "kind": kind, "spec": res.get("spec", ""),
           "cfg_tag": res.get("cfg_tag", "")}
    row.update(summarize_core(res["equity"]))
    row.update(exposure_stats(res["weights"]))
    row.update(trade_stats(res))
    row.update(cost_analysis(res))
    if bench_eq is not None:
        row.update(active_stats(bench_eq, res["equity"]))
    if phase is not None and bench_eq is not None:
        row.update(capture_ratios(phase, bench_eq, res["equity"]))
    row.update(behavior_stats(res, bench_eq if bench_eq is not None else res["equity"]))
    return row


# ---------------------------------------------------------------------------
# Bootstrap / Monte Carlo（§11.3）
# ---------------------------------------------------------------------------
def _boot_picks(n: int, block: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """分块生成 stationary block bootstrap 的重抽样索引（n_iter, n）。"""
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_iter, k))
    off = np.arange(block)[None, None, :]
    idx = (starts[:, :, None] + off).reshape(n_iter, -1) % n
    return idx[:, :n]


def _boot_stats(r: np.ndarray, picks: np.ndarray) -> tuple:
    """返回 (final_cum, maxdd) 数组。"""
    lr = np.log1p(r[picks])
    cs = np.cumsum(lr, axis=1)
    cum = np.exp(cs[:, -1])
    eq = np.exp(cs)
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    return cum, mdd


def block_bootstrap(strat: pd.Series, bench: pd.Series, n_iter: int = 10_000,
                    block: int = 20, seed: int = SEED, chunk: int = 500) -> dict:
    """按收益块重抽样的 Bootstrap：策略跑赢/跑输基准的概率与 CAGR/MaxDD 分位。

    以基准的日频收益为共同抽样路径，比较策略与基准在同一重抽样下的终值。
    """
    s, b = to_daily(strat), to_daily(bench)
    s, b = s.align(b, join="inner")
    if len(s) < block * 3:
        return {}
    rs = s.pct_change().dropna().to_numpy()
    rb = b.pct_change().dropna().to_numpy()
    m = min(len(rs), len(rb))
    rs, rb = rs[:m], rb[:m]
    rng = np.random.default_rng(seed)
    years = m / DAYS_PER_YEAR
    wins = 0
    sc, sm, bc, bm = [], [], [], []
    done = 0
    while done < n_iter:
        k = min(chunk, n_iter - done)
        picks = _boot_picks(m, block, k, rng)
        cs, ms = _boot_stats(rs, picks)
        cb, mb = _boot_stats(rb, picks)
        sc.append(cs); bc.append(cb); sm.append(ms); bm.append(mb)
        wins += int((cs > cb).sum())
        done += k
    sc = np.concatenate(sc); bc = np.concatenate(bc)
    sm = np.concatenate(sm); bm = np.concatenate(bm)
    cagr_s = sc ** (1 / years) - 1.0
    cagr_b = bc ** (1 / years) - 1.0
    return {
        "n_iter": n_iter, "block": block, "seed": seed,
        "prob_strategy_wins": wins / n_iter,
        "prob_strategy_loses": 1.0 - wins / n_iter,
        "cagr_p5": float(np.quantile(cagr_s, 0.05)), "cagr_p50": float(np.quantile(cagr_s, 0.50)),
        "cagr_p95": float(np.quantile(cagr_s, 0.95)),
        "maxdd_p5": float(np.quantile(sm, 0.05)), "maxdd_p50": float(np.quantile(sm, 0.50)),
        "maxdd_p95": float(np.quantile(sm, 0.95)),
        "bench_cagr_p5": float(np.quantile(cagr_b, 0.05)),
        "bench_cagr_p50": float(np.quantile(cagr_b, 0.50)),
        "bench_cagr_p95": float(np.quantile(cagr_b, 0.95)),
        "bench_maxdd_p50": float(np.quantile(bm, 0.50)),
    }


def monte_carlo(strat: pd.Series, n_iter: int = 10_000, block: int = 20,
                seed: int = SEED, chunk: int = 500) -> dict:
    """单序列块 Bootstrap 的 Monte Carlo 分布（§11.3 图形与分位）。"""
    s = to_daily(strat)
    if len(s) < block * 3:
        return {}
    r = s.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed + 1)
    n = len(r)
    years = n / DAYS_PER_YEAR
    cagrs, mdds, finals = [], [], []
    done = 0
    while done < n_iter:
        k = min(chunk, n_iter - done)
        picks = _boot_picks(n, block, k, rng)
        cum, mdd = _boot_stats(r, picks)
        cagrs.append(cum ** (1 / years) - 1.0)
        mdds.append(mdd); finals.append(cum)
        done += k
    cagrs = np.concatenate(cagrs); mdds = np.concatenate(mdds); finals = np.concatenate(finals)
    return {"cagr_mean": float(np.mean(cagrs)), "cagr_p5": float(np.quantile(cagrs, 0.05)),
            "cagr_p50": float(np.quantile(cagrs, 0.50)), "cagr_p95": float(np.quantile(cagrs, 0.95)),
            "maxdd_mean": float(np.mean(mdds)), "maxdd_p5": float(np.quantile(mdds, 0.05)),
            "maxdd_p50": float(np.quantile(mdds, 0.50)), "maxdd_p95": float(np.quantile(mdds, 0.95)),
            "final_p5": float(np.quantile(finals, 0.05)),
            "final_p50": float(np.quantile(finals, 0.50)),
            "final_p95": float(np.quantile(finals, 0.95)),
            "n_iter": n_iter, "block": block}


def rolling_metrics(eq: pd.Series, window_days: int) -> pd.DataFrame:
    e = to_daily(eq)
    r = e.pct_change()
    n = int(window_days)
    if len(e) <= n + 2:
        return pd.DataFrame()
    yrs = n / 365.25
    cagr = (e / e.shift(n)) ** (1 / yrs) - 1.0
    sharpe = (r.rolling(n).mean() / r.rolling(n).std()) * ANN
    mdd = e.rolling(n).apply(lambda x: float((x / np.maximum.accumulate(x) - 1).min()), raw=True)
    return pd.DataFrame({"cagr": cagr, "sharpe": sharpe, "max_dd": mdd}).dropna()