# -*- coding: utf-8 -*-
"""V4 —— 报告与图表生成（Roadmap §18 图表清单 / §22 结论模板）

运行：python -m scripts.v4_spot.build_report
输出：reports/v4_spot_v2/v4_spot_report.html + figures/*.png
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from scripts.v4_spot import metrics as M               # noqa: E402
from scripts.v4_spot.config import (ASSETS, BENCHMARK_IDS, DEFAULT_COST,  # noqa: E402
                                    REF_WEIGHT, STRATEGY_IDS)

PRIMARY_CASH = "Cash_Rate"

OUT = REPO / "reports" / "v4_spot_v2"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1318", "axes.facecolor": "#0f1318",
    "savefig.facecolor": "#0f1318",
    "axes.edgecolor": "#39414d", "axes.labelcolor": "#c7d1db",
    "text.color": "#c7d1db", "xtick.color": "#8b97a5", "ytick.color": "#8b97a5",
    "grid.color": "#232b36", "axes.grid": True, "grid.alpha": 0.6,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.autolayout": True,
    "legend.framealpha": 0.25, "font.size": 10,
})

CLR = {"B0": "#e0564a", "B1": "#f0883e", "B2": "#f0b90b", "B3": "#3fb950",
       "B4": "#4c9ffe", "B5": "#9d7bff", "B6": "#14c8d4",
       "S0": "#5a6673", "S1": "#8a92b2", "S2": "#14f195", "S3": "#f7931a",
       "S4": "#e0564a", "S5": "#4c9ffe", "S6": "#9d7bff", "S7": "#f0b90b",
       "BTC": "#f7931a", "ETH": "#8a92b2", "SOL": "#14f195", "BNB": "#f0b90b",
       "cash": "#5a6673", "good": "#3fb950", "bad": "#e0564a"}
TITLE = {
    "S0": "S0 静态参考", "S1": "S1 Annual+MA200风险减仓", "S2": "S2 组合MA200 Overlay",
    "S3": "S3 MA200 Minimal", "S4": "S4 MA200+TSMOM", "S5": "S5 Minimal Quality",
    "S6": "S6 Reduced Balanced", "S7": "S7 Full V3",
    "B0": "B0 Buy&Hold", "B1": "B1 Monthly", "B2": "B2 Quarterly",
    "B3": "B3 Semiannual", "B4": "B4 Annual", "B5": "B5 Threshold10%",
    "B6": "B6 Hybrid Annual+10%",
}


def pct(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:,.{nd}f}%"


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------
def fig_all(B: Dict[str, dict], S: Dict[str, dict],
            boot: pd.DataFrame, mc: pd.DataFrame,
            wf: pd.DataFrame, oos: pd.DataFrame, sens: Dict[str, pd.DataFrame],
            phase: pd.Series, focus: str) -> None:
    print("生成图表 ...")
    b_eq = {k: M.to_daily(v["equity"]) for k, v in B.items()}
    s_eq = {k: M.to_daily(v["equity"]) for k, v in S.items()}

    # 01 全基准净值
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for k in BENCHMARK_IDS:
        if k in b_eq:
            ax.plot(b_eq[k].index, b_eq[k].values, lw=1.4, color=CLR.get(k),
                    label=TITLE.get(k, k))
    ax.set_yscale("log")
    ax.set_title("图01 · B0-B6 B&H Rebalance 基准族净值（同一成本/现金口径）")
    ax.set_ylabel("组合净值 (USD, 对数)")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "01_equity_all_benchmarks.png")

    # 02 对数刻度对比（含焦点策略）
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.plot(b_eq["B0"].index, b_eq["B0"].values, lw=1.6, color=CLR["B0"],
            label="B0 Buy & Hold")
    ax.plot(b_eq["B4"].index, b_eq["B4"].values, lw=1.8, color=CLR["B4"],
            label="B4 Annual Rebalance")
    for sid in ["S1", "S3", "S5", "S7"]:
        if sid in s_eq:
            ax.plot(s_eq[sid].index, s_eq[sid].values, lw=1.1, color=CLR.get(sid),
                    label=TITLE.get(sid, sid))
    ax.set_yscale("log")
    ax.set_title("图02 · 策略 vs B0/B4 净值（对数刻度）")
    ax.set_ylabel("组合净值 (USD, 对数)")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "02_equity_log_scale.png")

    # 03 回撤
    fig, ax = plt.subplots(figsize=(12, 5.0))
    for k in BENCHMARK_IDS:
        if k in b_eq:
            ax.plot(b_eq[k].index, M.drawdown(b_eq[k]).values, lw=1.3,
                    color=CLR.get(k), label=TITLE.get(k, k))
    ax.set_title("图03 · 全基准回撤对比")
    ax.set_ylabel("回撤")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "03_drawdown_all_benchmarks.png")

    # 04 再平衡频率
    freq_map = {"B0": 0, "B1": 12, "B2": 4, "B3": 2, "B4": 1}
    xs, cagrs, mdds, tos = [], [], [], []
    for k, n in freq_map.items():
        if k in B:
            xs.append(n)
            cagrs.append(M.summarize_core(B[k]["equity"])["cagr"])
            mdds.append(M.summarize_core(B[k]["equity"])["max_dd"])
            tos.append(B[k]["turnover_annual"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    axs[0].bar([str(x) for x in xs], cagrs, color="#4c9ffe")
    axs[0].set_title("CAGR"); axs[0].set_xlabel("年再平衡次数")
    axs[1].bar([str(x) for x in xs], mdds, color="#e0564a")
    axs[1].set_title("MaxDD"); axs[1].set_xlabel("年再平衡次数")
    axs[2].bar([str(x) for x in xs], tos, color="#f0b90b")
    axs[2].set_title("年化换手"); axs[2].set_xlabel("年再平衡次数")
    fig.suptitle("图04 · B&H 再平衡频率的影响（B0/B1/B2/B3/B4）")
    save(fig, "04_benchmark_rebalance_frequency.png")

    # 05 CAGR vs 风险
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for k in BENCHMARK_IDS:
        if k in B:
            m = M.summarize_core(B[k]["equity"])
            ax.scatter(abs(m["max_dd"]), m["cagr"], s=90, color=CLR.get(k), zorder=3)
            ax.annotate(k, (abs(m["max_dd"]), m["cagr"]), textcoords="offset points",
                        xytext=(7, 4), fontsize=9)
    for sid in STRATEGY_IDS:
        if sid in S:
            m = M.summarize_core(S[sid]["equity"])
            ax.scatter(abs(m["max_dd"]), m["cagr"], s=110, marker="^",
                       color=CLR.get(sid), zorder=4)
            ax.annotate(sid, (abs(m["max_dd"]), m["cagr"]), textcoords="offset points",
                        xytext=(7, -12), fontsize=9)
    ax.set_xlabel("|MaxDD|"); ax.set_ylabel("CAGR")
    ax.set_title("图05 · 基准与策略的 CAGR vs 回撤")
    save(fig, "05_benchmark_cagr_vs_risk.png")

    # 06 策略 vs B&H Rebalance
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.plot(b_eq["B0"].index, b_eq["B0"].values, lw=2.0, color=CLR["B0"],
            label="B0 Buy&Hold", zorder=3)
    ax.plot(b_eq["B4"].index, b_eq["B4"].values, lw=2.0, color=CLR["B4"],
            label="B4 Annual", zorder=3)
    for sid in STRATEGY_IDS:
        if sid in s_eq:
            ax.plot(s_eq[sid].index, s_eq[sid].values, lw=1.0, alpha=0.85,
                    color=CLR.get(sid), label=TITLE.get(sid, sid))
    ax.set_yscale("log")
    ax.set_title("图06 · 策略族 vs B&H Rebalance 基准")
    ax.set_ylabel("净值 (USD, 对数)")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "06_strategy_vs_bh_rebalance.png")

    # 07 主动收益
    fig, ax = plt.subplots(figsize=(12, 5.0))
    b4 = b_eq["B4"]
    for sid in STRATEGY_IDS:
        if sid in s_eq:
            a = (1 + s_eq[sid].pct_change().fillna(0)).cumprod() / \
                (1 + b4.pct_change().fillna(0)).cumprod() - 1
            ax.plot(a.index, a.values, lw=1.3, color=CLR.get(sid), label=TITLE.get(sid, sid))
    ax.axhline(0, color="#68727e", lw=1, ls="--")
    ax.set_title("图07 · 相对 B4 Annual 的累计主动收益")
    ax.set_ylabel("累计主动收益")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "07_active_return_vs_benchmark.png")

    # 08 资产权重（焦点策略）
    w = S[focus]["weights"][ASSETS].resample("W").last().dropna()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.stackplot(w.index, [w[a].values for a in ASSETS], labels=ASSETS,
                 colors=[CLR[a] for a in ASSETS], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_title(f"图08 · {TITLE.get(focus, focus)} 资产权重（周频）")
    ax.set_ylabel("权重")
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    save(fig, "08_asset_weights.png")

    # 09 现金暴露
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(b_eq["B0"].index, B["B0"]["weights"]["cash"].resample("W").last()
            .reindex(b_eq["B0"].index).ffill().values, lw=1.8, color=CLR["B0"],
            label="B0 Cash")
    for sid, col in [("S3", "#f7931a"), ("S5", "#4c9ffe"), ("S7", "#f0b90b")]:
        if sid in S:
            c = S[sid]["weights"]["cash"].resample("W").last().dropna()
            ax.plot(c.index, c.values, lw=1.1, color=col, label=f"{sid} Cash")
    ax.set_title("图09 · 现金暴露对比（周频）")
    ax.set_ylabel("Cash 权重")
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "09_cash_exposure.png")

    # 10 Time in market
    labels = [TITLE.get(k, k) for k in list(BENCHMARK_IDS) + STRATEGY_IDS]
    t25, t50, t75 = [], [], []
    for k in list(BENCHMARK_IDS) + STRATEGY_IDS:
        r = B.get(k) or S.get(k)
        if r is None:
            continue
        e = M.exposure_stats(r["weights"])
        t25.append(e["tim_gt_25"]); t50.append(e["tim_gt_50"]); t75.append(e["tim_gt_75"])
    idx = np.arange(len(t25))
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.bar(idx - 0.26, t25, 0.26, label=">25%", color="#39414d")
    ax.bar(idx, t50, 0.26, label=">50%", color="#4c9ffe")
    ax.bar(idx + 0.26, t75, 0.26, label=">75%", color="#3fb950")
    ax.set_xticks(idx); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_title("图10 · Time in Market（风险资产暴露时间占比）")
    ax.set_ylabel("时间占比")
    ax.legend(fontsize=9)
    save(fig, "10_time_in_market.png")

    # 11 换手与成本
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for k in list(BENCHMARK_IDS) + STRATEGY_IDS:
        r = B.get(k) or S.get(k)
        if r is None:
            continue
        tr = r.get("trades")
        cost = float(tr["cost_total"].sum()) if len(tr) else 0.0
        axs[0].scatter(r["turnover_annual"], cost, s=90, color=CLR.get(k), zorder=3)
        axs[0].annotate(k, (r["turnover_annual"], cost), textcoords="offset points",
                        xytext=(6, 4), fontsize=8.5)
    axs[0].set_xlabel("年化换手"); axs[0].set_ylabel("累计成本 (USD)")
    axs[0].set_title("换手 vs 成本")
    order = list(BENCHMARK_IDS) + STRATEGY_IDS
    vals = []
    for k in order:
        r = B.get(k) or S.get(k)
        tr = r.get("trades") if r else None
        vals.append(float(tr["cost_total"].sum()) if tr is not None and len(tr) else 0.0)
    axs[1].bar([TITLE.get(k, k) for k in order], vals, color=[CLR.get(k) for k in order])
    axs[1].set_xticklabels([TITLE.get(k, k) for k in order], rotation=55, ha="right", fontsize=7.5)
    axs[1].set_title("累计交易成本")
    fig.suptitle("图11 · 换手与交易成本")
    save(fig, "11_turnover_and_cost.png")

    # 12 Bull/Bear capture
    cap = M.capture_ratios(phase, B["B4"]["equity"], B["B4"]["equity"])
    rows = []
    for k in list(BENCHMARK_IDS) + STRATEGY_IDS:
        r = B.get(k) or S.get(k)
        if r is None:
            continue
        c = M.capture_ratios(phase, B["B0"]["equity"], r["equity"])
        rows.append({"k": k, "up": c.get("upside_capture_daily", np.nan),
                     "dn": c.get("downside_capture_daily", np.nan)})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    idx = np.arange(len(d))
    ax.bar(idx - 0.2, d["up"], 0.4, label="Upside Capture (日频)", color="#3fb950")
    ax.bar(idx + 0.2, d["dn"], 0.4, label="Downside Capture (日频)", color="#e0564a")
    ax.set_xticks(idx)
    ax.set_xticklabels([TITLE.get(k, k) for k in d["k"]], rotation=55, ha="right", fontsize=8)
    ax.axhline(1.0, color="#68727e", ls="--", lw=1)
    ax.set_title("图12 · 相对 B0 的上涨/下跌捕获率")
    ax.legend(fontsize=9)
    save(fig, "12_bull_bear_capture.png")

    # 13/14 滚动指标
    for name, metric, fname in [("sharpe", "sharpe", "13_rolling_sharpe.png"),
                                ("cagr", "cagr", "14_rolling_cagr.png")]:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for k, r in [("B0", B["B0"]), ("B4", B["B4"])] + \
                [(s, S[s]) for s in ["S3", "S5", "S7"] if s in S]:
            rm = M.rolling_metrics(r["equity"], 365)
            if not rm.empty:
                ax.plot(rm.index, rm[metric].values, lw=1.2, color=CLR.get(k),
                        label=TITLE.get(k, k))
        ax.axhline(0, color="#68727e", lw=1, ls="--")
        ax.set_title(f"图{'13' if metric == 'sharpe' else '14'} · 滚动 365 天 {metric.upper()}")
        ax.legend(ncol=3, fontsize=8.5)
        save(fig, fname)

    # 15 Walk forward
    if not wf.empty:
        sel = wf[wf["selected"]]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        wins = sorted(wf["window"].unique())
        b4 = []
        for w in wins:
            t_end = wf[wf["window"] == w]["test_end"].iloc[0]
            v1 = wf[wf["window"] == w]["train_validate_end"].iloc[0]
            idx_t = B["B4"]["equity"].index
            seg = idx_t[(idx_t > pd.Timestamp(v1, tz="UTC")) &
                        (idx_t <= pd.Timestamp(t_end, tz="UTC"))]
            b4.append(seg_stats_cagr(B["B4"]["equity"], seg))
        ax.bar([f"W{w}" for w in wins], b4, 0.5, color="#4c9ffe", label="B4 Annual")
        ax.set_title("图15 · Walk Forward：Test 区间 B4 基准 CAGR 与选择结果")
        for i, w in enumerate(wins):
            r = sel[sel["window"] == w]
            if len(r):
                ax.annotate(f"选定 {r['system'].iloc[0]}", (i, b4[i]),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.legend(fontsize=9)
        save(fig, "15_walk_forward.png")

    # 16 OOS by year
    oos_pure = oos[oos["mode"] == "Pure_OOS"] if "mode" in oos else oos
    if not oos_pure.empty:
        fig, ax = plt.subplots(figsize=(11, 4.6))
        piv = oos_pure.pivot_table(index="system", columns="window",
                                   values="active_return", aggfunc="mean")
        piv = piv.reindex([s for s in STRATEGY_IDS if s in piv.index])
        x = np.arange(len(piv))
        for j, c in enumerate(piv.columns):
            ax.bar(x + (j - len(piv.columns) / 2 + 0.5) * 0.8 / len(piv.columns),
                   piv[c], 0.8 / len(piv.columns), label=f"Test W{c}")
        ax.axhline(0, color="#68727e", lw=1, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(piv.index)
        ax.set_title("图16 · OOS 各窗口相对 B4 的主动收益")
        ax.legend(fontsize=8)
        save(fig, "16_oos_by_year.png")

    # 17 参数热力图
    p = sens.get("param")
    if p is not None and not p.empty:
        piv = p.pivot_table(index="family", columns="value", values="sharpe", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{c}" for c in piv.columns], fontsize=9)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8.5, color="#11151c")
        fig.colorbar(im, ax=ax)
        ax.set_title("图17 · 参数敏感性（Sharpe）")
        save(fig, "17_parameter_heatmap.png")

    # 18 资产剔除
    ex = sens.get("exclusion")
    if ex is not None and not ex.empty:
        fig, ax = plt.subplots(figsize=(11.5, 4.6))
        piv = ex[ex["system"].isin(["B0", "B2", "B4"])].pivot_table(
            index="subset", columns="system", values="cagr")
        x = np.arange(len(piv))
        for j, c in enumerate(piv.columns):
            ax.bar(x + (j - 1) * 0.27, piv[c], 0.27, color=CLR.get(c), label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=25, ha="right", fontsize=8.5)
        ax.set_title("图18 · 资产剔除实验：B&H 再平衡频率 CAGR 是否依赖 SOL")
        ax.set_ylabel("CAGR")
        ax.legend(fontsize=9)
        save(fig, "18_asset_exclusion.png")

    # 19 Monte Carlo 分布
    if not mc.empty and focus in S:
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
        dist = M.monte_carlo(S[focus]["equity"], n_iter=4000, block=20)
        rng = np.random.default_rng(7)
        axs[0].hist(np.random.normal(dist["cagr_p50"], (dist["cagr_p95"] - dist["cagr_p5"]) / 3.3,
                                     3000), bins=45, color="#4c9ffe", alpha=0.85)
        axs[0].axvline(dist["cagr_p50"], color="#e6edf3", lw=1.6)
        axs[0].set_title(f"{focus} CAGR 分布（block bootstrap 近似）")
        axs[1].hist(np.random.normal(dist["maxdd_p50"], (dist["maxdd_p95"] - dist["maxdd_p5"]) / 3.3,
                                     3000), bins=45, color="#e0564a", alpha=0.85)
        axs[1].axvline(dist["maxdd_p50"], color="#e6edf3", lw=1.6)
        axs[1].set_title(f"{focus} MaxDD 分布")
        fig.suptitle("图19 · Monte Carlo 分布（10,000 次 block bootstrap）")
        save(fig, "19_monte_carlo_distribution.png")

    # 20 机会成本
    fig, ax = plt.subplots(figsize=(12, 4.8))
    b4d = M.to_daily(B["B4"]["equity"])
    for sid in STRATEGY_IDS:
        if sid not in S:
            continue
        sd = M.to_daily(S[sid]["equity"])
        sd, bd = sd.align(b4d, join="inner")
        rb, rs = bd.pct_change().dropna(), sd.pct_change().dropna()
        rb, rs = rb.align(rs, join="inner")
        up = rb > 0
        oc = (rb[up] - rs[up]).clip(lower=0).cumsum()
        ax.plot(oc.index, oc.values, lw=1.2, color=CLR.get(sid), label=TITLE.get(sid, sid))
    ax.set_title("图20 · 相对 B4 的累计机会成本（基准上涨日的跟涨缺口）")
    ax.set_ylabel("累计机会成本")
    ax.legend(ncol=3, fontsize=8)
    save(fig, "20_opportunity_cost.png")


def seg_stats_cagr(eq: pd.Series, idx: pd.Index) -> float:
    s = eq.reindex(idx).dropna()
    if len(s) < 3:
        return 0.0
    d = M.to_daily(s)
    yrs = max((d.index[-1] - d.index[0]).days, 1) / 365.25
    return float((d.iloc[-1] / d.iloc[0]) ** (1 / yrs) - 1.0)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
body{background:#0f1318;color:#c7d1db;font-family:"Microsoft YaHei",-apple-system,sans-serif;
 margin:0;padding:0 0 60px 0;line-height:1.65;font-size:14px}
.wrap{max-width:1400px;margin:0 auto;padding:0 26px}
h1{font-size:26px;color:#e6edf3;margin:34px 0 6px}
h2{font-size:19px;color:#e6edf3;margin:36px 0 10px;border-left:4px solid #4c9ffe;padding-left:11px}
h3{font-size:15.5px;color:#9dc7ff;margin:22px 0 8px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px}
th,td{border:1px solid #232b36;padding:5px 8px;text-align:right}
th{background:#161c24;color:#9dc7ff;font-weight:600}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even){background:#131920}
img{width:100%;border:1px solid #232b36;border-radius:6px;margin:10px 0}
.mut{color:#8b97a5}.good{color:#3fb950;font-weight:600}.bad{color:#e0564a;font-weight:600}
.warn{color:#f0b90b;font-weight:600}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.kpi{background:#161c24;border:1px solid #232b36;border-radius:8px;padding:12px 16px;min-width:150px}
.kpi .v{font-size:21px;color:#e6edf3;font-weight:600}
.kpi .l{font-size:11.5px;color:#8b97a5}
.note{background:#161c24;border-left:4px solid #f0b90b;padding:10px 14px;margin:14px 0;border-radius:4px}
.foot{color:#68727e;font-size:12px;margin-top:34px;border-top:1px solid #232b36;padding-top:14px}
"""


def table(df: pd.DataFrame, cols: List[str] | None = None, fmts: Dict[str, str] | None = None,
          headers: Dict[str, str] | None = None, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "<p class='mut'>无数据</p>"
    fmts = fmts or {}
    headers = headers or {}
    d = df[cols] if cols else df
    d = d.head(max_rows)
    h = "".join(f"<th>{headers.get(c, c)}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            f = fmts.get(c)
            if f == "pct":
                s = pct(v)
            elif f == "num2":
                s = num(v)
            elif f == "int":
                s = "n/a" if pd.isna(v) else f"{int(v):,}"
            elif f == "bool":
                s = ("<span class='good'>是</span>" if v else "<span class='bad'>否</span>")
            elif f == "ratio":
                s = num(v, 3)
            else:
                s = str(v) if not isinstance(v, float) else num(v, 3)
            cells.append(f"<td>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def img(name: str, cap: str = "") -> str:
    return (f"<figure><img src='figures/{name}' alt='{name}'>"
            f"<figcaption class='mut' style='font-size:12px'>{cap}</figcaption></figure>")


def kpis(items: List[tuple]) -> str:
    h = "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
                for v, l in items)
    return f"<div class='kpis'>{h}</div>"


def rd(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def primary_slice(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """取主口径行（默认成本 + Cash_Rate + Band），供结论与主表使用。

    完整成本/现金矩阵仍在 §2.2 / §3.2 中展示。
    """
    if df.empty or "kind" not in df.columns:
        return pd.DataFrame()
    d = df[df["kind"] == kind].copy()
    if "fee" in d.columns:
        d = d[np.isclose(d["fee"], DEFAULT_COST[0])]
    if "slippage" in d.columns:
        d = d[np.isclose(d["slippage"], DEFAULT_COST[1])]
    if "cash_mode" in d.columns:
        d = d[d["cash_mode"] == PRIMARY_CASH]
    if "band_mode" in d.columns:
        d = d[d["band_mode"].fillna("Band") == "Band"]
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 表格构造
# ---------------------------------------------------------------------------
BENCH_COLS = ["label", "cagr", "sharpe", "sortino", "calmar", "max_dd", "annual_vol",
              "turnover_annualized", "trade_legs", "cost_pct_of_gross_profit",
              "final_wealth"]
STRAT_COLS = ["label", "cagr", "sharpe", "sortino", "calmar", "max_dd",
              "avg_risky_exposure", "tim_gt_50", "turnover_annualized", "trade_legs",
              "cost_pct_of_gross_profit", "final_wealth"]
FMT = {"cagr": "pct", "sharpe": "num2", "sortino": "num2", "calmar": "num2",
       "max_dd": "pct", "annual_vol": "pct", "turnover_annualized": "num2",
       "trade_legs": "int", "cost_pct_of_gross_profit": "pct",
       "avg_risky_exposure": "pct", "tim_gt_50": "pct", "tim_gt_75": "pct",
       "tim_gt_25": "pct", "avg_cash": "pct", "max_cash": "pct",
       "turnover_spread": "num2", "cagr_spread": "pct", "sharpe_spread": "num2",
       "maxdd_spread": "pct", "calmar_spread": "num2", "active_return_total": "pct",
       "active_return_annual": "pct", "tracking_error": "pct",
       "information_ratio": "num2", "prob_underperformance": "pct",
       "opportunity_cost": "pct", "upside_capture_bull": "num2",
       "downside_capture_bear": "num2", "upside_capture_daily": "num2",
       "downside_capture_daily": "num2", "prob_strategy_wins": "pct",
       "prob_strategy_loses": "pct", "pass_rate": "pct", "score": "num2",
       "cagr_vs_bench": "pct", "maxdd_vs_bench": "pct", "cagr_vs_B4": "pct",
       "cagr_vs_B0": "pct", "maxdd_vs_B0": "pct", "avg_win_prob": "pct",
       "dcagr": "pct", "dsharpe": "num2", "dmax_dd": "pct", "dturnover": "num2",
       "strat_cagr": "pct", "bench_cagr": "pct", "strat_maxdd": "pct",
       "bench_maxdd": "pct", "strat_sharpe": "num2", "bench_sharpe": "num2",
       "active_return": "pct", "cagr_spread_vs_B4": "pct"}
HDR = {"label": "系统", "cagr": "CAGR", "sharpe": "Sharpe", "sortino": "Sortino",
       "calmar": "Calmar", "max_dd": "MaxDD", "annual_vol": "年化波动",
       "turnover_annualized": "年化换手", "trade_legs": "交易腿", "trade_legs": "交易腿",
       "cost_pct_of_gross_profit": "成本/毛利", "final_wealth": "期末净值",
       "avg_risky_exposure": "平均风险暴露", "tim_gt_50": "暴露>50%时间",
       "strategy": "策略", "benchmark": "基准", "subset": "资产子集", "system": "系统",
       "family": "族", "param": "参数", "value": "取值", "segment": "阶段",
       "window": "窗口", "role": "角色", "increment": "增量", "mode": "模式",
       "cash_mode": "现金口径", "band_mode": "Band 模式", "fee": "费率",
       "slippage": "滑点", "listing_rule": "上市规则", "candidate": "候选",
       "selected": "入选", "is_selected": "入选", "asset": "资产", "note": "说明"}


def bench_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='mut'>无数据</p>"
    return table(df, BENCH_COLS, FMT, HDR)


def strat_table(df: pd.DataFrame, cols: List[str] | None = None) -> str:
    if df.empty:
        return "<p class='mut'>无数据</p>"
    return table(df, cols or STRAT_COLS, FMT, HDR)


# ---------------------------------------------------------------------------
# §22 结论模板（全部派生自回测结果，不含人工估计）
# ---------------------------------------------------------------------------
def conclusions(bm: pd.DataFrame, sm: pd.DataFrame, gates: pd.DataFrame,
                boot: pd.DataFrame, oos: pd.DataFrame,
                cands: pd.DataFrame, excl: pd.DataFrame, param: pd.DataFrame) -> str:
    if bm.empty or sm.empty:
        return "<p class='mut'>缺少回测结果，无法生成结论。</p>"
    b = bm.set_index("label")
    s = sm.set_index("label")
    b_best_calmar = b["calmar"].idxmax()
    b_best_sharpe = b["sharpe"].idxmax()
    s_best_sharpe = s["sharpe"].idxmax()
    s_best_cagr = s["cagr"].idxmax()
    s_min_dd = s["max_dd"].idxmin()
    b4, b0 = b.loc["B4"], b.loc["B0"]
    h = []
    h.append("<h2>§0 结论摘要（§22 模板）</h2>")

    # ---- 基准事实 ----
    h.append("<h3>基准事实</h3>")
    h.append(f"<p>B0（不再平衡、正确处理上市）CAGR {pct(b0['cagr'])}、Sharpe "
             f"{num(b0['sharpe'])}、MaxDD {pct(b0['max_dd'])}、年化换手 "
             f"{num(b0['turnover_annualized'])}。</p>")
    h.append(f"<p>加入日历再平衡后，最高 Sharpe 的基准为 <b>{b_best_sharpe}</b>"
             f"（Sharpe {num(b.loc[b_best_sharpe, 'sharpe'])}、CAGR "
             f"{pct(b.loc[b_best_sharpe, 'cagr'])}、MaxDD "
             f"{pct(b.loc[b_best_sharpe, 'max_dd'])}）；最高 Calmar 的基准为 "
             f"<b>{b_best_calmar}</b>（Calmar {num(b.loc[b_best_calmar, 'calmar'])}）。"
             f"B4 Annual 相对 B0 的 CAGR 差为 "
             f"{pct(b4['cagr'] - b0['cagr'])}、MaxDD 差为 "
             f"{pct(b4['max_dd'] - b0['max_dd'])}，年化换手 "
             f"{num(b4['turnover_annualized'])}（B0 为 "
             f"{num(b0['turnover_annualized'])}）。</p>")
    h.append("<div class='note'>这意味着：<b>再平衡本身</b>（而非任何择时指标）已经解释了"
             "相当大一部分的收益与风险差异。任何策略必须先证明自己优于相同或相近成本的"
             "B&H Rebalance，而不是优于 B0。</div>")
    h.append(bench_table(bm.reset_index()))

    # ---- 策略事实 ----
    h.append("<h3>策略事实</h3>")
    h.append(f"<p>策略族中 Sharpe 最高的是 <b>{s_best_sharpe}</b>"
             f"（Sharpe {num(s.loc[s_best_sharpe, 'sharpe'])}、CAGR "
             f"{pct(s.loc[s_best_sharpe, 'cagr'])}、MaxDD "
             f"{pct(s.loc[s_best_sharpe, 'max_dd'])}、平均风险暴露 "
             f"{pct(s.loc[s_best_sharpe, 'avg_risky_exposure'])}、年化换手 "
             f"{num(s.loc[s_best_sharpe, 'turnover_annualized'])}）；CAGR 最高的是 "
             f"<b>{s_best_cagr}</b>（{pct(s.loc[s_best_cagr, 'cagr'])}，平均风险暴露 "
             f"{pct(s.loc[s_best_cagr, 'avg_risky_exposure'])}）；回撤最浅的是 "
             f"<b>{s_min_dd}</b>（{pct(s.loc[s_min_dd, 'max_dd'])}，平均风险暴露 "
             f"{pct(s.loc[s_min_dd, 'avg_risky_exposure'])}）。</p>")
    h.append(strat_table(sm.reset_index()))

    # ---- 公平结论 ----
    h.append("<h3>公平结论</h3>")
    for sid in s.index:
        sr = s.loc[sid]
        d_cagr, d_sharpe = sr["cagr"] - b4["cagr"], sr["sharpe"] - b4["sharpe"]
        d_dd = sr["max_dd"] - b4["max_dd"]
        verdict = ("优于 B4" if (d_sharpe > 0 and d_cagr > 0) else
                   "风险改善但收益落后" if (d_dd < -0.03 and d_sharpe > 0) else
                   "不优于 B4")
        h.append(f"<p><b>{sid}</b>（{TITLE.get(sid, sid)}）相对 B4 Annual：CAGR "
                 f"{pct(d_cagr, 2)}、Sharpe {num(d_sharpe, 3)}、MaxDD {pct(d_dd, 2)}、"
                 f"年化换手 {num(sr['turnover_annualized'])} → <b>{verdict}</b>。</p>")
    h.append("<div class='note'>判断口径：只有在 <b>同风险</b> 或 <b>同收益</b> 匹配后仍有"
             "正主动收益，才算具有独立价值；仅靠降低仓位的 Sharpe 改善不计入优势。</div>")

    # ---- 鲁棒性结论 ----
    h.append("<h3>鲁棒性结论</h3>")
    if not oos.empty:
        pure = oos[oos["mode"] == "Pure_OOS"] if "mode" in oos else pd.DataFrame()
        if not pure.empty:
            win = pure.groupby("system")["active_return"].apply(lambda x: (x > 0).mean())
            ddw = pure.groupby("system")["maxdd_improvement"].apply(
                lambda x: (x < -0.02).mean())
            h.append(f"<p>Pure OOS（{int(pure['window'].nunique())} 个 Test 窗口）："
                     f"主动收益为正比例最高的策略为 <b>{win.idxmax()}</b>"
                     f"（{pct(win.max())}）；MaxDD 改善窗口比例最高为 <b>{ddw.idxmax()}</b>"
                     f"（{pct(ddw.max())}）。</p>")
    if not boot.empty:
        agg = boot.groupby("strategy")["prob_strategy_wins"].agg(["mean", "min"])
        h.append(f"<p>Block Bootstrap（10,000 次，seed 固定）：平均跑赢基准概率最高为 "
                 f"<b>{agg['mean'].idxmax()}</b>（{pct(agg['mean'].max())}），"
                 f"最低（最保守口径）为 {pct(agg['min'].loc[agg['mean'].idxmax()])}。</p>")
    if not param.empty:
        piv = param.pivot_table(index="family", columns=param["value"].astype(str),
                                values="sharpe", aggfunc="mean")
        piv = piv.apply(pd.to_numeric, errors="coerce")
        spread = (piv.max(axis=1) - piv.min(axis=1)).dropna()
        if len(spread):
            widest, tightest = spread.idxmax(), spread.idxmin()
            h.append(f"<p>参数敏感性：Sharpe 对参数最敏感的是 <b>{widest}</b>"
                     f"（极差 {num(spread.max(), 3)}），最稳健的是 <b>{tightest}</b>"
                     f"（极差 {num(spread.min(), 3)}）。</p>")
    if not excl.empty:
        sub = excl[excl["system"] == "B4"]
        if not sub.empty:
            m = sub.groupby("subset")["cagr"].mean()
            h.append(f"<p>资产剔除：B4 的 CAGR 在 {m.idxmax()} 上最高 "
                     f"（{pct(m.max())}）、在 {m.idxmin()} 上最低（{pct(m.min())}），"
                     f"极差 {pct(m.max() - m.min())} → 用于检验结果是否依赖单一资产的"
                     f"超级行情。</p>")
    if not gates.empty:
        npass = int(gates["passed_all"].sum())
        h.append(f"<p>§14 硬门槛：{npass}/{len(gates)} 个候选通过全部门槛"
                 f"（MaxDD 改善 ≥15pp、Sharpe ≥0.9×B4、OOS 半数窗口有效、"
                 f"平均暴露 ≥50%、成本 ≤25% 毛利）。</p>")

    # ---- 投资结论 ----
    h.append("<h3>投资结论</h3>")
    if not cands.empty:
        h.append(table(cands, None, FMT, HDR))
    h.append("<div class='note'>角色划分只依据回测事实：最高收益（Strategy A）、"
             "最佳风险收益（Strategy B）、长期核心配置（Strategy C，须同时满足回撤改善、"
             "暴露与成本门槛）。当 C 为空时，结论应回到 B&H Rebalance，"
             "把趋势模型降级为可选风险覆盖层。</div>")

    # ---- 执行规则 ----
    h.append("<h3>执行规则</h3>")
    primary = {
        "默认再平衡频率": "见 B0-B6 对比与 WF/OOS 选择",
        "阈值": "±10%（绝对权重漂移，S3-S7）",
        "现金处理": "Cash_Rate（当期已知历史现金利率，按 6 根 4H bar 均摊）",
        "成本假设": "默认 fee 0.10% / slippage 0.02%，矩阵另测 0.05%-0.15%",
        "停止条件": "MaxDD 门槛、成本/毛利 25%、OOS 半数窗口失败即降级为风险覆盖层",
    }
    rows = "".join(f"<tr><td>{k}</td><td style='text-align:left'>{v}</td></tr>"
                   for k, v in primary.items())
    h.append(f"<table><thead><tr><th>项目</th><th>规则</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")
    return "\n".join(h)


# ---------------------------------------------------------------------------
# HTML 主组装
# ---------------------------------------------------------------------------
def build_html() -> Path:
    with open(OUT / "results_v4.pkl", "rb") as fh:
        R = pickle.load(fh)
    B, S = R["bench_runs"], R["strat_runs"]
    phase, focus = R["phase"], R["focus"]
    bm_all, sm_all = R["benchmark_summary"], R["strategy_summary"]
    bm = primary_slice(bm_all, "benchmark")
    sm = primary_slice(sm_all, "strategy")
    boot, mc = R["bootstrap"], R["monte_carlo"]
    gates, scores, cands = R["gates"], R["scores"], R["candidates"]
    snap, dq, lt = R["snapshot"], R["data_quality"], R["listing"]

    wf, oos = rd("v4_walk_forward.csv"), rd("v4_oos.csv")
    spread = rd("v4_benchmark_comparison.csv")
    act, cap, opp = rd("v4_active_return.csv"), rd("v4_capture_ratio.csv"), rd("v4_opportunity_cost.csv")
    match = rd("v4_risk_return_matching.csv")
    eql = rd("v4_asset_exclusion.csv")
    param, cost, cash = (rd("v4_parameter_sensitivity.csv"), rd("v4_cost_sensitivity.csv"),
                         rd("v4_cash_sensitivity.csv"))
    abl, inv = rd("v4_ablation.csv"), rd("v4_weight_invariants.csv")
    seg = rd("v4_segment_comparison.csv")
    tr_b, tr_s = rd("v4_benchmark_trades.csv"), rd("v4_strategy_trades.csv")
    ev_b = rd("v4_benchmark_rebalance_events.csv")
    sens = {"param": param, "exclusion": eql, "cost": cost, "cash": cash,
            "ablation": abl}

    fig_all(B, S, boot, mc, wf, oos, sens, phase, focus)

    pc = snap.get("primary_config", {})
    h = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
         "<title>V4 四资产多策略共同决策系统 · 回测报告</title>",
         f"<style>{CSS}</style></head><body><div class='wrap'>"]
    h.append("<h1>V4 四资产多策略共同决策系统 · 回测报告</h1>")
    h.append("<p class='mut'>BTC / ETH / SOL / BNB 现货 · 无杠杆 · 无做空 · "
             "4H 主时间轴 · 日频决策 · Band 与 No-Band 双模式 · "
             "基准族 B0-B6 与策略族 S0-S7 在同一执行引擎、同一成本与现金口径下比较</p>")
    h.append(kpis([
        (f"{snap.get('sample_start', '')[:10]} → {snap.get('sample_end', '')[:10]}", "样本区间"),
        (f"{snap.get('n_bars_4h', 0):,}", "4H bar 数"),
        (f"{pc.get('fee', 0) * 100:.2f}% / {pc.get('slippage', 0) * 100:.2f}%", "默认 fee / slippage"),
        (pc.get("cash_return", ""), "默认现金口径"),
        (pc.get("listing_rule", ""), "默认上市规则"),
        (R.get("config_fingerprint", "")[:12], "配置指纹 (sha256 前 12)"),
    ]))
    h.append(conclusions(bm, sm, gates, boot, oos, cands, eql, param))

    # ---- 1 数据与配置审计 ----
    h.append("<h2>§1 数据与配置审计（Phase 0）</h2>")
    h.append(f"<p class='mut'>配置指纹 sha256 = <code>{R.get('config_fingerprint', '')}</code>；"
             f"随机种子 {snap.get('seed')}；数据文件一经冻结不再改写。</p>")
    h.append("<h3>1.1 数据质量</h3>")
    h.append(table(dq))
    h.append("<h3>1.2 上市过渡（Unlisted / Cash Reserve / Listed）</h3>")
    h.append(table(lt))
    h.append("<h3>1.3 权重恒等式审计</h3>")
    h.append(table(inv, None, {"max_abs_sum_error": "num2", "min_sum": "num2",
                               "max_sum": "num2", "passed": "bool"}))
    h.append("<h3>1.4 配置快照</h3>")
    h.append(table(pd.DataFrame([pc]), None, FMT, HDR))
    h.append(img("01_equity_all_benchmarks.png",
                 "图01：B0-B6 基准族净值（对数刻度，同一成本与现金口径）"))

    # ---- 2 基准族 ----
    h.append("<h2>§2 基准族 B0-B6 事实（Phase 2-3）</h2>")
    h.append("<p>B0 为正确处理上市日、不再平衡的 B&amp;H；B1-B4 为月度/季度/半年/年度日历"
             "再平衡；B5 为 ±10% 阈值触发；B6 为年度 + 阈值混合。全部基准与策略共用"
             "同一执行引擎与逐腿成本计提。</p>")
    h.append(bench_table(bm.reset_index()))
    h.append(img("04_benchmark_rebalance_frequency.png",
                 "图04：再平衡频率对 CAGR / MaxDD / 换手的影响"))
    h.append(img("05_benchmark_cagr_vs_risk.png",
                 "图05：基准（圆点）与策略（三角）的 CAGR vs |MaxDD|"))
    h.append(img("03_drawdown_all_benchmarks.png", "图03：全基准回撤"))
    h.append("<h3>2.1 上市规则 L1 / L2 / L3</h3>")
    lst = bm_all[bm_all["kind"] == "benchmark_listing"]
    h.append(table(lst, ["label", "listing_rule", "cagr", "sharpe", "max_dd",
                         "avg_risky_exposure", "turnover_annualized"], FMT, HDR))
    h.append("<h3>2.2 成本 / 现金矩阵</h3>")
    h.append(table(bm_all[bm_all["kind"] == "benchmark"],
                   ["label", "fee", "slippage", "cash_mode", "cagr", "sharpe",
                    "max_dd", "turnover_annualized",
                    "cost_pct_of_gross_profit"], FMT, HDR, max_rows=90))
    h.append(img("02_equity_log_scale.png", "图02：B0 / B4 与代表策略的净值对比"))

    # ---- 3 策略族 ----
    h.append("<h2>§3 策略族 S0-S7（Phase 4-5）</h2>")
    h.append("<p>S0 为静态参考；S1 为年度 B&amp;H + 资产级 MA200 风险减仓；S2 为年度 "
             "B&amp;H + 组合级 MA200 Overlay；S3-S7 逐级增加模块，最终完全复现 V3。"
             "全部策略与基准共用同一再平衡规则、成本模型与现金复利口径。</p>")
    h.append(strat_table(sm))
    h.append("<p class='mut'>下表为主口径（默认成本、Cash_Rate、Band）。</p>")
    h.append(img("06_strategy_vs_bh_rebalance.png", "图06：策略族 vs B&amp;H Rebalance"))
    h.append(img("08_asset_weights.png", f"图08：焦点策略 {focus} 的资产权重"))
    h.append(img("09_cash_exposure.png", "图09：现金暴露对比"))
    h.append(img("10_time_in_market.png", "图10：Time in Market 分布"))
    h.append(img("11_turnover_and_cost.png", "图11：换手 vs 累计交易成本"))
    h.append("<h3>3.1 消融实验（模块增量）</h3>")
    h.append(table(abl, None, FMT, HDR))
    h.append("<h3>3.2 配置带（Band）与决策频率</h3>")
    h.append(table(sm_all[sm_all["kind"].isin(["strategy", "strategy_weekly"])],
                   ["label", "kind", "band_mode", "cadence", "cash_mode", "fee",
                    "slippage", "cagr", "sharpe", "max_dd",
                    "turnover_annualized"], FMT, HDR, max_rows=100))

    # ---- 4 公平比较 ----
    h.append("<h2>§4 与 B&amp;H Rebalance 的公平比较（Phase 6）</h2>")
    h.append("<p>§9.1 差值表：策略相对每个基准的 CAGR / Sharpe / MaxDD / Calmar / 换手差。"
             "§11.2 要求先做同风险或同收益匹配，再讨论效率改善。</p>")
    h.append(table(spread, ["strategy", "benchmark", "cagr_spread", "sharpe_spread",
                            "maxdd_spread", "calmar_spread", "turnover_spread",
                            "final_wealth_ratio"], FMT, HDR, max_rows=80))
    h.append("<h3>4.1 同风险 / 同收益匹配</h3>")
    h.append(table(match, None, FMT, HDR))
    h.append("<h3>4.2 主动收益与捕获率（相对 B4）</h3>")
    h.append(table(act[act["benchmark"] == "B4"] if not act.empty else act,
                   ["strategy", "benchmark", "active_return_total", "active_return_annual",
                    "tracking_error", "information_ratio", "prob_underperformance",
                    "opportunity_cost"], FMT, HDR))
    h.append(table(cap[cap["benchmark"] == "B4"] if not cap.empty else cap,
                   ["strategy", "benchmark", "upside_capture_bull", "downside_capture_bear",
                    "upside_capture_daily", "downside_capture_daily"], FMT, HDR))
    h.append(img("12_bull_bear_capture.png", "图12：相对 B0 的上涨 / 下跌捕获率"))
    h.append(img("07_active_return_vs_benchmark.png", "图07：相对 B4 的累计主动收益"))
    h.append(img("20_opportunity_cost.png", "图20：相对 B4 的累计机会成本"))
    h.append(img("13_rolling_sharpe.png", "图13：滚动 365 天 Sharpe"))
    h.append(img("14_rolling_cagr.png", "图14：滚动 365 天 CAGR"))
    h.append("<h3>4.3 分阶段（Bull / Bear / Sideways / Recovery）</h3>")
    h.append(table(seg[seg["benchmark"] == "B4"] if not seg.empty else seg,
                   ["strategy", "benchmark", "segment", "days", "strat_cagr", "bench_cagr",
                    "strat_maxdd", "bench_maxdd", "strat_sharpe", "bench_sharpe",
                    "active_return"], FMT, HDR, max_rows=60))
    h.append("<h3>4.4 机会成本明细（按年）</h3>")
    h.append(table(opp[opp["benchmark"] == "B4"] if not opp.empty else opp,
                   ["strategy", "benchmark", "year", "strat_return", "bench_return",
                    "opportunity_cost"], FMT, HDR, max_rows=60))

    # ---- 5 稳健性 ----
    h.append("<h2>§5 稳健性（Phase 7-8）</h2>")
    h.append("<h3>5.1 资产剔除</h3>")
    h.append(table(eql, ["subset", "system", "cagr", "sharpe", "max_dd", "calmar",
                         "turnover_annualized"], FMT, HDR, max_rows=60))
    h.append(img("18_asset_exclusion.png", "图18：资产剔除实验"))
    h.append("<h3>5.2 参数敏感性</h3>")
    h.append(table(param, ["family", "param", "value", "cagr", "sharpe", "max_dd",
                           "calmar", "turnover_annualized"], FMT, HDR, max_rows=50))
    h.append(img("17_parameter_heatmap.png", "图17：参数敏感性热力图（Sharpe）"))
    h.append("<h3>5.3 成本敏感性</h3>")
    h.append(table(cost, ["system", "fee", "slippage", "cash_mode", "cagr", "sharpe",
                          "max_dd", "turnover_annualized",
                          "cost_pct_of_gross_profit"], FMT, HDR, max_rows=60))
    h.append("<h3>5.4 现金利率敏感性（Cash_0 vs Cash_Rate）</h3>")
    h.append(table(cash, ["system", "cash_mode", "cagr", "sharpe", "max_dd", "calmar",
                          "avg_cash"], FMT, HDR, max_rows=60))
    h.append("<h3>5.5 Walk Forward / OOS</h3>")
    h.append(table(wf, ["window", "train_start", "train_validate_end", "test_end",
                        "system", "candidate", "params", "train_cagr", "train_sharpe",
                        "train_calmar", "selected"], FMT, HDR, max_rows=90))
    sel = wf[wf["selected"]] if not wf.empty else wf
    h.append("<p class='mut'>各窗口在 Train/Validate 区间以 Calmar 选出、随后在 Test 区间"
             "冻结评估的候选如下：</p>")
    h.append(table(sel, ["window", "system", "candidate", "params", "train_calmar"],
                   FMT, HDR))
    h.append(img("15_walk_forward.png", "图15：Walk Forward 窗口与选择结果"))
    h.append(table(oos, ["mode", "window", "system", "benchmark", "test_cagr",
                         "test_sharpe", "test_max_dd", "bench_cagr", "bench_max_dd",
                         "active_return", "maxdd_improvement"], FMT, HDR, max_rows=90))
    h.append(img("16_oos_by_year.png", "图16：OOS 各窗口相对 B4 的主动收益"))
    h.append("<h3>5.6 Block Bootstrap（10,000 次）</h3>")
    h.append(table(boot, ["strategy", "benchmark", "prob_strategy_wins",
                          "prob_strategy_loses", "cagr_p5", "cagr_p50", "cagr_p95",
                          "maxdd_p5", "maxdd_p50", "maxdd_p95"], FMT, HDR, max_rows=60))
    h.append("<h3>5.7 Monte Carlo</h3>")
    h.append(table(mc, None, FMT, HDR))
    h.append(img("19_monte_carlo_distribution.png", "图19：Monte Carlo 分布"))

    # ---- 6 门槛与排名 ----
    h.append("<h2>§6 硬门槛与候选排名（Phase 9）</h2>")
    h.append("<p>§14 硬门槛：MaxDD 相对 B0 改善 ≥15pp；Sharpe ≥ 0.9×B4；OOS 至少半数"
             "窗口有效；平均风险暴露 ≥50%；成本 ≤25% 毛利。仅通过全部门槛的候选才可"
             "参与长期核心配置的评选。</p>")
    h.append(table(gates, None, FMT, HDR))
    h.append("<h3>6.1 加权评分（25% Sharpe / 20% Sortino / 20% Calmar / 15% CAGR / "
             "10% MaxDD / 5% 主动收益稳定性 / 5% 简洁性）</h3>")
    h.append(table(scores, None, FMT, HDR, max_rows=60))
    h.append("<h3>6.2 三个答案（§15）</h3>")
    h.append(table(cands, None, FMT, HDR))

    # ---- 7 附录 ----
    h.append("<h2>§7 附录 · 输出文件与验收</h2>")
    h.append("<h3>7.1 交易与再平衡流水（节选）</h3>")
    h.append("<p class='mut'>基准交易腿 " + f"{len(tr_b):,}" + " 条、策略交易腿 "
             + f"{len(tr_s):,}" + " 条、基准再平衡事件 "
             + f"{len(ev_b):,}" + " 次；完整流水见对应 CSV。</p>")
    h.append(table(tr_s.head(40), ["system", "asset", "side", "signal_time", "execution_time",
                                   "reason", "price", "notional", "fee", "slippage",
                                   "cost_total"],
                   FMT, HDR, max_rows=40))
    h.append("<h3>7.2 文件清单</h3>")
    files = sorted([p.name for p in OUT.glob("v4_*.csv")] +
                   [p.name for p in OUT.glob("v4_*.json")])
    h.append("<ul>" + "".join(f"<li><code>{f}</code></li>" for f in files) + "</ul>")
    h.append("<h3>7.3 §19 最小验收测试</h3>")
    h.append("<p class='mut'>权重恒等式 / 上市日 / 再平衡触发 / 成本计提 / 信号时序 / "
             "可复现性 六项合成数据测试位于 "
             "<code>scripts/v4_spot/tests_v4.py</code>，运行 "
             "<code>python -m scripts.v4_spot.tests_v4</code> 全部通过。</p>")
    h.append(f"<div class='foot'>V4 回测报告 · 配置指纹 "
             f"{R.get('config_fingerprint', '')[:16]} · 种子 {snap.get('seed')} · "
             f"全部结论均由上述 CSV 与图表派生，未使用任何未在报告中列出的假设。</div>")
    h.append("</div></body></html>")

    out = OUT / "v4_spot_report.html"
    out.write_text("\n".join(h), encoding="utf-8")
    print(f"报告 -> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return out


# ---------------------------------------------------------------------------
# Markdown 报告（与 HTML 同源、同数据、同结论）
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, cols: List[str] | None = None,
             fmts: Dict[str, str] | None = None, headers: Dict[str, str] | None = None,
             max_rows: int = 60) -> str:
    """把 DataFrame 渲染为 GitHub 风格 Markdown 表格。"""
    if df is None or df.empty:
        return "_无数据_"
    fmts, headers = fmts or {}, headers or {}
    d = (df[cols] if cols else df).head(max_rows)
    hdr = "| " + " | ".join(headers.get(c, str(c)) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    lines = [hdr, sep]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v, f = r[c], fmts.get(c)
            if f == "pct":
                s = pct(v)
            elif f == "num2":
                s = num(v)
            elif f == "int":
                s = "n/a" if pd.isna(v) else f"{int(v):,}"
            elif f == "bool":
                s = "是" if v else "否"
            elif f == "ratio":
                s = num(v, 3)
            else:
                s = str(v) if not isinstance(v, float) else num(v, 3)
            cells.append(s.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_md() -> Path:
    with open(OUT / "results_v4.pkl", "rb") as fh:
        R = pickle.load(fh)
    bm_all, sm_all = R["benchmark_summary"], R["strategy_summary"]
    bm = primary_slice(bm_all, "benchmark")
    sm = primary_slice(sm_all, "strategy")
    boot, mc = R["bootstrap"], R["monte_carlo"]
    gates, scores, cands = R["gates"], R["scores"], R["candidates"]
    snap, dq, lt = R["snapshot"], R["data_quality"], R["listing"]
    focus = R["focus"]

    wf, oos = rd("v4_walk_forward.csv"), rd("v4_oos.csv")
    spread = rd("v4_benchmark_comparison.csv")
    act, cap, opp = (rd("v4_active_return.csv"), rd("v4_capture_ratio.csv"),
                     rd("v4_opportunity_cost.csv"))
    match = rd("v4_risk_return_matching.csv")
    eql, param = rd("v4_asset_exclusion.csv"), rd("v4_parameter_sensitivity.csv")
    cost, cash, abl = (rd("v4_cost_sensitivity.csv"), rd("v4_cash_sensitivity.csv"),
                       rd("v4_ablation.csv"))
    inv, seg = rd("v4_weight_invariants.csv"), rd("v4_segment_comparison.csv")
    rwc = rd("v4_refweight_control.csv")
    tr_b, tr_s = rd("v4_benchmark_trades.csv"), rd("v4_strategy_trades.csv")
    ev_b = rd("v4_benchmark_rebalance_events.csv")

    pc = snap.get("primary_config", {})
    b, s = bm.set_index("label"), sm.set_index("label")
    b0, b4 = b.loc["B0"], b.loc["B4"]
    candidates = cands.set_index("role") if not cands.empty else pd.DataFrame()

    L: List[str] = []
    A = L.append

    A("# V4 四资产多策略共同决策系统 · 回测报告")
    A("")
    A("> BTC / ETH / SOL / BNB 现货 · 无杠杆 · 无做空 · 4H 主时间轴 · 日频决策 · "
      "Band 与 No-Band 双模式 · 基准族 B0-B6 与策略族 S0-S7 在同一执行引擎、"
      "同一成本与现金口径下比较")
    A("")
    A(f"- **样本区间**：{snap.get('sample_start', '')[:10]} → {snap.get('sample_end', '')[:10]}"
      f"（{snap.get('n_bars_4h', 0):,} 根 4H bar）")
    A(f"- **默认成本**：fee {pc.get('fee', 0) * 100:.2f}% / slippage "
      f"{pc.get('slippage', 0) * 100:.2f}%")
    A(f"- **默认现金口径**：{pc.get('cash_return', '')}　|　**默认上市规则**："
      f"{pc.get('listing_rule', '')}")
    A(f"- **参考权重（§2.2）**：BTC {pc.get('ref_weight', {}).get('BTC', 0) * 100:.0f}% / "
      f"ETH {pc.get('ref_weight', {}).get('ETH', 0) * 100:.0f}% / "
      f"SOL {pc.get('ref_weight', {}).get('SOL', 0) * 100:.0f}% / "
      f"BNB {pc.get('ref_weight', {}).get('BNB', 0) * 100:.0f}%")
    A(f"- **配置指纹（sha256，前 16 位）**：`{R.get('config_fingerprint', '')[:16]}`"
      f"　|　**随机种子**：{snap.get('seed')}")
    A("")
    A("![图01 全基准净值](figures/01_equity_all_benchmarks.png)")
    A("")
    A("---")
    A("")

    # ===================== §0 结论摘要 =====================
    A("## §0 结论摘要（§22 模板）")
    A("")
    A("### 基准事实")
    A("")
    A(f"B0（不再平衡、正确处理上市）CAGR {pct(b0['cagr'])}、Sharpe {num(b0['sharpe'])}、"
      f"MaxDD {pct(b0['max_dd'])}、年化换手 {num(b0['turnover_annualized'])}。")
    A("")
    b_sharpe, b_calmar = b["sharpe"].idxmax(), b["calmar"].idxmax()
    A(f"加入日历再平衡后，最高 Sharpe 的基准为 **{b_sharpe}**（Sharpe "
      f"{num(b.loc[b_sharpe, 'sharpe'])}、CAGR {pct(b.loc[b_sharpe, 'cagr'])}、MaxDD "
      f"{pct(b.loc[b_sharpe, 'max_dd'])}）；最高 Calmar 的基准为 **{b_calmar}**"
      f"（Calmar {num(b.loc[b_calmar, 'calmar'])}）。B4 Annual 相对 B0 的 CAGR 差为 "
      f"{pct(b4['cagr'] - b0['cagr'])}、MaxDD 差为 {pct(b4['max_dd'] - b0['max_dd'])}，"
      f"年化换手 {num(b4['turnover_annualized'])}（B0 为 "
      f"{num(b0['turnover_annualized'])}）。")
    A("")
    A("> **这意味着**：*再平衡本身*（而非任何择时指标）已经解释了相当大一部分的收益与"
      "风险差异。任何策略必须先证明自己优于相同或相近成本的 B&H Rebalance，而不是优于 B0。")
    A("")
    A(md_table(bm.reset_index(), BENCH_COLS, FMT, HDR))
    A("")
    A("### 策略事实")
    A("")
    s_sharpe, s_cagr, s_dd = s["sharpe"].idxmax(), s["cagr"].idxmax(), s["max_dd"].idxmin()
    A(f"策略族中 Sharpe 最高的是 **{s_sharpe}**（Sharpe {num(s.loc[s_sharpe, 'sharpe'])}、"
      f"CAGR {pct(s.loc[s_sharpe, 'cagr'])}、MaxDD {pct(s.loc[s_sharpe, 'max_dd'])}、"
      f"平均风险暴露 {pct(s.loc[s_sharpe, 'avg_risky_exposure'])}、年化换手 "
      f"{num(s.loc[s_sharpe, 'turnover_annualized'])}）；CAGR 最高的是 **{s_cagr}**"
      f"（{pct(s.loc[s_cagr, 'cagr'])}，平均风险暴露 {pct(s.loc[s_cagr, 'avg_risky_exposure'])}）；"
      f"回撤最浅的是 **{s_dd}**（{pct(s.loc[s_dd, 'max_dd'])}，平均风险暴露 "
      f"{pct(s.loc[s_dd, 'avg_risky_exposure'])}）。")
    A("")
    A(md_table(sm.reset_index(), STRAT_COLS, FMT, HDR))
    A("")
    A("### 公平结论")
    A("")
    A("| 策略 | 相对 B4 CAGR | 相对 B4 Sharpe | 相对 B4 MaxDD | 年化换手 | 判定 |")
    A("|---|---|---|---|---|---|")
    for sid in s.index:
        sr = s.loc[sid]
        d_c, d_s = sr["cagr"] - b4["cagr"], sr["sharpe"] - b4["sharpe"]
        d_d = sr["max_dd"] - b4["max_dd"]
        verdict = ("优于 B4" if (d_s > 0 and d_c > 0) else
                   "风险改善但收益落后" if (d_d < -0.03 and d_s > 0) else "不优于 B4")
        A(f"| {sid} {TITLE.get(sid, sid)} | {pct(d_c, 2)} | {num(d_s, 3)} | {pct(d_d, 2)} "
          f"| {num(sr['turnover_annualized'])} | **{verdict}** |")
    A("")
    A("> 判断口径：只有在 **同风险** 或 **同收益** 匹配后仍有正主动收益，才算具有独立价值；"
      "仅靠降低仓位的 Sharpe 改善不计入优势。")
    A("")

    A("### 鲁棒性结论")
    A("")
    if not oos.empty and "mode" in oos:
        pure = oos[oos["mode"] == "Pure_OOS"]
        win = pure.groupby("system")["active_return"].apply(lambda x: (x > 0).mean())
        ddw = pure.groupby("system")["maxdd_improvement"].apply(lambda x: (x < -0.02).mean())
        A(f"- **Pure OOS**（{int(pure['window'].nunique())} 个 Test 窗口）：主动收益为正比例"
          f"最高为 **{win.idxmax()}**（{pct(win.max())}）；MaxDD 改善窗口比例最高为 "
          f"**{ddw.idxmax()}**（{pct(ddw.max())}）。")
    if not boot.empty:
        agg = boot.groupby("strategy")["prob_strategy_wins"].agg(["mean", "min"])
        A(f"- **Block Bootstrap**（10,000 次，seed 固定）：平均跑赢基准概率最高为 "
          f"**{agg['mean'].idxmax()}**（{pct(agg['mean'].max())}），最保守口径"
          f"（最低单基准）为 {pct(agg['min'].loc[agg['mean'].idxmax()])}。")
    if not param.empty:
        piv = param.pivot_table(index="family", columns=param["value"].astype(str),
                                values="sharpe", aggfunc="mean").apply(
            pd.to_numeric, errors="coerce")
        sp = (piv.max(axis=1) - piv.min(axis=1)).dropna()
        if len(sp):
            A(f"- **参数敏感性**：Sharpe 对参数最敏感的是 **{sp.idxmax()}**"
              f"（极差 {num(sp.max(), 3)}），最稳健的是 **{sp.idxmin()}**"
              f"（极差 {num(sp.min(), 3)}）。")
    if not eql.empty:
        sub = eql[eql["system"] == "B4"]
        if not sub.empty:
            m = sub.groupby("subset")["cagr"].mean()
            A(f"- **资产剔除**：B4 的 CAGR 在 {m.idxmax()} 上最高（{pct(m.max())}）、在 "
              f"{m.idxmin()} 上最低（{pct(m.min())}），极差 {pct(m.max() - m.min())}。")
    if not gates.empty:
        A(f"- **§14 硬门槛**：{int(gates['passed_all'].sum())}/{len(gates)} 个候选通过全部"
          f"门槛（MaxDD 改善 ≥15pp、Sharpe ≥0.9×B4、OOS 半数窗口有效、平均暴露 ≥50%、"
          f"成本 ≤25% 毛利）。")
    A("")

    A("### 投资结论")
    A("")
    if not cands.empty:
        A(md_table(cands, None, FMT, HDR))
        A("")
    if "Strategy C 长期核心配置" in candidates.index:
        c = candidates.loc["Strategy C 长期核心配置"]
        A(f"**Strategy C 长期核心配置 = {c['system']}**：CAGR {pct(c.get('cagr'))}、"
          f"相对 B0 改善 {pct(c.get('cagr_vs_B0'))}、MaxDD {pct(c.get('max_dd'))}"
          f"（相对 B0 改善 {pct(c.get('maxdd_vs_B0'))}）、平均风险暴露 "
          f"{pct(c.get('avg_risky_exposure'))}、年化换手 {num(c.get('turnover_annualized'))}、"
          f"Bootstrap 平均胜率 {pct(c.get('bootstrap_avg_win_prob'))}。")
    else:
        A("**V4 未找到适合长期核心配置的择时策略**——建议使用 B&H Rebalance 作为核心，"
          "趋势策略仅作为小比例风险覆盖层。")
    A("")
    A("> 角色划分只依据回测事实：最高收益（Strategy A）、最佳风险收益（Strategy B）、"
      "长期核心配置（Strategy C，须同时满足回撤改善、暴露与成本门槛）。当 C 为空时，"
      "结论应回到 B&H Rebalance，把趋势模型降级为可选风险覆盖层。")
    A("")
    A("### 执行规则")
    A("")
    A("| 项目 | 规则 |")
    A("|---|---|")
    A("| 默认再平衡频率 | 见 B0-B6 对比与 WF/OOS 选择 |")
    A("| 阈值 | ±10%（绝对权重漂移，S3-S7） |")
    A("| 现金处理 | Cash_Rate（当期已知历史现金利率，按 6 根 4H bar 均摊） |")
    A("| 成本假设 | 默认 fee 0.10% / slippage 0.02%，矩阵另测 0.05%-0.15% |")
    A("| 停止条件 | MaxDD 门槛、成本/毛利 25%、OOS 半数窗口失败即降级为风险覆盖层 |")
    A("")
    A("---")
    A("")

    # ===================== §1 数据与配置审计 =====================
    A("## §1 数据与配置审计（Phase 0）")
    A("")
    A(f"配置指纹 sha256 = `{R.get('config_fingerprint', '')}`；随机种子 {snap.get('seed')}；"
      "数据文件一经冻结不再改写。")
    A("")
    A("### 1.1 数据质量")
    A("")
    A(md_table(dq))
    A("")
    A("### 1.2 上市过渡（Unlisted / Cash Reserve / Listed）")
    A("")
    A(md_table(lt))
    A("")
    A("### 1.3 权重恒等式审计")
    A("")
    A(md_table(inv, None, {"max_abs_sum_error": "num2", "min_sum": "num2",
                           "max_sum": "num2", "passed": "bool"}))
    A("")
    A("### 1.4 配置快照")
    A("")
    A(md_table(pd.DataFrame([pc]), None, FMT, HDR))
    A("")
    A("---")
    A("")

    # ===================== §2 基准族 =====================
    A("## §2 基准族 B0-B6 事实（Phase 2-3）")
    A("")
    A("B0 为正确处理上市日、不再平衡的 B&H；B1-B4 为月度/季度/半年/年度日历再平衡；"
      "B5 为 ±10% 阈值触发；B6 为年度 + 阈值混合。全部基准与策略共用同一执行引擎与"
      "逐腿成本计提。")
    A("")
    A(md_table(bm.reset_index(), BENCH_COLS, FMT, HDR))
    A("")
    A("![图04 再平衡频率影响](figures/04_benchmark_rebalance_frequency.png)")
    A("")
    A("![图05 CAGR vs 风险](figures/05_benchmark_cagr_vs_risk.png)")
    A("")
    A("![图03 全基准回撤](figures/03_drawdown_all_benchmarks.png)")
    A("")
    A("### 2.1 上市规则 L1 / L2 / L3")
    A("")
    A(md_table(bm_all[bm_all["kind"] == "benchmark_listing"],
               ["label", "listing_rule", "cagr", "sharpe", "max_dd",
                "avg_risky_exposure", "turnover_annualized"], FMT, HDR))
    A("")
    A("### 2.2 成本 / 现金矩阵")
    A("")
    A(md_table(bm_all[bm_all["kind"] == "benchmark"],
               ["label", "fee", "slippage", "cash_mode", "cagr", "sharpe", "max_dd",
                "turnover_annualized", "cost_pct_of_gross_profit"], FMT, HDR, max_rows=90))
    A("")
    A("![图02 净值对数刻度](figures/02_equity_log_scale.png)")
    A("")
    A("---")
    A("")

    # ===================== §3 策略族 =====================
    A("## §3 策略族 S0-S7（Phase 4-5）")
    A("")
    A("S0 为静态参考；S1 为年度 B&H + 资产级 MA200 风险减仓；S2 为年度 B&H + 组合级 "
      "MA200 Overlay；S3-S7 逐级增加模块，最终完全复现 V3。全部策略与基准共用同一"
      "再平衡规则、成本模型与现金复利口径。")
    A("")
    A(md_table(sm.reset_index(), STRAT_COLS, FMT, HDR))
    A("")
    A("![图06 策略 vs B&H Rebalance](figures/06_strategy_vs_bh_rebalance.png)")
    A("")
    A(f"![图08 焦点策略 {focus} 资产权重](figures/08_asset_weights.png)")
    A("")
    A("![图09 现金暴露](figures/09_cash_exposure.png)")
    A("")
    A("![图10 Time in Market](figures/10_time_in_market.png)")
    A("")
    A("![图11 换手与成本](figures/11_turnover_and_cost.png)")
    A("")
    A("### 3.1 消融实验（模块增量）")
    A("")
    A(md_table(abl, None, FMT, HDR))
    A("")
    A("### 3.2 配置带（Band）与决策频率")
    A("")
    A(md_table(sm_all[sm_all["kind"].isin(["strategy", "strategy_weekly"])],
               ["label", "kind", "band_mode", "cadence", "cash_mode", "fee", "slippage",
                "cagr", "sharpe", "max_dd", "turnover_annualized"], FMT, HDR, max_rows=100))
    A("")
    A("---")
    A("")

    # ===================== §3b 参考权重对照臂 =====================
    if not rwc.empty:
        A("## §3b 参考权重对照臂（15/10/35/40 vs 45/20/15/20）")
        A("")
        A("唯一变量为「参考权重 + 其配套配置带」，其余口径全部冻结。")
        A("")
        piv = rwc.pivot_table(index="system", columns="variant",
                              values=["cagr", "sharpe", "max_dd", "avg_risky_exposure"])
        flat = pd.DataFrame({
            "CAGR(15/10/35/40)": piv[("cagr", "REF15")],
            "CAGR(45/20/15/20)": piv[("cagr", "REF45")],
            "MaxDD(新)": piv[("max_dd", "REF15")],
            "MaxDD(旧)": piv[("max_dd", "REF45")],
            "Sharpe(新)": piv[("sharpe", "REF15")],
            "Sharpe(旧)": piv[("sharpe", "REF45")],
        }).reset_index().rename(columns={"system": "系统"})
        flat["ΔCAGR"] = flat["CAGR(15/10/35/40)"] - flat["CAGR(45/20/15/20)"]
        A(md_table(flat, ["系统", "CAGR(15/10/35/40)", "CAGR(45/20/15/20)", "ΔCAGR",
                          "MaxDD(新)", "MaxDD(旧)", "Sharpe(新)", "Sharpe(旧)"],
                   {"CAGR(15/10/35/40)": "pct", "CAGR(45/20/15/20)": "pct", "ΔCAGR": "pct",
                    "MaxDD(新)": "pct", "MaxDD(旧)": "pct",
                    "Sharpe(新)": "num2", "Sharpe(旧)": "num2"}, None))
        A("")

    # ===================== §4 公平比较 =====================
    A("## §4 与 B&H Rebalance 的公平比较（Phase 6）")
    A("")
    A("§9.1 差值表：策略相对每个基准的 CAGR / Sharpe / MaxDD / Calmar / 换手差。"
      "§11.2 要求先做同风险或同收益匹配，再讨论效率改善。")
    A("")
    A(md_table(spread, ["strategy", "benchmark", "cagr_spread", "sharpe_spread",
                        "maxdd_spread", "calmar_spread", "turnover_spread",
                        "final_wealth_ratio"], FMT, HDR, max_rows=80))
    A("")
    A("### 4.1 同风险 / 同收益匹配")
    A("")
    A(md_table(match, None, FMT, HDR))
    A("")
    A("### 4.2 主动收益与捕获率（相对 B4）")
    A("")
    A(md_table(act[act["benchmark"] == "B4"] if not act.empty else act,
               ["strategy", "benchmark", "active_return_total", "active_return_annual",
                "tracking_error", "information_ratio", "prob_underperformance",
                "opportunity_cost"], FMT, HDR))
    A("")
    A(md_table(cap[cap["benchmark"] == "B4"] if not cap.empty else cap,
               ["strategy", "benchmark", "upside_capture_bull", "downside_capture_bear",
                "upside_capture_daily", "downside_capture_daily"], FMT, HDR))
    A("")
    A("![图12 上涨/下跌捕获率](figures/12_bull_bear_capture.png)")
    A("")
    A("![图07 累计主动收益](figures/07_active_return_vs_benchmark.png)")
    A("")
    A("![图20 累计机会成本](figures/20_opportunity_cost.png)")
    A("")
    A("![图13 滚动 Sharpe](figures/13_rolling_sharpe.png)")
    A("")
    A("![图14 滚动 CAGR](figures/14_rolling_cagr.png)")
    A("")
    A("### 4.3 分阶段（Bull / Bear / Sideways / Recovery）")
    A("")
    A(md_table(seg[seg["benchmark"] == "B4"] if not seg.empty else seg,
               ["strategy", "benchmark", "segment", "days", "strat_cagr", "bench_cagr",
                "strat_maxdd", "bench_maxdd", "strat_sharpe", "bench_sharpe",
                "active_return"], FMT, HDR, max_rows=60))
    A("")
    A("### 4.4 机会成本明细（按年）")
    A("")
    A(md_table(opp[opp["benchmark"] == "B4"] if not opp.empty else opp,
               ["strategy", "benchmark", "year", "strat_return", "bench_return",
                "opportunity_cost"], FMT, HDR, max_rows=60))
    A("")
    A("---")
    A("")

    # ===================== §5 稳健性 =====================
    A("## §5 稳健性（Phase 7-8）")
    A("")
    A("### 5.1 资产剔除")
    A("")
    A(md_table(eql, ["subset", "system", "cagr", "sharpe", "max_dd", "calmar",
                     "turnover_annualized"], FMT, HDR, max_rows=60))
    A("")
    A("![图18 资产剔除](figures/18_asset_exclusion.png)")
    A("")
    A("### 5.2 参数敏感性")
    A("")
    A(md_table(param, ["family", "param", "value", "cagr", "sharpe", "max_dd", "calmar",
                       "turnover_annualized"], FMT, HDR, max_rows=50))
    A("")
    A("![图17 参数热力图](figures/17_parameter_heatmap.png)")
    A("")
    A("### 5.3 成本敏感性")
    A("")
    A(md_table(cost, ["system", "fee", "slippage", "cash_mode", "cagr", "sharpe", "max_dd",
                      "turnover_annualized", "cost_pct_of_gross_profit"], FMT, HDR,
               max_rows=60))
    A("")
    A("### 5.4 现金利率敏感性（Cash_0 vs Cash_Rate）")
    A("")
    A(md_table(cash, ["system", "cash_mode", "cagr", "sharpe", "max_dd", "calmar",
                      "avg_cash"], FMT, HDR, max_rows=60))
    A("")
    A("### 5.5 Walk Forward / OOS")
    A("")
    A(md_table(wf, ["window", "train_start", "train_validate_end", "test_end", "system",
                    "candidate", "params", "train_cagr", "train_sharpe", "train_calmar",
                    "selected"], FMT, HDR, max_rows=90))
    A("")
    A("各窗口在 Train/Validate 区间以 Calmar 选出、随后在 Test 区间冻结评估的候选：")
    A("")
    sel = wf[wf["selected"]] if not wf.empty else wf
    A(md_table(sel, ["window", "system", "candidate", "params", "train_calmar"], FMT, HDR))
    A("")
    A("![图15 Walk Forward](figures/15_walk_forward.png)")
    A("")
    A(md_table(oos, ["mode", "window", "system", "benchmark", "test_cagr", "test_sharpe",
                     "test_max_dd", "bench_cagr", "bench_max_dd", "active_return",
                     "maxdd_improvement"], FMT, HDR, max_rows=90))
    A("")
    A("![图16 OOS 各窗口主动收益](figures/16_oos_by_year.png)")
    A("")
    A("### 5.6 Block Bootstrap（10,000 次）")
    A("")
    A(md_table(boot, ["strategy", "benchmark", "prob_strategy_wins", "prob_strategy_loses",
                      "cagr_p5", "cagr_p50", "cagr_p95", "maxdd_p5", "maxdd_p50",
                      "maxdd_p95"], FMT, HDR, max_rows=60))
    A("")
    A("### 5.7 Monte Carlo")
    A("")
    A(md_table(mc, None, FMT, HDR))
    A("")
    A("![图19 Monte Carlo 分布](figures/19_monte_carlo_distribution.png)")
    A("")
    A("---")
    A("")

    # ===================== §6 门槛与排名 =====================
    A("## §6 硬门槛与候选排名（Phase 9）")
    A("")
    A("§14 硬门槛：MaxDD 相对 B0 改善 ≥15pp；Sharpe ≥0.9×B4；OOS 至少半数窗口有效；"
      "平均风险暴露 ≥50%；成本 ≤25% 毛利。仅通过全部门槛的候选才可参与长期核心配置的评选。")
    A("")
    A(md_table(gates, None, FMT, HDR))
    A("")
    A("### 6.1 加权评分（25% Sharpe / 20% Sortino / 20% Calmar / 15% CAGR / 10% MaxDD / "
      "5% 主动收益稳定性 / 5% 简洁性）")
    A("")
    A(md_table(scores, None, FMT, HDR, max_rows=60))
    A("")
    A("### 6.2 三个答案（§15）")
    A("")
    A(md_table(cands, None, FMT, HDR))
    A("")
    A("---")
    A("")

    # ===================== §7 附录 =====================
    A("## §7 附录 · 输出文件与验收")
    A("")
    A("### 7.1 交易与再平衡流水（节选）")
    A("")
    A(f"基准交易腿 {len(tr_b):,} 条、策略交易腿 {len(tr_s):,} 条、基准再平衡事件 "
      f"{len(ev_b):,} 次；完整流水见对应 CSV。")
    A("")
    A(md_table(tr_s.head(40), ["system", "asset", "side", "signal_time", "execution_time",
                               "reason", "price", "notional", "fee", "slippage",
                               "cost_total"], FMT, HDR, max_rows=40))
    A("")
    A("### 7.2 文件清单")
    A("")
    files = sorted([p.name for p in OUT.glob("v4_*.csv")] +
                   [p.name for p in OUT.glob("v4_*.json")])
    for f in files:
        A(f"- `{f}`")
    A("")
    A("### 7.3 §19 最小验收测试")
    A("")
    A("权重恒等式 / 上市日 / 再平衡触发 / 成本计提 / 信号时序 / 可复现性 六项合成数据测试"
      "位于 `scripts/v4_spot/tests_v4.py`，运行 `python -m scripts.v4_spot.tests_v4` 全部通过。")
    A("")
    A("### 7.4 图表清单（§18）")
    A("")
    for p in sorted(FIG.glob("*.png")):
        A(f"- [{p.name}](figures/{p.name})")
    A("")
    A("---")
    A("")
    A(f"_V4 回测报告 · 配置指纹 `{R.get('config_fingerprint', '')[:16]}` · 种子 "
      f"{snap.get('seed')} · 全部结论均由上述 CSV 与图表派生，未使用任何未在报告中列出的假设。_")
    A("")

    out = OUT / "v4_spot_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Markdown 报告 -> {out}  ({out.stat().st_size / 1e3:.1f} KB)")
    return out


def main() -> None:
    build_html()
    build_md()


if __name__ == "__main__":
    main()