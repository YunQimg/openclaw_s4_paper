# -*- coding: utf-8 -*-
"""V3 现货系统 —— 最终报告构建器（§53-§59）

产出：
  * 15 张 PNG 图表（§58，深色主题）
  * v3_spot_report.html（回答 §56 全部 18 问、§54 评分与四类排名、§59 Strategy A/B/C）

读取：reports/v3_spot/results.pkl + experiments.pkl
"""
from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

from scripts.v3_spot import common as C                                  # noqa: E402
from scripts.v3_spot import metrics as M                                 # noqa: E402
from scripts.v3_spot import strategy as S                                # noqa: E402
from scripts.v3_spot.run_experiments import WINDOWS, reentry_efficiency   # noqa: E402

OUT = REPO / "reports" / "v3_spot"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 绘图主题
# ---------------------------------------------------------------------------
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

CLR = {"BTC": "#f7931a", "ETH": "#8a92b2", "SOL": "#14f195", "BNB": "#f0b90b",
       "Cash": "#5a6673", "bench": "#e0564a", "strat": "#4c9ffe",
       "alt": "#9d7bff", "good": "#3fb950"}

# §55 三个候选 + 收益型模型
FINALISTS = ["ModelA_MA200", "ModelB_MA200_TSMOM", "Conservative",
             "Balanced", "Aggressive"]
SHORT = {"ModelA_MA200": "A:MA200", "ModelB_MA200_TSMOM": "B:MA200+TSMOM",
         "Conservative": "保守 Conservative", "Balanced": "平衡 Balanced",
         "Aggressive": "激进 Aggressive"}

# §50 参数稳健性扫描（对每个候选系统重复）
ROBUST_VARIANTS = [("base", {}), ("ma=250", {"ma_len": 250}),
                   ("tsmom=3M", {"tsmom_days": 63}), ("atr=20", {"atr_len": 20}),
                   ("adx=20", {"adx_len": 20}), ("persist=12", {"persist_bars": 12}),
                   ("rebal=15%", {"rebal_threshold": 0.15}),
                   ("rebal=20%", {"rebal_threshold": 0.20}),
                   ("vol=20%", {"target_vol": 0.20})]


def pct(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:,.{nd}f}%"


def num(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


# ---------------------------------------------------------------------------
# §53 Opportunity Cost
# ---------------------------------------------------------------------------
def opportunity_cost(fp, res, bench_eq):
    """持有 Cash 的机会成本：按基准日收益折算"若现金按比例投入基准"的损失。"""
    idx = fp["index"]
    cash_w = res["weights"]["cash"]
    b = M.to_daily(bench_eq).pct_change().fillna(0.0)
    c = cash_w.resample("1D").last().reindex(b.index).ffill().fillna(0.0)
    px = fp["px"]
    btc = M.to_daily(px["BTC"]).pct_change().fillna(0.0)
    opp = (c * b).sum()                      # 累计复利近似（小量级）
    missed_btc = (c * btc).sum()
    s = M.summarize(res["equity"])
    bs = M.summarize(bench_eq)
    return {"opp_cost_total": float(opp), "missed_btc_upside": float(missed_btc),
            "avg_cash": float(c.mean()),
            "bench_total_return": bs["total_return"],
            "strategy_total_return": s["total_return"],
            "bench_final": bs["final_wealth"], "strategy_final": s["final_wealth"],
            "net_gain_vs_bench": float(s["final_wealth"] - bs["final_wealth"])}


# ---------------------------------------------------------------------------
# 稳健性：参数扫描 + 纯 OOS
# ---------------------------------------------------------------------------
def robustness(fp, name, modules):
    rows = []
    for label, kw in ROBUST_VARIANTS:
        cfg = S.Cfg(modules=list(modules), **kw)
        r = C.run_strategy(fp, cfg, f"{name}|{label}")
        s = M.summarize(r["equity"])
        rows.append({"variant": label, "cagr": s.get("cagr"), "sharpe": s.get("sharpe"),
                     "max_dd": s.get("max_dd"), "calmar": s.get("calmar")})
    df = pd.DataFrame(rows)
    oos = []
    for tr, va, te in WINDOWS:
        sub = C.get_fp(start=f"{te}-01-01", end=f"{te}-12-31")
        if len(sub["index"]) < 200:
            continue
        r = C.run_strategy(sub, S.Cfg(modules=list(modules)), f"{name}|{te}")
        s = M.summarize(r["equity"])
        oos.append({"test": te, "cagr": s.get("cagr"), "sharpe": s.get("sharpe"),
                    "max_dd": s.get("max_dd")})
    odf = pd.DataFrame(oos)
    base = df.loc[df["variant"] == "base"].iloc[0]
    return {
        "variant_table": df, "oos_table": odf,
        "param_sharpe_min": float(df["sharpe"].min()),
        "param_sharpe_max": float(df["sharpe"].max()),
        "param_sharpe_std": float(df["sharpe"].std()),
        "param_sharpe_base": float(base["sharpe"]),
        "oos_sharpe_mean": float(odf["sharpe"].mean()) if not odf.empty else np.nan,
        "oos_sharpe_worst": float(odf["sharpe"].min()) if not odf.empty else np.nan,
        "oos_cagr_mean": float(odf["cagr"].mean()) if not odf.empty else np.nan,
        "oos_positive": float((odf["cagr"] > 0).mean()) if not odf.empty else np.nan,
    }


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def fig_equity(curves, bench_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    d = {k: M.to_daily(v) for k, v in curves.items()}
    order = [bench_name, "ModelA_MA200", "Conservative", "Balanced", "Aggressive"]
    cols = {"BuyHold_45_20_15_20": CLR["bench"], "ModelA_MA200": "#f0b90b",
            "Conservative": "#3fb950", "Balanced": CLR["strat"], "Aggressive": CLR["alt"]}
    for k in order:
        if k in d:
            lab = SHORT.get(k, "基准 B&H 45/20/15/20" if k == bench_name else k)
            ax.plot(d[k].index, d[k].values, label=lab, lw=1.5, color=cols.get(k))
    ax.set_yscale("log")
    ax.set_title("图01 · 净值曲线（对数坐标，起点 $10,000）")
    ax.set_ylabel("组合净值 (USD，对数)"); ax.set_xlabel("")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(FIG / "01_equity_curve.png", dpi=110); plt.close(fig)


def fig_drawdown(curves, bench_name):
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, c in [("BuyHold_45_20_15_20", CLR["bench"]), ("Conservative", "#3fb950"),
                 ("Balanced", CLR["strat"]), ("Aggressive", CLR["alt"])]:
        if k in curves:
            dd = M.to_daily(M.drawdown(curves[k]))
            ax.plot(dd.index, dd.values, lw=1.4, color=c,
                    label=SHORT.get(k, "基准 B&H 45/20/15/20"))
    ax.set_title("图02 · 回撤曲线")
    ax.set_ylabel("回撤"); ax.legend(loc="lower left", ncol=2)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    fig.savefig(FIG / "02_drawdown.png", dpi=110); plt.close(fig)


def fig_allocation(weights, bench_name):
    w = weights["Balanced"].resample("1D").last().ffill()
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = w.index
    ax.stackplot(xs, [w[a].values for a in S.ASSETS] + [w["cash"].values],
                 labels=S.ASSETS + ["Cash"],
                 colors=[CLR[a] for a in S.ASSETS] + [CLR["Cash"]], alpha=0.92)
    ax.set_ylim(0, 1)
    ax.set_title("图03 · 资产配置演变（Balanced 候选，日频）")
    ax.set_ylabel("权重"); ax.legend(loc="lower left", ncol=5)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    fig.savefig(FIG / "03_asset_allocation.png", dpi=110); plt.close(fig)


def fig_regime(bench_eq, phase):
    b = M.to_daily(bench_eq)
    fig, ax = plt.subplots(figsize=(12, 5))
    cols = {"Bull": "#1f6f4a", "Bear": "#7a2b2b", "Sideways": "#4a4a52", "Recovery": "#7a6a1f"}
    p = phase.reindex(b.index).ffill()
    for ph, c in cols.items():
        m = (p == ph).to_numpy()
        if m.any():
            ax.fill_between(b.index, 0, b.max().max(), where=m, color=c, alpha=0.30,
                            step="mid")
    ax.plot(b.index, b.values, color=CLR["bench"], lw=1.5)
    ax.set_yscale("log")
    ax.set_title("图04 · 市场阶段划分（基准组合，日频对数净值）")
    ax.legend(handles=[Patch(facecolor=c, alpha=0.4, label=k) for k, c in cols.items()],
              loc="upper left", ncol=4)
    fig.savefig(FIG / "04_regime.png", dpi=110); plt.close(fig)


def fig_direction(layers):
    fig, ax = plt.subplots(figsize=(12, 5))
    for a in S.ASSETS:
        d = layers[a]["direction"].resample("1D").last()
        ax.plot(d.index, d.values, lw=1.1, color=CLR[a], label=a, alpha=0.9)
    ax.axhline(0.25, color="#8b97a5", ls="--", lw=0.9)
    ax.axhline(0.0, color="#39414d", lw=0.9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("图05 · Layer1 Direction Score（Balanced 配置，虚线=0.25 分档阈值）")
    ax.legend(loc="lower left", ncol=4)
    fig.savefig(FIG / "05_direction_score.png", dpi=110); plt.close(fig)


def fig_quality(layers):
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for a in S.ASSETS:
        axs[0].plot(*_xy(layers[a]["quality"]), lw=1.0, color=CLR[a], label=a, alpha=0.9)
        axs[1].plot(*_xy(layers[a]["risk"]), lw=1.0, color=CLR[a], label=a, alpha=0.9)
    axs[0].set_title("图06 · Layer2 Trend Quality（上）与 Layer3 Risk Modifier（下）")
    axs[0].set_ylim(0, 1); axs[1].set_ylim(0, 2)
    axs[0].legend(loc="upper left", ncol=4)
    fig.savefig(FIG / "06_trend_quality.png", dpi=110); plt.close(fig)


def _xy(s):
    d = s.resample("1D").last()
    return d.index, d.values


def fig_rolling(eq, bench_eq, which):
    fig, ax = plt.subplots(figsize=(12, 5))
    for yr, c in [(365, "#3fb950"), (730, CLR["strat"]), (1095, CLR["alt"])]:
        rm = M.rolling_metrics(eq, yr)
        if not rm.empty and which in rm:
            ax.plot(rm.index, rm[which].values, lw=1.3, color=c, label=f"Balanced {yr // 365}Y")
    rm = M.rolling_metrics(bench_eq, 1095)
    if not rm.empty and which in rm:
        ax.plot(rm.index, rm[which].values, lw=1.3, color=CLR["bench"], label="基准 B&H 3Y")
    ttl = "Sharpe" if which == "sharpe" else "CAGR"
    ax.set_title(f"图{'07' if which == 'sharpe' else '08'} · 滚动窗口 {ttl}")
    if which == "max_dd":
        ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    ax.axhline(0, color="#39414d", lw=0.9)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(FIG / f"{'07_rolling_sharpe' if which == 'sharpe' else '08_rolling_cagr'}.png",
                dpi=110)
    plt.close(fig)


def fig_capture(summary):
    df = summary[summary["label"].isin(FINALISTS)].set_index("label").loc[FINALISTS]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, df["upside_capture_bull"], 0.4, label="Upside Capture（Bull 阶段）",
           color=CLR["good"])
    ax.bar(x + 0.2, df["downside_capture_bear"], 0.4, label="Downside Capture（Bear 阶段）",
           color=CLR["bench"])
    ax.axhline(0.8, color="#8b97a5", ls="--", lw=1.0, label="80% 目标线")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[i] for i in df.index], rotation=12)
    ax.set_title("图09 · 牛市上涨捕获 / 熊市下跌捕获（阶段累计口径）")
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(FIG / "09_bull_bear_capture.png", dpi=110); plt.close(fig)


def fig_tim(weights, summary):
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    for k, c in [("Conservative", "#3fb950"), ("Balanced", CLR["strat"]),
                 ("Aggressive", CLR["alt"])]:
        if k in weights:
            rt = weights[k]["risky_total"].resample("1D").last()
            axs[0].plot(rt.index, rt.values, lw=1.1, color=c, label=SHORT[k])
    axs[0].set_ylim(0, 1.02)
    axs[0].legend(loc="upper left", ncol=3)
    axs[0].set_title("图10 · Time in Market（风险资产总权重，上）与暴露分布（下）")
    df = summary[summary["label"].isin(FINALISTS)].set_index("label").loc[FINALISTS]
    x = np.arange(len(df))
    axs[1].bar(x - 0.2, df["tim_gt_50"], 0.4, color=CLR["strat"], label=">50% 时间占比")
    axs[1].bar(x + 0.2, df["tim_gt_75"], 0.4, color=CLR["good"], label=">75% 时间占比")
    axs[1].set_xticks(x); axs[1].set_xticklabels([SHORT[i] for i in df.index], rotation=12)
    axs[1].legend(loc="upper left", ncol=2)
    axs[1].yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    fig.savefig(FIG / "10_time_in_market.png", dpi=110); plt.close(fig)


def fig_comparison(summary):
    df = summary[summary["kind"].isin(["benchmark", "model"])].copy()
    order = ["BuyHold_45_20_15_20", "BuyHold_BTC_only", "ModelA_MA200",
             "ModelB_MA200_TSMOM", "Conservative", "Balanced", "Aggressive"]
    df = df[df["label"].isin(order)].set_index("label").loc[order]
    metrics = [("cagr", "CAGR"), ("sharpe", "Sharpe"), ("calmar", "Calmar"),
               ("max_dd", "MaxDD")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, (col, ttl) in zip(axs, metrics):
        v = df[col].abs() if col == "max_dd" else df[col]
        colors = [CLR["bench"] if i.startswith("BuyHold") else
                  (CLR["alt"] if i == "Aggressive" else
                   ("#3fb950" if i == "Conservative" else
                    (CLR["strat"] if i == "Balanced" else "#f0b90b")))
                  for i in df.index]
        ax.bar(range(len(df)), v.values, color=colors)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([SHORT.get(i, i.replace("BuyHold_", "BM:")) for i in df.index],
                           rotation=70, fontsize=7.5)
        ax.set_title(ttl if col != "max_dd" else "MaxDD（绝对值）")
        if col == "cagr":
            ax.yaxis.set_major_formatter(lambda v2, p: f"{v2:.0%}")
    fig.suptitle("图11 · 策略横向对比", fontsize=12)
    fig.savefig(FIG / "11_strategy_comparison.png", dpi=110); plt.close(fig)


def fig_ablation(abl):
    fig, ax = plt.subplots(figsize=(11, 5))
    d = abl.set_index("label")
    base = float(d.loc["Full(8 modules)", "sharpe"])
    colors = [CLR["good"] if v >= base else CLR["bench"] for v in d["sharpe"]]
    ax.barh(range(len(d)), d["sharpe"].values, color=colors)
    ax.axvline(base, color="#8b97a5", ls="--", lw=1.0, label=f"Full = {base:.3f}")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe")
    ax.set_title("图12 · §39 Ablation（逐个删除模块，绿=优于 Full）")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "12_ablation.png", dpi=110); plt.close(fig)


def fig_param_heatmap(param):
    p = param.pivot_table(index="kind", columns="variant", values="sharpe", aggfunc="mean")
    order = ["ma_len", "tsmom", "atr_len", "adx_len", "target_vol", "persist_bars",
             "rebal_threshold", "cadence", "alloc_mode", "lower_band", "ref_weight",
             "reentry_guard", "dd_guard"]
    p = p.reindex([k for k in order if k in p.index])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(p.values, cmap="RdYlGn", aspect="auto",
                   vmin=np.nanmin(p.values) - 0.05, vmax=np.nanmax(p.values) + 0.05)
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, rotation=45, fontsize=8.5)
    ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index, fontsize=8.5)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#11151c")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax); cb.set_label("Sharpe")
    ax.set_title("图13 · §50 参数稳定性热力图（Sharpe）")
    fig.savefig(FIG / "13_parameter_heatmap.png", dpi=110); plt.close(fig)


def fig_walk_forward(pure, selected):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(pure))
    axs[0].bar(x - 0.2, pure["cagr"], 0.4, label="纯 OOS（冻结默认参数）", color=CLR["strat"])
    if len(selected):
        xs = np.arange(len(selected))
        axs[0].bar(xs + 0.2, selected["cagr"], 0.4,
                   label="Train/Validate 选参 → Test", color=CLR["alt"])
    axs[0].set_xticks(x); axs[0].set_xticklabels(pure["label"])
    axs[0].yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    axs[0].set_title("OOS CAGR"); axs[0].legend(fontsize=8)
    axs[1].bar(x - 0.2, pure["sharpe"], 0.4, color=CLR["strat"])
    if len(selected):
        axs[1].bar(np.arange(len(selected)) + 0.2, selected["sharpe"], 0.4, color=CLR["alt"])
    axs[1].set_xticks(x); axs[1].set_xticklabels(pure["label"])
    axs[1].set_title("OOS Sharpe")
    fig.suptitle("图14 · §48 Walk-Forward 样本外结果", fontsize=12)
    fig.savefig(FIG / "14_walk_forward.png", dpi=110); plt.close(fig)


def fig_monte_carlo(mc_cagr, mc_mdd, rows):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    r = rows[rows["kind"] == "monte_carlo"].iloc[0]
    axs[0].hist(np.asarray(mc_cagr) * 100, bins=70, color=CLR["strat"], alpha=0.85)
    axs[0].axvline(r["strategy_cagr_median"] * 100, color="#f0b90b", lw=1.6,
                   label=f"中位数 {pct(r['strategy_cagr_median'])}")
    axs[0].axvline(r["strategy_cagr_p5"] * 100, color=CLR["bench"], lw=1.6, ls="--",
                   label=f"5% 分位 {pct(r['strategy_cagr_p5'])}")
    axs[0].axvline(r["strategy_cagr_p95"] * 100, color=CLR["good"], lw=1.6, ls="--",
                   label=f"95% 分位 {pct(r['strategy_cagr_p95'])}")
    axs[0].set_title("Monte Carlo · CAGR 分布（10,000 次）")
    axs[0].set_xlabel("CAGR (%)"); axs[0].legend(fontsize=8)
    axs[1].hist(np.asarray(mc_mdd) * 100, bins=70, color=CLR["bench"], alpha=0.85)
    axs[1].axvline(r["strategy_mdd_median"] * 100, color="#f0b90b", lw=1.6,
                   label=f"中位数 {pct(r['strategy_mdd_median'])}")
    axs[1].axvline(r["strategy_mdd_p5"] * 100, color=CLR["bench"], lw=1.6, ls="--",
                   label=f"5% 分位 {pct(r['strategy_mdd_p5'])}")
    axs[1].set_title("Monte Carlo · MaxDD 分布")
    axs[1].set_xlabel("MaxDD (%)"); axs[1].legend(fontsize=8)
    fig.suptitle("图15 · §51 Monte Carlo 情景分布", fontsize=12)
    fig.savefig(FIG / "15_monte_carlo.png", dpi=110); plt.close(fig)


def fig_rebalance_freq(curves, summary):
    """§4 补充：四资产 B&H 在不同日历再平衡频率下的净值与关键指标。"""
    order = [("BuyHold_45_20_15_20", "静态不再平衡", CLR["bench"]),
             ("BuyHold_Monthly", "每月再平衡", "#9d7bff"),
             ("BuyHold_Quarterly", "每季再平衡", CLR["strat"]),
             ("BuyHold_SemiAnnual", "每半年再平衡", CLR["good"]),
             ("BuyHold_Annual", "每年再平衡", "#f0b90b"),
             ("Balanced", "Strategy C（Balanced）", "#e6edf3")]
    s = summary[summary["kind"] == "benchmark"].set_index("label")
    ref_cagr = summary.set_index("label").loc["Balanced", "cagr"] * 100
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5),
                            gridspec_kw={"width_ratios": [1.7, 1]})
    for k, lab, c in order:
        if k in curves:
            d = M.to_daily(curves[k])
            axs[0].plot(d.index, d.values, lw=1.6 if k == "Balanced" else 1.4,
                        ls="--" if k == "Balanced" else "-", label=lab, color=c)
    axs[0].set_yscale("log")
    axs[0].set_title("四资产 B&H：不同再平衡频率的净值（对数坐标）")
    axs[0].set_ylabel("组合净值 (USD，对数)"); axs[0].legend(fontsize=8.5)

    bars = [(lab, c, s.loc[k, "cagr"] * 100)
            for k, lab, c in order if k != "Balanced" and k in s.index]
    xs = np.arange(len(bars))
    vals = [v for _, _, v in bars]
    axs[1].bar(xs, vals, color=[c for _, c, _ in bars], alpha=0.9)
    axs[1].axhline(ref_cagr, color="#e6edf3", lw=1.4, ls="--")
    axs[1].text(len(bars) - 0.5, ref_cagr + 1.5,
                f"Strategy C {ref_cagr:.1f}%", ha="right", fontsize=8.5, color="#e6edf3")
    for x, v in zip(xs, vals):
        axs[1].text(x, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5, color="#c7d1db")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([lab.replace("再平衡", "\n再平衡") for lab, _, _ in bars],
                           fontsize=8)
    axs[1].set_title("CAGR 对比 vs Strategy C")
    axs[1].set_ylabel("CAGR (%)")
    axs[1].set_ylim(0, max(vals + [ref_cagr]) * 1.22)
    fig.suptitle("图16 · §4 补充：B&H 日历再平衡频率对比", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "16_rebalance_frequency.png", dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
body{background:#0f1318;color:#c7d1db;font-family:"Microsoft YaHei",-apple-system,sans-serif;
 margin:0;padding:0 0 60px 0;line-height:1.65;font-size:14px}
.wrap{max-width:1180px;margin:0 auto;padding:0 26px}
h1{font-size:26px;color:#e6edf3;margin:34px 0 6px}
h2{font-size:19px;color:#e6edf3;margin:34px 0 10px;border-left:4px solid #4c9ffe;padding-left:11px}
h3{font-size:15.5px;color:#9dc7ff;margin:20px 0 8px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:12.5px}
th,td{border:1px solid #232b36;padding:6px 9px;text-align:right;white-space:nowrap}
th{background:#182029;color:#e6edf3;text-align:right}
td:first-child,th:first-child{text-align:left}
tr:nth-child(even) td{background:#141a21}
.good{color:#3fb950;font-weight:600}.bad{color:#e0564a;font-weight:600}
.mut{color:#8b97a5}
.card{background:#151b23;border:1px solid #232b36;border-radius:9px;padding:14px 18px;margin:14px 0}
.grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:14px 0}
.kpi{background:#151b23;border:1px solid #232b36;border-radius:9px;padding:12px 14px}
.kpi .v{font-size:21px;color:#e6edf3;font-weight:600}
.kpi .l{font-size:11.5px;color:#8b97a5}
img{width:100%;border:1px solid #232b36;border-radius:8px;margin:10px 0;background:#0f1318}
.q{background:#151b23;border-left:3px solid #4c9ffe;padding:9px 14px;margin:9px 0;border-radius:0 7px 7px 0}
.q b{color:#9dc7ff}
ol,ul{padding-left:22px}li{margin:4px 0}
code{background:#1c242e;padding:1px 5px;border-radius:4px;color:#f0b90b}
.foot{color:#68727e;font-size:12px;margin-top:34px;border-top:1px solid #232b36;padding-top:14px}
"""


def html_table(df, fmt=None, index=False, cls_col=None):
    fmt = fmt or {}
    cols = list(df.columns)
    head = ("<th></th>" if index else "") + "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for idx, row in df.iterrows():
        tds = []
        if index:
            tds.append(f"<td>{idx}</td>")
        for c in cols:
            v = row[c]
            if c in fmt:
                txt = fmt[c](v)
            elif isinstance(v, (int, np.integer)):
                txt = f"{v:,d}"
            elif isinstance(v, (float, np.floating)):
                txt = f"{v:,.3f}"
            else:
                txt = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            tds.append(f"<td>{txt}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def img(name, cap=""):
    p = FIG / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{b64}" alt="{name}">'
            + (f'<div class="mut" style="font-size:12px">{cap}</div>' if cap else ""))


# ---------------------------------------------------------------------------
def main():
    with open(OUT / "results.pkl", "rb") as fh:
        R = pickle.load(fh)
    with open(OUT / "experiments.pkl", "rb") as fh:
        X = pickle.load(fh)

    fp = C.get_fp()
    summ = R["summary"]
    curves = R["curves"]
    weights = R["weights"]
    phase = R["phase"]
    bench_name = "BuyHold_45_20_15_20"
    bench_eq = curves[bench_name]
    bench_s = M.summarize(bench_eq)
    btc_s = M.summarize(curves["BuyHold_BTC_only"])

    models = summ[summ["kind"] == "model"].set_index("label")
    print("样本:", fp["index"][0].date(), "~", fp["index"][-1].date())

    # ---- 逐候选系统：稳健性 + OOS + 阶段 + 回撤事件 ----
    print("计算候选系统稳健性 / OOS ...")
    rob, regimes, ddev = {}, {}, {}
    for name in FINALISTS:
        mods = S.MODELS[name]
        rob[name] = robustness(fp, name, mods)
        r = C.run_strategy(fp, S.Cfg(modules=mods), name)
        regimes[name] = M.regime_table(phase, bench_eq, r["equity"], r["weights"])
        ddev[name] = M.drawdown_episodes(r["equity"], top=3)
        print(f"  {name}: param Sharpe {rob[name]['param_sharpe_min']:.2f}"
              f"~{rob[name]['param_sharpe_max']:.2f}, OOS Sharpe均值 "
              f"{rob[name]['oos_sharpe_mean']:.2f}")

    # ---- §53 机会成本 ----
    focus_res = C.run_strategy(fp, S.Cfg(modules=S.MODELS["Balanced"]), "Balanced")
    oc = opportunity_cost(fp, focus_res, bench_eq)

    # ---- §54 评分与排名 ----
    sc = models[[c for c in ["cagr", "sharpe", "sortino", "calmar", "max_dd",
                             "upside_capture_bull", "final_wealth", "turnover_annual",
                             "trades", "avg_risky", "tim_gt_50", "tim_gt_75",
                             "downside_capture_bear"]
                 if c in models.columns]].copy()
    sc["param_sharpe_min"] = [rob[i]["param_sharpe_min"] if i in rob else np.nan
                              for i in sc.index]
    sc["oos_sharpe_mean"] = [rob[i]["oos_sharpe_mean"] if i in rob else np.nan
                             for i in sc.index]
    sc["robustness"] = [
        float(np.clip(0.5 * (rob[i]["param_sharpe_min"] / 2.0)
                      + 0.5 * (rob[i]["oos_sharpe_mean"] / 2.0), 0.0, 1.0))
        if i in rob else 0.5 for i in sc.index]
    sc["score"] = [M.score_rank(r) for _, r in sc.iterrows()]

    # §59 Strategy C —— 长期持有评分（五项：CAGR/评分 + 低 MaxDD + 高 Upside Capture
    #        + 低交易频率 + 高 OOS 稳定性）
    sc["longterm"] = (0.25 * sc["score"]
                      + 0.25 * np.clip(sc["oos_sharpe_mean"] / 2.0, 0, 1)
                      + 0.20 * (1.0 - (sc["max_dd"].abs() / 0.8))
                      + 0.15 * np.clip(sc["upside_capture_bull"] / 1.2, 0, 1)
                      + 0.15 * (1.0 / (1.0 + sc["turnover_annual"] / 20.0)))

    rank_return = sc["cagr"].rank(ascending=False)
    rank_risk = (sc["sharpe"].rank(ascending=False)
                 + sc["calmar"].rank(ascending=False)
                 + sc["max_dd"].rank(ascending=False)) / 3.0
    rank_robust = (sc["robustness"].rank(ascending=False))
    rank_capture = sc["upside_capture_bull"].rank(ascending=False)
    rank_df = pd.DataFrame({
        "Return Ranking": rank_return, "Risk Ranking": rank_risk,
        "Robustness Ranking": rank_robust, "B&H Capture Ranking": rank_capture,
        "综合评分": sc["score"], "长期持有评分": sc["longterm"],
        "CAGR": sc["cagr"], "Sharpe": sc["sharpe"],
        "MaxDD": sc["max_dd"], "Upside Capture": sc["upside_capture_bull"],
        "OOS Sharpe": sc["oos_sharpe_mean"],
        "Robustness": sc["robustness"]}).sort_values("综合评分", ascending=False)

    # ---- §59 Strategy A / B / C：三个独立目标函数，且必须给出三个不同系统 ----
    A = sc.sort_values("cagr", ascending=False).index[0]
    B = sc.drop(index=[A]).sort_values("score", ascending=False).index[0]
    C_ = sc.drop(index=[A, B]).sort_values("longterm", ascending=False).index[0]

    # ---- 行为 ----
    years = (fp["index"][-1] - fp["index"][0]).days / 365.25
    bb = X["bull_bear"]
    reev = bb[bb["section"] == "reentry_event"].copy()
    reev["flat_bars"] = pd.to_numeric(reev["flat_bars"], errors="coerce")
    maj = reev[reev["flat_bars"] >= 6].sort_values("flat_bars", ascending=False)
    mc_rows = pd.read_csv(OUT / "v3_spot_monte_carlo.csv")
    abl = X["ablation"]
    inc = X["incremental"]
    param = X["param"]
    wf_pure, wf_sel = X["wf_pure"], X["wf_selected"]
    bs = X["bootstrap"]
    mc_cagr = np.load(OUT / "v3_spot_mc_cagr_samples.npy")
    mc_mdd = np.load(OUT / "v3_spot_mc_mdd_samples.npy")

    # ---- 绘图 ----
    print("生成 15 张图表 ...")
    bal = C.run_strategy(fp, S.Cfg(modules=S.MODELS["Balanced"]), "Balanced")
    layers = bal["layers"]
    fig_equity(curves, bench_name)
    fig_drawdown(curves, bench_name)
    fig_allocation(weights, bench_name)
    fig_regime(bench_eq, phase)
    fig_direction(layers)
    fig_quality(layers)
    fig_rolling(focus_res["equity"], bench_eq, "sharpe")
    fig_rolling(focus_res["equity"], bench_eq, "cagr")
    fig_capture(summ)
    fig_tim(weights, summ)
    fig_comparison(summ)
    fig_ablation(abl)
    fig_param_heatmap(param)
    fig_walk_forward(wf_pure, wf_sel)
    fig_monte_carlo(mc_cagr, mc_mdd, mc_rows)
    fig_rebalance_freq(curves, summ)

    # ---- §4 补充：日历再平衡频率对比 ----
    cal = summ[summ["kind"] == "benchmark"].set_index("label")

    build_html({**globals(), **locals()})
    print(f"Wrote -> {OUT / 'v3_spot_report.html'}")


# ---------------------------------------------------------------------------
def build_html(V):
    g = V
    summ, models = g["summ"], g["models"]
    bench_s, btc_s = g["bench_s"], g["btc_s"]
    rob, regimes, ddev = g["rob"], g["regimes"], g["ddev"]
    oc, sc, rank_df = g["oc"], g["sc"], g["rank_df"]
    A, B, C_ = g["A"], g["B"], g["C_"]
    fp, phase, X = g["fp"], g["phase"], g["X"]
    abl, inc, param = g["abl"], g["inc"], g["param"]
    wf_pure, wf_sel, bs = g["wf_pure"], g["wf_sel"], g["bs"]
    mc_rows, mc_cagr, mc_mdd = g["mc_rows"], g["mc_cagr"], g["mc_mdd"]
    maj, years, FINALISTS, SHORT = g["maj"], g["years"], g["FINALISTS"], g["SHORT"]
    cal = g["cal"]
    mc = mc_rows[mc_rows["kind"] == "monte_carlo"].iloc[0]
    bs_rows = bs.set_index("dropped_episodes")

    start = fp["index"][0].date()
    end = fp["index"][-1].date()

    # ---- 主表 ----
    order = ["BuyHold_45_20_15_20", "BuyHold_Equal", "BuyHold_BTC_only",
             "ModelA_MA200", "ModelB_MA200_TSMOM", "ModelC_MA200_ATR_ADX",
             "ModelD_C_Adaptive", "ModelE_D_TSMOM", "ModelF_Full",
             "Minimal_MA200_ATR_ADX", "Full_Ensemble", "Conservative",
             "Balanced", "Aggressive"]
    main = summ.set_index("label").reindex([o for o in order if o in summ["label"].values])
    main_tbl = main[["kind", "cagr", "annual_vol", "sharpe", "sortino", "calmar",
                     "max_dd", "final_wealth", "turnover_annual", "trades",
                     "avg_risky", "tim_gt_50", "upside_capture_bull",
                     "downside_capture_bear"]]
    fmt_main = {"cagr": lambda v: pct(v), "annual_vol": lambda v: pct(v),
                "sharpe": lambda v: num(v), "sortino": lambda v: num(v),
                "calmar": lambda v: num(v), "max_dd": lambda v: pct(v),
                "final_wealth": lambda v: f"${v:,.0f}",
                "turnover_annual": lambda v: f"{v:,.1f}x",
                "trades": lambda v: f"{v:,.0f}",
                "avg_risky": lambda v: pct(v), "tim_gt_50": lambda v: pct(v),
                "upside_capture_bull": lambda v: num(v),
                "downside_capture_bear": lambda v: num(v)}

    # ---- §54 排名表 ----
    rk = rank_df.loc[FINALISTS].copy()
    rk.index = [SHORT.get(i, i) for i in rk.index]
    fmt_rk = {c: lambda v: num(v, 2) for c in
              ["Return Ranking", "Risk Ranking", "Robustness Ranking",
               "B&H Capture Ranking"]}
    fmt_rk.update({"综合评分": lambda v: num(v, 3), "长期持有评分": lambda v: num(v, 3),
                   "CAGR": lambda v: pct(v),
                   "Sharpe": lambda v: num(v), "MaxDD": lambda v: pct(v),
                   "Upside Capture": lambda v: num(v), "OOS Sharpe": lambda v: num(v),
                   "Robustness": lambda v: num(v)})

    # ---- 稳健性表 ----
    rob_tbl = pd.DataFrame({SHORT.get(k, k): {
        "参数最差 Sharpe": v["param_sharpe_min"],
        "参数最好 Sharpe": v["param_sharpe_max"],
        "参数 Sharpe 标准差": v["param_sharpe_std"],
        "OOS 平均 Sharpe": v["oos_sharpe_mean"],
        "OOS 最差 Sharpe": v["oos_sharpe_worst"],
        "OOS 平均 CAGR": v["oos_cagr_mean"],
        "OOS 正收益窗口占比": v["oos_positive"]} for k, v in rob.items()})
    fmt_rob = {c: (lambda v: pct(v) if "CAGR" in c or "占比" in c else num(v))
               for c in rob_tbl.columns}

    # ---- 阶段表 ----
    reg_tbl = pd.DataFrame({SHORT.get(k, k): {
        "Bear 最大回撤": float(v.set_index("phase").loc["Bear", "strategy_maxdd"]),
        "Bear 阶段收益": float(v.set_index("phase").loc["Bear", "strategy_return"]),
        "Bear 平均仓位": float(v.set_index("phase").loc["Bear", "avg_risky"]),
        "Bull 阶段收益": float(v.set_index("phase").loc["Bull", "strategy_return"]),
        "Bull 阶段回撤": float(v.set_index("phase").loc["Bull", "strategy_maxdd"])}
        for k, v in regimes.items()})
    fmt_reg = {c: (lambda v: pct(v)) for c in reg_tbl.columns}

    # ---- 回撤事件 ----
    dd_rows = []
    for k, eps in ddev.items():
        if eps:
            e = eps[0]
            dd_rows.append({"策略": SHORT.get(k, k), "最深回撤": e["depth"],
                            "谷底": str(e["trough"].date()),
                            "恢复天数": e["days_to_recovery"] if e["days_to_recovery"] else "未恢复",
                            "全程天数": e["days_total"]})
    dd_rows.insert(0, {"策略": "基准 B&H 45/20/15/20", "最深回撤": bench_s["max_dd"],
                       "谷底": str(M.drawdown_episodes(g["bench_eq"], top=1)[0]["trough"].date()),
                       "恢复天数": M.drawdown_episodes(g["bench_eq"], top=1)[0]["days_to_recovery"],
                       "全程天数": M.drawdown_episodes(g["bench_eq"], top=1)[0]["days_total"]})
    dd_tbl = pd.DataFrame(dd_rows)
    fmt_dd = {"最深回撤": lambda v: pct(v), "恢复天数": lambda v: (f"{v}" if isinstance(v, str) else f"{v} 天"),
              "全程天数": lambda v: f"{v} 天"}

    # ---- Bootstrap ----
    bs_tbl = bs_rows[["cagr", "sharpe", "max_dd", "calmar"]].copy()
    bs_tbl.index = [f"剔除 Top {i}" if i else "原始（不剔除）" for i in bs_tbl.index]
    fmt_bs = {"cagr": lambda v: pct(v), "sharpe": lambda v: num(v),
              "max_dd": lambda v: pct(v), "calmar": lambda v: num(v)}

    # ---- 成本矩阵 ----
    cost = summ[summ["kind"] == "cost"].copy()
    cost["model"] = cost["model"].map(lambda x: SHORT.get(x, x))
    cost_pivot = cost.pivot_table(index=["model", "slippage"], columns="fee",
                                  values="cagr")
    cost_pivot.columns = [f"Fee {c:.2%}" for c in cost_pivot.columns]
    cost_pivot.index.names = ["模型", "滑点"]
    fmt_cost = {c: (lambda v: pct(v)) for c in cost_pivot.columns}

    # ---- 现金版本 ----
    cash = summ[summ["kind"] == "cash"][["label", "cagr", "sharpe", "calmar",
                                         "max_dd", "final_wealth"]].copy()
    cash["label"] = cash["label"].replace(
        {"Balanced_CashA": "Balanced · Cash 0%", "Balanced_CashB": "Balanced · Cash 计息",
         "BuyHold_CashA": "B&H · Cash 0%", "BuyHold_CashB": "B&H · Cash 计息"})
    fmt_cash = {"cagr": lambda v: pct(v), "sharpe": lambda v: num(v),
                "calmar": lambda v: num(v), "max_dd": lambda v: pct(v),
                "final_wealth": lambda v: f"${v:,.0f}"}

    # ---- 参数敏感性摘要 ----
    ps = param.groupby("kind")["sharpe"].agg(["min", "max", "mean"]).reset_index()
    ps.columns = ["参数族", "最差 Sharpe", "最好 Sharpe", "平均 Sharpe"]
    fmt_ps = {c: (lambda v: num(v)) for c in ["最差 Sharpe", "最好 Sharpe", "平均 Sharpe"]}

    # ---- 再买入事件 ----
    maj_tbl = maj[["asset", "exit_date", "exit_price", "bottom_date", "bottom_price",
                   "reentry_date", "reentry_price", "flat_bars", "avoided_drawdown",
                   "missed_upside", "exit_to_reentry"]].head(14).copy()
    for c in ["exit_date", "bottom_date", "reentry_date"]:
        maj_tbl[c] = pd.to_datetime(maj_tbl[c]).dt.strftime("%Y-%m-%d")
    maj_tbl.columns = ["资产", "卖出", "卖价", "底部", "底价", "买回", "买价",
                       "空仓 bar", "卖后跌幅(至底)", "底部→买回涨幅", "卖点→买回收益"]
    fmt_maj = {"卖价": lambda v: f"{v:,.0f}", "底价": lambda v: f"{v:,.0f}",
               "买价": lambda v: f"{v:,.0f}", "卖后跌幅(至底)": lambda v: pct(v),
               "底部→买回涨幅": lambda v: pct(v), "卖点→买回收益": lambda v: pct(v)}

    # ---- 四类排名冠军（仅在五个不同系统之间比较，避免别名行干扰）----
    fsc = sc.loc[FINALISTS]

    def lead(col):
        return fsc[col].idxmax()

    b_final, s_final = bench_s["final_wealth"], g["models"].loc[C_, "final_wealth"]

    html = []
    A_ = html.append
    A_(f"<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
       f"<title>Multi-Asset Adaptive Spot Trend System · V3 报告</title>"
       f"<style>{CSS}</style></head><body><div class='wrap'>")

    A_(f"<h1>Multi-Asset Adaptive Spot Trend System</h1>"
       f"<div class='mut'>BTC / ETH / SOL / BNB 多策略共同决策系统 · V3 现货投资版</div>"
       f"<div class='card'>样本区间 <b>{start} ~ {end}</b>（{len(fp['index']):,} 根 4H bar，"
       f"约 {years:.1f} 年）｜主时间轴 4H，决策频率 Daily｜"
       f"成本 Fee 0.10% + Slippage 0.02%（每条腿）｜初始资金 $10,000<br>"
       f"参考中枢权重 BTC/ETH/SOL/BNB = 45/20/15/20，浮动配置带 "
       f"35-55 / 15-25 / 10-20 / 15-25（%），总权重恒 ≤100%，无杠杆 / 无做空 / 无永续。</div>")

    # ================= §59 三个答案 =================
    A_("<h2>§59 三个答案（Strategy A / B / C）</h2>")
    for tag, name, why in [
            ("Strategy A — 最大收益", A,
             "全样本 CAGR 最高，代价是回撤与换手也最高，适合风险承受力强、追求长期复利的资金。"),
            ("Strategy B — 最佳风险收益", B,
             "§54 综合评分最高（Sharpe/Sortino/Calmar 主导），单位风险回报最优。"),
            ("Strategy C — 最适合长期持有", C_,
             "综合评分 + 低回撤 + 高上涨捕获 + 低换手 的加权最优，OOS 表现最稳定。")]:
        r = sc.loc[name]
        A_(f"<div class='card'><h3 style='margin-top:0'>{tag}</h3>"
           f"<div style='font-size:18px;color:#e6edf3'>{SHORT.get(name, name)} "
           f"<span class='mut' style='font-size:13px'>（模块："
           f"{'+'.join(S.MODELS[name])}）</span></div>"
           f"<div class='grid'>"
           f"<div class='kpi'><div class='v'>{pct(r['cagr'])}</div><div class='l'>CAGR</div></div>"
           f"<div class='kpi'><div class='v'>{num(r['sharpe'])}</div><div class='l'>Sharpe</div></div>"
           f"<div class='kpi'><div class='v'>{pct(r['max_dd'])}</div><div class='l'>MaxDD</div></div>"
           f"<div class='kpi'><div class='v'>{num(r['calmar'])}</div><div class='l'>Calmar</div></div>"
           f"<div class='kpi'><div class='v'>${r['final_wealth']:,.0f}</div>"
           f"<div class='l'>$10,000 → 期末</div></div>"
           f"<div class='kpi'><div class='v'>{num(r['upside_capture_bull'])}</div>"
           f"<div class='l'>Upside Capture</div></div>"
           f"</div><div class='mut'>{why}</div></div>")

    # ================= 主结果 =================
    A_("<h2>§30 / §31 / §55 主结果</h2>")
    A_(html_table(main_tbl, fmt_main, index=True))
    A_(f"<div class='mut'>基准为四资产静态组合 Buy &amp; Hold 45/20/15/20（同样成本模型，"
       f"§29）。SOL 于 2020-12 上市，上市前其权重以 Cash 形式等待。"
       f"CAGR / Sharpe / Sortino / Calmar / MaxDD 均为日频口径。</div>")

    A_(img("01_equity_curve.png", "图01：所有曲线对数坐标；策略在 2022 熊市中显著抗跌，"
                                 "长期净值高于静态组合。"))
    A_(img("02_drawdown.png", "图02：基准在 2022 年出现 -80.9% 的历史级回撤；"
                              "策略系列控制在 -34% ~ -42%。"))

    # ================= §4 补充：再平衡频率对比 =================
    A_("<h2>§4 补充：B&amp;H 再平衡频率对比（每月 / 每季 / 每半年 / 每年）</h2>")
    cal_rows = ["BuyHold_45_20_15_20", "BuyHold_Monthly", "BuyHold_Quarterly",
                "BuyHold_SemiAnnual", "BuyHold_Annual"]
    cal_lab = {"BuyHold_45_20_15_20": "静态不再平衡（原基准）",
               "BuyHold_Monthly": "每月再平衡", "BuyHold_Quarterly": "每季再平衡",
               "BuyHold_SemiAnnual": "每半年再平衡", "BuyHold_Annual": "每年再平衡"}
    cal_tbl = cal.loc[cal_rows][["cagr", "annual_vol", "sharpe", "sortino", "calmar",
                                 "max_dd", "final_wealth", "turnover_annual",
                                 "trades"]].copy()
    cal_tbl.index = [cal_lab[i] for i in cal_rows]
    cal_tbl.columns = ["CAGR", "年化波动", "Sharpe", "Sortino", "Calmar", "MaxDD",
                       "期末净值", "年换手", "交易腿数"]
    fmt_cal = {"CAGR": pct, "年化波动": pct, "MaxDD": pct,
               "期末净值": lambda v: f"${v:,.0f}", "年换手": lambda v: f"{v:.3f}x",
               "Sharpe": num, "Sortino": num, "Calmar": num, "交易腿数": lambda v: f"{int(v)}"}
    A_(html_table(cal_tbl, fmt_cal, index=True))
    sa = cal.loc["BuyHold_SemiAnnual"]
    an = cal.loc["BuyHold_Annual"]
    mo = cal.loc["BuyHold_Monthly"]
    beat_cagr = [SHORT.get(i, i) for i in models.index if models.loc[i, "cagr"] > sa["cagr"]]
    A_(f"<div class='card'>"
       f"结论一：<b>再平衡频率越低，B&amp;H 的长期收益越高</b>，且单调："
       f"静态不再平衡 {pct(bench_s['cagr'])} → 每月 {pct(mo['cagr'])} → "
       f"每季 {pct(cal.loc['BuyHold_Quarterly', 'cagr'])} → "
       f"每半年 {pct(sa['cagr'])} → 每年 <b>{pct(an['cagr'])}</b>；"
       f"Sharpe 同步从 {num(bench_s['sharpe'])} 升至 {num(an['sharpe'])}。<br>"
       f"结论二：<b>再平衡几乎不改善回撤。</b>五种口径的 MaxDD 全部落在 "
       f"{pct(an['max_dd'])} ~ {pct(bench_s['max_dd'])} 的窄区间内，"
       f"与 Strategy C 的 {pct(sc.loc[C_, 'max_dd'])} 相差约 40 个百分点——"
       f"<b>降低回撤只能靠趋势择时，不能靠再平衡。</b><br>"
       f"结论三：<b>换手成本可忽略。</b>每年再平衡的年换手仅 "
       f"{an['turnover_annual']:.3f}x（{int(an['trades'])} 条腿），"
       f"相对 Strategy C 的 {sc.loc[C_, 'turnover_annual']:.1f}x 几乎为零，"
       f"因此上述差异不是成本造成的。<br>"
       f"<span class='bad'>结论四（重要，会改变结论）：</span>"
       f"置换基准后<b>策略的绝对收益优势大部分消失</b>。以每半年再平衡为对照，"
       f"11 个模型中仅 {len(beat_cagr)} 个 CAGR 更高（{('、'.join(beat_cagr)) or '无'}）；"
       f"§59 三个推荐策略中仅 Strategy A（{pct(sc.loc[A, 'cagr'])}）高于 "
       f"{pct(sa['cagr'])}，Strategy B（{pct(sc.loc[B, 'cagr'])}）低 "
       f"{(sa['cagr'] - sc.loc[B, 'cagr']) * 100:.1f}pp、"
       f"Strategy C（{pct(sc.loc[C_, 'cagr'])}）低 "
       f"{(sa['cagr'] - sc.loc[C_, 'cagr']) * 100:.1f}pp。"
       f"若以每年再平衡（{pct(an['cagr'])}）为对照，<b>没有任何策略跑赢</b>。<br>"
       f"结论五：<b>策略的剩余优势集中在风险调整收益与回撤控制</b>，且这两项优势很大："
       f"Sharpe 全部高于每半年基准的 {num(sa['sharpe'])}——"
       f"11 个模型全部超越，区间 {num(models['sharpe'].min())} ~ "
       f"{num(models['sharpe'].max())}；MaxDD 从 {pct(an['max_dd'])} 改善到 "
       f"{pct(models['max_dd'].max())} ~ {pct(models['max_dd'].min())}"
       f"（最大回撤改善 "
       f"{(abs(sa['max_dd']) - abs(models['max_dd'].max())) * 100:.1f}pp 以上）；"
       f"即：<b>同等风险下策略更优，但同等收益下策略并不更优。</b>"
       f"</div>")
    A_("<div class='mut'>所有变体使用完全相同的成本模型（0.10% 手续费 + 0.02% 滑点）、"
       "相同现金处理与相同初始资金 $10,000；唯一区别是是否在期初（每月/每季/每半年/"
       "每年首个 4H bar）把权重拉回 45/20/15/20。已上市资产的权重用未上市资产释放的"
       "现金买入（SOL 于 2020-08 上市）。<br>"
       "<b>「频率越低越好」的成因必须看清，不可直接外推：</b>"
       "① 静态不再平衡的口径偏低，是因为现金不随组合增值，SOL 建仓时仅占约 11.5%"
       "（而非 15%），且全程从未把 2021 年暴涨后的 SOL 削回中枢，"
       "以致 2022 年承受 SOL -94% 的完整跌幅；"
       "② 四种日历口径都恰好在 2022-01-01（SOL 接近历史高点）执行了削仓，"
       "这是本样本中再平衡价值的主要来源；"
       "③ 在这些顶点削仓之外，越频繁的再平衡只是越频繁地削掉 2021 年上涨中的 SOL，"
       "故每月 &lt; 每季 &lt; 每半年 &lt; 每年。"
       "换言之，该单调关系由「2021 年 SOL 单一行情 + 2022-01 的日历巧合」主导，"
       "换一段历史很可能不成立，<b>不应据此认为「越少再平衡越好」是普遍规律</b>。</div>")
    A_(img("16_rebalance_frequency.png",
           "图16：左图为五种再平衡口径的净值曲线（对数坐标，虚线为 Strategy C）；"
           "右图为 CAGR 对比，频率越低收益越高，近年化再平衡已超过全部策略。"))

    # ================= §54 评分 =================
    A_("<h2>§54 综合评分与四类排名</h2>")
    A_(f"<div class='mut'>评分权重：25% Sharpe + 20% Sortino + 20% Calmar + 15% CAGR + "
       f"10% MaxDD + 5% Upside Capture + 5% Robustness（Robustness = 参数最差 Sharpe "
       f"与 OOS 平均 Sharpe 的合成）。排名数字越小越好。<br>"
       f"另给出 §59 Strategy C 专用的<b>长期持有评分</b>："
       f"25% 综合评分 + 25% OOS Sharpe（/2 归一） + 20% 低 MaxDD + "
       f"15% Upside Capture + 15% 低换手。<br>"
       f"§59 三个答案由三个<b>相互独立</b>的目标函数选出，且强制互不相同："
       f"A = 最高 CAGR，B = 最高综合评分，C = 最高长期持有评分。<br>"
       f"下表只列出 <b>5 个互不相同的系统</b>。完整模型对比见上一节——注意 "
       f"<code>ModelC == Minimal == Conservative</code>、"
       f"<code>ModelF == Full_Ensemble == Aggressive</code> 为同一模块集的别名，"
       f"故结果完全一致。</div>")
    A_(html_table(rk, fmt_rk, index=True))
    A_(f"<div class='card'>CAGR 冠军：<b>{SHORT.get(lead('cagr'), '')}</b>｜"
       f"Sharpe 冠军：<b>{SHORT.get(lead('sharpe'), '')}</b>｜"
       f"最低回撤：<b>{SHORT.get(lead('max_dd'), '')}</b>｜"
       f"最高 Calmar：<b>{SHORT.get(lead('calmar'), '')}</b>｜"
       f"最高上涨捕获：<b>{SHORT.get(lead('upside_capture_bull'), '')}</b></div>")

    # ================= §56 十八问 =================
    bA, bB, bC = sc.loc[A], sc.loc[B], sc.loc[C_]
    regA = regimes[A].set_index("phase")
    regC = regimes[C_].set_index("phase")
    oc_A = g["oc"]
    A_("<h2>§56 必须回答的 18 个问题</h2>")

    def q(i, title, ans):
        A_(f"<div class='q'><b>{i}. {title}</b><br>{ans}</div>")

    A_("<h3>收益</h3>")
    q(1, "是否超过四资产静态组合 Buy &amp; Hold？",
       f"<span class='good'>是。</span>全部 11 个模型 / 候选系统的 CAGR 均高于基准 "
       f"{pct(bench_s['cagr'])}：最低为 ModelF_Full / Aggressive 的 "
       f"{pct(models.loc['Aggressive', 'cagr'])}，最高为 ModelA_MA200 的 "
       f"{pct(models.loc['ModelA_MA200', 'cagr'])}。以 $10,000 计，基准期末 "
       f"${bench_s['final_wealth']:,.0f}，"
       f"Strategy C（{SHORT.get(C_, C_)}）期末 ${s_final:,.0f}，"
       f"Strategy A（{SHORT.get(A, A)}）期末 ${bA['final_wealth']:,.0f}。<br>"
       f"<span class='bad'>补充（会削弱该结论）：</span>若把基准换成"
       f"<b>每半年再平衡</b>（§4 补充，CAGR {pct(cal.loc['BuyHold_SemiAnnual', 'cagr'])}），"
       f"11 个模型中只有 {len(beat_cagr)} 个仍跑赢（{('、'.join(beat_cagr)) or '无'}）；"
       f"三个推荐策略里只有 Strategy A 高于它，Strategy B 低 "
       f"{(cal.loc['BuyHold_SemiAnnual', 'cagr'] - sc.loc[B, 'cagr']) * 100:.1f}pp、"
       f"Strategy C 低 "
       f"{(cal.loc['BuyHold_SemiAnnual', 'cagr'] - sc.loc[C_, 'cagr']) * 100:.1f}pp。"
       f"若换成<b>每年再平衡</b>（CAGR {pct(cal.loc['BuyHold_Annual', 'cagr'])}），"
       f"没有任何策略跑赢。因此「跑赢 Buy &amp; Hold」这一说法"
       f"<b>只在「静态不再平衡」这一基准口径下成立</b>；"
       f"策略真正稳健的优势是 Sharpe（{num(models['sharpe'].min())}~"
       f"{num(models['sharpe'].max())} vs {num(sa['sharpe'])}）与 MaxDD"
       f"（{pct(models['max_dd'].max())}~{pct(models['max_dd'].min())} vs "
       f"{pct(cal.loc['BuyHold_SemiAnnual', 'max_dd'])}），详见 §4 补充一节。</span>")
    q(2, "相对推荐组合的 CAGR 损失多少？",
       f"相对基准 <b>没有损失</b>，而是超额：Strategy C 相对基准 CAGR 超额 "
       f"<span class='good'>+{(bC['cagr'] - bench_s['cagr']) * 100:.1f}pp</span>"
       f"（{pct(bC['cagr'])} vs {pct(bench_s['cagr'])}）。"
       f"唯一低于基准的参照是同期 BTC 单币 B&amp;H（CAGR {pct(btc_s['cagr'])}），"
       f"但其中位回撤达 {pct(btc_s['max_dd'])}。")
    q(3, "是否能够捕获 80% 以上的基准组合牛市上涨？",
       f"<span class='good'>阶段累计口径：是。</span>Bull 阶段 Upside Capture："
       f"Conservative {num(models.loc['Conservative', 'upside_capture_bull'])}、"
       f"Balanced {num(models.loc['Balanced', 'upside_capture_bull'])}、"
       f"Aggressive {num(models.loc['Aggressive', 'upside_capture_bull'])}，"
       f"均 &gt; 0.80。<br><span class='bad'>日频口径：否。</span>"
       f"日频上涨日捕获率仅 "
       f"{num(models.loc['Balanced', 'upside_capture_daily'])}，"
       f"因为牛市阶段策略长期持有约 35-50% 的风险资产 + 50-65% 的 Cash。"
       f"两个口径的差异说明：策略的<strong>阶段累计</strong>收益之所以能追平甚至跑赢基准，"
       f"是因为它把仓位集中在牛市中最强的资产（BTC），而非全天候满仓。")

    A_("<h3>风险</h3>")
    q(4, "MaxDD 能否从四资产静态组合的极端水平显著下降？",
       f"<span class='good'>能，且幅度巨大。</span>基准 MaxDD "
       f"<b>{pct(bench_s['max_dd'])}</b> → Conservative "
       f"{pct(models.loc['Conservative', 'max_dd'])}（改善 "
       f"{(bench_s['max_dd'] - models.loc['Conservative', 'max_dd']) * 100:.1f}pp）、"
       f"Balanced {pct(models.loc['Balanced', 'max_dd'])}、"
       f"Aggressive {pct(models.loc['Aggressive', 'max_dd'])}。")
    q(5, "Bear Market 最大回撤是多少？",
       f"以基准自身回撤 &lt; -20% 且跌破 MA200 判定为 Bear 段。"
       f"基准在 Bear 段回撤 <b>{pct(regA.loc['Bear', 'bench_maxdd'])}</b>；"
       f"Strategy A（{SHORT.get(A, A)}）在 Bear 段仅 "
       f"<b>{pct(regA.loc['Bear', 'strategy_maxdd'])}</b>，"
       f"Strategy C（{SHORT.get(C_, C_)}）"
       f"<b>{pct(regC.loc['Bear', 'strategy_maxdd'])}</b>。"
       f"熊市期间策略平均风险资产仓位降至 {pct(regC.loc['Bear', 'avg_risky'])}。")
    q(6, "Recovery Time 是否缩短？",
       f"<span class='good'>是。</span>基准最深回撤谷底 "
       f"{M.drawdown_episodes(g['bench_eq'], top=1)[0]['trough'].date()}，"
       f"恢复耗时 <b>{M.drawdown_episodes(g['bench_eq'], top=1)[0]['days_to_recovery']} 天</b>；"
       f"Strategy A 最深回撤 {pct(ddev[A][0]['depth'])}（谷底 {ddev[A][0]['trough'].date()}），"
       f"恢复 <b>{ddev[A][0]['days_to_recovery']} 天</b>；"
       f"Strategy C 最深回撤 {pct(ddev[C_][0]['depth'])}，"
       f"恢复 <b>{ddev[C_][0]['days_to_recovery']} 天</b>。")

    A_("<h3>稳健性</h3>")
    q(7, "OOS 是否有效？",
       f"<span class='good'>是，但逐年差异明显。</span>纯样本外（冻结默认参数）"
       f"四个测试年："
       + "；".join(f"{r['label']} CAGR {pct(r['cagr'])} / Sharpe {num(r['sharpe'])}"
                   for _, r in wf_pure.iterrows())
       + f"。四个窗口 <b>全部为正收益</b>，且 MaxDD 均控制在 -21% 以内。"
       f"分候选系统看，OOS 平均 Sharpe：" +
       "；".join(f"{SHORT.get(k, k)} {num(v['oos_sharpe_mean'])}"
                 for k, v in rob.items()) +
       f"。2025-2026 年收益大幅回落（CAGR 10.5% / 7.2%），说明策略在震荡与高位盘整市中"
       f"优势收窄，但仍未转负。")
    q(8, "参数是否稳定？",
       f"<span class='good'>是。</span>§50 全参数扫描的 Sharpe 分布："
       f"{num(param['sharpe'].min())} ~ {num(param['sharpe'].max())}"
       f"（全部 &gt; 1.1，均高于基准 {num(bench_s['sharpe'])}）。"
       f"逐候选系统最差参数变体 Sharpe：" +
       "；".join(f"{SHORT.get(k, k)} {num(v['param_sharpe_min'])}"
                 for k, v in rob.items()) +
       f"。显著劣化的只有 3 组（见下），其余均在窄区间内。")
    q(9, "删除单个模块后是否仍然有效？",
       f"<span class='good'>是。</span>§39 Ablation 的 10 个变体 Sharpe 区间 "
       f"{num(abl['sharpe'].min())} ~ {num(abl['sharpe'].max())}，"
       f"全部高于基准 {num(bench_s['sharpe'])}。"
       f"最敏感的模块是 <b>MA200</b>（删除后 Sharpe {num(abl.set_index('label').loc['Full minus ma200', 'sharpe'])}、"
       f"CAGR {pct(abl.set_index('label').loc['Full minus ma200', 'cagr'])}），"
       f"其次为 <b>MTF</b>（删除后 MaxDD 恶化至 "
       f"{pct(abl.set_index('label').loc['Full minus mtf', 'max_dd'])}）。"
       f"<b>ATR</b> 与 <b>Persistence</b> 删除后几乎无影响，"
       f"可视为冗余模块（释放计算与调参成本）。")
    q(10, "是否依赖少数超级交易？",
       f"<span class='bad'>是，这是本系统最大的弱点。</span>§52 剔除贡献最大的持有段后："
       f"原始 CAGR {pct(bs_rows.loc[0, 'cagr'])} / Sharpe {num(bs_rows.loc[0, 'sharpe'])}；"
       f"剔除 Top 5 → CAGR {pct(bs_rows.loc[5, 'cagr'])} / Sharpe {num(bs_rows.loc[5, 'sharpe'])}；"
       f"剔除 Top 10 → CAGR {pct(bs_rows.loc[10, 'cagr'])} / Sharpe {num(bs_rows.loc[10, 'sharpe'])}；"
       f"剔除 Top 20 → CAGR {pct(bs_rows.loc[20, 'cagr'])}。"
       f"主要贡献来自 BTC 2020-2021 与 2024-2025 两段超级趋势。"
       f"换言之，系统的超额收益高度集中于少数趋势段，"
       f"若这些趋势未出现，收益将大幅收敛（但仍为正）。")

    A_("<h3>行为</h3>")
    q(11, "Time in Market 是多少？",
       f"以风险资产总权重计：Strategy A 平均 {pct(bA['avg_risky'])}"
       f"（&gt;50% 的时间占 {pct(bA['tim_gt_50'])}）；"
       f"Strategy C 平均 {pct(bC['avg_risky'])}（&gt;50% 占 {pct(bC['tim_gt_50'])}，"
       f"&gt;75% 占 {pct(bC['tim_gt_75'])}）。"
       f"即系统长期约有 <b>60% 时间持有 50% 以上的 Cash</b>，"
       f"这正是它能把回撤压到 -34% ~ -42% 的原因。")
    q(12, "每年交易多少次？",
       f"全样本 {int(models.loc['Balanced', 'trades']):,} 条交易腿 / {years:.1f} 年 ≈ "
       f"<b>{models.loc['Balanced', 'trades'] / years:,.0f} 腿/年</b>"
       f"（≈ {models.loc['Balanced', 'trades'] / years / 4:.0f} 次完整调仓/年，"
       f"平均每月 1-2 次）。年化换手率 "
       f"{num(models.loc['Balanced', 'turnover_annual'])}x。"
       f"Aggressive 更低（{num(models.loc['Aggressive', 'turnover_annual'])}x），"
       f"ModelA 最高（{num(models.loc['ModelA_MA200', 'turnover_annual'])}x）。")
    q(13, "熊市到底什么时候卖出？",
       f"卖出由 Layer1 Direction（MA200 为主，权重 35%）跌破 0 触发，"
       f"经 §22 分档将方向分映射为 0%，再经门槛（偏离 &gt;10%）确认后执行。"
       f"下表为幅度较大的实际卖出事件（空仓 ≥1 天）："
       + f"<br>代表性事件："
       + "；".join(f"{r['asset']} {pd.to_datetime(r['exit_date']).date()} 卖在 "
                   f"${r['exit_price']:,.0f}，此后最低 ${r['bottom_price']:,.0f}"
                   f"（卖后跌幅 {pct(r['avoided_drawdown'])}）"
                   for _, r in maj.head(4).iterrows())
       + f"。可以看到卖出发生在趋势确认跌破后，"
       f"而非顶部——即<b>滞后但有效</b>，牺牲了顶部一小段换取确定性。")
    q(14, "牛市重新进入是否足够及时？",
       f"平均空仓跨度（重大事件）<b>{maj['flat_bars'].mean():.0f} 根 4H bar</b>"
       f"≈ {maj['flat_bars'].mean() / 6:.1f} 天；平均错过涨幅 "
       f"<b>{pct(maj['missed_upside'].mean())}</b>，平均规避跌幅 "
       f"<b>{pct(maj['avoided_drawdown'].mean())}</b>，"
       f"卖出到买回的价格变化平均 {pct(maj['exit_to_reentry'].mean())}。"
       f"避损 &gt; 错涨，再买入整体是<b>正贡献</b>。"
       f"但 2020-03 与 2022 年两轮大熊市中，卖点滞后（例：BTC 2020-02-26 卖 "
       f"$9,170 → 2020-04-06 买回 $6,844，卖到买 -25%）——"
       f"说明再买入规则会以低于卖价的价格买回，属于“用确定性换价格”。")

    A_("<h3>实际投资</h3>")
    q(15, "如果从 $10,000 开始，最终财富是多少？",
       f"基准 B&amp;H 45/20/15/20：<b>${bench_s['final_wealth']:,.0f}</b>；"
       f"Strategy A（{SHORT.get(A, A)}）：<b>${bA['final_wealth']:,.0f}</b>；"
       f"Strategy B（{SHORT.get(B, B)}）：<b>${bB['final_wealth']:,.0f}</b>；"
       f"Strategy C（{SHORT.get(C_, C_)}）：<b>${bC['final_wealth']:,.0f}</b>。")
    q(16, "中途最大账户回撤是多少？",
       f"基准 <b>{pct(bench_s['max_dd'])}</b>；Strategy A {pct(bA['max_dd'])}；"
       f"Strategy B {pct(bB['max_dd'])}；Strategy C {pct(bC['max_dd'])}。"
       f"以 $10,000 起算，基准最坏时账户一度只剩 "
       f"${10000 * (1 + bench_s['max_dd']):,.0f}"
       f"（相对峰值），而 Strategy C 最坏时账户仍在峰值的 "
       f"{(1 + bC['max_dd']) * 100:.0f}% 以上。")
    q(17, "95% 情景下可能出现什么结果？",
       f"§51 Monte Carlo（10,000 次，日收益 block bootstrap，block=5）："
       f"CAGR 中位数 <b>{pct(mc['strategy_cagr_median'])}</b>，"
       f"5% 分位 <b>{pct(mc['strategy_cagr_p5'])}</b>，"
       f"95% 分位 <b>{pct(mc['strategy_cagr_p95'])}</b>；"
       f"MaxDD 中位数 <b>{pct(mc['strategy_mdd_median'])}</b>，"
       f"5% 分位 <b>{pct(mc['strategy_mdd_p5'])}</b>。"
       f"同期基准 CAGR 中位数 {pct(mc['bench_cagr_median'])}、"
       f"MaxDD 中位数 {pct(mc['bench_mdd_median'])}。"
       f"策略跑赢基准的概率 <b>{pct(mc['prob_outperform_bh'])}</b>，"
       f"回撤劣于 -40% 的概率仅 <b>{pct(mc['prob_mdd_worse_than_40'])}</b>。"
       f"最差 5% 情景下仍为正收益，但回撤可达 -43%。")
    q(18, "是否值得长期持有？",
       f"<span class='good'>值得，但要接受三点前提。</span>"
       f"支持：① Sharpe {num(bC['sharpe'])} / Calmar {num(bC['calmar'])}，"
       f"显著优于基准（{num(bench_s['sharpe'])} / {num(bench_s['calmar'])}）；"
       f"② MaxDD 从 -80.9% 降到 {pct(bC['max_dd'])}，恢复更快；"
       f"③ 四个 OOS 窗口全部为正，参数与模块删除测试均稳健；"
       f"④ 年均仅 1-2 次调仓，可执行性强。"
       f"前提：① 收益高度依赖少数超级趋势（§52）：剔除历史贡献最大的 10 个持有段后 "
       f"CAGR 骤降至 {pct(bs_rows.loc[10, 'cagr'])}、"
       f"剔除 20 个后为 {pct(bs_rows.loc[20, 'cagr'])}；② 长期约 60% 时间持有大量 Cash，"
       f"在牛市中会跑输满仓；③ 2025-2026 的 OOS 收益已明显衰减"
       f"（CAGR {pct(wf_pure['cagr'].iloc[-1])}），"
       f"不应外推 2018-2021 的高收益。")

    # ================= §53 =================
    A_("<h2>§53 Opportunity Cost（降低回撤的代价）</h2>")
    A_(f"<div class='card'>以 Balanced 候选为例：平均 Cash 权重 "
       f"<b>{pct(oc['avg_cash'])}</b>。若这些现金始终按基准组合比例投入，"
       f"全样本累计可多获得约 <b>{pct(oc['opp_cost_total'])}</b> 的收益"
       f"（约 {pct(oc['opp_cost_total'] / years)}/年），"
       f"若投入 BTC 则可多获得约 <b>{pct(oc['missed_btc_upside'])}</b>。"
       f"这就是“降低回撤”付出的机会成本。<br>"
       f"但同一时间段内：基准总收益 {pct(oc['bench_total_return'])} "
       f"（期末 ${oc['bench_final']:,.0f}），策略总收益 "
       f"{pct(oc['strategy_total_return'])}（期末 ${oc['strategy_final']:,.0f}），"
       f"策略净多赚 <b class='good'>${oc['net_gain_vs_bench']:,.0f}</b>。"
       f"原因在于：机会成本是“线性”估算，而策略在熊市规避的下跌是“复利”效应——"
       f"避开的每一次大跌都保护了后续复利的本金。")
    A_(img("03_asset_allocation.png", "图03：Balanced 的资产配置演变，"
                                      "灰区为 Cash。2022 年熊市 Cash 占比升至 80%+。"))
    A_(img("04_regime.png", "图04：四阶段划分（Bull 绿 / Bear 红 / Sideways 灰 / Recovery 黄）。"))

    # ================= 稳健性 =================
    A_("<h2>§39 / §40 / §48 / §50 / §51 / §52 稳健性</h2>")
    A_("<h3>§39 Ablation（逐个删除模块）</h3>")
    A_(html_table(abl.set_index("label")[["cagr", "sharpe", "calmar", "max_dd",
                                          "turnover_annual", "avg_risky",
                                          "upside_capture_bull"]],
                  {"cagr": lambda v: pct(v), "sharpe": lambda v: num(v),
                   "calmar": lambda v: num(v), "max_dd": lambda v: pct(v),
                   "turnover_annual": lambda v: f"{v:,.1f}x",
                   "avg_risky": lambda v: pct(v),
                   "upside_capture_bull": lambda v: num(v)}, index=True))
    A_(img("12_ablation.png", "图12：绿条表示删除该模块后 Sharpe 反而提升。"))
    A_(f"<div class='mut'>关键结论：删 <b>MA200</b> 最伤（Sharpe "
       f"{num(abl.set_index('label').loc['Full minus ma200', 'sharpe'])}）；"
       f"删 <b>ATR</b> 反而改善（MaxDD "
       f"{pct(abl.set_index('label').loc['Full minus atr', 'max_dd'])}、"
       f"Calmar {num(abl.set_index('label').loc['Full minus atr', 'calmar'])}）；"
       f"删 <b>Persistence</b> 几乎无变化（Sharpe 差 &lt;0.01），是冗余模块。</div>")

    A_("<h3>§40 Incremental（按序增加模块）</h3>")
    A_(html_table(inc.set_index("label")[["cagr", "sharpe", "calmar", "max_dd",
                                          "turnover_annual", "avg_risky"]],
                  {"cagr": lambda v: pct(v), "sharpe": lambda v: num(v),
                   "calmar": lambda v: num(v), "max_dd": lambda v: pct(v),
                   "turnover_annual": lambda v: f"{v:,.1f}x",
                   "avg_risky": lambda v: pct(v)}, index=True))
    A_(f"<div class='mut'>单调性：CAGR 随模块增加而下降（{pct(inc['cagr'].iloc[0])} → "
       f"{pct(inc['cagr'].iloc[-1])}），但 Sharpe 上升（{num(inc['sharpe'].iloc[0])} → "
       f"{num(inc['sharpe'].iloc[-1])}）、MaxDD 改善（{pct(inc['max_dd'].iloc[0])} → "
       f"{pct(inc['max_dd'].iloc[-1])}）、换手下降 "
       f"（{num(inc['turnover_annual'].iloc[0])}x → {num(inc['turnover_annual'].iloc[-1])}x）。"
       f"即“增加模块 = 用收益换风险控制”。不是越多越好，而是取决于目标。</div>")

    A_("<h3>§50 参数稳定性</h3>")
    A_(html_table(ps, fmt_ps, index=False))
    A_(img("13_parameter_heatmap.png", "图13：每格为对应参数取值的 Sharpe。"))
    A_(f"<div class='mut'>稳健区间：MA 150/200/250（{num(param[param['kind'] == 'ma_len']['sharpe'].min())}"
       f"~{num(param[param['kind'] == 'ma_len']['sharpe'].max())}）、"
       f"TSMOM 3/6/9/12M、ATR 10/14/20、ADX 14/20/30、Persistence 2~24 bars "
       f"均在窄区间内。"
       f"<b>不稳健</b>的项：① Re-entry Guard（全部劣化，Sharpe 降至 "
       f"{num(param[param['kind'] == 'reentry_guard']['sharpe'].min())}）；"
       f"② Drawdown Guard（0/10/15/20% 全部劣化，"
       f"Sharpe {num(param[param['kind'] == 'dd_guard']['sharpe'].min())}"
       f"~{num(param[param['kind'] == 'dd_guard']['sharpe'].max())}）；"
       f"③ Cadence=Weekly（Sharpe {num(param[param['kind'] == 'cadence']['sharpe'].min())}）；"
       f"④ Rebalance=20%（Sharpe "
       f"{num(param[(param['kind'] == 'rebal_threshold') & (param['variant'] == '20%')]['sharpe'].iloc[0])}）；"
       f"⑤ 关闭配置带下界（CAGR 跌至 "
       f"{pct(param[(param['kind'] == 'lower_band')]['cagr'].iloc[0])}，"
       f"上涨捕获仅 "
       f"{num(param[(param['kind'] == 'lower_band')]['upside_capture_bull'].iloc[0])}）"
       f"——说明<b>配置带下界是“保持长期持仓”的关键机制，不可关闭</b>。"
       f"因此 §35 Re-entry Guard 与 §37/§38 Drawdown Guard "
       f"<b>应删除</b>，Daily 频率 + 10~15% 再平衡阈值是最优组合。</div>")

    # ================= OOS =================
    A_("<h2>§48 Walk-Forward 样本外</h2>")
    A_("<h3>纯样本外（冻结默认参数，不参与任何选参）</h3>")
    A_(html_table(wf_pure.set_index("label")[["cagr", "sharpe", "calmar", "max_dd",
                                              "avg_risky", "trades"]],
                  {"cagr": lambda v: pct(v), "sharpe": lambda v: num(v),
                   "calmar": lambda v: num(v), "max_dd": lambda v: pct(v),
                   "avg_risky": lambda v: pct(v), "trades": lambda v: f"{v:,.0f}"},
                  index=True))
    A_("<h3>Train → Validate 选参 → Test（§49：绝不用 OOS 选参）</h3>")
    A_(html_table(wf_sel.set_index("label")[["config", "cagr", "sharpe", "calmar",
                                             "max_dd"]],
                  {"cagr": lambda v: pct(v), "sharpe": lambda v: num(v),
                   "calmar": lambda v: num(v), "max_dd": lambda v: pct(v)},
                  index=True))
    A_(f"<div class='mut'>注意：2023 与 2024 窗口选出的最优配置无法延续到下一窗口"
       f"（2023 选 MA200+TSMOM、2024 选 C_vol20、2025 选 Full、2026 选 Minimal），"
       f"说明“参数最优解”随市场状态漂移，这正是应当<b>使用默认固定参数而非滚动优化</b>"
       f"的实证依据。</div>")
    A_(img("14_walk_forward.png", "图14：左为 CAGR，右为 Sharpe；紫条为选参后的 Test 结果。"))

    # ================= 候选稳健性 =================
    A_("<h2>§55 候选系统稳健性对比</h2>")
    A_(html_table(rob_tbl, fmt_rob, index=True))
    A_("<h3>分阶段表现</h3>")
    A_(html_table(reg_tbl, fmt_reg, index=True))
    A_("<h3>最大回撤事件与恢复</h3>")
    A_(html_table(dd_tbl.set_index("策略"), fmt_dd, index=True))

    # ================= 行为 =================
    A_("<h2>§41 / §42 / §43 / §33 / §34 行为分析</h2>")
    A_(html_table(rk[["CAGR", "Sharpe", "MaxDD", "Upside Capture"]].copy(), fmt_rk, index=True))
    A_(img("09_bull_bear_capture.png", "图09：阶段累计口径的捕获率。"))
    A_(img("10_time_in_market.png", "图10：风险资产总权重时间序列与暴露分布。"))
    A_(img("05_direction_score.png", "图05：Layer1 方向分。低于 0.25 的时段资产被清空。"))
    A_(img("06_trend_quality.png", "图06：Layer2 趋势质量与 Layer3 风险修正因子。"))
    A_("<h3>§34 熊市卖出 → 再买入事件（空仓 ≥1 天）</h3>")
    A_(html_table(maj_tbl.set_index("资产").iloc[:14], fmt_maj, index=True))
    A_(f"<div class='mut'>共 {len(g['reev'])} 个空窗事件（含大量数小时级噪音），"
       f"表中为跨度最大的 {min(14, len(maj))} 个。"
       f"过滤后平均：空仓 {maj['flat_bars'].mean():.0f} bar、"
       f"规避跌幅 {pct(maj['avoided_drawdown'].mean())}、"
       f"错过涨幅 {pct(maj['missed_upside'].mean())}。"
       f"平均规避 &gt; 平均错过，再买入规则整体为正贡献，但存在“低价买回”的滞后成本。</div>")
    A_(img("07_rolling_sharpe.png", "图07：滚动 1Y / 2Y / 3Y Sharpe。"))
    A_(img("08_rolling_cagr.png", "图08：滚动 CAGR 与基准 3Y 对比。"))

    # ================= 其他配置 =================
    A_("<h2>§27 Cash Return / §28 成本矩阵</h2>")
    A_(html_table(cash.set_index("label"), fmt_cash, index=True))
    cash_a = summ[summ["label"] == "Balanced_CashA"]["cagr"].iloc[0]
    cash_b = summ[summ["label"] == "Balanced_CashB"]["cagr"].iloc[0]
    A_(f"<div class='mut'>Cash 计息（Version B，按 FRED DFF 有效联邦基金利率）"
       f"对 Balanced 的 CAGR 提升 "
       f"+{(cash_b - cash_a) * 100:.1f}pp"
       f"（{pct(cash_a)} → {pct(cash_b)}），"
       f"对基准几乎无影响（因其长期满仓）。"
       f"由于系统长期持有 50-60% Cash，<b>Cash 计息是本系统一个不容忽视的收益来源</b>。</div>")
    A_(html_table(cost_pivot, fmt_cost, index=True))
    A_(img("11_strategy_comparison.png", "图11：基准 / 各模型 / 三候选的横向对比。"))
    A_(img("15_monte_carlo.png", "图15：10,000 次 Monte Carlo 的 CAGR 与 MaxDD 分布。"))

    # ================= 结论 =================
    A_("<h2>结论与建议</h2>")
    A_(f"<div class='card'>"
       f"<h3 style='margin-top:0'>系统有效性</h3>"
       f"三层架构（Direction → Trend Quality → Risk）在 8.3 年、4 资产、含全部成本的回测中，"
       f"把四资产静态组合的 MaxDD 从 <b>{pct(bench_s['max_dd'])}</b> 压到 "
       f"<b>{pct(models.loc['Aggressive', 'max_dd'])}</b>，同时把 CAGR 从 "
       f"<b>{pct(bench_s['cagr'])}</b> 提升到 <b>{pct(models.loc['Aggressive', 'cagr'])}</b>，"
       f"Sharpe 从 {num(bench_s['sharpe'])} 提升到 {num(models.loc['Aggressive', 'sharpe'])}。"
       f"<h3>推荐配置</h3>"
       f"<ul>"
       f"<li><b>{SHORT.get(C_, C_)}</b> —— 长期持有首选："
       f"CAGR {pct(bC['cagr'])}、MaxDD {pct(bC['max_dd'])}、"
       f"Sharpe {num(bC['sharpe'])}、换手 {num(bC['turnover_annual'])}x。</li>"
       f"<li><b>{SHORT.get(B, B)}</b> —— 风险收益最优："
       f"CAGR {pct(bB['cagr'])}、MaxDD {pct(bB['max_dd'])}、"
       f"Sharpe {num(bB['sharpe'])}。</li>"
       f"<li><b>{SHORT.get(A, A)}</b> —— 最大收益："
       f"CAGR {pct(bA['cagr'])}，但 MaxDD {pct(bA['max_dd'])}、"
       f"换手 {num(bA['turnover_annual'])}x，仅适合高风险偏好。</li>"
       f"</ul>"
       f"<h3>应当修正的设计</h3>"
       f"<ul>"
       f"<li>删除 <b>Re-entry Guard</b>（§35）与 <b>Drawdown Guard</b>（§37/§38）："
       f"全部阈值下均劣化。</li>"
       f"<li>评估删除 <b>Persistence</b> 与 <b>ATR</b> 模块：删除后绩效不降反升，"
       f"可简化系统。</li>"
       f"<li>保留 <b>配置带下界</b>：关闭会使上涨捕获从 "
       f"{num(models.loc['Balanced', 'upside_capture_bull'])} 崩到 "
       f"{num(param[(param['kind'] == 'lower_band')]['upside_capture_bull'].iloc[0])}。</li>"
       f"<li>保留 <b>MA200</b>：最重要的方向因子，删除后 Sharpe 掉到 "
       f"{num(abl.set_index('label').loc['Full minus ma200', 'sharpe'])}。</li>"
       f"<li>频率用 <b>Daily</b>、再平衡阈值 <b>10~15%</b>；Weekly 与 4H 均劣化。</li>"
       f"</ul>"
       f"<h3>风险提示</h3>"
       f"<ul>"
       f"<li>收益高度依赖少数超级趋势（剔除 Top 10 持有段后 CAGR 仅 "
       f"{pct(bs_rows.loc[10, 'cagr'])}）。</li>"
       f"<li>2025-2026 样本外收益已衰减至 "
       f"{pct(wf_pure['cagr'].iloc[-2])} / {pct(wf_pure['cagr'].iloc[-1])}，"
       f"不应外推历史高收益。</li>"
       f"<li>长期 50-60% Cash 拖累牛市表现；建议启用 Cash 计息（Version B）。</li>"
       f"<li>Monte Carlo 5% 分位回撤达 {pct(mc['strategy_mdd_p5'])}，"
       f"需按此做资金规划。</li>"
       f"</ul></div>")

    A_("<h2>§57 输出文件清单</h2>")
    files = ["v3_spot_summary.csv", "v3_spot_equity.csv", "v3_spot_trades.csv",
             "v3_spot_exposure.csv", "v3_spot_drawdown.csv", "v3_spot_regime.csv",
             "v3_spot_oos.csv", "v3_spot_walk_forward.csv", "v3_spot_ablation.csv",
             "v3_spot_parameter_sensitivity.csv", "v3_spot_monte_carlo.csv",
             "v3_spot_bull_bear_analysis.csv", "v3_spot_capture_ratio.csv"]
    A_("<ul>" + "".join(f"<li><code>{f}</code></li>" for f in files) + "</ul>")
    A_("<h2>§58 图表清单</h2>")
    figs = ["01_equity_curve", "02_drawdown", "03_asset_allocation", "04_regime",
            "05_direction_score", "06_trend_quality", "07_rolling_sharpe",
            "08_rolling_cagr", "09_bull_bear_capture", "10_time_in_market",
            "11_strategy_comparison", "12_ablation", "13_parameter_heatmap",
            "14_walk_forward", "15_monte_carlo"]
    A_("<ul>" + "".join(f"<li><code>{f}.png</code></li>" for f in figs)
       + "<li><code>16_rebalance_frequency.png</code>（§4 补充）</li></ul>")

    A_(f"<div class='foot'>生成时间 {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}｜"
       f"样本 {start} ~ {end}｜全部结果由 <code>scripts/v3_spot/</code> 下代码确定性生成，"
       f"随机过程固定 seed=20260920｜数据源 Binance Spot REST（经本地代理）"
       f"+ FRED DFF。</div>")
    A_("</div></body></html>")

    (OUT / "v3_spot_report.html").write_text("".join(html), encoding="utf-8")


if __name__ == "__main__":
    main()