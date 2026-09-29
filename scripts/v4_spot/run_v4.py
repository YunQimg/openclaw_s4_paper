# -*- coding: utf-8 -*-
"""V4 —— 主编排：基准修复 -> 策略族 -> 公平比较 -> 稳健性 -> 输出（Roadmap §20 Phase 0-9）

运行：python -m scripts.v4_spot.run_v4
输出：reports/v4_spot_v2/v4_*.csv + results_v4.pkl
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.v4_spot import benchmarks as B          # noqa: E402
from scripts.v4_spot import data as D                # noqa: E402
from scripts.v4_spot import engine as E              # noqa: E402
from scripts.v4_spot import metrics as M             # noqa: E402
from scripts.v4_spot import strategies as S          # noqa: E402
from scripts.v4_spot.config import (ASSETS, BAND, BENCHMARK_IDS, COST_MATRIX,  # noqa: E402
                                    DEFAULT_COST, HYBRIDS, LEGACY_BAND,
                                    LEGACY_REF45, LISTING_RULES, MA_LENS,
                                    REF_WEIGHT, REBAL_FREQS_SENS,
                                    RISK_MULTIPLIERS, SEED, STRATEGY_IDS,
                                    THRESHOLDS, TSMOM_LENS, RebalanceSpec,
                                    V4Cfg, config_fingerprint)

OUT = REPO / "reports" / "v4_spot_v2"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                cash_return="Cash_Rate", band_mode="Band",
                listing_rule="L1", cadence="daily", rebal_threshold=0.10)
FOCUS = "S5"                       # 明细输出对象
SIMPLICITY = {"S0": 1.00, "S1": 0.90, "S2": 0.80, "S3": 0.85,
              "S4": 0.70, "S5": 0.60, "S6": 0.50, "S7": 0.30}

WF_WINDOWS: List[Tuple[str, str, str, str]] = [
    ("2018-06-01", "2021-12-31", "2022-12-31", "2023-12-31"),
    ("2019-01-01", "2022-12-31", "2023-12-31", "2024-12-31"),
    ("2020-01-01", "2023-12-31", "2024-12-31", "2025-12-31"),
    ("2021-01-01", "2024-12-31", "2025-12-31", "2026-09-20"),
]

_log_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _log_t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 运行缓存
# ---------------------------------------------------------------------------
class Runner:
    """按 (系统, 配置) 缓存回测结果，避免重复仿真。"""

    def __init__(self, fp: dict):
        self.fp = fp
        self._cache: Dict[str, dict] = {}
        self._cr: Dict[str, pd.Series] = {}

    def cash(self, mode: str) -> pd.Series:
        if mode not in self._cr:
            self._cr[mode] = D.cash_rate_series(self.fp["index"], mode)
        return self._cr[mode]

    def run(self, sys_id: str, cfg: V4Cfg,
            spec: Optional[RebalanceSpec] = None,
            ref_weight: Optional[Dict[str, float]] = None,
            freq_override: Optional[str] = None,
            base: Optional[str] = None) -> dict:
        fam = base or sys_id.split("_")[0].split("@")[0]
        rw = ref_weight or cfg.ref_weight
        c = cfg if ref_weight is None else V4Cfg(**{**cfg.__dict__, "ref_weight": rw})
        bo = json.dumps(c.band_override, sort_keys=True) if c.band_override else ""
        key = (f"{sys_id}|{c.tag()}|{spec.key() if spec else ''}|{freq_override}"
               f"|{sorted(rw.items())}|band={bo}")
        if key in self._cache:
            return self._cache[key]
        if fam in STRATEGY_IDS or fam in S.V3_MODULES:
            if fam == "S0":
                tv, sp = S.build(self.fp, c, "S0")
            elif fam == "S1":
                tv, sp = S.targets_s1(self.fp, c)
            elif fam == "S2":
                tv, sp = S.targets_s2(self.fp, c, freq=freq_override)
            else:
                tv, sp = S.targets_v3(self.fp, c, fam)
            spec = spec or sp
        else:
            tv = B.constant_targets(self.fp, rw)
            spec = spec or B.default_specs()[fam]
        res = E.simulate(self.fp, tv, c, spec, cash_rate=self.cash(c.cash_return), label=sys_id)
        self._cache[key] = res
        return res


# ---------------------------------------------------------------------------
# §11.1 分段
# ---------------------------------------------------------------------------
SEGMENTS = ["full", "Bull", "Bear", "Sideways", "Recovery"]


def segment_masks(phase: pd.Series) -> Dict[str, pd.Index]:
    out = {"full": phase.index}
    for s in SEGMENTS[1:]:
        out[s] = phase.index[phase == s]
    return out


def slice_res(res: dict, idx: pd.Index) -> dict:
    return {"equity": res["equity"].reindex(idx).dropna(),
            "weights": res["weights"].reindex(idx).dropna(),
            "trades": res.get("trades", pd.DataFrame()),
            "turnover_annual": res.get("turnover_annual", 0.0),
            "turnover_total": res.get("turnover_total", 0.0),
            "n_trades": res.get("n_trades", 0),
            "n_rebalances": res.get("n_rebalances", 0),
            "spec": res.get("spec", ""), "cfg_tag": res.get("cfg_tag", "")}


def seg_stats(eq: pd.Series, idx: pd.Index) -> dict:
    s = eq.reindex(idx).dropna()
    if len(s) < 3:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "calmar": np.nan,
                "ret": np.nan}
    d = M.to_daily(s)
    yrs = max((d.index[-1] - d.index[0]).days, 1) / 365.25
    r = d.pct_change().dropna()
    cagr = (d.iloc[-1] / d.iloc[0]) ** (1 / yrs) - 1.0 if d.iloc[0] > 0 else np.nan
    mdd = M.max_drawdown(d)
    return {"cagr": float(cagr),
            "sharpe": float(r.mean() / r.std() * M.ANN) if r.std() > 1e-12 else 0.0,
            "max_dd": float(mdd),
            "calmar": float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0,
            "ret": float(d.iloc[-1] / d.iloc[0] - 1.0)}


# ---------------------------------------------------------------------------
# Phase 0：配置与数据冻结
# ---------------------------------------------------------------------------
def phase0(fp: dict) -> dict:
    log("Phase 0 冻结配置 / 数据质量 / 上市过渡")
    dq = D.data_quality(fp)
    dq.to_csv(OUT / "v4_data_quality.csv", index=False)
    lt = D.listing_transition(fp)
    lt["cash_reserve_weight"] = [
        float(REF_WEIGHT[a]) if not fp["tradable"][a].all() else 0.0 for a in ASSETS]
    lt["listed_bars_after_entry_L1"] = [
        int(fp["tradable"][a].sum()) for a in ASSETS]
    lt.to_csv(OUT / "v4_listing_transition.csv", index=False)
    snap = {"primary_config": PRIMARY.to_dict(),
            "cost_matrix": [list(c) for c in COST_MATRIX],
            "listing_rules": LISTING_RULES, "thresholds": THRESHOLDS,
            "hybrids": [list(h) for h in HYBRIDS],
            "sample_start": str(fp["index"][0]), "sample_end": str(fp["index"][-1]),
            "n_bars_4h": len(fp["index"]), "seed": SEED,
            "ref_weight": REF_WEIGHT, "band_mode_supported": ["Band", "No-Band"],
            "cash_modes": ["Cash_0", "Cash_Rate"], "assets": ASSETS}
    (OUT / "v4_config_snapshot.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "v4_reproducibility.json").write_text(json.dumps(
        {"config_fingerprint_sha256": config_fingerprint(PRIMARY), "seed": SEED},
        indent=2), encoding="utf-8")
    return {"data_quality": dq, "listing": lt, "snapshot": snap}


# ---------------------------------------------------------------------------
# Phase 2-3：基准族
# ---------------------------------------------------------------------------
def phase_benchmarks(R: Runner) -> Tuple[pd.DataFrame, Dict[str, dict], pd.DataFrame]:
    log("Phase 2-3 基准族 B0-B6 + 成本/现金矩阵 + 审计")
    fp = R.fp
    rows, default_runs = [], {}
    for bid, spec in B.default_specs().items():
        for fee, slip in COST_MATRIX:
            for cash in ["Cash_0", "Cash_Rate"]:
                cfg = V4Cfg(fee=fee, slippage=slip, cash_return=cash,
                            band_mode="Band", listing_rule="L1")
                res = R.run(bid, cfg, spec)
                if (fee, slip) == DEFAULT_COST and cash == PRIMARY.cash_return:
                    default_runs[bid] = res
                row = M.metrics_row(bid, "benchmark", res)
                row.update({"fee": fee, "slippage": slip, "cash_mode": cash,
                            "rebalance_freq": spec.freq or "never",
                            "threshold": spec.threshold})
                rows.append(row)
    # 上市规则敏感性（L1/L2/L3）
    for rule in LISTING_RULES:
        for bid in ["B0", "B4"]:
            cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                        cash_return=PRIMARY.cash_return, listing_rule=rule)
            res = R.run(bid, cfg, B.default_specs()[bid])
            row = M.metrics_row(bid, "benchmark_listing", res)
            row.update({"listing_rule": rule, "cash_mode": cfg.cash_return})
            rows.append(row)
    # B5/B6 完整档位
    for bid, spec in B.extended_specs().items():
        for cash in ["Cash_0", "Cash_Rate"]:
            cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1], cash_return=cash)
            res = R.run(bid, cfg, spec)
            row = M.metrics_row(bid, "benchmark_extended", res)
            row.update({"cash_mode": cash, "basis": spec.basis,
                        "threshold": spec.threshold, "rebalance_freq": spec.freq or ""})
            rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "v4_benchmark_summary.csv", index=False)

    bench = default_runs["B0"]
    phase = M.classify_phases(bench["equity"])
    masks = segment_masks(phase)

    eq_df = pd.DataFrame({k: M.to_daily(v["equity"]) for k, v in default_runs.items()})
    eq_df.index.name = "timestamp"
    eq_df.to_csv(OUT / "v4_benchmark_equity.csv")

    wdf = {k: v["weights"].resample("1D").last() for k, v in default_runs.items()}
    w_out = pd.concat(wdf, axis=1)
    w_out.columns = [f"{b}_{c}" for b, c in w_out.columns]
    w_out.index.name = "timestamp"
    w_out.to_csv(OUT / "v4_benchmark_weights.csv")

    tr_out = []
    for k, v in default_runs.items():
        t = v["trades"]
        if len(t):
            t = t.copy()
            t.insert(0, "system", k)
            tr_out.append(t)
    pd.concat(tr_out).to_csv(OUT / "v4_benchmark_trades.csv", index=False)

    ev_out = []
    for k, v in default_runs.items():
        e = v["events"]
        if len(e):
            ev_out.append(e)
    pd.concat(ev_out).to_csv(OUT / "v4_benchmark_rebalance_events.csv", index=False)

    inv = []
    for k, v in default_runs.items():
        d = D.weight_invariants(v["weights"])
        d["system"] = k
        inv.append(d)
    pd.DataFrame(inv).to_csv(OUT / "v4_weight_invariants.csv", index=False)
    return summ, default_runs, phase


# ---------------------------------------------------------------------------
# Phase 5：策略族
# ---------------------------------------------------------------------------
def phase_strategies(R: Runner, bench_eq: pd.Series, phase: pd.Series
                     ) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    log("Phase 5 策略族 S0-S7 + 成本/现金/配置带矩阵")
    rows, default_runs = [], {}
    for sid in STRATEGY_IDS:
        for band in ["Band", "No-Band"]:
            for fee, slip in COST_MATRIX:
                for cash in ["Cash_0", "Cash_Rate"]:
                    if band == "No-Band" and (fee, slip) != DEFAULT_COST:
                        continue
                    cfg = V4Cfg(fee=fee, slippage=slip, cash_return=cash,
                                band_mode=band, listing_rule="L1")
                    res = R.run(sid, cfg)
                    if band == "Band" and (fee, slip) == DEFAULT_COST \
                            and cash == PRIMARY.cash_return:
                        default_runs[sid] = res
                    row = M.metrics_row(sid, "strategy", res, bench_eq, phase)
                    row.update({"fee": fee, "slippage": slip, "cash_mode": cash,
                                "band_mode": band})
                    rows.append(row)
    # cadence：Daily vs Weekly
    for sid in STRATEGY_IDS:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return, cadence="weekly")
        res = R.run(f"{sid}_weekly", cfg)
        row = M.metrics_row(sid, "strategy_weekly", res, bench_eq, phase)
        row.update({"cadence": "weekly", "cash_mode": cfg.cash_return})
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "v4_strategy_summary.csv", index=False)

    eq_df = pd.DataFrame({k: M.to_daily(v["equity"]) for k, v in default_runs.items()})
    eq_df.index.name = "timestamp"
    eq_df.to_csv(OUT / "v4_strategy_equity.csv")

    w_out = pd.concat({k: v["weights"].resample("1D").last() for k, v in default_runs.items()},
                      axis=1)
    w_out.columns = [f"{b}_{c}" for b, c in w_out.columns]
    w_out.index.name = "timestamp"
    w_out.to_csv(OUT / "v4_strategy_weights.csv")

    dd_out = pd.concat({k: M.drawdown(M.to_daily(v["equity"])) for k, v in default_runs.items()},
                       axis=1)
    dd_out.index.name = "timestamp"
    dd_out.to_csv(OUT / "v4_strategy_drawdown.csv")

    tr_out = []
    for k, v in default_runs.items():
        t = v["trades"]
        if len(t):
            t = t.copy()
            t.insert(0, "system", k)
            tr_out.append(t)
    pd.concat(tr_out).to_csv(OUT / "v4_strategy_trades.csv", index=False)

    sig = default_runs[FOCUS]["desired"].resample("1D").last()
    act = default_runs[FOCUS]["weights"].resample("1D").last()
    sig.columns = [f"target_{c}" for c in sig.columns]
    act.columns = [f"actual_{c}" for c in act.columns]
    sig.join(act).to_csv(OUT / "v4_strategy_signals.csv")
    return summ, default_runs


# ---------------------------------------------------------------------------
# Phase 4：参考权重对照臂（新默认 15/10/35/40 vs 历史 45/20/15/20）
# ---------------------------------------------------------------------------
def phase_refweight_control(R: Runner) -> pd.DataFrame:
    """对照臂：唯一变量为「参考权重 + 其配套配置带」，其余口径全部冻结。

    REF15 —— §2.2 当前默认（B-Balanced 15/10/35/40 + 新带）
    REF45 —— V4.0~V4.3 已发布锚点（45/20/15/20 + 历史带），只作为对照，不改写历史结论
    """
    log("Phase 4 参考权重对照臂：15/10/35/40 vs 45/20/15/20")
    variants = {
        "REF15": (dict(REF_WEIGHT), dict(BAND), "15/10/35/40"),
        "REF45": (dict(LEGACY_REF45), dict(LEGACY_BAND), "45/20/15/20"),
    }
    rows = []
    for vname, (rw, band, label) in variants.items():
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return, band_mode="Band",
                    listing_rule="L1", cadence="daily", rebal_threshold=0.10,
                    ref_weight=dict(rw), band_override=dict(band))
        base = R.run("B0", cfg, ref_weight=rw)
        for sid in list(BENCHMARK_IDS) + list(STRATEGY_IDS):
            if sid in STRATEGY_IDS:
                res = R.run(sid, cfg, ref_weight=rw)
            else:
                res = R.run(sid, cfg, B.default_specs()[sid], ref_weight=rw)
            m = M.metrics_row(sid, "refweight_control", res, base["equity"])
            rows.append({
                "variant": vname, "ref_weight": label, "system": sid,
                "group": "Benchmark" if sid in BENCHMARK_IDS else "Strategy",
                "band": "|".join(f"{a}:{band[a][0]:.2f}-{band[a][1]:.2f}" for a in ASSETS),
                "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                "calmar": m["calmar"], "max_dd": m["max_dd"],
                "final_wealth": m["final_wealth"],
                "avg_risky_exposure": m["avg_risky_exposure"],
                "turnover_annualized": m["turnover_annualized"],
                "n_rebalances": m["rebalance_count"],
                "cost_pct_of_gross_profit": m["cost_pct_of_gross_profit"],
            })
        log(f"  完成 {vname}（{label}）")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v4_refweight_control.csv", index=False)
    piv = df.pivot_table(index="system", columns="variant",
                         values=["cagr", "sharpe", "max_dd", "avg_risky_exposure"])
    print("\n=== 参考权重对照臂 ===")
    print(piv.round(4).to_string())
    return df


# ---------------------------------------------------------------------------
# Phase 6：公平比较
# ---------------------------------------------------------------------------
def phase_compare(R: Runner, bench_runs: Dict[str, dict], strat_runs: Dict[str, dict],
                  phase: pd.Series) -> Dict[str, pd.DataFrame]:
    log("Phase 6 策略 vs B&H Rebalance 公平比较")
    masks = segment_masks(phase)
    bench_eq = bench_runs["B0"]["equity"]
    bmetrics = {b: M.metrics_row(b, "benchmark", r) for b, r in bench_runs.items()}
    smetrics = {s: M.metrics_row(s, "strategy", r, bench_eq, phase)
                for s, r in strat_runs.items()}

    # ---- §9.1 差值表（CAGR/Sharpe/MaxDD/Calmar/Turnover）----
    spread_rows = []
    for s, sr in smetrics.items():
        for b, br in bmetrics.items():
            spread_rows.append({
                "strategy": s, "benchmark": b,
                "cagr_spread": sr["cagr"] - br["cagr"],
                "sharpe_spread": sr["sharpe"] - br["sharpe"],
                "maxdd_spread": sr["max_dd"] - br["max_dd"],
                "calmar_spread": sr["calmar"] - br["calmar"],
                "turnover_spread": sr["turnover_annualized"] - br["turnover_annualized"],
                "final_wealth_ratio": (sr["final_wealth"] / br["final_wealth"]
                                       if br["final_wealth"] else np.nan),
            })
    spread = pd.DataFrame(spread_rows)
    spread.to_csv(OUT / "v4_benchmark_comparison.csv", index=False)

    # ---- §11.1 分段比较 ----
    seg_rows = []
    for s, r in strat_runs.items():
        for b in ["B0", "B3", "B4"]:
            for seg in SEGMENTS:
                idx = masks[seg]
                ss = seg_stats(r["equity"], idx)
                bs = seg_stats(bench_runs[b]["equity"], idx)
                seg_rows.append({"strategy": s, "benchmark": b, "segment": seg,
                                 "days": int(len(idx) / 6),
                                 "strat_cagr": ss["cagr"], "bench_cagr": bs["cagr"],
                                 "strat_maxdd": ss["max_dd"], "bench_maxdd": bs["max_dd"],
                                 "strat_sharpe": ss["sharpe"], "bench_sharpe": bs["sharpe"],
                                 "active_return": ss["ret"] - bs["ret"]})
    pd.DataFrame(seg_rows).to_csv(OUT / "v4_segment_comparison.csv", index=False)

    # ---- Active Return / Capture / Opportunity cost ----
    ar_rows, cap_rows, opp_rows = [], [], []
    for s, r in strat_runs.items():
        for b in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
            be = bench_runs[b]["equity"]
            a = M.active_stats(be, r["equity"])
            ar_rows.append({"strategy": s, "benchmark": b, **a})
            cap = M.capture_ratios(phase, be, r["equity"])
            cap_rows.append({"strategy": s, "benchmark": b, **cap})
            # 机会成本：按年度统计
            sd, bd = M.to_daily(r["equity"]), M.to_daily(be)
            sd, bd = sd.align(bd, join="inner")
            for y, idx_y in sd.groupby(sd.index.year).groups.items():
                si, bi = sd.loc[idx_y], bd.loc[idx_y]
                rb = bi.pct_change().dropna()
                rs = si.pct_change().dropna()
                rb, rs = rb.align(rs, join="inner")
                up = rb > 0
                opp_rows.append({
                    "strategy": s, "benchmark": b, "year": int(y),
                    "strat_return": float(si.iloc[-1] / si.iloc[0] - 1.0),
                    "bench_return": float(bi.iloc[-1] / bi.iloc[0] - 1.0),
                    "opportunity_cost": float((rb[up] - rs[up]).clip(lower=0).sum())
                    if up.any() else 0.0,
                })
    a_df, c_df, o_df = (pd.DataFrame(ar_rows), pd.DataFrame(cap_rows), pd.DataFrame(opp_rows))
    a_df.to_csv(OUT / "v4_active_return.csv", index=False)
    c_df.to_csv(OUT / "v4_capture_ratio.csv", index=False)
    o_df.to_csv(OUT / "v4_opportunity_cost.csv", index=False)

    # ---- §11.2 同风险 / 同收益匹配 ----
    match_rows = []
    for s, sr in smetrics.items():
        b_risk = min(bmetrics, key=lambda b: abs(bmetrics[b]["max_dd"] - sr["max_dd"]))
        b_ret = min(bmetrics, key=lambda b: abs(bmetrics[b]["cagr"] - sr["cagr"]))
        br_r, br_t = bmetrics[b_risk], bmetrics[b_ret]
        match_rows.append({
            "strategy": s,
            "risk_matched_benchmark": b_risk,
            "risk_matched_bench_maxdd": br_r["max_dd"],
            "strategy_maxdd": sr["max_dd"],
            "risk_matched_cagr_spread": sr["cagr"] - br_r["cagr"],
            "risk_matched_sharpe_spread": sr["sharpe"] - br_r["sharpe"],
            "return_matched_benchmark": b_ret,
            "return_matched_bench_cagr": br_t["cagr"],
            "return_matched_maxdd_spread": sr["max_dd"] - br_t["max_dd"],
            "return_matched_sharpe_spread": sr["sharpe"] - br_t["sharpe"],
        })
    pd.DataFrame(match_rows).to_csv(OUT / "v4_risk_return_matching.csv", index=False)
    return {"spread": spread, "smetrics": pd.DataFrame(list(smetrics.values())),
            "bmetrics": pd.DataFrame(list(bmetrics.values())),
            "active": a_df, "capture": c_df, "opp": o_df}


# ---------------------------------------------------------------------------
# Phase 7：剔除 / 敏感性 / ablation
# ---------------------------------------------------------------------------
def phase_sensitivity(R: Runner, phase: pd.Series) -> Dict[str, pd.DataFrame]:
    log("Phase 7 资产剔除 / 成本 / 现金 / 参数敏感性 / ablation")
    fp = R.fp
    out: Dict[str, pd.DataFrame] = {}

    # ---- §9.2 资产剔除（基准口径 = §2.2 当前默认参考权重，剔除后按剩余资产归一）----
    def _drop(*drop: str) -> Dict[str, float]:
        return {a: (0.0 if a in drop else float(REF_WEIGHT[a])) for a in ASSETS}

    subsets = {
        "ALL4": dict(REF_WEIGHT),
        "BTC_ETH_BNB": _drop("SOL"),
        "BTC_ETH_SOL": _drop("BNB"),
        "BTC_SOL_BNB": _drop("ETH"),
        "ETH_SOL_BNB": _drop("BTC"),
        "EqualWeight4": {a: 0.25 for a in ASSETS},
        "BTC_ETH_only": _drop("SOL", "BNB"),
    }
    rows = []
    for name, w in subsets.items():
        tot = sum(w.values())
        wn = {k: v / tot for k, v in w.items()} if tot > 0 else w
        for bid, spec in B.default_specs().items():
            if bid in ("B5", "B6"):
                continue
            cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                        cash_return=PRIMARY.cash_return, ref_weight=wn)
            res = R.run(bid, cfg, spec, ref_weight=wn)
            m = M.metrics_row(bid, "exclusion", res)
            rows.append({"subset": name, "system": bid, "cagr": m["cagr"],
                         "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                         "calmar": m["calmar"], "turnover_annualized": m["turnover_annualized"]})
        for sid in ["S1", "S5"]:
            cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                        cash_return=PRIMARY.cash_return, ref_weight=wn)
            res = R.run(sid, cfg, ref_weight=wn)
            m = M.metrics_row(sid, "exclusion", res)
            rows.append({"subset": name, "system": sid, "cagr": m["cagr"],
                         "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                         "calmar": m["calmar"], "turnover_annualized": m["turnover_annualized"]})
    out["exclusion"] = pd.DataFrame(rows)
    out["exclusion"].to_csv(OUT / "v4_asset_exclusion.csv", index=False)

    # ---- §13 参数敏感性 ----
    rows = []
    for ma in MA_LENS:
        for sid in ["S3", "S5"]:
            cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                        cash_return=PRIMARY.cash_return, ma_len=ma)
            res = R.run(f"{sid}_ma{ma}", cfg)
            m = M.metrics_row(sid, "param", res)
            rows.append({"family": sid, "param": "ma_len", "value": ma,
                         "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                         "calmar": m["calmar"], "turnover_annualized": m["turnover_annualized"]})
    for t in TSMOM_LENS:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return, tsmom_days=t)
        res = R.run(f"S4_tsmom{t}", cfg)
        m = M.metrics_row("S4", "param", res)
        rows.append({"family": "S4", "param": "tsmom_days", "value": t,
                     "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                     "calmar": m["calmar"], "turnover_annualized": m["turnover_annualized"]})
    for k in RISK_MULTIPLIERS:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return, risk_multiplier=k)
        res = R.run(f"S1_rm{k}", cfg)
        m = M.metrics_row("S1", "param", res)
        rows.append({"family": "S1", "param": "risk_multiplier", "value": k,
                     "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                     "calmar": m["calmar"], "turnover_annualized": m["turnover_annualized"],
                     "avg_risky_exposure": m.get("avg_risky_exposure")})
    for th in THRESHOLDS:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return, rebal_threshold=th)
        res = R.run(f"S3_thr{th}", cfg, spec=RebalanceSpec("threshold", None, th))
        m = M.metrics_row("S3", "param", res)
        rows.append({"family": "S3", "param": "rebal_threshold", "value": th,
                     "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                     "calmar": m["calmar"], "turnover_annualized": m["turnover_annualized"]})
    for f in REBAL_FREQS_SENS:
        for sid in ["B4"]:
            res = R.run(f"{sid}_freq{f}", V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                                                cash_return=PRIMARY.cash_return),
                        spec=RebalanceSpec("calendar", f))
            m = M.metrics_row(sid, "param", res)
            rows.append({"family": "B_HA", "param": "rebalance_freq", "value": f,
                         "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                         "calmar": m["calmar"],
                         "turnover_annualized": m["turnover_annualized"]})
    out["param"] = pd.DataFrame(rows)
    out["param"].to_csv(OUT / "v4_parameter_sensitivity.csv", index=False)

    # ---- 成本 / 现金敏感性（B 与 S 对照）----
    rows = []
    for bid in ["B0", "B4"]:
        for fee, slip in COST_MATRIX:
            for cash in ["Cash_0", "Cash_Rate"]:
                cfg = V4Cfg(fee=fee, slippage=slip, cash_return=cash)
                res = R.run(bid, cfg, B.default_specs()[bid])
                m = M.metrics_row(bid, "cost_sens", res)
                rows.append({"system": bid, "fee": fee, "slippage": slip,
                             "cash_mode": cash, "cagr": m["cagr"],
                             "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                             "turnover_annualized": m["turnover_annualized"],
                             "cost_pct_of_gross_profit": m["cost_pct_of_gross_profit"]})
    for sid in STRATEGY_IDS:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return)
        res = R.run(sid, cfg)
        m = M.metrics_row(sid, "cost_sens", res)
        rows.append({"system": sid, "fee": cfg.fee, "slippage": cfg.slippage,
                     "cash_mode": cfg.cash_return, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "max_dd": m["max_dd"],
                     "turnover_annualized": m["turnover_annualized"],
                     "cost_pct_of_gross_profit": m["cost_pct_of_gross_profit"]})
    cs = pd.DataFrame(rows)
    cs.to_csv(OUT / "v4_cost_sensitivity.csv", index=False)
    out["cost"] = cs

    cash_rows = []
    for sid in STRATEGY_IDS + ["B0", "B1", "B2", "B3", "B4"]:
        for cash in ["Cash_0", "Cash_Rate"]:
            cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1], cash_return=cash)
            if sid.startswith("B"):
                res = R.run(sid, cfg, B.default_specs()[sid])
            else:
                res = R.run(sid, cfg)
            m = M.metrics_row(sid, "cash_sens", res)
            cash_rows.append({"system": sid, "cash_mode": cash, "cagr": m["cagr"],
                              "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                              "calmar": m["calmar"], "avg_cash": m["avg_cash"]})
    ch = pd.DataFrame(cash_rows)
    ch.to_csv(OUT / "v4_cash_sensitivity.csv", index=False)
    out["cash"] = ch

    # ---- ablation：策略族增量 ----
    chain = [("S3", "MA200"), ("S4", "+ TSMOM6M"), ("S5", "+ ATR"), ("S6", "+ Adaptive"),
             ("S7", "+ MTF/Persistence (Full V3)")]
    rows, prev = [], None
    for sid, note in chain:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return)
        res = R.run(sid, cfg)
        m = M.metrics_row(sid, "ablation", res)
        row = {"system": sid, "increment": note, "cagr": m["cagr"], "sharpe": m["sharpe"],
               "sortino": m["sortino"], "calmar": m["calmar"], "max_dd": m["max_dd"],
               "avg_risky_exposure": m["avg_risky_exposure"],
               "turnover_annualized": m["turnover_annualized"]}
        if prev is not None:
            row.update({"dcagr": m["cagr"] - prev["cagr"],
                        "dsharpe": m["sharpe"] - prev["sharpe"],
                        "dmax_dd": m["max_dd"] - prev["max_dd"],
                        "dturnover": m["turnover_annualized"] - prev["turnover_annualized"]})
        prev = row
        rows.append(row)
    out["ablation"] = pd.DataFrame(rows)
    out["ablation"].to_csv(OUT / "v4_ablation.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Phase 8：Walk Forward / OOS / Bootstrap / Monte Carlo
# ---------------------------------------------------------------------------
WF_CANDIDATES: List[Tuple[str, dict, Optional[str]]] = [
    ("B0", {}, None), ("B1", {}, None), ("B2", {}, None),
    ("B3", {}, None), ("B4", {}, None), ("B5", {}, None), ("B6", {}, None),
    ("S1", {"risk_multiplier": 0.0}, None), ("S1", {"risk_multiplier": 0.5}, None),
    ("S2", {"breadth_floor": 0.0}, None), ("S2", {"breadth_floor": 0.25}, None),
    ("S3", {"rebal_threshold": 0.05}, None), ("S3", {"rebal_threshold": 0.10}, None),
    ("S3", {"rebal_threshold": 0.20}, None),
    ("S5", {"ma_len": 150}, None), ("S5", {"ma_len": 200}, None),
    ("S7", {}, None),
]


def phase_walkforward(R: Runner, strat_runs: Dict[str, dict], bench_runs: Dict[str, dict]
                      ) -> Dict[str, pd.DataFrame]:
    log("Phase 8 Walk Forward / OOS / Bootstrap / Monte Carlo")
    fp = R.fp
    cand: Dict[str, dict] = {}
    for sid, kw, _ in WF_CANDIDATES:
        cfg = V4Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return=PRIMARY.cash_return, **kw)
        key = f"{sid}@{sorted(kw.items())}"
        if sid in STRATEGY_IDS:
            res = R.run(key, cfg, base=sid)
        else:
            res = R.run(sid, cfg, B.default_specs()[sid])
        cand[key] = {"system": sid, "params": kw, "res": res}

    wf_rows, sel_rows = [], []
    for w_i, (t0, t1, v1, te) in enumerate(WF_WINDOWS, start=1):
        tr_idx = fp["index"][(fp["index"] >= pd.Timestamp(t0, tz="UTC")) &
                             (fp["index"] <= pd.Timestamp(v1, tz="UTC"))]
        te_idx = fp["index"][(fp["index"] > pd.Timestamp(v1, tz="UTC")) &
                             (fp["index"] <= pd.Timestamp(te, tz="UTC"))]
        best, best_score = None, -np.inf
        for key, c in cand.items():
            s = seg_stats(c["res"]["equity"], tr_idx)
            score = s["calmar"] if s["calmar"] == s["calmar"] else -np.inf
            wf_rows.append({"window": w_i, "train_start": t0, "train_validate_end": v1,
                            "test_end": te, "candidate": key, "system": c["system"],
                            "params": json.dumps(c["params"]),
                            "train_cagr": s["cagr"], "train_sharpe": s["sharpe"],
                            "train_max_dd": s["max_dd"], "train_calmar": s["calmar"],
                            "selected": False})
            if score > best_score:
                best, best_score = key, score
        for r in wf_rows[-len(cand):]:
            if r["candidate"] == best:
                r["selected"] = True
        # 冻结后在 Test 区间评估
        for key, c in cand.items():
            s_te = seg_stats(c["res"]["equity"], te_idx)
            sel_rows.append({
                "window": w_i, "system": c["system"], "candidate": key,
                "params": json.dumps(c["params"]), "is_selected": key == best,
                "test_start": str(te_idx[0].date()) if len(te_idx) else "",
                "test_end": str(te_idx[-1].date()) if len(te_idx) else "",
                "test_cagr": s_te["cagr"], "test_sharpe": s_te["sharpe"],
                "test_max_dd": s_te["max_dd"], "test_calmar": s_te["calmar"],
            })
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(OUT / "v4_walk_forward.csv", index=False)
    oos = pd.DataFrame(sel_rows)

    # ---- OOS：Pure OOS（固定 V4 默认规则）vs Selected OOS ----
    rows = []
    bench_test = {}
    for w_i, (t0, t1, v1, te) in enumerate(WF_WINDOWS, start=1):
        te_idx = fp["index"][(fp["index"] > pd.Timestamp(v1, tz="UTC")) &
                             (fp["index"] <= pd.Timestamp(te, tz="UTC"))]
        for b in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
            s = seg_stats(bench_runs[b]["equity"], te_idx)
            bench_test[(w_i, b)] = s
        for sid in STRATEGY_IDS:
            s = seg_stats(strat_runs[sid]["equity"], te_idx)
            for b in ["B0", "B3", "B4"]:
                bs = bench_test[(w_i, b)]
                rows.append({"mode": "Pure_OOS", "window": w_i, "system": sid,
                             "benchmark": b, "test_cagr": s["cagr"],
                             "test_sharpe": s["sharpe"], "test_max_dd": s["max_dd"],
                             "bench_cagr": bs["cagr"], "bench_max_dd": bs["max_dd"],
                             "active_return": s["ret"] - bs["ret"],
                             "maxdd_improvement": s["max_dd"] - bs["max_dd"]})
        sel = oos[(oos["window"] == w_i) & oos["is_selected"]]
        sel_sys = sel["system"].iloc[0] if len(sel) else None
        if sel_sys:
            s = seg_stats(cand[sel["candidate"].iloc[0]]["res"]["equity"], te_idx)
            for b in ["B0", "B3", "B4"]:
                bs = bench_test[(w_i, b)]
                rows.append({"mode": "Selected_OOS", "window": w_i, "system": sel_sys,
                             "benchmark": b, "test_cagr": s["cagr"], "test_sharpe": s["sharpe"],
                             "test_max_dd": s["max_dd"], "bench_cagr": bs["cagr"],
                             "bench_max_dd": bs["max_dd"],
                             "active_return": s["ret"] - bs["ret"],
                             "maxdd_improvement": s["max_dd"] - bs["max_dd"]})
    oos_full = pd.concat([pd.DataFrame(rows), oos], ignore_index=True)
    oos_full.to_csv(OUT / "v4_oos.csv", index=False)

    # ---- §11.3 Bootstrap：S0-S7 vs B0-B6 ----
    br, mr = [], []
    for sid in STRATEGY_IDS:
        se = strat_runs[sid]["equity"]
        for b in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
            r = M.block_bootstrap(se, bench_runs[b]["equity"], n_iter=10_000,
                                  block=20, seed=SEED)
            if r:
                br.append({"strategy": sid, "benchmark": b, **r})
        mc = M.monte_carlo(se, n_iter=10_000, block=20, seed=SEED)
        if mc:
            mr.append({"strategy": sid, **mc})
    pd.DataFrame(br).to_csv(OUT / "v4_bootstrap.csv", index=False)
    pd.DataFrame(mr).to_csv(OUT / "v4_monte_carlo.csv", index=False)
    return {"wf": wf, "oos": oos_full, "bootstrap": pd.DataFrame(br),
            "monte_carlo": pd.DataFrame(mr)}


# ---------------------------------------------------------------------------
# §14/§15 硬门槛 + 评分 + 候选选择
# ---------------------------------------------------------------------------
def _nz(x: float, cap: float) -> float:
    return float(np.clip(x / cap, -1.0, 1.0))


def phase_ranking(smetrics: pd.DataFrame, bmetrics: pd.DataFrame,
                  oos: pd.DataFrame, bootstrap: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    log("Phase 9 硬门槛 / 评分 / 候选排名")
    b0 = bmetrics[bmetrics["label"] == "B0"].iloc[0]
    b4 = bmetrics[bmetrics["label"] == "B4"].iloc[0]
    rows, score_rows = [], []
    for _, s in smetrics.iterrows():
        sid = s["label"]
        pure = oos[(oos["mode"] == "Pure_OOS") & (oos["system"] == sid)]
        n_win = int(len(pure))
        pos = int((pure["active_return"] > 0).sum())
        dd_better = int((pure["maxdd_improvement"] > 0.02).sum())
        half = max(n_win // 2, 1)
        gates = {
            "gate_maxdd_vs_B0_15pp": bool((s["max_dd"] - b0["max_dd"]) >= 0.15),
            "gate_sharpe_vs_B4_0.90": bool(s["sharpe"] >= 0.90 * b4["sharpe"]),
            "gate_oos_half_windows": bool(pos >= half or dd_better >= half),
            "gate_exposure_50": bool(s.get("avg_risky_exposure", 0) >= 0.50),
            "gate_cost_25pct": bool(s["cost_pct_of_gross_profit"] <= 0.25),
        }
        gates["passed_all"] = all(gates.values())
        gates["note_defensive"] = bool(not gates["gate_exposure_50"])
        rows.append({"strategy": sid, **gates,
                     "oos_windows": n_win, "oos_positive_active": pos,
                     "oos_maxdd_better": dd_better, "cagr": s["cagr"],
                     "sharpe": s["sharpe"], "max_dd": s["max_dd"],
                     "avg_risky_exposure": s.get("avg_risky_exposure"),
                     "cost_pct_of_gross_profit": s["cost_pct_of_gross_profit"]})
        # 评分（相对 B0 / B3 / B4 分别计算）
        for bkey, bref in (("B0", b0), ("B3", bmetrics[bmetrics["label"] == "B3"].iloc[0]),
                           ("B4", b4)):
            active_stab = pos / n_win if n_win else 0.0
            score = (0.25 * _nz(s["sharpe"], 2.0) + 0.20 * _nz(s["sortino"], 3.0)
                     + 0.20 * _nz(s["calmar"], 1.5) + 0.15 * _nz(s["cagr"], 0.5)
                     + 0.10 * float(np.clip(1.0 - abs(s["max_dd"]) / 0.8, -1.0, 1.0))
                     + 0.05 * active_stab + 0.05 * SIMPLICITY.get(sid, 0.5))
            score_rows.append({"strategy": sid, "benchmark": bkey, "score": score,
                               "sharpe": s["sharpe"], "sortino": s["sortino"],
                               "calmar": s["calmar"], "cagr": s["cagr"],
                               "max_dd": s["max_dd"], "active_stability": active_stab,
                               "simplicity": SIMPLICITY.get(sid, 0.5),
                               "cagr_vs_bench": s["cagr"] - bref["cagr"],
                               "maxdd_vs_bench": s["max_dd"] - bref["max_dd"]})
    gates_df, scores = pd.DataFrame(rows), pd.DataFrame(score_rows)
    gates_df.to_csv(OUT / "v4_hard_gates.csv", index=False)
    scores.to_csv(OUT / "v4_candidate_scores.csv", index=False)

    # ---- §15 三个答案 ----
    bs = bootstrap
    picks = []
    if len(bs):
        agg = bs.groupby("strategy")["prob_strategy_wins"].mean().rename("avg_win_prob")
    else:
        agg = pd.Series(dtype=float)
    strat = smetrics.set_index("label")
    strat = strat.join(agg)
    valid = strat.index.tolist()
    if valid:
        a = strat["cagr"].idxmax()
        b = strat["sharpe"].idxmax()
        core_pool = [x for x in valid if gates_df.set_index("strategy").loc[x, "gate_maxdd_vs_B0_15pp"]
                     and gates_df.set_index("strategy").loc[x, "gate_exposure_50"]
                     and gates_df.set_index("strategy").loc[x, "gate_cost_25pct"]]
        c = strat.loc[core_pool]["calmar"].idxmax() if core_pool else None
        picks.append({
            "role": "Strategy A 最高收益", "system": a,
            "cagr": strat.loc[a, "cagr"], "cagr_vs_B4": strat.loc[a, "cagr"] - b4["cagr"],
            "max_dd": strat.loc[a, "max_dd"],
            "avg_risky_exposure": strat.loc[a, "avg_risky_exposure"],
            "turnover_annualized": strat.loc[a, "turnover_annualized"],
            "bootstrap_avg_win_prob": strat.loc[a].get("avg_win_prob"),
        })
        picks.append({
            "role": "Strategy B 最佳风险收益", "system": b,
            "sharpe": strat.loc[b, "sharpe"], "sortino": strat.loc[b, "sortino"],
            "calmar": strat.loc[b, "calmar"], "cagr": strat.loc[b, "cagr"],
            "max_dd": strat.loc[b, "max_dd"],
            "avg_risky_exposure": strat.loc[b, "avg_risky_exposure"],
            "bootstrap_avg_win_prob": strat.loc[b].get("avg_win_prob"),
        })
        if c:
            picks.append({
                "role": "Strategy C 长期核心配置", "system": c,
                "cagr": strat.loc[c, "cagr"], "cagr_vs_B4": strat.loc[c, "cagr"] - b4["cagr"],
                "cagr_vs_B0": strat.loc[c, "cagr"] - b0["cagr"],
                "sharpe": strat.loc[c, "sharpe"], "max_dd": strat.loc[c, "max_dd"],
                "maxdd_vs_B0": strat.loc[c, "max_dd"] - b0["max_dd"],
                "avg_risky_exposure": strat.loc[c, "avg_risky_exposure"],
                "turnover_annualized": strat.loc[c, "turnover_annualized"],
                "bootstrap_avg_win_prob": strat.loc[c].get("avg_win_prob"),
            })
        else:
            picks.append({"role": "Strategy C 长期核心配置", "system": "NONE",
                          "cagr": np.nan,
                          "note": "无候选同时满足 MaxDD 改善>=15pp、暴露>=50%、成本<=25% 门槛"})
    picks_df = pd.DataFrame(picks)
    picks_df.to_csv(OUT / "v4_final_candidates.csv", index=False)
    return {"gates": gates_df, "scores": scores, "candidates": picks_df}


# ---------------------------------------------------------------------------
def main() -> None:
    log("装载数据面板")
    fp = D.get_panel()
    log(f"样本 {fp['index'][0].date()} ~ {fp['index'][-1].date()} ({len(fp['index'])} 根 4H)")
    R = Runner(fp)

    p0 = phase0(fp)
    bench_summ, bench_runs, phase = phase_benchmarks(R)
    log(f"基准完成：{len(bench_summ)} 行")

    bench_eq = bench_runs["B0"]["equity"]
    strat_summ, strat_runs = phase_strategies(R, bench_eq, phase)
    log(f"策略完成：{len(strat_summ)} 行")

    rwc_out = phase_refweight_control(R)

    cmp_out = phase_compare(R, bench_runs, strat_runs, phase)
    sens_out = phase_sensitivity(R, phase)
    wf_out = phase_walkforward(R, strat_runs, bench_runs)
    rank_out = phase_ranking(cmp_out["smetrics"], cmp_out["bmetrics"],
                             wf_out["oos"], wf_out["bootstrap"])

    with open(OUT / "results_v4.pkl", "wb") as fh:
        pickle.dump({
            "bench_runs": bench_runs, "strat_runs": strat_runs, "phase": phase,
            "benchmark_summary": bench_summ, "strategy_summary": strat_summ,
            "smetrics": cmp_out["smetrics"], "bmetrics": cmp_out["bmetrics"],
            "bootstrap": wf_out["bootstrap"], "monte_carlo": wf_out["monte_carlo"],
            "gates": rank_out["gates"], "scores": rank_out["scores"],
            "candidates": rank_out["candidates"], "snapshot": p0["snapshot"],
            "refweight_control": rwc_out,
            "data_quality": p0["data_quality"], "listing": p0["listing"],
            "config_fingerprint": config_fingerprint(PRIMARY),
            "focus": FOCUS,
        }, fh)

    log("候选选择：")
    print(rank_out["candidates"].to_string(index=False))
    log(f"输出目录 {OUT}")
    for f in sorted(OUT.glob("v4_*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()