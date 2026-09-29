# -*- coding: utf-8 -*-
"""V3 现货系统 —— 主回测（§4 基准 / §30 Model A-F / §31 Minimal vs Full / §55 候选系统）"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.v3_spot import common as C      # noqa: E402
from scripts.v3_spot import metrics as M     # noqa: E402
from scripts.v3_spot import strategy as S    # noqa: E402

OUT = REPO / "reports" / "v3_spot"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["ModelA_MA200", "ModelB_MA200_TSMOM", "ModelC_MA200_ATR_ADX",
               "ModelD_C_Adaptive", "ModelE_D_TSMOM", "ModelF_Full",
               "Minimal_MA200_ATR_ADX", "Full_Ensemble",
               "Conservative", "Balanced", "Aggressive"]

# 主要展示用的曲线（报告图表核心）
MAIN_CURVES = ["BuyHold_45_20_15_20", "ModelA_MA200", "ModelC_MA200_ATR_ADX",
               "Balanced", "Aggressive"]
# §4 补充基准：不同日历再平衡频率的四资产 B&H（freq 为日历周期码）
CAL_BMS = {"BuyHold_Monthly": "M", "BuyHold_Quarterly": "Q",
           "BuyHold_SemiAnnual": "2Q", "BuyHold_Annual": "Y"}
FOCUS = "Balanced"          # 交易流水/暴露/回撤明细的对象


def full_summary(res: dict, eq: pd.Series, bench_eq: pd.Series | None = None,
                 bench_phase: pd.Series | None = None) -> dict:
    s = M.summarize(eq)
    if not s:
        return {}
    tim = M.time_in_market(res["weights"])
    row = {"label": res.get("label", ""), "cagr": s["cagr"], "annual_vol": s["annual_vol"],
           "sharpe": s["sharpe"], "sortino": s["sortino"], "calmar": s["calmar"],
           "max_dd": s["max_dd"], "total_return": s["total_return"],
           "final_wealth": s["final_wealth"], "worst_year": s["worst_year"],
           "turnover_annual": res.get("turnover_annual", 0.0),
           "trades": int(len(res["trades"])), **tim}
    if bench_phase is not None:
        row.update(M.capture_ratios(bench_phase, bench_eq, eq))
        eps = M.drawdown_episodes(eq, top=1)
        row["dd_recovery_days"] = eps[0]["days_to_recovery"] if eps else np.nan
    return row


def main() -> None:
    fp = C.get_fp()
    base = S.Cfg()
    print(f"样本: {fp['index'][0].date()} ~ {fp['index'][-1].date()}  ({len(fp['index'])} 根 4H)")

    runs, rows, curves, weights_store = {}, [], {}, {}

    # ---------------- 基准 ----------------
    bms = {
        "BuyHold_45_20_15_20": S.REF_WEIGHT,
        "BuyHold_Equal": {a: 0.25 for a in S.ASSETS},
        "BuyHold_BTC_only": {"BTC": 1.0, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0},
    }
    for name, w in bms.items():
        r = C.run_benchmark(fp, base, w)
        runs[name] = r
        curves[name] = r["equity"]
        rows.append({"kind": "benchmark", **full_summary({"label": name, **r}, r["equity"])})

    bench = runs["BuyHold_45_20_15_20"]
    bench_eq = bench["equity"]
    phase = M.classify_phases(bench_eq)
    bench_phase = phase

    # ---------------- §4 补充基准：日历再平衡频率对比 ----------------
    for name, freq in CAL_BMS.items():
        r = C.run_benchmark_calendar(fp, base, freq, S.REF_WEIGHT)
        runs[name] = r
        curves[name] = r["equity"]
        rows.append({"kind": "benchmark", **full_summary({"label": name, **r}, r["equity"])})
        print(f"  {name}: {len(r['trades'])} 条交易腿, 年换手 {r['turnover_annual']:.3f}")

    # ---------------- 模型 A-F / Minimal / Full / 候选 ----------------
    for name in MODEL_ORDER:
        cfg = S.Cfg(modules=S.MODELS[name])
        r = C.run_strategy(fp, cfg, name)
        runs[name] = r
        curves[name] = r["equity"]
        weights_store[name] = r["weights"]
        rows.append({"kind": "model", **full_summary({**r, "label": name}, r["equity"],
                                                     bench_eq, bench_phase)})

    # ---------------- §27 Cash Return A/B（对 Balanced）----------------
    for mode in ["A", "B"]:
        cfg = S.Cfg(modules=S.MODELS["Balanced"], cash_return=mode)
        r = C.run_strategy(fp, cfg, f"Balanced_Cash{mode}")
        runs[f"Balanced_Cash{mode}"] = r
        rows.append({"kind": "cash", **full_summary({**r, "label": f"Balanced_Cash{mode}"},
                                                    r["equity"], bench_eq, bench_phase)})
    for mode in ["A", "B"]:
        cfg = S.Cfg(cash_return=mode)
        r = C.run_benchmark(fp, cfg, S.REF_WEIGHT)
        rows.append({"kind": "cash", **full_summary({"label": f"BuyHold_Cash{mode}", **r},
                                                    r["equity"], bench_eq, bench_phase)})

    # ---------------- §28 成本矩阵（Model C / Balanced / Aggressive）----------------
    for name in ["ModelC_MA200_ATR_ADX", "Balanced", "Aggressive"]:
        for fee in [0.0005, 0.0010]:
            for slip in [0.0, 0.0002, 0.0005]:
                cfg = S.Cfg(modules=S.MODELS[name], fee=fee, slippage=slip)
                r = C.run_strategy(fp, cfg, f"{name}_fee{fee}_slip{slip}")
                rows.append({"kind": "cost", "model": name, "fee": fee, "slippage": slip,
                             **full_summary({**r, "label": f"{name}_fee{fee}_slip{slip}"},
                                            r["equity"], bench_eq, bench_phase)})

    # ---------------- 汇总 ----------------
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "v3_spot_summary.csv", index=False)

    # ---------------- 曲线类输出 ----------------
    eq_df = pd.DataFrame({k: v for k, v in curves.items()})
    eq_df.index.name = "timestamp"
    eq_df.to_csv(OUT / "v3_spot_equity.csv")

    dd_df = pd.DataFrame({k: M.drawdown(v) for k, v in curves.items()})
    dd_df.index.name = "timestamp"
    dd_df.to_csv(OUT / "v3_spot_drawdown.csv")

    exp = weights_store[FOCUS].copy()
    exp.index.name = "timestamp"
    exp.to_csv(OUT / "v3_spot_exposure.csv")

    tr = runs[FOCUS]["trades"].copy()
    if not tr.empty:
        tr["date"] = pd.to_datetime(tr["date"])
    tr.to_csv(OUT / "v3_spot_trades.csv", index=False)

    reg = M.regime_table(bench_phase, bench_eq, runs[FOCUS]["equity"], runs[FOCUS]["weights"])
    reg.to_csv(OUT / "v3_spot_regime.csv", index=False)

    cap_rows = []
    for name in MAIN_CURVES + ["Full_Ensemble", "Conservative", "ModelE_D_TSMOM"]:
        if name in runs:
            cap = M.capture_ratios(bench_phase, bench_eq, runs[name]["equity"])
            cap_rows.append({"label": name, **cap})
    pd.DataFrame(cap_rows).to_csv(OUT / "v3_spot_capture_ratio.csv", index=False)

    # ---------------- 关键对象持久化（供报告构建）----------------
    with open(OUT / "results.pkl", "wb") as fh:
        pickle.dump({"summary": summ, "curves": curves,
                     "weights": weights_store, "phase": phase,
                     "trades": runs[FOCUS]["trades"]}, fh)

    # ---------------- 控制台摘要 ----------------
    cols = ["kind", "label", "cagr", "sharpe", "sortino", "calmar", "max_dd",
            "final_wealth", "turnover_annual", "trades", "avg_risky",
            "tim_gt_50", "upside_capture_bull", "downside_capture_bear"]
    show = summ[[c for c in cols if c in summ.columns]]
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(show.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nWrote -> {OUT}")


if __name__ == "__main__":
    main()