# -*- coding: utf-8 -*-
"""V3 现货系统 —— 实验套件

包含：
  §39 Ablation Test       逐个删除模块
  §40 Incremental Test    按顺序增加模块
  §50 参数稳定性           MA/TSMOM/ATR/ADX/Persistence/Rebalance/Cadence/目标波动率/配置带
  §48 Walk-Forward        冻结默认参数的纯样本外 + Train→Validate→Test 参数选择
  §51 Monte Carlo          10,000 次日收益 block bootstrap
  §52 Bootstrap            剔除 Top 5/10/20 持有段贡献
  §33/§34 熊市与再买入效率  Exit / Bottom / Re-entry 价格与 Missed Upside
  §37/§38 Drawdown Guard   0 / 10 / 15 / 20% 对比
  §41-§46 Time-in-Market / Capture / Recovery / Rolling
"""
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
RNG_SEED = 20260920
N_MC = 10_000


# ---------------------------------------------------------------------------
def evaluate(fp: dict, cfg: S.Cfg, label: str, phase=None, bench_eq=None) -> dict:
    r = C.run_strategy(fp, cfg, label)
    s = M.summarize(r["equity"])
    if not s:
        return {"label": label}
    row = {"label": label, "cfg_modules": "+".join(cfg.modules_of()),
           "cagr": s["cagr"], "annual_vol": s["annual_vol"], "sharpe": s["sharpe"],
           "sortino": s["sortino"], "calmar": s["calmar"], "max_dd": s["max_dd"],
           "total_return": s["total_return"], "final_wealth": s["final_wealth"],
           "turnover_annual": r["turnover_annual"], "trades": len(r["trades"]),
           **M.time_in_market(r["weights"])}
    if phase is not None and bench_eq is not None:
        row.update(M.capture_ratios(phase, bench_eq, r["equity"]))
    row["_equity"] = r["equity"]
    row["_weights"] = r["weights"]
    return row


def clean(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    return df


# ---------------------------------------------------------------------------
# §39 Ablation / §40 Incremental
# ---------------------------------------------------------------------------
def ablation(fp: dict, phase, bench_eq) -> pd.DataFrame:
    rows = []
    rows.append(evaluate(fp, S.Cfg(modules=S.ALL_MODULES), "Full(8 modules)", phase, bench_eq))
    for m in S.ALL_MODULES:
        mods = [x for x in S.ALL_MODULES if x != m]
        rows.append(evaluate(fp, S.Cfg(modules=mods), f"Full minus {m}", phase, bench_eq))
    rows.append(evaluate(fp, S.Cfg(modules=["ma200"]), "Only MA200", phase, bench_eq))
    return clean(rows)


def incremental(fp: dict, phase, bench_eq) -> pd.DataFrame:
    order = ["ma200", "atr", "adx", "adaptive", "tsmom6m", "tsmom12m", "mtf", "persistence"]
    rows = []
    mods: list = []
    for m in order:
        mods = mods + [m]
        rows.append(evaluate(fp, S.Cfg(modules=list(mods)),
                             f"+{m}", phase, bench_eq))
    return clean(rows)


# ---------------------------------------------------------------------------
# §50 参数稳定性
# ---------------------------------------------------------------------------
def parameter_sensitivity(fp: dict, phase, bench_eq) -> pd.DataFrame:
    base = list(S.ALL_MODULES)
    rows = []

    def add(kind, label, **kw):
        cfg = S.Cfg(modules=kw.pop("modules", base), **kw)
        r = evaluate(fp, cfg, f"{kind}:{label}", phase, bench_eq)
        r["kind"] = kind
        r["variant"] = label
        rows.append(r)

    for n in [150, 200, 250]:
        add("ma_len", str(n), ma_len=n)
    add("tsmom", "6M_only", modules=[m for m in base if m != "tsmom12m"])
    add("tsmom", "12M_only", modules=[m for m in base if m != "tsmom6m"])
    add("tsmom", "6M+12M", modules=base)
    add("tsmom", "3M", tsmom_days=63)
    add("tsmom", "9M", tsmom_days=189)
    for n in [10, 14, 20]:
        add("atr_len", str(n), atr_len=n)
    for n in [14, 20, 30]:
        add("adx_len", str(n), adx_len=n)
    for v in [0.10, 0.15, 0.20, 0.25]:
        add("target_vol", f"{v:.0%}", target_vol=v)
    for b in [2, 4, 6, 12, 24]:
        add("persist_bars", str(b), persist_bars=b)
    for th in [0.05, 0.10, 0.15, 0.20]:
        add("rebal_threshold", f"{th:.0%}", rebal_threshold=th)
    for cd in ["4h", "daily", "weekly"]:
        add("cadence", cd, cadence=cd)
    for mode in ["bucket", "linear"]:
        add("alloc_mode", mode, alloc_mode=mode)
    add("lower_band", "off", enforce_lower_band=False)
    # §32 配置带敏感性：中枢上下移动
    for shift, tag in [(-0.10, "BTC 35%"), (0.0, "BTC 45%"), (0.10, "BTC 55%")]:
        ref = dict(S.REF_WEIGHT)
        band = dict(S.BAND)
        others = ["ETH", "SOL", "BNB"]
        ref["BTC"] = 0.45 + shift
        rest = 1.0 - ref["BTC"]
        base_rest = sum(S.REF_WEIGHT[a] for a in others)
        for a in others:
            ref[a] = S.REF_WEIGHT[a] / base_rest * rest
        for a in S.ASSETS:
            lo, hi = S.BAND[a]
            band[a] = (lo + shift, hi + shift)
        add("ref_weight", tag, ref_weight=ref, band=band)
    # §35 Re-entry Guard 阈值
    for g in [0.05, 0.10, 0.15, 0.20]:
        add("reentry_guard", f"{g:.0%}", reentry_guard=g)
    # §37/§38 Drawdown Guard
    for g in [0.0, 0.10, 0.15, 0.20]:
        add("dd_guard", ("off" if g == 0 else f"{g:.0%}"), dd_guard=g)
    return clean(rows)


# ---------------------------------------------------------------------------
# §48 Walk-Forward（冻结参数纯 OOS + Train/Validate/Test 参数选择）
# ---------------------------------------------------------------------------
WINDOWS = [("2017-2021", "2022", "2023"),
           ("2018-2022", "2023", "2024"),
           ("2019-2023", "2024", "2025"),
           ("2020-2024", "2025", "2026")]

# 参数候选集（仅在 Train/Validation 上选择，Test 完全冻结）
CANDIDATES = [
    ("Minimal", ["ma200", "atr", "adx"], {}),
    ("Minimal_weekly", ["ma200", "atr", "adx"], {"cadence": "weekly", "rebal_threshold": 0.15}),
    ("Balanced", ["ma200", "atr", "adx", "adaptive", "persistence"], {}),
    ("Balanced_weekly", ["ma200", "atr", "adx", "adaptive", "persistence"],
     {"cadence": "weekly", "rebal_threshold": 0.15}),
    ("Full", S.ALL_MODULES, {}),
    ("Full_weekly", S.ALL_MODULES, {"cadence": "weekly", "rebal_threshold": 0.15}),
    ("MA200_TSMOM", ["ma200", "tsmom6m", "tsmom12m"], {}),
    ("C_tight", ["ma200", "atr", "adx"], {"rebal_threshold": 0.20}),
    ("C_vol20", ["ma200", "atr", "adx"], {"target_vol": 0.20}),
    ("Full_persist12", S.ALL_MODULES, {"persist_bars": 12}),
]


def _sub(fp: dict, lo: str, hi: str) -> dict:
    return C.get_fp(start=lo, end=hi)


def walk_forward(fp: dict, phase, bench_eq) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (纯 OOS 逐窗口结果, Train/Validate 选参 -> Test 结果)。"""
    pure, selected = [], []
    default_cfg = S.Cfg()
    for tr, va, te in WINDOWS:
        tr_lo = f"{tr.split('-')[0]}-01-01"
        tr_hi = f"{tr.split('-')[1]}-12-31"
        va_lo, va_hi = f"{va}-01-01", f"{va}-12-31"
        te_lo, te_hi = f"{te}-01-01", f"{te}-12-31"

        # (1) 冻结默认参数，直接看 Test 段（纯 OOS）
        fpt = _sub(fp, te_lo, te_hi)
        if len(fpt["index"]) > 200:
            r = evaluate(fpt, default_cfg, f"{te}")
            r.update({"train": tr, "validate": va, "test": te, "config": "Default(8 modules)"})
            pure.append(r)

        # (2) 在 Train 上排序，在 Validate 上确认，冻结后应用到 Test
        fptr = _sub(fp, tr_lo, tr_hi)
        fpva = _sub(fp, va_lo, va_hi)
        best, best_key = None, None
        for name, mods, kw in CANDIDATES:
            try:
                a = evaluate(fptr, S.Cfg(modules=list(mods), **kw), name)
                b = evaluate(fpva, S.Cfg(modules=list(mods), **kw), name)
            except Exception:  # noqa: BLE001
                continue
            if not a.get("sharpe"):
                continue
            key = 0.5 * (a["sharpe"] + b.get("sharpe", a["sharpe"]))
            if best_key is None or key > best_key:
                best_key, best = key, (name, mods, kw)
        if best is None:
            continue
        name, mods, kw = best
        if len(fpt["index"]) > 200:
            r = evaluate(fpt, S.Cfg(modules=list(mods), **kw), f"{te}:{name}")
            r.update({"train": tr, "validate": va, "test": te,
                      "config": name, "select_score": best_key,
                      "frozen_params": str(kw)})
            selected.append(r)
    return clean(pure), clean(selected)


# ---------------------------------------------------------------------------
# §51 Monte Carlo / §52 Bootstrap
# ---------------------------------------------------------------------------
def monte_carlo(eq: pd.Series, bench: pd.Series, n_sims: int = N_MC,
                block: int = 5) -> dict:
    """日收益 block bootstrap（stationary block），比较策略与基准的分布。"""
    rng = np.random.default_rng(RNG_SEED)
    se = M.to_daily(eq)
    sb = M.to_daily(bench)
    idx = se.index.intersection(sb.index)
    rs = se.reindex(idx).pct_change().dropna().to_numpy()
    rb = sb.reindex(idx).pct_change().dropna().to_numpy()
    n = len(rs)
    if n < 50:
        return {}
    years = n / 365.25
    n_blocks = int(np.ceil(n / block))

    def sim(path: np.ndarray) -> tuple[float, float]:
        cum = np.cumprod(1.0 + path)
        cagr = cum[-1] ** (1 / years) - 1.0
        peak = np.maximum.accumulate(cum)
        mdd = float((cum / peak - 1.0).min())
        return cagr, mdd

    res_s, res_b, better = [], [], 0
    for _ in range(n_sims):
        starts = rng.integers(0, n - block, size=n_blocks)
        pick = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        cs, ms = sim(rs[pick])
        cb, mb = sim(rb[pick])
        res_s.append((cs, ms))
        res_b.append((cb, mb))
        better += 1 if cs > cb else 0
    a = np.array(res_s)
    b = np.array(res_b)
    return {
        "sims": n_sims,
        "strategy_cagr_median": float(np.median(a[:, 0])),
        "strategy_cagr_p5": float(np.percentile(a[:, 0], 5)),
        "strategy_cagr_p95": float(np.percentile(a[:, 0], 95)),
        "strategy_mdd_median": float(np.median(a[:, 1])),
        "strategy_mdd_p5": float(np.percentile(a[:, 1], 5)),
        "strategy_mdd_p95": float(np.percentile(a[:, 1], 95)),
        "bench_cagr_median": float(np.median(b[:, 0])),
        "bench_mdd_median": float(np.median(b[:, 1])),
        "prob_underperform_bh": 1.0 - better / n_sims,
        "prob_outperform_bh": better / n_sims,
        "prob_mdd_worse_than_40": float((a[:, 1] < -0.40).mean()),
        "cagr_samples": a[:, 0],
        "mdd_samples": a[:, 1],
    }


def bootstrap_drop(fp: dict, cfg: S.Cfg, res: dict, phase, bench_eq) -> pd.DataFrame:
    """§52 剔除贡献最大的 Top-N 持有段后重算绩效。

    做法：把组合日收益中"被剔除持有段"的贡献扣掉（其余部分保持不变），
    再用剩余日收益重建净值曲线。
    """
    px = fp["px"]
    eps = M.contribution_episodes(res["weights"], px)
    daily = M.daily_contributions(res["weights"], px)          # 4H，各资产贡献
    eq_d = M.to_daily(res["equity"])
    rets = eq_d.pct_change().fillna(0.0)

    def drop_and_rerun(n_drop: int) -> dict:
        dropped = pd.Series(0.0, index=daily.index)
        for _, e in eps.head(n_drop).iterrows():
            m = (daily.index >= e["start"]) & (daily.index <= e["end"])
            dropped.loc[m] += daily.loc[m, e["asset"]]
        dropped_d = dropped.resample("1D").sum().reindex(rets.index).fillna(0.0)
        new_r = (rets - dropped_d).fillna(0.0)
        eq = float(eq_d.iloc[0]) * (1.0 + new_r).cumprod()
        s = M.summarize(eq)
        cap = M.capture_ratios(phase, bench_eq, eq) if phase is not None else {}
        return {"dropped_episodes": n_drop, "cagr": s.get("cagr"), "sharpe": s.get("sharpe"),
                "max_dd": s.get("max_dd"), "calmar": s.get("calmar"),
                "total_return": s.get("total_return"), **cap}

    base = M.summarize(res["equity"])
    rows = [{"dropped_episodes": 0, "cagr": base.get("cagr"), "sharpe": base.get("sharpe"),
             "max_dd": base.get("max_dd"), "calmar": base.get("calmar"),
             "total_return": base.get("total_return")}]
    for n in [5, 10, 20]:
        rows.append(drop_and_rerun(n))
    df = pd.DataFrame(rows)
    df["top_contributors"] = "; ".join(
        f"{r.asset}@{r.start.date()}({r.contribution:.1%})" for _, r in eps.head(5).iterrows())
    return df


# ---------------------------------------------------------------------------
# §33/§34 熊市行为与再买入效率
# ---------------------------------------------------------------------------
def reentry_efficiency(fp: dict, res: dict) -> pd.DataFrame:
    px = fp["px"]
    rows = []
    for a in S.ASSETS:
        w = res["weights"][a]
        p = px[a]
        prev = 0.0
        exit_i = None
        for i, (t, wv) in enumerate(w.items()):
            if prev > 1e-6 and wv <= 1e-6:
                exit_i = i
                exit_t, exit_p = t, p.iloc[i]
            elif prev <= 1e-6 and wv > 1e-6 and exit_i is not None:
                seg = p.iloc[exit_i:i + 1]
                bottom = float(seg.min())
                bottom_t = seg.idxmin()
                reentry_p = float(p.iloc[i])
                rows.append({
                    "asset": a, "exit_date": exit_t, "exit_price": float(exit_p),
                    "bottom_date": bottom_t, "bottom_price": bottom,
                    "reentry_date": t, "reentry_price": reentry_p,
                    "flat_bars": i - exit_i,
                    "avoided_drawdown": exit_p / bottom - 1.0,
                    "missed_upside": reentry_p / bottom - 1.0,
                    "exit_to_reentry": reentry_p / exit_p - 1.0,
                })
                exit_i = None
            prev = wv
    df = pd.DataFrame(rows)
    return df


def bull_bear_analysis(fp: dict, cfg: S.Cfg, res: dict, bench: dict) -> pd.DataFrame:
    """§33/§47：分阶段 + 每资产熊市行为 + 单资产 B&H 对照。"""
    rows = []
    phase = M.classify_phases(bench["equity"])
    reg = M.regime_table(phase, bench["equity"], res["equity"], res["weights"])
    for _, r in reg.iterrows():
        rows.append({"section": "regime", **r.to_dict()})

    # 单一资产 B&H 对照（§4 / §33）
    s_all = M.summarize(res["equity"])
    for a in S.ASSETS:
        sub = {"BTC": {"BTC": 1.0}, "ETH": {"ETH": 1.0},
               "SOL": {"SOL": 1.0}, "BNB": {"BNB": 1.0}}[a]
        bsub = C.run_benchmark(fp, cfg, sub)
        sb = M.summarize(bsub["equity"])
        rows.append({"section": "single_asset_bh", "asset": a,
                     "bench_return": sb.get("total_return"), "strategy_return": None,
                     "bench_maxdd": sb.get("max_dd"), "strategy_maxdd": None,
                     "cagr": sb.get("cagr"), "sharpe": sb.get("sharpe")})

    # 策略在各资产上的暴露与"熊市卖出"统计
    for a in S.ASSETS:
        w = res["weights"][a]
        bt = res["trades"]
        sells = bt[(bt["asset"] == a) & (bt["action"] == "sell")] if not bt.empty else bt
        rows.append({"section": "asset_exposure", "asset": a,
                     "avg_weight": float(w.mean()), "max_weight": float(w.max()),
                     "time_in_market": float((w > 1e-6).mean()),
                     "sell_legs": int(len(sells)),
                     "bench_return": None, "strategy_return": None,
                     "bench_maxdd": None, "strategy_maxdd": None,
                     "cagr": s_all.get("cagr"), "sharpe": s_all.get("sharpe")})

    # §34 再买入效率（事件级）
    re_eff = reentry_efficiency(fp, res)
    if not re_eff.empty:
        for _, r in re_eff.iterrows():
            rows.append({"section": "reentry_event", **r.to_dict()})
        rows.append({"section": "reentry_summary",
                     "asset": "ALL", "flat_bars": float(re_eff["flat_bars"].mean()),
                     "avoided_drawdown": float(re_eff["avoided_drawdown"].mean()),
                     "missed_upside": float(re_eff["missed_upside"].mean()),
                     "exit_to_reentry": float(re_eff["exit_to_reentry"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    fp = C.get_fp()
    cfg = S.Cfg()
    bench = C.run_benchmark(fp, cfg)
    bench_eq = bench["equity"]
    phase = M.classify_phases(bench_eq)
    print("样本:", fp["index"][0].date(), "~", fp["index"][-1].date())

    print("[1/7] Ablation ...")
    abl = ablation(fp, phase, bench_eq)
    abl.to_csv(OUT / "v3_spot_ablation.csv", index=False)

    print("[2/7] Incremental ...")
    inc = incremental(fp, phase, bench_eq)
    inc.to_csv(OUT / "v3_spot_ablation_incremental.csv", index=False)

    print("[3/7] Parameter sensitivity ...")
    ps = parameter_sensitivity(fp, phase, bench_eq)
    ps.to_csv(OUT / "v3_spot_parameter_sensitivity.csv", index=False)

    print("[4/7] Walk-forward ...")
    pure, sel = walk_forward(fp, phase, bench_eq)
    pure.to_csv(OUT / "v3_spot_walk_forward.csv", index=False)
    sel.to_csv(OUT / "v3_spot_oos.csv", index=False)

    print("[5/7] Monte Carlo + Bootstrap ...")
    focus = S.Cfg(modules=S.MODELS["Full_Ensemble"])
    res_focus = C.run_strategy(fp, focus, "Full_Ensemble")
    mc = monte_carlo(res_focus["equity"], bench_eq)
    mc_row = {k: v for k, v in mc.items() if not k.endswith("_samples")}
    mc_row["kind"] = "monte_carlo"
    bs = bootstrap_drop(fp, focus, res_focus, phase, bench_eq)
    bs["kind"] = "bootstrap_drop_top"
    pd.concat([pd.DataFrame([mc_row]), bs], ignore_index=True).to_csv(
        OUT / "v3_spot_monte_carlo.csv", index=False)
    np.save(OUT / "v3_spot_mc_cagr_samples.npy", mc.get("cagr_samples", np.array([])))
    np.save(OUT / "v3_spot_mc_mdd_samples.npy", mc.get("mdd_samples", np.array([])))

    print("[6/7] Bull/Bear + re-entry ...")
    bb = bull_bear_analysis(fp, focus, res_focus, bench)
    bb.to_csv(OUT / "v3_spot_bull_bear_analysis.csv", index=False)

    print("[7/7] 保存中间对象 ...")
    with open(OUT / "experiments.pkl", "wb") as fh:
        pickle.dump({"ablation": abl, "incremental": inc, "param": ps,
                     "wf_pure": pure, "wf_selected": sel, "monte_carlo": mc_row,
                     "bootstrap": bs, "bull_bear": bb,
                     "focus_equity": res_focus["equity"],
                     "focus_weights": res_focus["weights"],
                     "bench_equity": bench_eq, "phase": phase}, fh)
    print(f"Wrote -> {OUT}")
    print("\n=== Ablation ===")
    print(abl[["label", "cagr", "sharpe", "calmar", "max_dd", "turnover_annual",
               "avg_risky", "upside_capture_bull"]].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n=== Walk-forward (pure OOS) ===")
    print(pure[["label", "cagr", "sharpe", "max_dd"]].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n=== Selected OOS ===")
    print(sel[["label", "config", "cagr", "sharpe", "max_dd"]].to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    main()