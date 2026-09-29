# -*- coding: utf-8 -*-
"""V5 —— 主编排：三参考权重 × (带相容性 + 基准族 + 策略族)（Roadmap §20）

Phase 划分（Roadmap §20）：
  Phase 0  配置与数据冻结
  Phase 1  §2.2.2 / §19.2 配置带相容性审计
  Phase 2  §4.8 / §5.3   B0-B6 × 3 参考权重（36 个基准臂）
  Phase 3  §2.1 / §19.1  组合恒等式
  Phase 4  §20 Phase 4   **R3 对 V4 基线的复现闸门**（硬闸门）

Phase 4 是 V5 的关键闸门：若 R3 不能复现 V4 的基线结果，说明执行口径已被破坏，
V5 的全部三臂对比均无效。闸门未过时**不得**进入 Phase 5 及以后。

运行：python -m scripts.v5_spot.run_v5 --through 4
输出：reports/v5_spot/v5_*.csv + results_v5_phase4.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v4_spot import metrics as M4                      # noqa: E402
from scripts.v4_spot.config import RebalanceSpec               # noqa: E402
from scripts.v5_spot import benchmarks as B                    # noqa: E402
from scripts.v5_spot import data as D                          # noqa: E402
from scripts.v5_spot import engine as E                        # noqa: E402
from scripts.v5_spot import metrics as MT                      # noqa: E402
from scripts.v5_spot import strategies as S                    # noqa: E402
from scripts.v5_spot.config import (ASSETS, BAND_V4, BAND_V5,  # noqa: E402
                                    BENCHMARK_IDS, CAL_FREQS,
                                    CASH_MODES, COST_MATRIX,
                                    DEFAULT_COST, ENVELOPE, LISTING_RULES,
                                    LRR_CAP, LRR_PERTURB_PP, MA_LENS,
                                    REF_IDS, REF_LABEL, REF_ROLE,
                                    REF_WEIGHTS, SEED, STRATEGY_IDS,
                                    TSMOM_LENS, V5Cfg, WF_WINDOWS,
                                    config_fingerprint, envelope_of)

OUT = REPO / "reports" / "v5_spot"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = V5Cfg(fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                cash_return="Cash_Rate", listing_rule="L1",
                band_group="V5-Band", cadence="daily", rebal_threshold=0.10)

_log_t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _log_t0:7.1f}s] {msg}", flush=True)


def wf(df: pd.DataFrame, name: str) -> None:
    p = OUT / name
    df.to_csv(p, index=False, encoding="utf-8-sig")
    log(f"  -> {name}  ({len(df)} 行)")


# ---------------------------------------------------------------------------
# 运行缓存（三臂共用同一份面板与现金序列）
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, fp: dict):
        self.fp = fp
        self._cache: Dict[str, dict] = {}
        self._cr: Dict[str, pd.Series] = {}

    def cash(self, mode: str) -> pd.Series:
        if mode not in self._cr:
            self._cr[mode] = D.cash_rate_series(self.fp["index"], mode)
        return self._cr[mode]

    def bench(self, ref: str, bid: str, cfg: Optional[V5Cfg] = None) -> dict:
        c = cfg or V5Cfg(**{**PRIMARY.__dict__, "ref": ref})
        c = V5Cfg(**{**c.__dict__, "ref": ref})
        spec = B.default_specs()[bid]
        key = f"B|{ref}|{bid}|{c.tag()}|{c.ref_weight_override}"
        if key not in self._cache:
            self._cache[key] = B.run_benchmark(self.fp, c, bid, spec,
                                               self.cash(c.cash_return))
        return self._cache[key]

    def bench_ext(self, ref: str, spec_key: str, cfg: Optional[V5Cfg] = None) -> dict:
        c = cfg or V5Cfg(**{**PRIMARY.__dict__, "ref": ref})
        c = V5Cfg(**{**c.__dict__, "ref": ref})
        spec = B.extended_specs()[spec_key]
        key = f"BE|{ref}|{spec_key}|{c.tag()}"
        if key not in self._cache:
            self._cache[key] = B.run_benchmark(self.fp, c, spec_key, spec,
                                               self.cash(c.cash_return))
        return self._cache[key]

    def strat(self, ref: str, sid: str, cfg: Optional[V5Cfg] = None) -> dict:
        c = cfg or V5Cfg(**{**PRIMARY.__dict__, "ref": ref})
        c = V5Cfg(**{**c.__dict__, "ref": ref})
        key = f"S|{ref}|{sid}|{c.tag()}|{c.ref_weight_override}"
        if key not in self._cache:
            tv, spec = S.build(self.fp, c, sid)
            self._cache[key] = E.simulate(self.fp, tv, c, spec,
                                          cash_rate=self.cash(c.cash_return),
                                          label=f"{ref}:{sid}")
        return self._cache[key]

    def bench_or_strat(self, ref: str, sys_id: str, cfg: Optional[V5Cfg] = None,
                       ref_weight: Optional[Dict[str, float]] = None) -> dict:
        """按编号自动分派：B* 走基准、S* 走策略。供 Phase 7 剔除/敏感性复用。"""
        c = cfg or V5Cfg(**{**PRIMARY.__dict__, "ref": ref})
        if ref_weight is not None:
            c = V5Cfg(**{**c.__dict__, "ref": ref, "ref_weight_override": ref_weight})
        if sys_id in BENCHMARK_IDS:
            return self.bench(ref, sys_id, c)
        return self.strat(ref, sys_id, c)


# ---------------------------------------------------------------------------
# 指标记录
# ---------------------------------------------------------------------------
def summarize(res: dict, sys_id: str, kind: str, ref: str,
              extra: Optional[dict] = None) -> dict:
    """用 V4 metrics_row 出整行（含暴露/交易/成本/行为），再叠加 V5 字段。"""
    row = M4.metrics_row(sys_id, kind, res)
    row.update({
        "system": sys_id, "kind": kind, "ref": ref,
        "ref_label": REF_LABEL.get(ref, ""),
        "ref_role": REF_ROLE.get(ref, ""),
        "band_clip_bars": int(res.get("band_clip_bars", 0)),
        "invariant_passed": bool(res.get("invariant_passed", True)),
        "invariant_max_err": float(res.get("invariant_max_err", np.nan)),
        "signal_time_ok": bool(res.get("signal_time_ok", True)),
    })
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Phase 0：配置与数据冻结
# ---------------------------------------------------------------------------
def phase0(rn: Runner) -> dict:
    log("=" * 74)
    log("Phase 0 —— 配置与数据冻结")
    log("=" * 74)
    fp = rn.fp
    env = envelope_of()
    band_rows = []
    for a in ASSETS:
        band_rows.append({
            "asset": a,
            "ref_min": env[a][0], "ref_max": env[a][1],
            "envelope_width_pct": (env[a][1] - env[a][0]) * 100,
            "band_v5_lo": BAND_V5[a][0], "band_v5_hi": BAND_V5[a][1],
            "band_v5_width_pct": (BAND_V5[a][1] - BAND_V5[a][0]) * 100,
            "band_over_expand_pct": ((BAND_V5[a][1] - BAND_V5[a][0])
                                     - (env[a][1] - env[a][0])) * 100,
            "band_v4_lo": BAND_V4[a][0], "band_v4_hi": BAND_V4[a][1],
        })
    wf(pd.DataFrame(band_rows), "v5_band_envelope.csv")

    dq = D.data_quality(fp)
    wf(dq, "v5_data_quality.csv")

    # §17 审计文件：上市过渡与 Cash Reserve（三参考权重）
    lt = D.listing_transition(fp)
    wf(lt, "v5_listing_transition.csv")

    snap = {
        "phase": 0,
        "sample_start": str(fp["index"][0]), "sample_end": str(fp["index"][-1]),
        "n_bars_4h": int(len(fp["index"])), "assets": ASSETS,
        "ref_weights": REF_WEIGHTS, "ref_roles": REF_ROLE,
        "band_v5": {a: list(BAND_V5[a]) for a in ASSETS},
        "band_v4": {a: list(BAND_V4[a]) for a in ASSETS},
        "default_cost": {"fee": DEFAULT_COST[0], "slippage": DEFAULT_COST[1]},
        "cost_matrix": [list(c) for c in COST_MATRIX],
        "cash_primary": PRIMARY.cash_return,
        "listing_rule": PRIMARY.listing_rule,
        "seed": SEED, "n_bootstrap": MT.N_BOOTSTRAP, "boot_block": MT.BOOT_BLOCK,
        "primary_cfg_tag": PRIMARY.tag(),
    }
    (OUT / "v5_config_snapshot.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  样本 {snap['sample_start'][:10]} ~ {snap['sample_end'][:10]} "
        f"({snap['n_bars_4h']} 根 4H)")
    log("  -> v5_config_snapshot.json")
    return {"snapshot": snap, "envelope": env}


# ---------------------------------------------------------------------------
# Phase 1：配置带相容性审计（§2.2.2 / §2.3 / §19.2）
# ---------------------------------------------------------------------------
def phase1(rn: Runner) -> dict:
    log("=" * 74)
    log("Phase 1 —— 配置带相容性审计（§2.2.2 / §19.2）")
    log("=" * 74)
    ref_audit = D.band_audit_reference()
    wf(ref_audit, "v5_band_audit_reference.csv")
    # 参考权重 × 带边界**间距**汇总。此前被写成 `v5_band_audit.csv`，但 §17 把
    # 该文件名定义为「三参考权重 × V5带/V4带 的裁剪事件」——名实不符，按名取数会
    # 拿错表。现改名为 `v5_band_gap.csv`，`v5_band_audit.csv` 留给真正的裁剪事件
    # 审计（Phase 7 产出，见下）。
    gap = D.band_gap_summary()
    wf(gap, "v5_band_gap.csv")

    v5 = ref_audit[ref_audit["band_group"] == "V5-Band"]
    v4 = ref_audit[ref_audit["band_group"] == "V4-Band-Strict"]
    res = {
        "v5_min_gap_pp": float(v5["gap"].min() * 100),
        "v5_gap_ok_all": bool(v5["gap_ok"].all()),
        "v5_on_boundary_count": int(v5["on_boundary"].sum()),
        "v5_all_refs_inside": bool((v5["weight"] >= v5["band_lo"] - 1e-9).all()
                                   and (v5["weight"] <= v5["band_hi"] + 1e-9).all()),
        "v4_on_boundary_by_ref": {
            r: sorted(v4[(v4["ref"] == r) & v4["on_boundary"]]["asset"].tolist())
            for r in REF_IDS},
        "pairwise": MT.reference_pairwise_distance(),
    }
    wf(res["pairwise"], "v5_reference_pairwise_distance.csv")

    log(f"  V5 带最小边界间距 = {res['v5_min_gap_pp']:.1f}pp "
        f"(要求 >= 5pp, 通过={res['v5_gap_ok_all']})")
    log(f"  V5 带触界项 = {res['v5_on_boundary_count']} (要求 0)")
    log(f"  三参考权重全部落在 V5 带内 = {res['v5_all_refs_inside']}")
    for r in REF_IDS:
        log(f"  V4 带下 {r} 压界资产 = {res['v4_on_boundary_by_ref'][r]}")
    log("  三参考权重两两距离 = "
        + ", ".join(f"{x['ref_a']}-{x['ref_b']}:{x['tv_distance_pct']:.0f}pp"
                    for _, x in res["pairwise"].iterrows()))
    return res


# ---------------------------------------------------------------------------
# Phase 2：B0-B6 × 3 参考权重（36 个基准臂）
# ---------------------------------------------------------------------------
def phase2(rn: Runner) -> dict:
    log("=" * 74)
    log("Phase 2 —— B0-B6 × R13/R3/R9（§4.8）")
    log("=" * 74)
    rows, eq_frames, w_frames, e_frames = [], [], [], []
    for ref in REF_IDS:
        for bid in BENCHMARK_IDS:
            res = rn.bench(ref, bid)
            sid = f"{ref}:{bid}"
            rows.append(summarize(res, sid, "benchmark", ref, {
                "benchmark": bid, "rebalance_freq": res.get("spec", ""),
                "ref_role": REF_ROLE.get(ref, "")}))
            e = res["equity"]
            e.name = sid
            eq_frames.append(e)
            e = res.get("events", pd.DataFrame())
            if len(e):
                e = e.copy()
                e["system"] = sid
                e["ref"] = ref
                e_frames.append(e)
    sm = pd.DataFrame(rows)
    wf(sm, "v5_benchmark_summary.csv")

    eq = pd.concat(eq_frames, axis=1)
    eq.index.name = "timestamp"
    eq.index = eq.index.astype(str)
    eq.to_csv(OUT / "v5_benchmark_equity.csv", encoding="utf-8-sig")
    log(f"  -> v5_benchmark_equity.csv  ({eq.shape[0]} 行 × {eq.shape[1]} 臂)")

    # §17 基准权重序列（列名与策略权重一致：{ref}:{bid}_{asset}）
    bw = pd.concat({f"{ref}:{bid}": rn.bench(ref, bid)["weights"].resample("1D").last()
                    for ref in REF_IDS for bid in BENCHMARK_IDS}, axis=1)
    bw.columns = ["_".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
                  for c in bw.columns]
    bw.index.name = "timestamp"
    bw.to_csv(OUT / "v5_benchmark_weights.csv", encoding="utf-8-sig")
    log(f"  -> v5_benchmark_weights.csv  ({bw.shape[0]} 行 × {bw.shape[1]} 列)")

    # §17 基准成交明细
    btr = []
    for ref in REF_IDS:
        for bid in BENCHMARK_IDS:
            t = rn.bench(ref, bid).get("trades", pd.DataFrame())
            if len(t):
                t = t.copy()
                t.insert(0, "system", f"{ref}:{bid}")
                t.insert(1, "ref", ref)
                btr.append(t)
    if btr:
        pd.concat(btr, ignore_index=True).to_csv(
            OUT / "v5_benchmark_trades.csv", index=False, encoding="utf-8-sig")
        log("  -> v5_benchmark_trades.csv")

    ev = pd.concat(e_frames, ignore_index=True) if e_frames else pd.DataFrame()
    wf(ev, "v5_benchmark_rebalance_events.csv")

    # 带事件汇总（§19.2）
    band_rows = []
    for ref in REF_IDS:
        for bid in BENCHMARK_IDS:
            be = rn.bench(ref, bid).get("band_events", pd.DataFrame())
            band_rows.append({
                "system": f"{ref}:{bid}", "ref": ref, "benchmark": bid,
                "band_group": "V5-Band",
                "n_clip_assets": int(be["asset"].nunique()) if len(be) else 0,
                "n_bars_clipped": int(be["n_bars_clipped"].sum()) if len(be) else 0,
                "n_bars_out_of_band": int(be["n_bars_out_of_band"].sum()) if len(be) else 0,
                "halt_required": bool(len(be) and be["n_bars_clipped"].sum() > 0),
            })
    ba = pd.DataFrame(band_rows)
    wf(ba, "v5_benchmark_band_events.csv")

    log(f"  基准臂 = {len(sm)}（预期 21 = 7 × 3）")
    log(f"  V5-Band 下总裁剪事件 = {int(ba['n_bars_clipped'].sum())}（必须为 0）")
    return {"summary": sm, "band": ba, "equity": eq}


# ---------------------------------------------------------------------------
# Phase 3：组合恒等式（§2.1 / §19.1）
# ---------------------------------------------------------------------------
def phase3(rn: Runner) -> dict:
    log("=" * 74)
    log("Phase 3 —— 组合恒等式（§2.1 / §19.1）")
    log("=" * 74)
    rows = []
    for ref in REF_IDS:
        for bid in BENCHMARK_IDS:
            res = rn.bench(ref, bid)
            inv = D.full_invariant_audit(res["weights"], PRIMARY.band,
                                         res.get("desired"))
            rows.append({
                "system": f"{ref}:{bid}", "ref": ref, "kind": "benchmark",
                "max_abs_sum_error": inv["max_abs_sum_error"],
                "negative_asset_bars": inv["negative_asset_bars"],
                "negative_cash_bars": inv["negative_cash_bars"],
                "max_asset_weight": inv["max_asset_weight"],
                "band_upper_violations": inv["band_upper_violations"],
                "passed": bool(inv["passed"]),
            })
    inv_df = pd.DataFrame(rows)
    wf(inv_df, "v5_weight_invariants.csv")
    ok = bool(inv_df["passed"].all())
    log(f"  全部通过 = {ok}；最大误差 = {inv_df['max_abs_sum_error'].max():.2e}")
    return {"invariants": inv_df, "passed": ok}


# ---------------------------------------------------------------------------
# Phase 4：R3 对 V4 基线复现闸门（硬闸门）
# ---------------------------------------------------------------------------
V4_BASELINE = REPO / "reports" / "v4_spot_v2"
GATE_TOL = 1e-6


def phase4_gate(rn: Runner) -> dict:
    log("=" * 74)
    log("Phase 4 —— R3 对 V4 基线复现闸门（硬闸门）")
    log("=" * 74)
    cfg_v4 = V5Cfg(ref="R3", fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                   cash_return="Cash_Rate", listing_rule="L1",
                   band_group="V5-Band", cadence="daily", rebal_threshold=0.10)
    # V4 基线 cfg_tag 形如 Band|Cash_Rate|L1|fee=0.0010|slip=0.0002
    v4_tag = (f"Band|Cash_Rate|L1|fee={DEFAULT_COST[0]:.4f}"
              f"|slip={DEFAULT_COST[1]:.4f}")

    rows = []
    # 基线消歧：同一 cfg_tag 下 S3-S7 还存在 cadence=weekly 的变体行，
    # 必须只取主口径 cadence=daily（在 CSV 中表现为 NaN，因为主口径不写该列）。
    # 基准族同理存在 No-Band 对照行（band_mode 列区分）。
    METRICS = ("cagr", "sharpe", "max_dd", "total_return", "calmar", "sortino")

    def _pick_primary(df: pd.DataFrame, label: str, band_mode: str) -> Optional[pd.Series]:
        sub = df[df["label"] == label]
        if not len(sub):
            return None
        if "band_mode" in sub.columns:
            bm = sub["band_mode"].fillna("Band")
            sub = sub[bm == band_mode]
        if "cadence" in sub.columns:
            # 主口径 daily 在基线中写作 NaN
            cad = sub["cadence"]
            sub = sub[cad.isna() | (cad.astype(str) == "daily")]
        if not len(sub):
            return None
        return sub.iloc[0]

    # --- 基准 ---
    b4 = pd.read_csv(V4_BASELINE / "v4_benchmark_summary.csv")
    b4 = b4[(b4["cfg_tag"] == v4_tag) & (b4["label"].isin(BENCHMARK_IDS))]
    for bid in BENCHMARK_IDS:
        r = _pick_primary(b4, bid, "Band")
        if r is None:
            log(f"  [跳过] 基线缺失 benchmark {bid}")
            continue
        res = rn.bench("R3", bid, cfg_v4)
        got = M4.summarize_core(res["equity"])
        for m in METRICS:
            rows.append({
                "family": "benchmark", "system": bid, "metric": m,
                "v4_baseline": float(r[m]), "v5_r3": float(got[m]),
                "abs_diff": abs(float(got[m]) - float(r[m])),
                "passed": bool(abs(float(got[m]) - float(r[m])) <= GATE_TOL),
            })
    # --- 策略 ---
    s4 = pd.read_csv(V4_BASELINE / "v4_strategy_summary.csv")
    s4 = s4[(s4["cfg_tag"] == v4_tag) & (s4["label"].isin(STRATEGY_IDS))]
    for sid in STRATEGY_IDS:
        r = _pick_primary(s4, sid, "Band")
        if r is None:
            log(f"  [跳过] 基线缺失 strategy {sid}")
            continue
        res = rn.strat("R3", sid)
        got = M4.summarize_core(res["equity"])
        for m in METRICS:
            rows.append({
                "family": "strategy", "system": sid, "metric": m,
                "v4_baseline": float(r[m]), "v5_r3": float(got[m]),
                "abs_diff": abs(float(got[m]) - float(r[m])),
                "passed": bool(abs(float(got[m]) - float(r[m])) <= GATE_TOL),
            })
    gate = pd.DataFrame(rows)
    wf(gate, "v5_phase4_gate.csv")

    n_fail = int((~gate["passed"]).sum())
    worst = gate.sort_values("abs_diff", ascending=False).head(10)
    log(f"  比对项 = {len(gate)}；失败 = {n_fail}")
    if n_fail:
        log("  差异最大的 10 项：")
        for _, x in worst.iterrows():
            log(f"    {x['family']:<9} {x['system']:<3} {x['metric']:<13}"
                f" V4={x['v4_baseline']:+.6f} V5={x['v5_r3']:+.6f} "
                f"|Δ|={x['abs_diff']:.2e}")
    passed = n_fail == 0
    log(f"  闸门结果 = {'PASS 口径零漂移' if passed else 'FAIL 执行口径已被破坏'}")
    if not passed:
        log("  V5 的全部三臂对比无效，不得进入 Phase 5 及以后。")
    return {"gate": gate, "passed": passed, "n_fail": n_fail}


# ---------------------------------------------------------------------------
# Phase 5：S0-S7 × 三参考权重策略族（Roadmap §7）
# ---------------------------------------------------------------------------
SEGMENTS = ["full", "Bull", "Bear", "Sideways", "Recovery"]


def segment_masks(phase: pd.Series) -> Dict[str, pd.Index]:
    out = {"full": phase.index}
    for s in SEGMENTS[1:]:
        out[s] = phase.index[phase == s]
    return out


def seg_stats(eq: pd.Series, idx: pd.Index,
              weights: Optional[pd.DataFrame] = None) -> dict:
    """分段统计。

    §11.4 防低波动伪影：分段（OOS 窗口 / 牛熊段）内若策略近乎空仓，
    日收益序列退化为近似常数，Sharpe 会爆炸成 16~241 的伪影值。
    必须与 Phase 9 全期口径**一致**地置为 NaN，否则会污染 OOS 表
    与报告结论（此前仅全期表加防护，分段表漏防）。
    """
    s = eq.reindex(idx).dropna()
    if len(s) < 3:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                "calmar": np.nan, "ret": np.nan}
    d = M4.to_daily(s)
    yrs = max((d.index[-1] - d.index[0]).days, 1) / 365.25
    r = d.pct_change().dropna()
    cagr = (d.iloc[-1] / d.iloc[0]) ** (1 / yrs) - 1.0 if d.iloc[0] > 0 else np.nan
    mdd = M4.max_drawdown(d)
    expo = None
    if weights is not None and len(weights):
        w = weights.reindex(idx).dropna()
        if len(w):
            expo = float(M4.exposure_stats(w)["avg_risky_exposure"])
    return {"cagr": float(cagr),
            "sharpe": float(MT.safe_sharpe(r, expo)),
            "max_dd": float(mdd),
            "calmar": float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0,
            "ret": float(d.iloc[-1] / d.iloc[0] - 1.0)}


def phase5(rn: Runner, bench: pd.DataFrame) -> dict:
    """S0-S7 × R13/R3/R9 = 24 个策略臂（§7）。

    纪律（§7）：模块开关在三臂之间完全一致；参考权重是唯一变量。
    本相同时产出成本/现金敏感性所需的配置矩阵（§6/§5.2）。
    """
    log("=" * 74)
    log("Phase 5 —— S0-S7 × R13/R3/R9（§7）")
    log("=" * 74)
    fp = rn.fp
    b0 = {r: rn.bench(r, "B0") for r in REF_IDS}
    phase = M4.classify_phases(b0["R3"]["equity"])

    rows, default_runs = [], {}
    for ref in REF_IDS:
        for sid in STRATEGY_IDS:
            res = rn.strat(ref, sid)
            default_runs[(ref, sid)] = res
            row = summarize(res, f"{ref}:{sid}", "strategy", ref,
                            {"strategy": sid, "cadence": "daily"})
            rows.append(row)
    sm = pd.DataFrame(rows)
    wf(sm, "v5_strategy_summary.csv")

    # 明细序列（默认口径：daily / Cash_Rate / V5-Band）
    # 注意：to_daily/drawdown 返回 Series，其 name 必须显式设置，
    # 否则 concat 后会出现三级列（key, None, 0）导致扁平化失败。
    def _eq(k, v):
        s = M4.to_daily(v["equity"])
        s.name = f"{k[0]}:{k[1]}"
        return s

    def _dd(k, v):
        s = M4.drawdown(M4.to_daily(v["equity"]))
        s.name = f"{k[0]}:{k[1]}"
        return s

    for name, fn in (("v5_strategy_equity.csv", _eq),
                     ("v5_strategy_drawdown.csv", _dd),
                     ("v5_strategy_weights.csv",
                      lambda k, v: v["weights"].resample("1D").last())):
        out = pd.concat({f"{k[0]}:{k[1]}": fn(k, v)
                         for k, v in default_runs.items()}, axis=1)
        out.columns = ["_".join(str(x) for x in c) if isinstance(c, tuple) else str(c)
                       for c in out.columns]
        out.index.name = "timestamp"
        out.to_csv(OUT / name, encoding="utf-8-sig")
        log(f"  -> {name}  ({out.shape[0]} 行 × {out.shape[1]} 列)")

    tr = []
    for (ref, sid), v in default_runs.items():
        t = v.get("trades", pd.DataFrame())
        if len(t):
            t = t.copy()
            t.insert(0, "system", f"{ref}:{sid}")
            t.insert(1, "ref", ref)
            tr.append(t)
    if tr:
        pd.concat(tr, ignore_index=True).to_csv(
            OUT / "v5_strategy_trades.csv", index=False, encoding="utf-8-sig")
        log("  -> v5_strategy_trades.csv")

    # 信号 vs 实际（焦点策略 S3，用于 §19.6 目视核验）
    # §17 通例要求所有 v5_ 输出带 `ref` 列，故按 ref 分列输出。
    #
    # 注意：`desired` 是**应用带之后的期望目标权重**，其数值依臂不同是设计使然
    # （风险打开时 target 即该臂参考权重，R13 的 BTC 15% ≠ R9 的 25%），因此
    # **不能**用「目标权重逐位相同」来证明臂中性；「去带后归一化比例
    # r = target_a / w^ref_a 逐位相同」同样不成立（S4–S7 重新归一化，映射非线性）。
    # 臂中性见下方对**信号层两个不变量**的核验。
    focus_sid = "S3"
    parts = []
    for ref in REF_IDS:
        if (ref, focus_sid) not in default_runs:
            continue
        sig = default_runs[(ref, focus_sid)]["desired"].resample("1D").last()
        act = default_runs[(ref, focus_sid)]["weights"].resample("1D").last()
        sig.columns = [f"target_{c}" for c in sig.columns]
        act.columns = [f"actual_{c}" for c in act.columns]
        t = sig.join(act).reset_index()
        t.insert(0, "ref", ref)
        parts.append(t)
    if parts:
        sig_all = pd.concat(parts, ignore_index=True)
        wf(sig_all, "v5_strategy_signals.csv")

    # §19.6/§7 臂中性核验：判据只能落在**信号层**（带约束与组合归一化之前）的决策量上。
    # 为什么不能用「目标权重 / 参考权重」逐位比较（实测偏差最高 2.7e-01，均为机制性）：
    #   * 带约束是**绝对**上下限（`_apply_band` 的 `min(col, hi)`），同一个上界会咬在
    #     各臂不同的目标值上；
    #   * S4–S7 在资产级开关（自身趋势 / 质量门）之后会对**仍在场的资产重新归一化**，
    #     于是「参考权重 → 目标权重」的映射不是逐资产成比例的。
    # 上述两点都在信号**之后**发生，属策略机制，不是信号污染。
    # 因此这里核验两个真正的信号层不变量（均用 V4-Band-Raw 去带重建，排除带约束）：
    #   ① 资产开关掩码 `target_a > 0`（何时持有何资产）—— 跨臂必须逐格一致；
    #   ② `RebalanceSpec`（信号节奏 mode/freq/threshold/basis）—— 跨臂必须一致。
    raw_cfgs = {r: V5Cfg(**{**PRIMARY.__dict__, "ref": r,
                            "band_group": "V4-Band-Raw"}) for r in REF_IDS}
    dev_mask = 0
    n_spec: List[int] = []
    for sid in STRATEGY_IDS:
        masks: Dict[str, np.ndarray] = {}
        specs: Dict[str, str] = {}
        for ref in REF_IDS:
            tv, spec = S.build(fp, raw_cfgs[ref], sid)
            cols = [a for a in ASSETS if a in tv.columns]
            masks[ref] = np.asarray(tv[cols].fillna(0.0) > 0.0)
            specs[ref] = spec.key()
        for ref in REF_IDS[1:]:
            dev_mask += int((masks[ref] != masks[REF_IDS[0]]).sum())
        n_spec.append(len(set(specs.values())))
    log(f"  §19.6/§7 臂中性核验（去带重建）：S0–S7 资产开关掩码 target_a>0 的跨臂不一致"
        f"格数 = {dev_mask}（必须为 0）；RebalanceSpec 跨臂取值数 = "
        f"{sorted(set(n_spec))}（必须全为 1）")

    # 成本 / 现金敏感性（每个参考权重 × 成本档 × 现金档）
    crows = []
    for ref in REF_IDS:
        for fee, slip in COST_MATRIX:
            for cash in CASH_MODES:
                cfg = V5Cfg(**{**PRIMARY.__dict__, "ref": ref, "fee": fee,
                               "slippage": slip, "cash_return": cash})
                for sid in STRATEGY_IDS:
                    res = rn.strat(ref, sid, cfg)
                    m = M4.summarize_core(res["equity"])
                    crows.append({
                        "ref": ref, "strategy": sid, "fee": fee, "slippage": slip,
                        "cost_per_leg": fee + slip, "cash_mode": cash,
                        "cagr": m["cagr"], "sharpe": m["sharpe"],
                        "max_dd": m["max_dd"], "final_wealth": m["final_wealth"]})
    cost_df = pd.DataFrame(crows)
    wf(cost_df, "v5_cost_sensitivity.csv")
    wf(cost_df[cost_df["cash_mode"] == PRIMARY.cash_return].copy(),
       "v5_cash_sensitivity.csv")

    log(f"  策略臂 = {len(sm)}（预期 24 = 8 × 3）")
    log(f"  V5-Band 下三臂裁剪事件 = {int(sm['band_clip_bars'].sum())}（必须为 0）")
    log(f"  恒等式失败 = {int((~sm['invariant_passed']).sum())}")
    return {"summary": sm, "runs": default_runs, "phase": phase,
            "cost": cost_df, "bench_b0": b0}


# ---------------------------------------------------------------------------
# Phase 6：策略 vs B&H Rebalance 三路公平比较（§9 / §11）
# ---------------------------------------------------------------------------
def phase6(rn: Runner, ph5: dict, ph2: dict) -> dict:
    log("=" * 74)
    log("Phase 6 —— 策略 vs B&H Rebalance 三路公平比较（§9/§11）")
    log("=" * 74)
    phase = ph5["phase"]
    masks = segment_masks(phase)
    runs = ph5["runs"]
    bench_runs = ph5["bench_b0"]

    # ---- §9.1 差值表：每个参考权重内，策略 × B0-B6 ----
    spread_rows = []
    for ref in REF_IDS:
        bm = {b: M4.summarize_core(rn.bench(ref, b)["equity"])
              for b in BENCHMARK_IDS}
        for sid in STRATEGY_IDS:
            sr = M4.summarize_core(runs[(ref, sid)]["equity"])
            for b in BENCHMARK_IDS:
                br = bm[b]
                spread_rows.append({
                    "ref": ref, "strategy": sid, "benchmark": b,
                    "cagr_spread": sr["cagr"] - br["cagr"],
                    "sharpe_spread": sr["sharpe"] - br["sharpe"],
                    "maxdd_spread": sr["max_dd"] - br["max_dd"],
                    "calmar_spread": sr["calmar"] - br["calmar"],
                    "final_wealth_ratio": (sr["final_wealth"] / br["final_wealth"]
                                           if br["final_wealth"] else np.nan),
                })
    spread = pd.DataFrame(spread_rows)
    wf(spread, "v5_benchmark_comparison.csv")

    # ---- §11.1 分段比较 ----
    seg_rows = []
    for ref in REF_IDS:
        for sid in STRATEGY_IDS:
            for b in ["B0", "B3", "B4"]:
                br = rn.bench(ref, b)
                for seg in SEGMENTS:
                    idx = masks[seg]
                    ss = seg_stats(runs[(ref, sid)]["equity"], idx,
                                   runs[(ref, sid)]["weights"])
                    bs = seg_stats(br["equity"], idx, br["weights"])
                    seg_rows.append({
                        "ref": ref, "strategy": sid, "benchmark": b,
                        "segment": seg, "days": int(len(idx) / 6),
                        "strat_cagr": ss["cagr"], "bench_cagr": bs["cagr"],
                        "strat_maxdd": ss["max_dd"], "bench_maxdd": bs["max_dd"],
                        "strat_sharpe": ss["sharpe"], "bench_sharpe": bs["sharpe"],
                        "active_return": ss["ret"] - bs["ret"]})
    wf(pd.DataFrame(seg_rows), "v5_segment_comparison.csv")

    # ---- Active Return / Capture / Opportunity cost ----
    ar, cap, opp = [], [], []
    for ref in REF_IDS:
        for sid in STRATEGY_IDS:
            for b in BENCHMARK_IDS:
                be = rn.bench(ref, b)["equity"]
                se = runs[(ref, sid)]["equity"]
                a = M4.active_stats(be, se)
                ar.append({"ref": ref, "strategy": sid, "benchmark": b, **a})
                c = M4.capture_ratios(phase, be, se)
                cap.append({"ref": ref, "strategy": sid, "benchmark": b, **c})
                sd, bd = M4.to_daily(se), M4.to_daily(be)
                sd, bd = sd.align(bd, join="inner")
                for y, idx_y in sd.groupby(sd.index.year).groups.items():
                    si, bi = sd.loc[idx_y], bd.loc[idx_y]
                    rb = bi.pct_change().dropna()
                    rs = si.pct_change().dropna()
                    rb, rs = rb.align(rs, join="inner")
                    up = rb > 0
                    opp.append({
                        "ref": ref, "strategy": sid, "benchmark": b, "year": int(y),
                        "strat_return": float(si.iloc[-1] / si.iloc[0] - 1.0),
                        "bench_return": float(bi.iloc[-1] / bi.iloc[0] - 1.0),
                        "opportunity_cost": float((rb[up] - rs[up]).clip(lower=0).sum())
                        if up.any() else 0.0})
    wf(pd.DataFrame(ar), "v5_active_return.csv")
    wf(pd.DataFrame(cap), "v5_capture_ratio.csv")
    wf(pd.DataFrame(opp), "v5_opportunity_cost.csv")

    # ---- §11.2 同风险 / 同收益匹配（在每个参考权重内匹配）----
    match_rows = []
    for ref in REF_IDS:
        bm = {b: M4.summarize_core(rn.bench(ref, b)["equity"])
              for b in BENCHMARK_IDS}
        for sid in STRATEGY_IDS:
            sr = M4.summarize_core(runs[(ref, sid)]["equity"])
            b_risk = min(bm, key=lambda b: abs(bm[b]["max_dd"] - sr["max_dd"]))
            b_ret = min(bm, key=lambda b: abs(bm[b]["cagr"] - sr["cagr"]))
            match_rows.append({
                "ref": ref, "strategy": sid,
                "risk_matched_benchmark": b_risk,
                "risk_matched_bench_maxdd": bm[b_risk]["max_dd"],
                "strategy_maxdd": sr["max_dd"],
                "risk_matched_cagr_spread": sr["cagr"] - bm[b_risk]["cagr"],
                "risk_matched_sharpe_spread": sr["sharpe"] - bm[b_risk]["sharpe"],
                "return_matched_benchmark": b_ret,
                "return_matched_bench_cagr": bm[b_ret]["cagr"],
                "return_matched_maxdd_spread": sr["max_dd"] - bm[b_ret]["max_dd"],
                "return_matched_sharpe_spread": sr["sharpe"] - bm[b_ret]["sharpe"],
            })
    wf(pd.DataFrame(match_rows), "v5_risk_return_matching.csv")

    log(f"  差值表 = {len(spread)} 行；分段 = {len(seg_rows)} 行")
    log(f"  Active Return = {len(ar)} 行；Capture = {len(cap)} 行")
    return {"spread": spread,
            "active": pd.DataFrame(ar), "capture": pd.DataFrame(cap),
            "matching": pd.DataFrame(match_rows)}


# ---------------------------------------------------------------------------
# Phase 7：资产剔除 / 参数敏感性 / 配置带对照 / LRR 邻域剖面（§9.2 / §13）
# ---------------------------------------------------------------------------
def phase7(rn: Runner, ph5: dict, ph2: dict) -> dict:
    log("=" * 74)
    log("Phase 7 —— 资产剔除 / 参数 / 配置带对照 / LRR（§9.2/§13）")
    log("=" * 74)
    fp = rn.fp
    runs = ph5["runs"]
    phase = ph5["phase"]

    # ---- §9.2 资产剔除（三参考权重各自剔除，剔除后按剩余资产归一）----
    def _drop(base: Dict[str, float], *drop: str) -> Dict[str, float]:
        w = {a: (0.0 if a in drop else float(base[a])) for a in ASSETS}
        tot = sum(w.values())
        return {k: (v / tot if tot > 0 else v) for k, v in w.items()}

    rows = []
    for ref in REF_IDS:
        base_rw = REF_WEIGHTS[ref]
        subsets = {"ALL4": dict(base_rw)}
        for a in ASSETS:
            subsets[f"drop_{a}"] = _drop(base_rw, a)
        subsets["EqualWeight4"] = {a: 0.25 for a in ASSETS}
        # §9.4 清单中的「只保留 BTC / ETH」是**同时剔除两个资产**的二资产子集，
        # 与 drop_SOL / drop_BNB 的单剔不同，必须单列（此前漏）。
        subsets["BTC_ETH_only"] = _drop(base_rw, "SOL", "BNB")
        for name, wn in subsets.items():
            cfg = V5Cfg(**{**PRIMARY.__dict__, "ref": ref, "ref_weight_override": wn})
            # B0–B4 全频率入表：§9.4 要求核查「B&H CAGR 单调性是否只在含 SOL
            # 样本中出现」，只跑 B4 无法回答（此前漏）。
            for sysid in ["B0", "B1", "B2", "B3", "B4", "S3", "S5"]:
                res = rn.bench_or_strat(ref, sysid, cfg, ref_weight=wn)
                m = M4.summarize_core(res["equity"])
                rows.append({
                    "ref": ref, "subset": name, "system": sysid,
                    "cagr": m["cagr"], "sharpe": m["sharpe"],
                    "max_dd": m["max_dd"], "calmar": m["calmar"],
                    "turnover_annualized": m.get("turnover_annualized", np.nan)})
    excl = pd.DataFrame(rows)
    wf(excl, "v5_asset_exclusion.csv")

    # ---- §13 参考权重邻域扰动 + LRR（V5 核心风险点）----
    lrr_rows, nbr_rows = [], []
    for ref in REF_IDS:
        perturbs = MT.perturb_weights(ref, LRR_PERTURB_PP[0])
        perturbs.update({k: v for k, v in MT.perturb_weights(ref, LRR_PERTURB_PP[1]).items()})
        base_res = runs[(ref, "S4")]
        base_score = MT.composite_score(
            {"system": "S4", **M4.summarize_core(base_res["equity"])})
        nbr_scores: Dict[str, float] = {}
        for nkey, w in perturbs.items():
            cfg = V5Cfg(**{**PRIMARY.__dict__, "ref": ref, "ref_weight_override": w})
            tv, spec = S.build(fp, cfg, "S4")
            rr = E.simulate(fp, tv, cfg, spec, cash_rate=rn.cash(cfg.cash_return),
                            label=f"{ref}:S4@{nkey[:18]}")
            sc = MT.composite_score({"system": "S4", **M4.summarize_core(rr["equity"])})
            nbr_scores[nkey] = sc
            nbr_rows.append({"ref": ref, "neighbor": nkey, "score": sc,
                             "weights": json.dumps(w)})
        lt = MT.lrr_table({ref: base_score}, {ref: nbr_scores}, ref)
        lt["system"] = "S4"
        lt["n_perturb_levels"] = len(LRR_PERTURB_PP)
        lrr_rows.append(lt)
    lrr = pd.concat(lrr_rows, ignore_index=True)
    wf(lrr, "v5_reference_lrr.csv")
    wf(pd.DataFrame(nbr_rows), "v5_reference_lrr_detail.csv")

    log("  LRR（Local Robustness Ratio，§13）")
    for _, x in lrr.iterrows():
        flag = "  <<< 局部低谷" if x["lrr_raw"] > LRR_CAP + 1e-12 else ""
        log(f"    {x['ref']}: raw={x['lrr_raw']:.4f} "
            f"capped={x['lrr_capped']:.4f} 邻居={int(x['n_neighbors'])} "
            f"判定={x['lrr_verdict']}{flag}")

    # ---- 配置带对照：V5-Band vs No-Band vs V4-Band-Strict vs V4-Band-Raw ----
    b_rows = []
    for ref in REF_IDS:
        for grp in ["V5-Band", "No-Band", "V4-Band-Strict", "V4-Band-Raw"]:
            mode = "No-Band" if grp == "No-Band" else "Band"
            cfg = V5Cfg(**{**PRIMARY.__dict__, "ref": ref,
                           "band_group": grp, "band_mode": mode})
            for sysid in ["B4", "S3", "S4"]:
                if sysid == "B4":
                    res = B.run_benchmark(fp, cfg, "B4",
                                          B.default_specs()["B4"],
                                          rn.cash(cfg.cash_return))
                else:
                    tv, spec = S.build(fp, cfg, sysid)
                    res = E.simulate(fp, tv, cfg, spec,
                                     cash_rate=rn.cash(cfg.cash_return),
                                     label=f"{ref}:{sysid}")
                m = M4.summarize_core(res["equity"])
                be = res.get("band_events", pd.DataFrame())
                b_rows.append({
                    "ref": ref, "band_group": grp, "system": sysid,
                    "cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"],
                    "calmar": m["calmar"],
                    "avg_risky_exposure": M4.exposure_stats(res["weights"])["avg_risky_exposure"],
                    # 注意：V4-Band-Strict 下 R13/R9 的参考权重**恰好压在** V4 带边界上，
                    # 属「压界」而非「严格越界」，故 band_clip_bars 为 0 但
                    # n_bars_out_of_band 不为 0。两列必须并列展示，否则会被误读为
                    # 「带约束未生效」。语义见 engine.band_audit 文档。
                    "band_clip_bars": int(res.get("band_clip_bars", 0)),
                    "n_bars_out_of_band": (int(be["n_bars_out_of_band"].sum())
                                           if len(be) else 0),
                    "band_event_assets": int(be["asset"].nunique()) if len(be) else 0,
                    "invariant_passed": bool(res.get("invariant_passed", True)),
                })
    bm = pd.DataFrame(b_rows)
    wf(bm, "v5_band_comparison.csv")
    log(f"  配置带对照 = {len(bm)} 行；"
        f"V5-Band 裁剪 = {int(bm[bm.band_group == 'V5-Band']['band_clip_bars'].sum())}")

    # ---- §17 核心：`v5_band_audit.csv` = 三参考权重 × {V5带 / V4带 / V4-Band-Raw /
    # No-Band} × 代表系统 的**裁剪事件**审计。此前该文件名被 Phase 1 的「带边界
    # 间距汇总」占用（名实不符）；现改由本表承担，并把两类事件显式分列：
    #   n_bars_clipped     —— 严格越界（真实裁剪）
    #   n_bars_out_of_band —— 恰好压界（§19.2 口径，非裁剪）
    # 二者必须并列看，否则会把「压界」误读成「带未生效」。
    ba_full = bm[["ref", "band_group", "system", "band_clip_bars",
                  "n_bars_out_of_band", "band_event_assets"]].copy()
    ba_full = ba_full.rename(columns={"band_clip_bars": "n_bars_clipped"})
    ba_full["strict_clipped"] = ba_full["n_bars_clipped"] > 0
    ba_full["boundary_touch_only"] = ((~ba_full["strict_clipped"])
                                      & (ba_full["n_bars_out_of_band"] > 0))
    ba_full["audit_verdict"] = np.where(
        ba_full["strict_clipped"], "严格越界（需停止并复查带宽）",
        np.where(ba_full["boundary_touch_only"],
                 "恰好压界（非裁剪，§19.2 口径）", "无事件"))
    wf(ba_full, "v5_band_audit.csv")
    log(f"  带审计：严格越界 = {int(ba_full['n_bars_clipped'].sum())}；"
        f"压界（非裁剪）= {int(ba_full['n_bars_out_of_band'].sum())}")

    # ---- §13 参数敏感性（每个参考权重）----
    p_rows = []
    for ref in REF_IDS:
        for ma in MA_LENS:
            cfg = V5Cfg(**{**PRIMARY.__dict__, "ref": ref, "ma_len": ma})
            for sid in ["S3", "S5"]:
                tv, spec = S.build(fp, cfg, sid)
                rr = E.simulate(fp, tv, cfg, spec, cash_rate=rn.cash(cfg.cash_return),
                                label=f"{ref}:{sid}_ma{ma}")
                m = M4.summarize_core(rr["equity"])
                p_rows.append({"ref": ref, "family": sid, "param": "ma_len",
                               "value": ma, "cagr": m["cagr"], "sharpe": m["sharpe"],
                               "max_dd": m["max_dd"], "calmar": m["calmar"]})
        for ts in TSMOM_LENS:
            cfg = V5Cfg(**{**PRIMARY.__dict__, "ref": ref, "tsmom_days": ts})
            tv, spec = S.build(fp, cfg, "S4")
            rr = E.simulate(fp, tv, cfg, spec, cash_rate=rn.cash(cfg.cash_return),
                            label=f"{ref}:S4_ts{ts}")
            m = M4.summarize_core(rr["equity"])
            p_rows.append({"ref": ref, "family": "S4", "param": "tsmom_days",
                           "value": ts, "cagr": m["cagr"], "sharpe": m["sharpe"],
                           "max_dd": m["max_dd"], "calmar": m["calmar"]})
    wf(pd.DataFrame(p_rows), "v5_parameter_sensitivity.csv")

    # ---- ablation：策略族增量（三臂）----
    chain = [("S3", "MA200"), ("S4", "+ TSMOM6M"), ("S5", "+ ATR"),
             ("S6", "+ Adaptive"), ("S7", "+ MTF/Persistence (Full V3)")]
    a_rows = []
    for ref in REF_IDS:
        prev = None
        for sid, note in chain:
            m = M4.summarize_core(runs[(ref, sid)]["equity"])
            row = {"ref": ref, "system": sid, "increment": note,
                   "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
                   "calmar": m["calmar"], "max_dd": m["max_dd"],
                   "turnover_annualized": m.get("turnover_annualized", np.nan)}
            if prev is not None:
                row.update({"dcagr": m["cagr"] - prev["cagr"],
                            "dsharpe": m["sharpe"] - prev["sharpe"],
                            "dmax_dd": m["max_dd"] - prev["max_dd"]})
            prev = row
            a_rows.append(row)
    wf(pd.DataFrame(a_rows), "v5_ablation.csv")

    return {"exclusion": excl, "lrr": lrr, "band": bm}


# ---------------------------------------------------------------------------
# Phase 8：Walk Forward / OOS / Bootstrap / Monte Carlo（§11.3 / §12）
# ---------------------------------------------------------------------------
# V5 的 walk-forward 与 V4 的关键区别（§12 第 3 条）：
#   候选集是**参考权重 × 策略**，参考权重必须在 Train/Validate 上选出，
#   再冻结到 Test 评估 —— 这正是 V5 要回答的核心问题。
WF_STRATEGIES: List[str] = ["S2", "S3", "S4", "S5"]


def phase8(rn: Runner, ph5: dict, ph2: dict) -> dict:
    log("=" * 74)
    log("Phase 8 —— Walk Forward / OOS / Bootstrap / MC（§11.3/§12）")
    log("=" * 74)
    fp = rn.fp
    phase = ph5["phase"]
    runs = ph5["runs"]

    # ---- §12 Walk Forward：在 Train/Validate 上选参考权重+策略（冻结后测）----
    wf_rows, sel_rows = [], []
    for w_i, (t0, t1, v1, te) in enumerate(WF_WINDOWS, start=1):
        tr_idx = fp["index"][(fp["index"] >= pd.Timestamp(t0, tz="UTC")) &
                             (fp["index"] <= pd.Timestamp(v1, tz="UTC"))]
        te_idx = fp["index"][(fp["index"] > pd.Timestamp(v1, tz="UTC")) &
                             (fp["index"] <= pd.Timestamp(te, tz="UTC"))]
        best, best_score = None, -np.inf
        for ref in REF_IDS:
            for sid in WF_STRATEGIES:
                s = seg_stats(runs[(ref, sid)]["equity"], tr_idx,
                              runs[(ref, sid)]["weights"])
                score = s["calmar"] if s["calmar"] == s["calmar"] else -np.inf
                wf_rows.append({
                    "window": w_i, "train_start": t0, "train_validate_end": v1,
                    "test_end": te, "ref": ref, "system": sid,
                    "train_cagr": s["cagr"], "train_sharpe": s["sharpe"],
                    "train_max_dd": s["max_dd"], "train_calmar": s["calmar"],
                    "selected": False})
                if score > best_score:
                    best, best_score = (ref, sid), score
        for r in wf_rows[-len(REF_IDS) * len(WF_STRATEGIES):]:
            r["selected"] = (r["ref"], r["system"]) == best
        s_te = seg_stats(runs[best]["equity"], te_idx, runs[best]["weights"])
        sel_rows.append({
            "window": w_i, "ref_selected": best[0], "system_selected": best[1],
            "train_start": t0, "test_start": str(te_idx[0].date()) if len(te_idx) else "",
            "test_end": str(te_idx[-1].date()) if len(te_idx) else "",
            "test_cagr": s_te["cagr"], "test_sharpe": s_te["sharpe"],
            "test_max_dd": s_te["max_dd"], "test_calmar": s_te["calmar"]})
    wfdf = pd.DataFrame(wf_rows)
    wf(wfdf, "v5_walk_forward.csv")
    sel = pd.DataFrame(sel_rows)

    # ---- OOS：Pure（固定三臂默认）vs Selected（Train/Validate 选出）----
    rows = []
    for w_i, (t0, t1, v1, te) in enumerate(WF_WINDOWS, start=1):
        te_idx = fp["index"][(fp["index"] > pd.Timestamp(v1, tz="UTC")) &
                             (fp["index"] <= pd.Timestamp(te, tz="UTC"))]
        # §11.4 同臂对比纪律：基准必须与策略同参考权重。
        # 此前三臂共用 R3 的基准（bench_cagr 三臂完全相同），
        # 使 R13/R9 的 active_return 相对错误的基准计算。
        btest = {r: {b: seg_stats(rn.bench(r, b)["equity"], te_idx,
                                  rn.bench(r, b)["weights"])
                     for b in ["B0", "B3", "B4"]}
                 for r in REF_IDS}
        for ref in REF_IDS:
            for sid in STRATEGY_IDS:
                s = seg_stats(runs[(ref, sid)]["equity"], te_idx,
                              runs[(ref, sid)]["weights"])
                for b, bs in btest[ref].items():
                    rows.append({
                        "mode": "Pure_OOS", "window": w_i, "ref": ref,
                        "system": sid, "benchmark": b,
                        "test_cagr": s["cagr"], "test_sharpe": s["sharpe"],
                        "test_max_dd": s["max_dd"], "bench_cagr": bs["cagr"],
                        "bench_max_dd": bs["max_dd"],
                        "active_return": s["ret"] - bs["ret"],
                        "maxdd_improvement": s["max_dd"] - bs["max_dd"]})
        sx = sel[sel["window"] == w_i].iloc[0]
        sel_ref = sx["ref_selected"]
        s = seg_stats(runs[(sel_ref, sx["system_selected"])]["equity"], te_idx,
                      runs[(sel_ref, sx["system_selected"])]["weights"])
        for b, bs in btest[sel_ref].items():
            rows.append({
                "mode": "Selected_OOS", "window": w_i,
                "ref": sx["ref_selected"], "system": sx["system_selected"],
                "benchmark": b, "test_cagr": s["cagr"], "test_sharpe": s["sharpe"],
                "test_max_dd": s["max_dd"], "bench_cagr": bs["cagr"],
                "bench_max_dd": bs["max_dd"],
                "active_return": s["ret"] - bs["ret"],
                "maxdd_improvement": s["max_dd"] - bs["max_dd"]})
    wf(pd.DataFrame(rows), "v5_oos.csv")

    # 参考权重 OOS 稳定性小结（§12 核心产出）
    oos_rows = []
    for ref in REF_IDS:
        sub = pd.DataFrame(rows)
        sub = sub[(sub["mode"] == "Pure_OOS") & (sub["ref"] == ref)]
        for sid in STRATEGY_IDS:
            sd = sub[sub["system"] == sid]
            if not len(sd):
                continue
            oos_rows.append({
                "ref": ref, "system": sid,
                "n_windows": int(sd["window"].nunique()),
                "mean_test_cagr": float(sd["test_cagr"].mean()),
                "mean_test_sharpe": float(sd["test_sharpe"].mean()),
                "worst_test_max_dd": float(sd["test_max_dd"].min()),
                "mean_active_return": float(sd["active_return"].mean()),
                "n_windows_beat_B4": int((sd[sd["benchmark"] == "B4"]["active_return"] > 0).sum())})
    wf(pd.DataFrame(oos_rows), "v5_oos_stability.csv")

    # ---- §11.3 Bootstrap / Monte Carlo：三臂共用种子与抽样块 ----
    br, mr = [], []
    for ref in REF_IDS:
        for sid in STRATEGY_IDS:
            se = runs[(ref, sid)]["equity"]
            for b in BENCHMARK_IDS:
                r = M4.block_bootstrap(se, rn.bench(ref, b)["equity"],
                                       n_iter=MT.N_BOOTSTRAP, block=MT.BOOT_BLOCK,
                                       seed=SEED)
                if r:
                    br.append({"ref": ref, "strategy": sid, "benchmark": b,
                               "seed": SEED, "block": MT.BOOT_BLOCK, **r})
            mc = M4.monte_carlo(se, n_iter=MT.N_BOOTSTRAP, block=MT.BOOT_BLOCK,
                                seed=SEED)
            if mc:
                mr.append({"ref": ref, "strategy": sid, "seed": SEED, **mc})
    wf(pd.DataFrame(br), "v5_bootstrap.csv")
    wf(pd.DataFrame(mr), "v5_monte_carlo.csv")

    log("  参考权重选择（Walk Forward）：")
    for _, x in sel.iterrows():
        log(f"    窗口{x['window']}: 选中 {x['ref_selected']}:{x['system_selected']} "
            f"→ Test CAGR={x['test_cagr']:+.4f} Sharpe={x['test_sharpe']:.4f}")
    return {"wf": wfdf, "oos": pd.DataFrame(rows), "bootstrap": pd.DataFrame(br),
            "monte_carlo": pd.DataFrame(mr), "selected": sel}


# ---------------------------------------------------------------------------
# Phase 9：硬门槛 / 评分 / 三臂两两对比 / 最终中枢选择（§14）
# ---------------------------------------------------------------------------
def phase9(rn: Runner, ph5: dict, ph6: dict, ph7: dict, ph8: dict) -> dict:
    log("=" * 74)
    log("Phase 9 —— 硬门槛 / 评分 / 三臂对比 / 中枢选择（§14）")
    log("=" * 74)
    sm = ph5["summary"]
    runs = ph5["runs"]
    oos = ph8["oos"]
    boot = ph8["bootstrap"]
    lrr = ph7["lrr"]

    # ---- §11.4(a) 统一参考分布：全臂（基准+策略）并集作为一个 pool ----
    bm_rows = []
    for ref in REF_IDS:
        for bid in BENCHMARK_IDS:
            m = M4.summarize_core(rn.bench(ref, bid)["equity"])
            bm_rows.append({"system": f"{ref}:{bid}", "kind": "benchmark",
                            "ref": ref, "benchmark": bid, **m})
    bm_df = pd.DataFrame(bm_rows)
    pool = pd.concat([bm_df, sm], ignore_index=True)
    log(f"  统一参考分布 = {len(pool)} 个臂（基准 {len(bm_df)} + 策略 {len(sm)}）")

    # 三臂各自的 B0/B4 参照
    b0 = {r: bm_df[(bm_df.ref == r) & (bm_df.benchmark == "B0")].iloc[0] for r in REF_IDS}
    b4 = {r: bm_df[(bm_df.ref == r) & (bm_df.benchmark == "B4")].iloc[0] for r in REF_IDS}

    # ---- §14 硬门槛（逐臂）----
    gate_rows = []
    for _, s in sm.iterrows():
        ref, sid = s["ref"], s["strategy"]
        pure = oos[(oos["mode"] == "Pure_OOS") & (oos["ref"] == ref) &
                   (oos["system"] == sid)]
        n_win = int(pure["window"].nunique())
        pos = int((pure[pure.benchmark == "B4"]["active_return"] > 0).sum())
        dd_better = int((pure["maxdd_improvement"] > 0.02).sum())
        half = max(n_win // 2, 1)
        gates = {
            "gate_maxdd_vs_B0_15pp": bool((s["max_dd"] - b0[ref]["max_dd"]) >= 0.15),
            "gate_sharpe_vs_B4_0.90": bool(s["sharpe"] >= 0.90 * b4[ref]["sharpe"]),
            "gate_oos_half_windows": bool(pos >= half or dd_better >= half),
            "gate_exposure_50": bool(s.get("avg_risky_exposure", 0) >= 0.50),
            "gate_cost_25pct": bool(s.get("cost_pct_of_gross_profit", 0) <= 0.25),
            "gate_invariants": bool(s["invariant_passed"]),
            "gate_band_clean": int(s["band_clip_bars"]) == 0,
            "gate_no_lookahead": bool(s["signal_time_ok"]),
        }
        gates["passed_all"] = all(v for k, v in gates.items() if k != "passed_all")
        gate_rows.append({
            "ref": ref, "strategy": sid, **gates,
            "oos_windows": n_win, "oos_positive_active": pos,
            "oos_maxdd_better": dd_better,
            "cagr": s["cagr"], "sharpe": s["sharpe"], "sortino": s["sortino"],
            "calmar": s["calmar"], "max_dd": s["max_dd"],
            "avg_risky_exposure": s.get("avg_risky_exposure"),
            "cost_pct_of_gross_profit": s.get("cost_pct_of_gross_profit"),
            "maxdd_vs_B0": s["max_dd"] - b0[ref]["max_dd"],
            "sharpe_vs_B4": s["sharpe"] / b4[ref]["sharpe"] if b4[ref]["sharpe"] else np.nan,
        })
    gate = pd.DataFrame(gate_rows)
    wf(gate, "v5_hard_gates.csv")
    log(f"  硬门槛通过 = {int(gate['passed_all'].sum())}/{len(gate)} 臂")
    for g in ("gate_maxdd_vs_B0_15pp", "gate_sharpe_vs_B4_0.90",
              "gate_cost_25pct", "gate_band_clean", "gate_no_lookahead"):
        log(f"    {g:<26} 通过 {int(gate[g].sum()):>2}/{len(gate)}")

    # ---- §14.2 评分：统一参考分布百分位 ----
    # §11.4 防伪影：把低波动/近空仓产生的爆炸型 Sharpe 置为 NaN，
    # 使其不参与百分位与加权（否则 S1 的 OOS Sharpe=65 会污染整池）。
    sm = sm.copy()
    bad_sharpe = []
    for i, s in sm.iterrows():
        res = runs[(s["ref"], s["strategy"])]
        d = M4.to_daily(res["equity"])
        expo = float(M4.exposure_stats(res["weights"])["avg_risky_exposure"])
        if not MT.sharpe_reliable(d.pct_change().dropna(), expo):
            bad_sharpe.append(f"{s['ref']}:{s['strategy']}(std={float(d.pct_change().std()):.2e})")
            sm.loc[i, "sharpe"] = np.nan
            sm.loc[i, "sortino"] = np.nan
    if bad_sharpe:
        log(f"  §11.4 剔除不可信 Sharpe 的臂（近空仓/极低波动）：{len(bad_sharpe)} 个")
        for x in bad_sharpe:
            log(f"    {x}")

    scored = MT.score_table(sm, pool=pool)
    # active_stability 取自 OOS 胜率
    stab = {}
    for _, s in sm.iterrows():
        pure = oos[(oos["mode"] == "Pure_OOS") & (oos["ref"] == s["ref"]) &
                   (oos["system"] == s["strategy"]) & (oos["benchmark"] == "B4")]
        stab[(s["ref"], s["strategy"])] = (
            float((pure["active_return"] > 0).mean()) if len(pure) else np.nan)
    scored["active_stability"] = [stab[(r, x)] for r, x in
                                  zip(scored["ref"], scored["strategy"])]
    scored["final_score"] = [
        MT.composite_score(
            {**r, "active_stability": r["active_stability"]},
            {k: r[f"rank_pct_{k}"] for k in ("sharpe", "sortino", "calmar", "cagr")})
        for r in scored.to_dict("records")]
    # 附 LRR（S4 的邻域剖面按臂套用）
    lrr_map = {str(x["ref"]): float(x["lrr_raw"]) for _, x in lrr.iterrows()}
    scored["lrr_raw"] = scored["ref"].map(lrr_map)
    scored["lrr_capped"] = scored["lrr_raw"].clip(upper=LRR_CAP)
    keep = ["ref", "strategy", "final_score", "cagr", "sharpe", "sortino", "calmar",
            "max_dd", "rank_pct_sharpe", "rank_pct_sortino", "rank_pct_calmar",
            "rank_pct_cagr", "active_stability", "lrr_raw", "lrr_capped"]
    sc = scored[keep].sort_values("final_score", ascending=False)
    wf(sc, "v5_candidate_scores.csv")

    # ---- §10 三参考权重两两对比 ----
    pair_rows = []
    for i, ra in enumerate(REF_IDS):
        for rb in REF_IDS[i + 1:]:
            for sid in STRATEGY_IDS:
                a = sm[(sm.ref == ra) & (sm.strategy == sid)].iloc[0]
                bb = sm[(sm.ref == rb) & (sm.strategy == sid)].iloc[0]
                pair_rows.append({
                    "ref_a": ra, "ref_b": rb, "strategy": sid,
                    "dcagr": a["cagr"] - bb["cagr"],
                    "dsharpe": a["sharpe"] - bb["sharpe"],
                    "dmaxdd": a["max_dd"] - bb["max_dd"],
                    "dcalmar": a["calmar"] - bb["calmar"],
                    "dexposure": (a.get("avg_risky_exposure", np.nan)
                                  - bb.get("avg_risky_exposure", np.nan)),
                    "dturnover": (a.get("turnover_annualized", np.nan)
                                  - bb.get("turnover_annualized", np.nan)),
                    "dcost_pct": (a.get("cost_pct_of_gross_profit", np.nan)
                                  - bb.get("cost_pct_of_gross_profit", np.nan)),
                })
    pair = pd.DataFrame(pair_rows)
    wf(pair, "v5_reference_comparison.csv")

    # 臂级汇总（每个参考权重的平均表现）
    agg_rows = []
    for ref in REF_IDS:
        sub = sm[sm.ref == ref]
        ssub = sc[sc.ref == ref]
        pure = oos[(oos["mode"] == "Pure_OOS") & (oos["ref"] == ref) &
                   (oos["benchmark"] == "B4")]
        agg_rows.append({
            "ref": ref, "ref_label": REF_LABEL[ref], "role": REF_ROLE[ref],
            "mean_cagr": float(sub["cagr"].mean()),
            "mean_sharpe": float(sub["sharpe"].mean()),
            "mean_max_dd": float(sub["max_dd"].mean()),
            "mean_calmar": float(sub["calmar"].mean()),
            "mean_exposure": float(sub["avg_risky_exposure"].mean()),
            "best_score": float(ssub["final_score"].max()),
            "best_strategy": str(ssub.iloc[0]["strategy"]),
            "oos_mean_active": float(pure["active_return"].mean()),
            "oos_beat_B4_windows": int((pure["active_return"] > 0).sum()),
            "lrr_raw": lrr_map.get(ref, np.nan),
            "lrr_verdict": "非常平坦" if lrr_map.get(ref, 0) >= 0.98 else "见 v5_reference_lrr.csv",
        })
    agg = pd.DataFrame(agg_rows)
    wf(agg, "v5_reference_summary.csv")
    log("  三臂臂级汇总：")
    for _, x in agg.iterrows():
        log(f"    {x['ref']:<4} 均CAGR={x['mean_cagr']:+.4f} 均Sharpe={x['mean_sharpe']:.4f} "
            f"均MaxDD={x['mean_max_dd']:+.4f} 最佳={x['best_strategy']}"
            f"({x['best_score']:.1f}) OOS超额={x['oos_mean_active']:+.4f}")

    # ---- 最终中枢选择（§14 + §12：训练期与样本外并重）----
    picks = []
    # 按评分
    top = sc.iloc[0]
    picks.append({
        "role": "评分最优（统一参考分布）",
        "ref": top["ref"], "system": top["strategy"],
        "final_score": top["final_score"], "cagr": top["cagr"],
        "sharpe": top["sharpe"], "max_dd": top["max_dd"],
        "evidence": "统一 pool 百分位加权得分最高"})
    # 训练期（WF）被选中最多
    sel = ph8["selected"]
    rc = sel["ref_selected"].value_counts()
    wf_ref = str(rc.index[0])
    wf_sys = str(sel[sel.ref_selected == wf_ref]["system_selected"]
                 .value_counts().index[0])
    wf_row = sc[(sc.ref == wf_ref) & (sc.strategy == wf_sys)]
    if not len(wf_row):
        wf_row = sc[sc.ref == wf_ref].head(1)
    wf_row = wf_row.iloc[0]
    picks.append({
        "role": "Walk-Forward 选中最多（训练期口径）",
        "ref": wf_ref, "system": wf_sys,
        "final_score": float(wf_row["final_score"]),
        "cagr": float(wf_row["cagr"]), "sharpe": float(wf_row["sharpe"]),
        "max_dd": float(wf_row["max_dd"]),
        "evidence": f"4 窗中被选中 {int(rc.iloc[0])} 次"})
    # 样本外最强
    oos_best = agg.sort_values("oos_mean_active", ascending=False).iloc[0]
    # §11.4(c) 口径纪律：ref/system 必须与 cagr/sharpe/max_dd 来自同一行。
    # 此前 cagr/sharpe/max_dd 误取 agg 的「臂级 8 策略均值」，
    # 而 system 取的是该臂评分最高的策略，导致报告 §6 出现
    # 「R9 S4」配「R9 臂均值 70.0%/1.420/-54.0%」的错配（真值 84.9%/1.631/-51.1%）。
    ob_row = sm[(sm.ref == oos_best["ref"]) &
                (sm.strategy == oos_best["best_strategy"])].iloc[0]
    ob_score = float(sc[(sc.ref == ob_row["ref"]) &
                        (sc.strategy == ob_row["strategy"])]["final_score"].iloc[0])
    picks.append({
        "role": "样本外最强（Pure OOS vs B4）",
        "ref": ob_row["ref"], "system": ob_row["strategy"],
        "final_score": ob_score,
        "cagr": float(ob_row["cagr"]), "sharpe": float(ob_row["sharpe"]),
        "max_dd": float(ob_row["max_dd"]),
        "evidence": f"OOS 平均超额 {oos_best['oos_mean_active']:+.4f}，"
                    f"跑赢 B4 {int(oos_best['oos_beat_B4_windows'])} 个窗口观测"
                    f"（臂级口径，8 策略均值 CAGR {oos_best['mean_cagr']:.4f}）"})
    # 最低回撤
    dd_best = sm.loc[sm["max_dd"].idxmax()]
    picks.append({
        "role": "回撤最小",
        "ref": dd_best["ref"], "system": dd_best["strategy"],
        "final_score": float(sc[(sc.ref == dd_best["ref"]) &
                                (sc.strategy == dd_best["strategy"])]["final_score"].iloc[0]),
        "cagr": dd_best["cagr"], "sharpe": dd_best["sharpe"],
        "max_dd": dd_best["max_dd"],
        "evidence": f"MaxDD={dd_best['max_dd']:+.4f} 为全部 24 臂中最浅"})
    picks_df = pd.DataFrame(picks)
    wf(picks_df, "v5_final_candidates.csv")

    return {"gates": gate, "scores": sc, "pairwise": pair, "aggregate": agg,
            "candidates": picks_df, "pool_size": len(pool)}


# ---------------------------------------------------------------------------
# Phase 10：冻结执行规则 + §22 结论要点（面向报告的可机读摘要）
# ---------------------------------------------------------------------------
def _oos_excess_note(oos: pd.DataFrame) -> str:
    """§22 披露 3：样本外平均超额为负的归因，**数字全部取自 v5_oos.csv**。

    此前该段落为手写常量，一旦基准口径修复就会与新数字脱节；
    改为数据驱动，报告引用的每个数值都可回溯到 OOS 表。
    超额一律相对**同参考权重**的基准（§11.4 同臂对比纪律）。
    """
    pure = oos[oos["mode"] == "Pure_OOS"]
    # 口径必须与 v5_reference_summary.csv 的 oos_mean_active 一致：只取 B4
    # （§11.4 同臂对比纪律下的主基准）。此前对 B0/B3/B4 取均值，导致同一份
    # 报告里同一指标出现两个数值（表 -0.573 vs 正文 -0.422）。
    exc = pure[pure["benchmark"] == "B4"].groupby("ref")["active_return"].mean()
    w1 = pure[(pure["window"] == 1) & (pure["benchmark"] == "B4")]
    w1_b4 = w1.groupby("ref")["bench_cagr"].max()
    w4 = pure[pure["window"] == 4]
    parts = []
    for r in REF_IDS:
        sub = w4[w4["ref"] == r]
        if not len(sub):
            continue
        best = sub.loc[sub["test_cagr"].idxmax()]
        b4 = float(sub[sub["benchmark"] == "B4"]["bench_cagr"].iloc[0])
        parts.append(f"{r} 最佳 {best['system']} {best['test_cagr'] * 100:+.1f}% "
                     f"vs B4 {b4 * 100:+.1f}%")
    return (
        "三臂的 Pure OOS 平均超额（相对 B4）均为负（"
        + "、".join(f"{r} {exc[r]:+.3f}" for r in REF_IDS if r in exc)
        + "），原因是窗口1（2023 测试年，牛市延续）满仓 B&H 基准 CAGR 极高（"
        + "、".join(f"{r} B4 {w1_b4[r] * 100:.0f}%" for r in REF_IDS if r in w1_b4)
        + "），任何带风险控制的策略都必然大幅跑输，该窗口主导了简单均值。"
        "逐窗口看，窗口4（2026）策略显著占优："
        + "；".join(parts)
        + "。因此『平均超额为负』不等于『策略无效』，须结合窗口与风险指标判读。"
        "注意：该口径与 `v5_reference_summary.csv` 的 `oos_mean_active` 完全一致，"
        "全部超额均相对**同参考权重**的基准计算（§11.4 同臂对比纪律）。")


def phase10(rn: Runner, ph9: dict, ph5: dict, ph7: dict, ph8: dict) -> dict:
    log("=" * 74)
    log("Phase 10 —— 冻结执行规则 + §22 结论要点")
    log("=" * 74)
    agg = ph9["aggregate"]
    sc = ph9["scores"]
    lrr = ph7["lrr"]

    # ---- 参考权重推荐：训练期与样本外综合（不得只看单一证据）----
    agg = agg.copy()
    agg["rank_train"] = agg["best_score"].rank(ascending=False, method="min")
    agg["rank_oos"] = agg["oos_mean_active"].rank(ascending=False, method="min")
    agg["combined_rank"] = agg["rank_train"] + agg["rank_oos"]
    agg = agg.sort_values(["combined_rank", "best_score"],
                          ascending=[True, False]).reset_index(drop=True)
    chosen = agg.iloc[0]
    # 并列检测：combined_rank 相同即为并列，不得凭排序位置宣称胜出
    n_tied = int((agg["combined_rank"] == chosen["combined_rank"]).sum())
    is_tied = n_tied > 1
    wf(agg, "v5_reference_selection.csv")

    # ---- §21 尾句兜底：三者包络的中位中枢 ----
    # 「如果三个参考权重无法区分，则不得为了给出结论而强行选出唯一中枢；
    #   应输出三权重不可区分，并以三者包络的中位中枢 + V5 默认带作为执行口径。」
    # 判定「无法区分」的口径：训练期名次与样本外名次不一致（排序证据互相矛盾），
    # 或综合名次并列。命中时兜底口径与推荐臂**同时**输出，不得只给唯一解。
    mid_hub = {a: float(np.median([REF_WEIGHTS[r][a] for r in REF_IDS]))
               for a in ASSETS}
    mid_hub = {a: round(v, 4) for a, v in mid_hub.items()}
    mid_cash = round(max(0.0, 1.0 - sum(mid_hub.values())), 4)
    ambiguous = bool(is_tied or chosen["rank_train"] != chosen["rank_oos"])
    fallback = {
        "applicable": ambiguous,
        "reference_weight": mid_hub,
        "cash": mid_cash,
        "band": {a: list(BAND_V5[a]) for a in ASSETS},
        "band_group": "V5-Band",
        "note": ("训练期与样本外排序不一致（或综合名次并列）→ 按 §21 尾句视为"
                 "『三参考权重无法区分』，不得强行选出唯一中枢；"
                 "此处给出的中位中枢为三者权重的逐资产中位数，"
                 "与 V5 默认带共同构成兜底执行口径，与推荐臂并列输出。"
                 ) if ambiguous else "训练期与样本外排序一致，兜底口径不适用。",
    }

    # ---- 冻结执行规则 ----
    rules = {
        "reference_weight": {a: REF_WEIGHTS[chosen["ref"]][a] for a in ASSETS},
        "reference_id": chosen["ref"], "reference_label": REF_LABEL[chosen["ref"]],
        "band": {a: list(BAND_V5[a]) for a in ASSETS},
        "band_group": "V5-Band",
        "rebalance": {"S3": "threshold ±10% absolute", "S4": "threshold ±10% absolute",
                      "B4": "annual calendar"},
        "cash": PRIMARY.cash_return,
        "listing_rule": PRIMARY.listing_rule,
        "cost": {"fee": PRIMARY.fee, "slippage": PRIMARY.slippage,
                 "per_leg": PRIMARY.cost_per_leg},
        "cadence": PRIMARY.cadence, "exec_lag_bars": PRIMARY.exec_lag_bars,
        "stop_conditions": [
            "V5-Band 下出现任何 band_clip > 0 → 立即停止并复查带宽",
            "组合恒等式 |sum(asset)+cash-1| > 1e-6 → 停止",
            "signal_time >= execution_time（首根建仓除外）→ 停止",
            "任一资产上市前出现该资产成交腿 → 停止",
        ],
        "selection_basis": {
            "combined_rank": float(chosen["combined_rank"]),
            "train_best_score": float(chosen["best_score"]),
            "oos_mean_active": float(chosen["oos_mean_active"]),
            "train_and_oos_agree": bool(chosen["rank_train"] == chosen["rank_oos"]),
            "is_tied": bool(is_tied),
            "n_tied_arms": int(n_tied),
            "tie_note": ("三臂综合名次并列（训练期与样本外排序恰好互补），"
                         "此时按 best_score 次序取首个，属**并列**而非胜出；"
                         "最终参考权重的取舍须结合 §22 共同短板与投资目标判断，"
                         "不得宣称该臂在所有维度占优。") if is_tied else "",
        },
        "fallback_mid_hub": fallback,
    }
    (OUT / "v5_execution_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  -> v5_execution_rules.json（推荐参考权重 {chosen['ref']} "
        f"{REF_LABEL[chosen['ref']]}）")
    log(f"     训练期名次={int(chosen['rank_train'])} 样本外名次={int(chosen['rank_oos'])} "
        f"两者一致={rules['selection_basis']['train_and_oos_agree']}")

    # ---- §22 结论要点（可机读，供报告直接引用）----
    sm = ph5["summary"]
    gate = ph9["gates"]
    fails_all = gate[~gate["passed_all"]]
    never_fail = [g for g in ("gate_band_clean", "gate_no_lookahead", "gate_invariants")
                  if bool(gate[g].all())]
    common_short = {
        "maxdd_all_arms_worst": float(sm["max_dd"].min()),
        "maxdd_median": float(sm["max_dd"].median()),
        "n_arms_maxdd_worse_than_-50pct": int((sm["max_dd"] < -0.50).sum()),
        "n_arms_total": int(len(sm)),
        "lrr_all": {str(x["ref"]): float(x["lrr_raw"]) for _, x in lrr.iterrows()},
        "lrr_convention_caveat": (
            "V5 的 LRR 邻域为 ±5pp/±10pp 保和扰动并受 V5 带过滤（10-12 个邻居），"
            "V4.8 的 LRR 基于完整权重候选网格。两者口径不同，"
            "V5 的 LRR 更接近 1 主要是邻域更窄的机械结果，不可直接断言『更鲁棒』。"),
        "sharpe_artifact_guard": {
            "min_daily_std": MT.MIN_DAILY_STD,
            "min_avg_exposure": MT.MIN_AVG_EXPOSURE,
            "note": ("近空仓窗口（如 S1 在 2023/2026 窗口平均风险暴露≈0）"
                     "的日收益近乎恒定，Sharpe 会爆炸到 16~241 的伪影值。"
                     "V5 将其置为 NaN，不参与统一参考分布百分位与评分；"
                     "该防护同时作用于全期表与分段/OOS 表，两侧口径一致。"),
        },
        "oos_negative_excess_explained": _oos_excess_note(ph8["oos"]),
    }
    conclusion = {
        "frozen_rules": rules,
        "gates": {
            "n_arms": int(len(gate)),
            "n_passed": int(gate["passed_all"].sum()),
            "never_failed_across_arms": never_fail,
            "failed_arms": fails_all[["ref", "strategy"]].to_dict("records"),
        },
        "common_shortcomings": common_short,
        "top_scored": sc.head(5)[["ref", "strategy", "final_score", "cagr",
                                  "sharpe", "max_dd"]].to_dict("records"),
        "reference_ranking": agg.sort_values("combined_rank")[
            ["ref", "ref_label", "rank_train", "rank_oos", "combined_rank",
             "oos_mean_active"]].to_dict("records"),
    }
    (OUT / "v5_conclusion.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    log("  -> v5_conclusion.json")

    # ---- 共同短板（§22 要求正面披露）----
    log("  §22 共同短板：")
    log(f"    24 臂 MaxDD 中位数 = {common_short['maxdd_median']:+.4f}，"
        f"最差 = {common_short['maxdd_all_arms_worst']:+.4f}")
    log(f"    MaxDD 深于 -50% 的臂数 = "
        f"{common_short['n_arms_maxdd_worse_than_-50pct']}/{common_short['n_arms_total']}")
    log(f"    重复验收项从未失败：{', '.join(never_fail)}")
    return {"rules": rules, "conclusion": conclusion}


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------
def write_repro(rn: Runner, phases: dict) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": SEED,
        "fingerprints": {r: config_fingerprint(V5Cfg(**{**PRIMARY.__dict__, "ref": r}))
                         for r in REF_IDS},
        "shared_data_sha256": {
            f.name: __import__("hashlib").sha256(f.read_bytes()).hexdigest()
            for f in sorted((REPO / "data" / "v3_spot").glob("*.csv"))},
        "primary_cfg_tag": PRIMARY.tag(),
        "phase4_gate_passed": bool(phases.get("phase4", {}).get("passed", False)),
    }
    (OUT / "v5_reproducibility.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    log("  -> v5_reproducibility.json")
    for r, v in payload["fingerprints"].items():
        log(f"     {r} 指纹 {v[:24]}...")


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=4,
                    help="执行到第几个 Phase（默认 4）")
    args = ap.parse_args(argv)

    log("V5 回测启动 —— 三参考权重公平对比")
    fp = D.get_panel()
    rn = Runner(fp)

    phases: Dict[str, dict] = {}
    phases["phase0"] = phase0(rn)
    phases["phase1"] = phase1(rn)
    phases["phase2"] = phase2(rn)
    phases["phase3"] = phase3(rn)

    if args.through >= 4:
        phases["phase4"] = phase4_gate(rn)

    # 硬闸门：Phase 4 未过则禁止进入 Phase 5 及以后（Roadmap §20 Phase 4）
    gate_ok = phases.get("phase4", {}).get("passed", False)
    if args.through >= 5:
        if not gate_ok:
            log("Phase 4 闸门未通过 —— 拒绝执行 Phase 5 及以后。")
            return 2
        phases["phase5"] = phase5(rn, phases["phase2"]["summary"])
        if args.through >= 6:
            phases["phase6"] = phase6(rn, phases["phase5"], phases["phase2"])
        if args.through >= 7:
            phases["phase7"] = phase7(rn, phases["phase5"], phases["phase2"])
        if args.through >= 8:
            phases["phase8"] = phase8(rn, phases["phase5"], phases["phase2"])
        if args.through >= 9:
            phases["phase9"] = phase9(rn, phases["phase5"], phases["phase6"],
                                      phases["phase7"], phases["phase8"])
        if args.through >= 10:
            phases["phase10"] = phase10(rn, phases["phase9"], phases["phase5"],
                                        phases["phase7"], phases["phase8"])

    write_repro(rn, phases)

    with (OUT / "results_v5_phase4.pkl").open("wb") as f:
        pickle.dump({
            "run": rn,
            "summary": phases.get("phase2", {}).get("summary"),
            "phase4_gate": phases.get("phase4", {}).get("gate"),
            "strategy_summary": phases.get("phase5", {}).get("summary"),
        }, f)
    log("  -> results_v5_phase4.pkl")

    log("=" * 74)
    g = phases.get("phase4", {})
    if "phase4" in phases:
        log(f"Phase 4 闸门：{'PASS' if g['passed'] else 'FAIL'}")
    done = [k for k in ("phase1", "phase2", "phase3", "phase4", "phase5", "phase6",
                        "phase7", "phase8", "phase9", "phase10") if k in phases]
    log(f"已完成：{', '.join(done)}")
    log("=" * 74)
    return 0 if gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())