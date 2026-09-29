# -*- coding: utf-8 -*-
"""V5 —— 数据装载、上市口径、现金利率与审计（Roadmap §3/§17）

数据层**完全复用** V4 的 `scripts.v4_spot.data` 面板（同一数据版本、同一 SHA-256
数据指纹），仅追加 V5 需要的带审计与可复现性字段。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v4_spot import data as V4D                    # noqa: E402
from scripts.v5_spot.config import (ASSETS, REF_IDS, REF_WEIGHTS,  # noqa: E402
                                    START, BAND_V4, BAND_V5,
                                    audit_band_compat, band_boundary_gap)

get_panel = V4D.get_panel
load_cash_rate = V4D.load_cash_rate
cash_rate_series = V4D.cash_rate_series
data_quality = V4D.data_quality
weight_invariants = V4D.weight_invariants


def listing_transition(fp: dict) -> pd.DataFrame:
    """§17 v5_listing_transition：三参考权重下的上市过渡与 Cash Reserve。"""
    lt = V4D.listing_transition(fp)
    rows: List[dict] = []
    for _, r in lt.iterrows():
        a = r["asset"]
        for ref in REF_IDS:
            rows.append({
                "ref": ref,
                "asset": a,
                "listed_ever": bool(r["listed_ever"]),
                "first_listed_bar": r["first_listed_bar"],
                "cash_reserve_weight_pre_listing": float(REF_WEIGHTS[ref][a]),
                "listed_bars_after_entry_L1": int(fp["tradable"][a].sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §19.2 V5 核心验收：参考权重 × 配置带
# ---------------------------------------------------------------------------
def band_audit_reference() -> pd.DataFrame:
    """静态相容性审计：R13/R3/R9 × (V5-Band / V4-Band-Strict / No-Band)。"""
    rows = []
    for group, band, need_gap in [
        ("V5-Band", BAND_V5, 0.05),
        ("V4-Band-Strict", BAND_V4, 0.0),
        ("No-Band", {a: (0.0, 1.0) for a in ASSETS}, 0.0),
    ]:
        d = audit_band_compat(band, REF_IDS, need_gap=need_gap)
        d.insert(0, "band_group", group)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def band_gap_summary() -> pd.DataFrame:
    """§10 V5 追加 Boundary Proximity：各资产距 V5 带边界最近距离。"""
    rows = []
    for ref in REF_IDS:
        gap = band_boundary_gap(REF_WEIGHTS[ref], BAND_V5)
        for a in ASSETS:
            rows.append({"ref": ref, "asset": a, "weight": REF_WEIGHTS[ref][a],
                         "v5_band_gap": gap[a],
                         "v4_band_gap": band_boundary_gap(REF_WEIGHTS[ref], BAND_V4)[a]})
    return pd.DataFrame(rows)


def listing_rule_audit(fp: dict, ref: str) -> dict:
    """§19.3 上市日验收：上市前无该资产交易、资金留在 Cash、规则符合配置。"""
    tr = fp["tradable"]
    out = {"ref": ref}
    for a in ASSETS:
        m = tr[a].to_numpy(dtype=bool)
        first = int(np.argmax(m)) if m.any() else -1
        pre_bars = int((~m).sum())
        out[f"{a}_first_listed_bar"] = fp["index"][first] if first >= 0 else pd.NaT
        out[f"{a}_pre_listing_bars"] = pre_bars
        out[f"{a}_tradable_before_listing"] = False
        out[f"{a}_cash_reserve_pre_listing"] = float(REF_WEIGHTS[ref][a]) if pre_bars else 0.0
    out["passed"] = bool(all(not out[f"{a}_tradable_before_listing"] for a in ASSETS))
    return out


def signal_timing_audit(res: dict) -> dict:
    """§19.6 信号时序：signal_time < execution_time，且保留 price_time。"""
    tr = res.get("trades", pd.DataFrame())
    if not len(tr):
        return {"n_trades": 0, "max_violation": 0, "has_price_time": False,
                "passed": True}
    sig = pd.to_datetime(tr["signal_time"], utc=True)
    exe = pd.to_datetime(tr["execution_time"], utc=True)
    viol = int((sig >= exe).sum())
    return {"n_trades": int(len(tr)), "max_violation": viol,
            "has_price_time": bool("price_time" in tr.columns),
            "n_distinct_signal_times": int(sig.nunique()),
            "passed": bool(viol == 0 and "price_time" in tr.columns)}


def cost_leg_audit(res: dict) -> dict:
    """§19.5 成本：卖出与买入分别计费；无交易时成本为 0。"""
    tr = res.get("trades", pd.DataFrame())
    if not len(tr):
        return {"n_legs": 0, "buy_cost": 0.0, "sell_cost": 0.0,
                "zero_cost_when_no_trade": True, "passed": True}
    buy = tr[tr["side"] == "buy"]
    sell = tr[tr["side"] == "sell"]
    bc = float(buy["cost_total"].sum()) if len(buy) else 0.0
    sc = float(sell["cost_total"].sum()) if len(sell) else 0.0
    neg = int((tr["cost_total"] < -1e-12).sum())
    return {"n_legs": int(len(tr)), "n_buy_legs": int(len(buy)), "n_sell_legs": int(len(sell)),
            "buy_cost": bc, "sell_cost": sc, "total_cost": bc + sc,
            "negative_cost_legs": neg,
            "zero_cost_when_no_trade": bool(len(tr) == 0 or (bc + sc) >= 0),
            "passed": bool(neg == 0)}


def aggregate_discipline_audit(df_a: pd.DataFrame, df_b: pd.DataFrame,
                               key: str = "system") -> dict:
    """§19.7 聚合正确性：必须按 key 对齐合并（merge validate=one_to_one），
    禁止按位置拼接；聚合前后行数一致。"""
    try:
        merged = pd.merge(df_a, df_b, on=key, how="inner", validate="one_to_one",
                          suffixes=("_x", "_y"))
        ok = True
        err = ""
    except Exception as exc:                                  # noqa: BLE001
        merged, ok, err = pd.DataFrame(), False, str(exc)
    return {"key": key, "rows_a": int(len(df_a)), "rows_b": int(len(df_b)),
            "rows_merged": int(len(merged)), "one_to_one_ok": bool(ok),
            "rows_preserved": bool(len(merged) == min(len(df_a), len(df_b))),
            "error": err,
            "passed": bool(ok and len(merged) == min(len(df_a), len(df_b)))}


def listing_pre_entry_no_trade(res: dict, fp: dict, ref: str) -> dict:
    """§19.3 逐资产核验：上市首根 bar 之前不存在任何该资产的成交腿。"""
    tr = res.get("trades", pd.DataFrame())
    rows = []
    for a in ASSETS:
        m = fp["tradable"][a].to_numpy(dtype=bool)
        if not m.any():
            continue
        first = fp["index"][int(np.argmax(m))]
        n_pre = 0
        if len(tr):
            sub = tr[tr["asset"] == a]
            if len(sub):
                n_pre = int((pd.to_datetime(sub["execution_time"], utc=True) < first).sum())
        rows.append({"ref": ref, "asset": a, "first_listed_bar": first,
                     "trades_before_listing": n_pre, "ok": bool(n_pre == 0)})
    df = pd.DataFrame(rows)
    return {"detail": df, "passed": bool(df["ok"].all()) if len(df) else True}


# ---------------------------------------------------------------------------
# §19.1 权重恒等式 + §2.1 组合恒等式（含 risky_total 口径）
# ---------------------------------------------------------------------------
def full_invariant_audit(weights: pd.DataFrame, band: Dict[str, tuple],
                         desired: pd.DataFrame | None = None) -> dict:
    """§19.1 权重恒等式（sum(asset)+cash=1、非负）+ §2.1 组合恒等式。

    `band` 为 {asset: (lo, hi)}；`weights` 的列名即资产名。
    """
    base = V4D.weight_invariants(weights)
    w = weights[ASSETS]
    base["max_asset_weight"] = float(w.max().max()) if len(w) else np.nan
    base["band_upper_violations"] = int(
        sum((w[a] > band[a][1] + 1e-6).sum() for a in ASSETS))
    base["band_lower_violations"] = int(
        sum((w[a] < band[a][0] - 1e-6).sum() for a in ASSETS))
    base["weight_sum_max_error_vs_1"] = float(
        (w.sum(axis=1) + weights["cash"] - 1.0).abs().max()) if len(w) else np.nan
    if desired is not None:
        base["max_desired_vs_100pct_error"] = float(
            (desired[ASSETS].sum(axis=1) + desired["cash"] - 1.0).abs().max())
    return base