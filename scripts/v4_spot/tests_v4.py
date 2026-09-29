# -*- coding: utf-8 -*-
"""V4 —— 最小验收测试（Roadmap §19）

在合成数据上验证：权重恒等式 / 上市日 / 再平衡触发 / 成本 / 信号时序 / 可复现性。
运行：python -m scripts.v4_spot.tests_v4
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from scripts.v4_spot import engine as E
from scripts.v4_spot.config import ASSETS, REF_WEIGHT, RebalanceSpec, V4Cfg
from scripts.v4_spot.data import weight_invariants

FREQ = "4h"


def synth_panel(n_days: int = 800, sol_list_day: int = 300,
                drift: float = 0.0) -> dict:
    """构造合成面板：恒定/缓慢漂移价格，SOL 在第 sol_list_day 天才可交易。"""
    idx = pd.date_range("2019-01-01", periods=n_days * 6, freq=FREQ, tz="UTC")
    px = pd.DataFrame(index=idx, columns=ASSETS, dtype=float)
    for j, a in enumerate(ASSETS):
        base = 100.0 + 50.0 * j
        ramp = 1.0 + drift * (j + 1) * np.arange(len(idx)) / len(idx)
        px[a] = base * ramp
    sol_cut = idx[sol_list_day * 6]
    px.loc[px.index < sol_cut, "SOL"] = np.nan
    feats = {a: pd.DataFrame({"close": px[a]}, index=idx) for a in ASSETS}
    return {"index": idx, "px": px, "feats": feats, "tradable": px.notna(),
            "assets": list(ASSETS)}


def flat_targets(fp: dict, w: Dict[str, float] | None = None) -> pd.DataFrame:
    w = dict(w or REF_WEIGHT)
    idx = fp["index"]
    df = pd.DataFrame({a: float(w.get(a, 0.0)) for a in ASSETS}, index=idx)
    df["cash"] = 1.0 - df[ASSETS].sum(axis=1)
    return df


def t_weights(fp, cfg) -> None:
    """§19.1 权重恒等式。"""
    for spec in [RebalanceSpec("never"), RebalanceSpec("calendar", "M"),
                 RebalanceSpec("threshold", None, 0.05)]:
        r = E.simulate(fp, flat_targets(fp), cfg, spec, label="t")
        v = weight_invariants(r["weights"])
        assert v["passed"], f"weight invariant failed for {spec.key()}: {v}"
    print("  [PASS] 19.1 权重恒等式 (sum=1, all>=0)")


def t_listing(fp, cfg) -> None:
    """§19.2 上市日处理。"""
    sol_first = fp["tradable"]["SOL"].idxmax()
    r = E.simulate(fp, flat_targets(fp), cfg, RebalanceSpec("never"), label="t")
    w = r["weights"]
    before = w.loc[w.index < sol_first, "SOL"]
    assert float(before.max()) < 1e-9, "SOL 在上市前被交易"
    cash_before = w.loc[w.index < sol_first, "cash"]
    assert abs(float(cash_before.iloc[-1]) - REF_WEIGHT["SOL"]) < 0.02, \
        f"上市前 SOL 目标权重未保留为 Cash: {float(cash_before.iloc[-1]):.4f}"
    after = w.loc[w.index >= sol_first, "SOL"]
    assert float(after.max()) > REF_WEIGHT["SOL"] - 0.05, "SOL 上市后未按 L1 建仓"
    ev = r["events"]
    assert (ev["reason"] == "listing_entry").any(), "缺少上市进入事件"
    print("  [PASS] 19.2 上市日 (上市前无交易 / Cash Reserve / L1 建仓)")


def t_rebalance(fp, cfg) -> None:
    """§19.3 日历再平衡触发。"""
    drift = synth_panel(drift=0.4)
    checks = [("M", "monthly"), ("Q", "quarterly"), ("2Q", "semiannual"), ("Y", "annual")]
    for freq, name in checks:
        r = E.simulate(drift, flat_targets(drift), cfg,
                       RebalanceSpec("calendar", freq), label="t")
        ev = r["events"]
        cal = ev[ev["reason"] == "calendar"]
        months = set(cal["time"].dt.month.tolist())
        if freq == "M":
            assert len(cal) >= 24, f"{name} 触发次数过少: {len(cal)}"
        elif freq == "Q":
            assert months <= {1, 4, 7, 10} and len(cal) >= 8, f"{name} 边界错误: {months}"
        elif freq == "2Q":
            assert months <= {1, 7}, f"{name} 边界错误: {months}"
        else:
            assert months <= {1}, f"{name} 边界错误: {months}"
    print("  [PASS] 19.3 再平衡触发 (Monthly/Quarterly/Semiannual/Annual 边界)")


def t_cost(fp, cfg) -> None:
    """§19.4 成本口径。"""
    quiet = synth_panel(drift=0.0)
    r0 = E.simulate(quiet, flat_targets(quiet),
                    V4Cfg(fee=0.0, slippage=0.0), RebalanceSpec("never"), label="t")
    assert abs(r0["turnover_total"] - sum(REF_WEIGHT.values())) < 0.02, \
        "零成本下的初始换手异常"
    assert float(r0["trades"]["cost_total"].sum()) == 0.0, "零费率下仍产生成本"
    # 无交易 -> 成本为 0
    r_none = E.simulate(quiet, flat_targets(quiet), V4Cfg(), RebalanceSpec("never"), label="t")
    n_first = len(r_none["trades"])
    tr = r_none["trades"]
    assert tr["fee"].min() >= 0 and tr["slippage"].min() >= 0, "成本符号错误"
    assert set(tr["side"].unique()) <= {"buy", "sell"}, "非法交易方向"
    # 买卖分别计费
    c = V4Cfg(fee=0.001, slippage=0.0005)
    drift = synth_panel(drift=0.5)
    rd = E.simulate(drift, flat_targets(drift), c,
                    RebalanceSpec("threshold", None, 0.02), label="t")
    sells = rd["trades"][rd["trades"]["side"] == "sell"]
    buys = rd["trades"][rd["trades"]["side"] == "buy"]
    assert len(sells) > 0 and len(buys) > 0, "阈值再平衡未产生双边交易"
    for _, row in rd["trades"].head(200).iterrows():
        exp_fee = row["notional"] * (1 + c.slippage) * c.fee if row["side"] == "buy" \
            else row["notional"] * (1 - c.slippage) * c.fee
        assert abs(row["fee"] - exp_fee) < 1e-6 * max(row["notional"], 1), "费用公式不一致"
    assert n_first >= 3
    print("  [PASS] 19.4 成本 (双边计费 / 零交易零成本 / 费用公式一致)")


def t_timing(fp, cfg) -> None:
    """§19.5 信号时序 signal_time <= execution_time，且非初始腿严格小于。"""
    d = synth_panel(drift=0.3)
    r = E.simulate(d, flat_targets(d), cfg, RebalanceSpec("threshold", None, 0.02), label="t")
    tr = r["trades"]
    assert (tr["execution_time"] >= tr["signal_time"]).all(), "存在未来函数（执行早于信号）"
    non_init = tr[tr["reason"] != "initial"]
    if not non_init.empty:
        assert (non_init["execution_time"] > non_init["signal_time"]).all(), \
            "非初始腿未按 t+1 执行"
    assert (tr["price_time"] == tr["execution_time"]).all(), "price_time 与执行时间不一致"
    print("  [PASS] 19.5 信号时序 (signal_time < execution_time, t+1 执行)")


def t_repro(fp, cfg) -> None:
    """§19.6 可复现性。"""
    d = synth_panel(drift=0.3)
    a = E.simulate(d, flat_targets(d), cfg, RebalanceSpec("calendar", "Q"), label="t")
    b = E.simulate(d, flat_targets(d), cfg, RebalanceSpec("calendar", "Q"), label="t")
    assert a["equity"].equals(b["equity"]), "同配置结果不可复现"
    ha = pd.util.hash_pandas_object(a["equity"]).sum()
    hb = pd.util.hash_pandas_object(b["equity"]).sum()
    assert ha == hb, "hash 不一致"
    print("  [PASS] 19.6 可复现性 (同配置同结果同 hash)")


def t_no_leverage(fp, cfg) -> None:
    """补充：目标权重和 >100% 时超配部分必须转入 Cash。"""
    idx = fp["index"]
    over = pd.DataFrame({a: 0.5 for a in ASSETS}, index=idx)
    r = E.simulate(fp, over, cfg, RebalanceSpec("calendar", "M"), label="t")
    assert weight_invariants(r["weights"])["passed"], "超配未转入 Cash"
    assert float(r["weights"]["cash"].min()) >= -1e-9, "出现负现金（隐含杠杆）"
    print("  [PASS] 补充: 超配转 Cash / 无负现金")


def main() -> None:
    print("=== V4 最小验收测试 (§19) ===")
    fp = synth_panel()
    cfg = V4Cfg(fee=0.001, slippage=0.0002)
    t_weights(fp, cfg)
    t_listing(fp, cfg)
    t_rebalance(fp, cfg)
    t_cost(fp, cfg)
    t_timing(fp, cfg)
    t_repro(fp, cfg)
    t_no_leverage(fp, cfg)
    print("=== 全部通过 ===")


if __name__ == "__main__":
    main()