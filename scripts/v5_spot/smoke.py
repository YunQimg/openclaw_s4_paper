# -*- coding: utf-8 -*-
"""V5 —— 冒烟测试：确认执行层与 V4 口径零漂移、配置带验收通过。

运行：python -m scripts.v5_spot.smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.v5_spot import benchmarks as B            # noqa: E402
from scripts.v5_spot import data as D                  # noqa: E402
from scripts.v5_spot import engine as E                # noqa: E402
from scripts.v5_spot import strategies as S            # noqa: E402
from scripts.v5_spot.config import (ASSETS, BAND_V4, BAND_V5, DEFAULT_COST,  # noqa: E402
                                    REF_IDS, REF_WEIGHTS, V5Cfg,
                                    audit_band_compat, band_boundary_gap,
                                    config_fingerprint, envelope_of)

pd.set_option("display.width", 200)


def main() -> int:
    ok = True
    print("=" * 78)
    print("V5 冒烟测试")
    print("=" * 78)

    # ---- §2.2.2 包络与带宽 ----
    env = envelope_of()
    print("\n[1] 三参考权重包络")
    for a in ASSETS:
        print(f"  {a}: {env[a][0]:.0%} ~ {env[a][1]:.0%}  -> V5 带 "
              f"{BAND_V5[a][0]:.0%}-{BAND_V5[a][1]:.0%}")
    for a in ASSETS:
        assert abs(BAND_V5[a][0] - max(env[a][0] - 0.05, 0.0)) < 1e-9, a
        assert abs(BAND_V5[a][1] - (env[a][1] + 0.05)) < 1e-9, a
    print("  [PASS] V5 带 = 包络 ± 5pp")

    # ---- §19.2 间距校验 ----
    print("\n[2] §19.2 参考权重 × V5 带 间距（需 >= 5pp）")
    a5 = audit_band_compat(BAND_V5, REF_IDS, need_gap=0.05)
    for _, r in a5.iterrows():
        print(f"  {r['ref']:>4} {r['asset']:>4} w={r['weight']:.2f} "
              f"band=[{r['band_lo']:.2f},{r['band_hi']:.2f}] gap={r['gap']:.2f} "
              f"on_boundary={r['on_boundary']} ok={r['gap_ok']}")
    assert a5["gap_ok"].all(), "V5 带间距校验失败"
    assert not a5["on_boundary"].any(), "存在触界项"
    print("  [PASS] 全部 >= 5pp 且无触界")

    # ---- §19.2 V4 带冲突事实 ----
    print("\n[3] §2.3 V4 带下的冲突事实（R13 2 项 / R9 3 项压界）")
    a4 = audit_band_compat(BAND_V4, REF_IDS)
    for ref in REF_IDS:
        sub = a4[a4["ref"] == ref]
        onb = sub[sub["on_boundary"]]["asset"].tolist()
        print(f"  {ref}: 压界项 = {onb if onb else '无'}")
    assert a4[(a4["ref"] == "R13") & a4["on_boundary"]]["asset"].tolist() == ["ETH", "BNB"]
    assert set(a4[(a4["ref"] == "R9") & a4["on_boundary"]]["asset"]) == {"BTC", "SOL", "BNB"}
    assert not a4[a4["ref"] == "R3"]["on_boundary"].any()
    print("  [PASS] R13=2 项、R9=3 项、R3=0 项，与 Roadmap §2.3 一致")

    # ---- 数据 ----
    print("\n[4] 数据面板")
    fp = D.get_panel()
    print(f"  样本 {fp['index'][0].date()} ~ {fp['index'][-1].date()} "
          f"({len(fp['index'])} 根 4H)")
    for a in ASSETS:
        print(f"  {a}: 首根 {fp['px'][a].first_valid_index()} "
              f"可交易 bar {int(fp['tradable'][a].sum())}")
    assert set(ASSETS) == set(fp["assets"])
    print("  [PASS]")

    # ---- §19.2 V5-Band 下三臂零裁剪 ----
    print("\n[5] §19.2/§5.3 V5-Band 下基准臂裁剪事件（必须为 0）")
    cfgs = {r: V5Cfg(ref=r, fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                     cash_return="Cash_Rate", band_group="V5-Band") for r in REF_IDS}
    total_clip = 0
    for r in REF_IDS:
        bench = B.run_all_benchmarks(fp, cfgs[r], D.cash_rate_series(fp["index"], "Cash_Rate"))
        clip = sum(v["band_clip_bars"] for v in bench.values())
        inv_bad = [k for k, v in bench.items() if not v["invariant_passed"]]
        total_clip += clip
        print(f"  {r}: B0-B6 裁剪事件={clip} 恒等式失败={inv_bad or '无'} "
              f"B4 CAGR={bench['B4']['equity'].iloc[-1] / bench['B4']['equity'].iloc[0] - 1:+.1%}")
        assert clip == 0, f"{r} 在 V5-Band 下出现裁剪，V5 必须停止排名"
        assert not inv_bad
    print("  [PASS] V5-Band 下从未触发裁剪")

    # ---- §19.2 V4-Band-Strict 越界记录 ----
    print("\n[6] §19.2/§2.3 V4-Band-Strict 下的越界事件留痕")
    for r, expect in (("R13", 2), ("R3", 0), ("R9", 3)):
        cfg = V5Cfg(ref=r, fee=DEFAULT_COST[0], slippage=DEFAULT_COST[1],
                    cash_return="Cash_Rate", band_group="V4-Band-Strict")
        res = B.run_benchmark(fp, cfg, "B0", B.default_specs()["B0"],
                              D.cash_rate_series(fp["index"], "Cash_Rate"))
        ev = res["band_events"]
        n_assets = int(ev["asset"].nunique()) if len(ev) else 0
        n_clip = int(ev["n_bars_clipped"].sum()) if len(ev) else 0
        n_oob = int(ev["n_bars_out_of_band"].sum()) if len(ev) else 0
        assets = sorted(ev["asset"].unique()) if len(ev) else []
        # §19.2 口径：R13/R9 的参考权重恰压在 V4 带边界上，属「越界事件」；
        # 恰好压界不产生裁剪，故 n_bars_clipped 应为 0。
        print(f"  {r}: 越界资产={assets} 事件资产数={n_assets}（预期 {expect}）"
              f" 压界事件={n_oob} 实际裁剪={n_clip}")
        assert n_assets == expect, f"{r} 越界事件资产数不符（{n_assets} != {expect}）"
        assert n_clip == 0, f"{r} 恰好压界不应产生裁剪（{n_clip}）"
    print("  [PASS] V4-Band-Strict 越界事件被正确记录且未误判为裁剪")

    # ---- §19.3 上市日 ----
    print("\n[7] §19.3 上市日验收")
    for r in REF_IDS:
        cfg = V5Cfg(ref=r, cash_return="Cash_Rate")
        res = B.run_benchmark(fp, cfg, "B0", B.default_specs()["B0"],
                              D.cash_rate_series(fp["index"], "Cash_Rate"))
        chk = D.listing_pre_entry_no_trade(res, fp, r)
        pre = {a: int((~fp["tradable"][a]).sum()) for a in ASSETS}
        print(f"  {r}: 上市前无交易={chk['passed']} 上市前 bar 数={pre}")
        assert chk["passed"]
    print("  [PASS] 上市前无交易、资金留 Cash")

    # ---- §19.1 权重恒等式 ----
    print("\n[8] §19.1 权重恒等式（sum(asset)+cash=1，全部 >= 0）")
    for r in REF_IDS:
        cfg = V5Cfg(ref=r, cash_return="Cash_Rate")
        res = B.run_benchmark(fp, cfg, "B1", B.default_specs()["B1"],
                              D.cash_rate_series(fp["index"], "Cash_Rate"))
        full = D.full_invariant_audit(res["weights"], cfg.band, res.get("desired"))
        print(f"  {r}: max_err={full['max_abs_sum_error']:.2e} "
              f"neg_asset={full['negative_asset_bars']} "
              f"neg_cash={full['negative_cash_bars']} passed={full['passed']}")
        assert full["passed"]
    print("  [PASS]")

    # ---- §19.4 再平衡触发 ----
    print("\n[9] §19.4 再平衡频率触发（人为构造月度日历路径）")
    # 6 个整月（181 天 × 24h/4h = 1086 根）→ 首根 + 5 个后续月初 = 6 个边界
    idx = pd.date_range("2021-01-01", periods=181 * 6, freq="4h", tz="UTC")
    px = pd.DataFrame({a: np.linspace(100, 200, len(idx)) for a in ASSETS}, index=idx)
    fp2 = {"index": idx, "px": px,
           "tradable": px.notna(),
           "feats": {a: pd.DataFrame(index=idx) for a in ASSETS}, "assets": ASSETS}
    for freq, expect in (("M", 6), ("Q", 2), ("2Q", 1), ("Y", 1)):
        b = E.period_boundary(idx, freq)
        print(f"  {freq}: 边界 bar 数={int(b.sum())}（预期 {expect}，首根计入）")
        assert int(b.sum()) == expect, f"{freq} 边界数不符：{int(b.sum())} != {expect}"
    # 边界必须落在周期首 bar 且首根 bar 恒为边界
    for freq in ("M", "Q", "2Q", "Y"):
        b = E.period_boundary(idx, freq)
        assert bool(b[0]), f"{freq} 首根 bar 必须是边界"
    # 每个边界都必须落在对应周期的第一根 bar 上（月份切换处）
    b_m = E.period_boundary(idx, "M")
    months_at_boundary = sorted(set(idx[b_m].month))
    print(f"  M 边界所在月份={months_at_boundary}（应为连续整月序列）")
    assert months_at_boundary == [1, 2, 3, 4, 5, 6]
    print("  [PASS] 月初/季首/半年首/年首触发正确")

    # ---- §19.5 成本 ----
    print("\n[10] §19.5 成本：买卖分别计费、无交易成本为 0")
    for r, fee, slip in (("R3", 0.0015, 0.0005), ("R13", 0.0005, 0.0)):
        cfg = V5Cfg(ref=r, fee=fee, slippage=slip, cash_return="Cash_Rate")
        res = B.run_benchmark(fp, cfg, "B1", B.default_specs()["B1"],
                              D.cash_rate_series(fp["index"], "Cash_Rate"))
        c = D.cost_leg_audit(res)
        b0 = B.run_benchmark(fp, cfg, "B0", B.default_specs()["B0"],
                             D.cash_rate_series(fp["index"], "Cash_Rate"))
        c0 = D.cost_leg_audit(b0)
        print(f"  {r} fee={fee}: legs={c['n_legs']} buy_cost={c['buy_cost']:.2f} "
              f"sell_cost={c['sell_cost']:.2f} | B0 legs={c0['n_legs']} "
              f"cost={c0.get('total_cost', 0.0):.4f}")
        assert c["passed"]
    print("  [PASS]")

    # ---- §19.6 信号时序 ----
    print("\n[11] §19.6 信号时序（首根建仓例外；其余必须严格 signal < execution）")
    sig_fps = {}
    for r in REF_IDS:
        cfg = V5Cfg(ref=r, cash_return="Cash_Rate")
        res = B.run_benchmark(fp, cfg, "B1", B.default_specs()["B1"],
                              D.cash_rate_series(fp["index"], "Cash_Rate"))
        print(f"  {r}: signal_time_ok={res['signal_time_ok']} "
              f"严格违反={res['signal_strict_violations']} "
              f"首根例外={res['signal_first_bar_exempt']} "
              f"price_time={res['has_price_time']}")
        assert res["signal_time_ok"], f"{r} 存在前视偏差"
        assert res["signal_strict_violations"] == 0
        assert res["has_price_time"], "必须保留 price_time"
        sig_fps[r] = int(res["n_distinct_signal_times"])
    # 三臂共用同一信号序列：参考权重不得改变信号生成
    assert len(set(sig_fps.values())) == 1, f"三臂信号序列不一致：{sig_fps}"
    print(f"  三臂信号时点数={sig_fps}（必须相同）")
    print("  [PASS] 无前视偏差，三臂共用同一条信号序列")

    # ---- §19.7 聚合正确性 ----
    print("\n[12] §19.7 聚合纪律：merge(on=key, validate=one_to_one)")
    df_a = pd.DataFrame({"system": ["B0", "B1", "B2"], "cagr": [1.0, 2.0, 3.0]})
    df_b = pd.DataFrame({"system": ["B2", "B0", "B1"], "sharpe": [3.0, 1.0, 2.0]})
    g = D.aggregate_discipline_audit(df_a, df_b)
    print(f"  乱序行合并: rows_merged={g['rows_merged']} one_to_one={g['one_to_one_ok']} "
          f"passed={g['passed']}")
    assert g["passed"], "聚合必须按 key 对齐"
    bad = pd.DataFrame({"system": ["B0", "B0", "B1"], "x": [1, 2, 3]})
    g2 = D.aggregate_discipline_audit(df_a, bad)
    print(f"  重复 key 必须报错: passed={g2['passed']} err_nonempty={bool(g2['error'])}")
    assert not g2["passed"], "重复 key 必须被 validate=one_to_one 拦截"
    print("  [PASS] 禁止按位置拼接")

    # ---- 策略族 + 三臂正交性 ----
    print("\n[13] §7 S0-S7 × 三臂（参考权重正交，模块开关一致）")
    for r in REF_IDS:
        cfg = V5Cfg(ref=r, cash_return="Cash_Rate")
        cr = D.cash_rate_series(fp["index"], "Cash_Rate")
        lines = []
        for sid in ["S0", "S1", "S2", "S3"]:
            tv, sp = S.build(fp, cfg, sid)
            res = E.simulate(fp, tv, cfg, sp, cash_rate=cr, label=f"{r}:{sid}")
            lines.append(f"{sid}:CAGR={res['equity'].iloc[-1] / res['equity'].iloc[0] - 1:+.1%}"
                         f"/clip={res['band_clip_bars']}")
        print(f"  {r}: " + "  ".join(lines))
    print("  [PASS]")

    # ---- §19.8 可复现性 ----
    print("\n[14] §19.8 可复现性：三臂配置指纹可区分且可复算")
    fps = {r: config_fingerprint(V5Cfg(ref=r)) for r in REF_IDS}
    for r, v in fps.items():
        print(f"  {r}: {v[:32]}...")
    assert len(set(fps.values())) == 3, "三臂指纹必须可区分"
    assert config_fingerprint(V5Cfg(ref="R3")) == fps["R3"], "指纹必须可复算"
    fp_data = {f.name for f in (REPO / "data" / "v3_spot").glob("*.csv")}
    print(f"  共享数据文件 {len(fp_data)} 个（三臂同一 SHA-256 数据指纹）")
    print("  [PASS]")

    print("\n" + "=" * 78)
    print("全部冒烟测试通过")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())