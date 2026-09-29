# -*- coding: utf-8 -*-
"""V4 —— 参考权重探针：EqualWeight4（25/25/25/25） vs 45/20/15/20

回答问题：把参考权重从 45/20/15/20 换成 4 资产平均配置后，
          B&H 基准（B0/B4）与 S0-S7 overlay 的收益/风险如何变化。

对照设计（4 个变体，同一引擎 / 同一成本 / 同一现金口径 / 同一数据）：
    REF45 × Band / No-Band
    EW4   × Band / No-Band
其中 Band 沿用 §2.2 的绝对配置带（以 45/20/15/20 为中枢标定），
No-Band 关闭配置带，用于剔除「ETF 带与等权中枢不匹配」的干扰。

运行：python -m scripts.v4_spot.run_refweight_probe
输出：reports/v4_spot/v4_refweight_sensitivity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.v4_spot import data as D          # noqa: E402
from scripts.v4_spot import metrics as M       # noqa: E402
from scripts.v4_spot.config import (ASSETS, LEGACY_BAND,  # noqa: E402
                                    NO_BAND, V4Cfg)
from scripts.v4_spot.run_v4 import Runner, PRIMARY  # noqa: E402

# 本探针复现 V4 已发布口径（以 45/20/15/20 为中枢标定的历史配置带），
# 故自带冻结的 OUT 与配置带，不随 §2.2 新默认（15/10/35/40）漂移。
OUT = REPO / "reports" / "v4_spot"

WEIGHTS = {
    "REF45": {"BTC": 0.45, "ETH": 0.20, "SOL": 0.15, "BNB": 0.20},
    "EW4": {a: 0.25 for a in ASSETS},
}
BAND_MODES = ["Band", "No-Band"]
BANDS = {"Band": dict(LEGACY_BAND), "No-Band": dict(NO_BAND)}
SYSTEMS = ["B0", "B4", "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
BOOT_ITER = 5000


def main() -> None:
    fp = D.get_panel()
    print(f"样本 {fp['index'][0].date()} ~ {fp['index'][-1].date()} "
          f"({len(fp['index'])} 根 4H)", flush=True)
    R = Runner(fp)
    rows = []
    for wname, w in WEIGHTS.items():
        for bm in BAND_MODES:
            cfg = V4Cfg(**{**PRIMARY.__dict__, "ref_weight": w, "band_mode": bm,
                           "band_override": BANDS[bm]})
            base = R.run("B0", cfg, ref_weight=w)
            for sid in SYSTEMS:
                res = R.run(sid, cfg, ref_weight=w)
                m = M.metrics_row(sid, "refweight", res)
                boot = M.block_bootstrap(res["equity"], base["equity"],
                                         n_iter=BOOT_ITER, seed=20260920)
                rows.append({
                    "variant": wname, "band_mode": bm, "system": sid,
                    "group": "Benchmark" if sid.startswith("B") else "Strategy",
                    "ref_weight": "/".join(f"{w[a]:.2f}" for a in ASSETS),
                    "cagr": m["cagr"], "vol_annual": m["annual_vol"],
                    "sharpe": m["sharpe"], "sortino": m["sortino"],
                    "calmar": m["calmar"], "max_dd": m["max_dd"],
                    "final_equity": m["final_wealth"],
                    "avg_risky_exposure": m["avg_risky_exposure"],
                    "turnover_annualized": m["turnover_annualized"],
                    "n_rebalances": m["rebalance_count"],
                    "prob_wins_vs_B0": boot.get("prob_strategy_wins", np.nan),
                })
            print(f"  完成 {wname} × {bm}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v4_refweight_sensitivity.csv", index=False)

    # 主口径（Band）下的并排对比
    band = df[df["band_mode"] == "Band"]
    piv = band.pivot_table(index="system", columns="variant",
                           values=["cagr", "sharpe", "max_dd", "avg_risky_exposure"])
    print("\n=== Band 口径：REF45 vs EW4 ===")
    print(piv.round(4).to_string())
    print(f"\n输出 {OUT / 'v4_refweight_sensitivity.csv'}（{len(df)} 行）")


if __name__ == "__main__":
    main()