# -*- coding: utf-8 -*-
"""V3 现货系统 —— 冒烟测试：跑通 基准 / Model C / Model F，打印关键指标。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.v3_spot import common as C      # noqa: E402
from scripts.v3_spot import metrics as M     # noqa: E402
from scripts.v3_spot import strategy as S    # noqa: E402

t0 = time.time()
fp = C.get_fp()
print(f"bars={len(fp['index'])}  {fp['index'][0]} -> {fp['index'][-1]}  "
      f"build={time.time() - t0:.1f}s")

cfg = S.Cfg()
bh = C.run_benchmark(fp, cfg)
print("\n[B&H 45/20/15/20]")
print({k: round(v, 4) for k, v in M.summarize(bh["equity"]).items()
       if isinstance(v, float)})

for name in ["ModelA_MA200", "ModelC_MA200_ATR_ADX", "ModelF_Full", "Minimal_MA200_ATR_ADX"]:
    c = S.Cfg(modules=S.MODELS[name])
    res = C.run_strategy(fp, c, name)
    s = M.summarize(res["equity"])
    tim = M.time_in_market(res["weights"])
    print(f"\n[{name}] CAGR={s['cagr']:.2%} Sharpe={s['sharpe']:.2f} "
          f"MaxDD={s['max_dd']:.2%} Calmar={s['calmar']:.2f} "
          f"trades={len(res['trades'])} turnover={res['turnover_annual']:.2f} "
          f"avg_risky={tim['avg_risky']:.1%} tim>50={tim['tim_gt_50']:.1%}")
print(f"\ntotal {time.time() - t0:.1f}s")