# -*- coding: utf-8 -*-
"""V5 —— 策略候选族 S0-S7 × 三参考权重（Roadmap §7）

规则**完全继承** V4 `scripts.v4_spot.strategies`（继承 V3 指标层），
参考权重作为独立维度与该编号**正交**。

V5 追加纪律（§7）：
  * S1-S7 的模块开关在三臂之间必须完全一致；
  * 禁止「为某个参考权重单独调参」——那会把参考权重比较变成新的历史参数优化。
本模块因此**不暴露任何 ref 相关的分支**：参考权重只通过 `cfg.ref_weight` 进入。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v3_spot import strategy as V3S                    # noqa: E402
from scripts.v4_spot import strategies as V4S                  # noqa: E402
from scripts.v4_spot.config import RebalanceSpec               # noqa: E402
from scripts.v4_spot.config import V4Cfg                       # noqa: E402
from scripts.v5_spot.config import ASSETS, V5Cfg               # noqa: E402
from scripts.v5_spot.engine import _to_v4_cfg                  # noqa: E402

STRATEGY_TITLES: Dict[str, str] = {
    "S0": "Static Reference（No Rebalance）",
    "S1": "Annual B&H + MA200 资产级风险减仓",
    "S2": "B&H Rebalance + 组合级 MA200 Overlay",
    "S3": "MA200 Minimal（带 + 日频 + 阈值再平衡）",
    "S4": "MA200 + TSMOM 6M",
    "S5": "Minimal Quality（MA200 + ATR + ADX）",
    "S6": "Reduced Balanced（MA200 + ADX + Adaptive）",
    "S7": "Full V3 Replication（仅复现性确认）",
}
DEFAULT_SPECS: Dict[str, RebalanceSpec] = dict(V4S.DEFAULT_SPECS)
V3_MODULES: Dict[str, List[str]] = dict(V4S.V3_MODULES)


def _v4_proxy(cfg: V5Cfg) -> V4Cfg:
    """策略**信号层**代理配置 —— 必须与 V4 逐位一致（Roadmap §7）。

    V3 策略层（S3-S7 走的 `targets_v3` → `V3S.build_strategy`）**同时**用
    `cfg.band` 做两件事（见 `scripts/v4_spot/strategies.py::_v3_cfg`）：

      1. `band=cfg.band`              —— 参与目标区间 `lo/hi` 的构造（信号生成）
      2. `enforce_lower_band`         —— 上行趋势中把权重托到带下界（信号生成）

    因此若把 V5 带直接透传给信号层，**V5 的带宽就会改变信号本身** —— 这违反
    §7「参考权重与配置带不得改变信号生成」，且实测会让 S4-S7 的目标权重产生
    恰好等于外扩量 5pp 的偏差。

    这里强制信号层使用 **V4 冻结带 + V4 原生 `enforce_lower_band` 语义**，
    使 S0-S7 的信号与 V4 基线逐位一致；V5 的配置带约束**只在信号生成之后**
    由 `_apply_band` 施加。
    """
    from scripts.v4_spot.config import BAND as V4_BAND
    import dataclasses
    v4 = _to_v4_cfg(cfg)
    return dataclasses.replace(v4, band_override=dict(V4_BAND))


def _apply_band(tv: pd.DataFrame, cfg: V5Cfg) -> pd.DataFrame:
    """V5 配置带约束（Roadmap §2.2.3 / §2.3）—— **在信号生成之后**施加。

    为什么不能对绝对权重做 `clip(lower=lo)`
    ----------------------------------------
    配置带的语义是约束「相对参考权重的**主动偏离**」，不是禁止资产离场。
    V4 的策略层通过**按比例缩放**表达减仓（见 `targets_s1`：`tv[a] = rw[a] * scale`，
    再经 `_norm(panic=1.0)` 只封顶不加底），因此：

    * 若对绝对权重直接 `clip(lower=lo)`，策略**故意持有的 0 权重**
      （S1/S3 的 MA200 风险减仓日、上市前资产）会被强行托到带下界，
      风险控制机制被整段摧毁 —— 这是本模块曾犯过的错误，由 Phase 4 闸门拦下。
    * 正确做法：上限约束增持；下界只对「参考权重本身低于带下界」的情形生效，
      减仓方向一律允许到 0（减仓是降低风险，带约束不应阻止）。

    口径
    ----
    * `V5-Band`        —— 按 V5 带施加比例约束；三参考权重均满足 ≥5pp 间距，
                          故对 B&H 型策略恒为 no-op（§19.2 要求 0 次裁剪）。
    * `V4-Band-Strict` —— 按 V4 带施加；R13/R9 的参考权重恰压在 V4 带边界上，
                          会被上界截断并留痕（§2.3）。
    * `V4-Band-Raw`    —— 显式声明允许越界，不作任何带约束（仅审计记录）。
    * `No-Band`        —— 不作任何带约束。
    """
    out = tv.copy()
    if not cfg.enforce_band:
        return out
    rw = cfg.ref_weight
    for a in ASSETS:
        w_ref = float(rw.get(a, 0.0))
        lo, hi = cfg.band[a]
        col = out[a].astype(float)
        # 1) 上限：增持不得超过带上限
        col = np.minimum(col, hi)
        # 2) 下限：只把「不低于带下界」施加于**参考权重本身**。
        #    参考权重若已低于带下界（V4 带下 R13 的 ETH/BNB、R9 的 BTC/SOL/BNB），
        #    目标不得低于该下界 —— 这是 §2.3 要求留痕的裁剪情形。
        #    参考权重在带内时（V5 带下的全部三臂），下界**不约束**目标，
        #    否则会把策略的风险减仓强行托回，摧毁减仓机制。
        floor = lo if w_ref < lo - 1e-12 else 0.0
        col = np.maximum(col, floor)
        out.loc[:, a] = col
    s = out[ASSETS].sum(axis=1)
    over = s > 1.0
    if over.any():
        out.loc[over, ASSETS] = out.loc[over, ASSETS].div(s[over], axis=0)
    out["cash"] = 1.0 - out[ASSETS].sum(axis=1)
    return out


def build(fp: dict, cfg: V5Cfg, sid: str) -> Tuple[pd.DataFrame, RebalanceSpec]:
    """按策略编号返回（目标权重, 再平衡规则）；参考权重只来自 cfg.ref。"""
    if sid in ("S0", "S1", "S2"):
        tv, spec = V4S.build(fp, _v4_proxy(cfg), sid)
    elif sid in V3_MODULES:
        tv, spec = V4S.targets_v3(fp, _v4_proxy(cfg), sid)
    else:
        raise KeyError(sid)
    return _apply_band(tv, cfg), spec


def build_all(fp: dict, cfg: V5Cfg, ids: List[str] | None = None
              ) -> Dict[str, Tuple[pd.DataFrame, RebalanceSpec]]:
    return {s: build(fp, cfg, s) for s in (ids or list(DEFAULT_SPECS))}