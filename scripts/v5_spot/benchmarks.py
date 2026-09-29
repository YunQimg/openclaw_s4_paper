# -*- coding: utf-8 -*-
"""V5 —— B&H Rebalance 基准族 B0-B6 × 三参考权重（Roadmap §4/§5）

与 V4 基准的唯一区别：**每一个基准都要在 R13 / R3 / R9 下各跑一遍**。
参考权重只改变目标权重，不改变信号生成，也不改变再平衡触发规则。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v5_spot.config import (ASSETS, HYBRIDS, REF_IDS, REF_WEIGHTS,
                                    THRESHOLDS, RebalanceSpec, V5Cfg)
from scripts.v5_spot.engine import simulate


def default_specs() -> Dict[str, RebalanceSpec]:
    """七个基准（B5/B6 采用申报的默认档位，继承 V4 §4）。"""
    return {
        "B0": RebalanceSpec("never"),
        "B1": RebalanceSpec("calendar", "M"),
        "B2": RebalanceSpec("calendar", "Q"),
        "B3": RebalanceSpec("calendar", "2Q"),
        "B4": RebalanceSpec("calendar", "Y"),
        "B5": RebalanceSpec("threshold", None, 0.10, "absolute"),
        "B6": RebalanceSpec("calendar_thresh", "Y", 0.10, "absolute"),
    }


def extended_specs() -> Dict[str, RebalanceSpec]:
    """§4.6/§4.7 阈值（4 档 × abs/rel）与混合（3 档）基准的完整档位。"""
    out: Dict[str, RebalanceSpec] = {}
    for t in THRESHOLDS:
        out[f"B5_abs_{int(t * 100)}"] = RebalanceSpec("threshold", None, t, "absolute")
        out[f"B5_rel_{int(t * 100)}"] = RebalanceSpec("threshold", None, t, "relative")
    for freq, t in HYBRIDS:
        out[f"B6_{freq}_{int(t * 100)}"] = RebalanceSpec("calendar_thresh", freq, t, "absolute")
    return out


def constant_targets(fp: dict, ref: str,
                     override: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """恒定目标权重（未上市资产由引擎按 Cash Reserve 处理）。

    override 为 Phase 7 资产剔除/敏感性所用的**参考权重覆盖**。此前该参数
    未从 cfg 透传，导致剔除子集下的 B* 基准全部退回 `REF_WEIGHTS[ref]`，
    各子集的 B4 净值逐位相同（影子复现 ALL4），剔除实验的基准侧无效。
    策略侧（strategies.build）一直读取 override，故只有基准侧受影响。
    """
    w = dict(override) if override else REF_WEIGHTS[ref]
    idx = fp["index"]
    df = pd.DataFrame({a: float(w.get(a, 0.0)) for a in ASSETS}, index=idx)
    df["cash"] = 1.0 - df[ASSETS].sum(axis=1)
    return df


def run_benchmark(fp: dict, cfg: V5Cfg, bid: str, spec: RebalanceSpec,
                  cash_rate=None) -> dict:
    tgt = constant_targets(fp, cfg.ref, cfg.ref_weight_override)
    return simulate(fp, tgt, cfg, spec, cash_rate=cash_rate,
                    label=f"{cfg.ref}:{bid}")


def run_all_benchmarks(fp: dict, cfg: V5Cfg, cash_rate=None) -> Dict[str, dict]:
    return {bid: run_benchmark(fp, cfg, bid, sp, cash_rate)
            for bid, sp in default_specs().items()}


def arm_keys(ref: str) -> List[str]:
    """某一臂的基准臂标识（用于 3 × 12 = 36 个基准臂，Roadmap §4.8）。"""
    keys = [f"{ref}:{b}" for b in default_specs()]
    keys += [f"{ref}:{b}" for b in extended_specs()]
    return keys