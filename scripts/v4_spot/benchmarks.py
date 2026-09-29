# -*- coding: utf-8 -*-
"""V4 —— B&H Rebalance 基准族 B0-B6（Roadmap §4/§5）

所有基准共用同一份参考权重与同一执行引擎，仅"再平衡触发规则"不同。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from scripts.v4_spot.config import (ASSETS, HYBRIDS, REF_WEIGHT, THRESHOLDS,
                                    RebalanceSpec, V4Cfg)
from scripts.v4_spot.engine import simulate


def default_specs() -> Dict[str, RebalanceSpec]:
    """§8 核心实验矩阵中的七个基准（B5/B6 采用申报的默认档位）。"""
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
    """§4.6/§4.7 阈值与混合基准的完整档位。"""
    out: Dict[str, RebalanceSpec] = {}
    for t in THRESHOLDS:
        out[f"B5_abs_{int(t * 100)}"] = RebalanceSpec("threshold", None, t, "absolute")
        out[f"B5_rel_{int(t * 100)}"] = RebalanceSpec("threshold", None, t, "relative")
    for freq, t in HYBRIDS:
        out[f"B6_{freq}_{int(t * 100)}"] = RebalanceSpec("calendar_thresh", freq, t, "absolute")
    return out


def constant_targets(fp: dict, weights: Dict[str, float] | None = None) -> pd.DataFrame:
    """恒定目标权重（未上市资产的权重由引擎按 Cash Reserve 处理）。"""
    w = dict(weights or REF_WEIGHT)
    idx = fp["index"]
    df = pd.DataFrame({a: float(w.get(a, 0.0)) for a in ASSETS}, index=idx)
    df["cash"] = 1.0 - df[ASSETS].sum(axis=1)
    return df


def run_benchmark(fp: dict, cfg: V4Cfg, bid: str, spec: RebalanceSpec,
                  cash_rate=None, weights: Dict[str, float] | None = None) -> dict:
    tgt = constant_targets(fp, weights)
    return simulate(fp, tgt, cfg, spec, cash_rate=cash_rate, label=bid)


def run_all_benchmarks(fp: dict, cfg: V4Cfg, cash_rate=None) -> Dict[str, dict]:
    specs = default_specs()
    return {bid: run_benchmark(fp, cfg, bid, sp, cash_rate) for bid, sp in specs.items()}