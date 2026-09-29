# -*- coding: utf-8 -*-
"""V4 —— 策略候选族 S0-S7（Roadmap §7）

S1/S2 为 V4 新写的可解释规则；S3-S7 复用 V3 指标层（scripts/v3_spot/strategy.py）
以验证"同一套因子在修复后的基准与统一执行口径下"的表现。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.v3_spot import strategy as V3S
from scripts.v4_spot.config import ASSETS, RebalanceSpec, V4Cfg

STRATEGY_TITLES: Dict[str, str] = {
    "S0": "Static Reference 15/10/35/40（不再平衡）",
    "S1": "Annual B&H + MA200 资产级风险减仓",
    "S2": "B&H Rebalance + 组合级 MA200 Overlay",
    "S3": "MA200 Minimal（带 + 日频 + 阈值再平衡）",
    "S4": "MA200 + TSMOM 6M",
    "S5": "Minimal Quality（MA200 + ATR + ADX）",
    "S6": "Reduced Balanced（MA200 + ADX + Adaptive）",
    "S7": "Full V3 Replication",
}
DEFAULT_SPECS: Dict[str, RebalanceSpec] = {
    "S0": RebalanceSpec("never"),
    "S1": RebalanceSpec("calendar", "Y"),
    "S2": RebalanceSpec("calendar_thresh", "Y", 0.10, "absolute"),
    "S3": RebalanceSpec("threshold", None, 0.10, "absolute"),
    "S4": RebalanceSpec("threshold", None, 0.10, "absolute"),
    "S5": RebalanceSpec("threshold", None, 0.10, "absolute"),
    "S6": RebalanceSpec("threshold", None, 0.10, "absolute"),
    "S7": RebalanceSpec("threshold", None, 0.10, "absolute"),
}
V3_MODULES: Dict[str, List[str]] = {
    "S3": ["ma200"],
    "S4": ["ma200", "tsmom6m"],
    "S4b": ["ma200", "tsmom6m", "tsmom12m"],
    "S5": ["ma200", "atr", "adx"],
    "S6": ["ma200", "adx", "adaptive"],
    "S7": list(V3S.ALL_MODULES),
}


def _ref(cfg: V4Cfg) -> Dict[str, float]:
    """参考权重（归一化到 1）——策略与基准共用同一份 cfg.ref_weight。"""
    w = {a: float(cfg.ref_weight.get(a, 0.0)) for a in ASSETS}
    tot = sum(w.values())
    return {a: (v / tot if tot > 0 else 0.0) for a, v in w.items()}


def ma_sign(fp: dict, cfg: V4Cfg, asset: str) -> pd.Series:
    f = fp["feats"][asset]
    k = f"ma_sign_{cfg.ma_len}"
    s = f[k] if k in f.columns else f["ma200_sign"]
    return s.reindex(fp["index"]).fillna(0.0)


def _norm(df: pd.DataFrame, panic: float = 1.0) -> pd.DataFrame:
    """归一化到 <= panic（现货无杠杆）。"""
    s = df[ASSETS].sum(axis=1)
    over = s > panic
    if over.any():
        df.loc[over, ASSETS] = df.loc[over, ASSETS].div(s[over], axis=0)
    df["cash"] = panic - df[ASSETS].sum(axis=1)
    return df


def targets_s1(fp: dict, cfg: V4Cfg, risk_mult: float | None = None) -> Tuple[pd.DataFrame, RebalanceSpec]:
    """S1：Annual B&H + MA200 资产级风险减仓。

    Close > MA200 -> 保持参考目标；Close < MA200 -> 目标权重 × risk_multiplier。
    """
    m = cfg.risk_multiplier if risk_mult is None else risk_mult
    idx = fp["index"]
    rw = _ref(cfg)
    tv = pd.DataFrame(index=idx)
    for a in ASSETS:
        sign = ma_sign(fp, cfg, a)
        scale = np.where(sign > 0, 1.0, m)
        tv[a] = rw[a] * scale
        tv.loc[fp["px"][a].isna(), a] = 0.0
    tv = _norm(tv)
    spec = DEFAULT_SPECS["S1"]
    if m != cfg.risk_multiplier:
        spec = spec
    return tv, spec


def targets_s2(fp: dict, cfg: V4Cfg, freq: str | None = None) -> Tuple[pd.DataFrame, RebalanceSpec]:
    """S2：B&H Rebalance + 组合级 MA200 Overlay。

    组合级信号 = 满足 Close > MA200 的资产比例 breadth ∈ {0,.25,.5,.75,1}；
    整体风险暴露缩放 scale = floor + (1-floor) × breadth，仅降低暴露、不改变相对权重。
    """
    idx = fp["index"]
    rw = _ref(cfg)
    signs = pd.DataFrame({a: ma_sign(fp, cfg, a) for a in ASSETS}, index=idx)
    listed = fp["px"].notna()
    above = ((signs > 0) & listed).sum(axis=1)
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    breadth = (above / n_listed).fillna(0.0)
    scale = (cfg.breadth_floor + (1.0 - cfg.breadth_floor) * breadth).clip(0.0, 1.0)
    tv = pd.DataFrame({a: rw[a] * scale for a in ASSETS}, index=idx)
    tv = _norm(tv)
    # S2 = B&H Rebalance (B6 Hybrid: Annual + ±10% threshold) + 组合级 MA200 Overlay
    # 必须用 calendar_thresh 才能让 MA200 Overlay 的目标权重变化通过阈值触发及时执行；
    # 纯 calendar(Y) 会使 Overlay 退化为仅在年度再平衡时才应用 scale，失去动态保护意义。
    spec = RebalanceSpec("calendar_thresh", freq or "Y", cfg.rebal_threshold,
                         cfg.threshold_basis)
    return tv, spec


def _v3_cfg(cfg: V4Cfg, modules: List[str]) -> V3S.Cfg:
    return V3S.Cfg(
        modules=list(modules),
        ma_len=cfg.ma_len,
        tsmom_days=cfg.tsmom_days,
        tsmom_days2=cfg.tsmom_days2,
        ref_weight=dict(cfg.ref_weight),
        band=cfg.band,
        enforce_lower_band=(cfg.band_mode == "Band"),
        cadence=cfg.cadence,
        rebal_threshold=cfg.rebal_threshold,
        fee=cfg.fee,
        slippage=cfg.slippage,
        cash_return="B" if cfg.cash_return == "Cash_Rate" else "A",
    )


def targets_v3(fp: dict, cfg: V4Cfg, key: str) -> Tuple[pd.DataFrame, RebalanceSpec]:
    """复用 V3 指标层构造 S3-S7 的目标权重。"""
    v3cfg = _v3_cfg(cfg, V3_MODULES[key])
    st = V3S.build_strategy(fp, v3cfg)
    return st["targets"], DEFAULT_SPECS.get(key, RebalanceSpec("threshold", None, 0.10))


def build(fp: dict, cfg: V4Cfg, sid: str) -> Tuple[pd.DataFrame, RebalanceSpec]:
    """按策略编号返回（目标权重, 再平衡规则）。"""
    if sid == "S0":
        idx = fp["index"]
        rw = _ref(cfg)
        tv = pd.DataFrame({a: rw[a] for a in ASSETS}, index=idx)
        tv = tv.mask(fp["px"].isna(), 0.0)
        return _norm(tv), DEFAULT_SPECS["S0"]
    if sid == "S1":
        return targets_s1(fp, cfg)
    if sid == "S2":
        return targets_s2(fp, cfg)
    if sid in V3_MODULES:
        return targets_v3(fp, cfg, sid)
    raise KeyError(sid)