# -*- coding: utf-8 -*-
"""V3 现货多策略系统 —— 三层决策架构与配置模型（§17-§24, §30-§32, §55）"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.v3_spot.indicators import ASSETS, BARS_PER_DAY, persistence_state

REF_WEIGHT = {"BTC": 0.45, "ETH": 0.20, "SOL": 0.15, "BNB": 0.20}
BAND = {"BTC": (0.35, 0.55), "ETH": (0.15, 0.25),
        "SOL": (0.10, 0.20), "BNB": (0.15, 0.25)}

DIR_WEIGHTS = {"ma200": 0.35, "tsmom6m": 0.15, "tsmom12m": 0.10,
               "mtf": 0.25, "adaptive": 0.15}
QUAL_WEIGHTS = {"adx": 0.40, "adaptive": 0.35, "atr": 0.25}

ALL_MODULES = ["ma200", "tsmom6m", "tsmom12m", "atr", "adx",
               "adaptive", "mtf", "persistence"]

MODELS: Dict[str, List[str]] = {
    # §30 核心策略组合
    "ModelA_MA200": ["ma200"],
    "ModelB_MA200_TSMOM": ["ma200", "tsmom6m", "tsmom12m"],
    "ModelC_MA200_ATR_ADX": ["ma200", "atr", "adx"],
    "ModelD_C_Adaptive": ["ma200", "atr", "adx", "adaptive"],
    "ModelE_D_TSMOM": ["ma200", "tsmom6m", "tsmom12m", "atr", "adx", "adaptive"],
    "ModelF_Full": ALL_MODULES,
    # §31 Minimal vs Full
    "Minimal_MA200_ATR_ADX": ["ma200", "atr", "adx"],
    "Full_Ensemble": ALL_MODULES,
    # §55 最终候选
    "Conservative": ["ma200", "atr", "adx"],
    "Balanced": ["ma200", "atr", "adx", "adaptive", "persistence"],
    "Aggressive": ALL_MODULES,
}


@dataclass
class Cfg:
    """策略与执行参数。"""
    modules: List[str] = field(default_factory=lambda: list(ALL_MODULES))
    # 指标参数（§50 参数稳定性测试对象）
    ma_len: int = 200
    tsmom_days: int = 126
    tsmom_days2: int = 252
    atr_len: int = 14
    adx_len: int = 14
    target_vol: float = 0.15
    persist_bars: int = 4
    # 配置
    ref_weight: Dict[str, float] = field(default_factory=lambda: dict(REF_WEIGHT))
    band: Dict[str, tuple] = field(default_factory=lambda: dict(BAND))
    alloc_mode: str = "bucket"          # bucket(§22 分档) | linear
    enforce_lower_band: bool = True
    use_mtf: bool | None = None
    # 执行
    cadence: str = "daily"              # 4h | daily | weekly
    rebal_threshold: float = 0.10       # §25
    fee: float = 0.0010                 # §28
    slippage: float = 0.0002
    cash_return: str = "A"              # A=0% | B=风险利率
    # 可选研究开关（§34-§38）
    dd_guard: float = 0.0               # 0=关闭；否则为组合回撤阈值
    dd_guard_scale: float = 0.75
    reentry_guard: float = 0.0          # 0=关闭；否则为从低点反弹阈值
    dca_reentry: bool = False
    initial: float = 10_000.0

    @property
    def cost_per_leg(self) -> float:
        return self.fee + self.slippage

    def modules_of(self) -> List[str]:
        mods = list(self.modules)
        if self.use_mtf is False and "mtf" in mods:
            mods.remove("mtf")
        if self.use_mtf is True and "mtf" not in mods:
            mods.append("mtf")
        return mods


# ---------------------------------------------------------------------------
# Layer 1/2/3 因子矩阵
# ---------------------------------------------------------------------------
def _wavg(parts: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """按可用分量重新归一化的加权平均（缺失分量不计权重）。"""
    num = None
    den = None
    for k, w in weights.items():
        if w <= 0 or k not in parts:
            continue
        v = parts[k]
        ok = v.notna()
        contrib = v.fillna(0.0) * w * ok.astype(float)
        num = contrib if num is None else num + contrib
        den = (w * ok.astype(float)) if den is None else den + w * ok.astype(float)
    if num is None or den is None:
        return pd.Series(np.nan, index=next(iter(parts.values())).index)
    return num / den.replace(0.0, np.nan)


def build_layer_matrices(feats: Dict[str, pd.DataFrame], cfg: Cfg) -> Dict[str, pd.DataFrame]:
    """对每个资产输出 direction / quality / risk 三条序列（按 cfg 选择参数变体）。"""
    mods = set(cfg.modules_of())
    ma_sign = f"ma_sign_{cfg.ma_len}"
    t_short = f"tsmom_{cfg.tsmom_days}"
    t_long = f"tsmom_{cfg.tsmom_days2}"
    rv = f"realized_vol_{cfg.atr_len}"
    at = f"atr_trend_{cfg.atr_len}"
    adxs = f"adx_score_{cfg.adx_len}"
    out = {}
    for a in ASSETS:
        f = feats[a]
        base_sign = f[ma_sign] if ma_sign in f.columns else f["ma200_sign"]
        vols = f[rv] if rv in f.columns else f["realized_vol"]
        atr_trend = f[at] if at in f.columns else f["atr_trend"]
        adx_score = f[adxs] if adxs in f.columns else f["adx_score"]

        parts1: Dict[str, pd.Series] = {}
        if "ma200" in mods:
            parts1["ma200"] = base_sign
        if "tsmom6m" in mods and cfg.tsmom_days > 0:
            parts1["tsmom6m"] = f[t_short] if t_short in f.columns else f["tsmom_short"]
        if "tsmom12m" in mods and cfg.tsmom_days2 > 0:
            parts1["tsmom12m"] = f[t_long] if t_long in f.columns else f["tsmom_long"]
        if "mtf" in mods:
            parts1["mtf"] = f["mtf"]
        if "adaptive" in mods:
            parts1["adaptive"] = f["adaptive"]
        direction = _wavg(parts1, DIR_WEIGHTS)

        parts2: Dict[str, pd.Series] = {}
        if "adx" in mods:
            parts2["adx"] = adx_score
        if "adaptive" in mods:
            parts2["adaptive"] = f["adaptive_quality"]
        if "atr" in mods:
            parts2["atr"] = atr_trend
        if parts2:
            quality = _wavg(parts2, QUAL_WEIGHTS)
        else:
            quality = pd.Series(1.0, index=f.index)
        quality = quality.fillna(0.0) if parts2 else quality

        # Layer 3：ATR 波动率缩放 × Persistence Modifier
        # §12/§23：低波动 -> 较高配置；高波动 -> 较低配置。
        # 现货无杠杆且 100% 上限，绝对波动率目标不可达，因此用"相对化"实现：
        #   rel = vol_median(point-in-time) / realized_vol          （该资产自身波动中枢的相对位置）
        #   target_scale = target_vol / 15%                          （§12 目标波动率档位）
        #   vol_factor = clip(rel × target_scale, 0.25, 2.0)
        if "atr" in mods:
            vm = f["vol_median"] if "vol_median" in f.columns else vols
            rel = (vm / vols).clip(0.5, 1.25)
            vol_factor = (rel * (cfg.target_vol / 0.15)).clip(0.25, 2.0)
            vol_factor = vol_factor.fillna(1.0)
        else:
            vol_factor = pd.Series(1.0, index=f.index)
        if "persistence" in mods:
            ps = persistence_state(base_sign, cfg.persist_bars)
            run = ps["persist_run"].to_numpy(dtype=float)
            confirm = ps["persist_confirm"].to_numpy(dtype=float)
            warn = ((confirm > 0) & (run > 0) & (run < cfg.persist_bars)).astype(float)
            persist_mod = pd.Series(1.0 - 0.25 * warn, index=f.index)
        else:
            persist_mod = pd.Series(1.0, index=f.index)
        risk = (vol_factor * persist_mod).clip(0.0, 2.0)

        out[a] = pd.DataFrame({
            "direction": direction,
            "quality": quality,
            "risk": risk,
            "vol_factor": vol_factor,
            "persist_mod": persist_mod,
            "close": f["close"],
            "realized_vol": vols,
        })
    return out


def direction_to_scale(d: pd.Series, mode: str = "bucket") -> pd.Series:
    """§22：Direction Score -> 配置比例（0~1）。"""
    d = d.astype(float)
    if mode == "linear":
        return d.clip(0.0, 1.0)
    s = pd.Series(0.0, index=d.index)
    s = s.mask(d > 0.0, 0.25)
    s = s.mask(d > 0.25, 0.50)
    s = s.mask(d > 0.50, 0.75)
    s = s.mask(d > 0.75, 1.00)
    s = s.mask(d.isna(), np.nan)
    return s


# ---------------------------------------------------------------------------
# 目标权重（仅在决策 bar 上计算）
# ---------------------------------------------------------------------------
def decision_bars(idx: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    """允许做出决策的 bar（4H 每根 / 每日首根 / 每周首根）。"""
    if cadence == "4h":
        return idx
    if cadence == "daily":
        key = np.asarray(idx.normalize().astype("int64"))
    elif cadence == "weekly":
        key = np.asarray(idx.to_period("W-SUN").astype("int64"))
    else:
        raise ValueError(cadence)
    _, pos = np.unique(key, return_index=True)
    return idx[np.sort(pos)]


def compute_targets(layers: Dict[str, pd.DataFrame], px: pd.DataFrame, cfg: Cfg,
                    bars: pd.DatetimeIndex,
                    bounce: Dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """在给定决策 bar 上计算目标权重（含归一化、配置带截断、Cash）。"""
    scale = {a: direction_to_scale(layers[a]["direction"].reindex(bars), cfg.alloc_mode)
             for a in ASSETS}
    rows = []
    for t in bars:
        raw, hi, lo, dirs, ok = {}, {}, {}, {}, {}
        for a in ASSETS:
            price = px.at[t, a]
            d = scale[a].at[t]
            q = layers[a]["quality"].at[t]
            r = layers[a]["risk"].at[t]
            tradable = (not np.isnan(price)) and (not np.isnan(d)) \
                and ("ma200" not in cfg.modules_of() or not np.isnan(layers[a]["direction"].at[t]))
            ok[a] = bool(tradable)
            raw[a] = 0.0 if not tradable else float(cfg.ref_weight[a] * d * q * r)
            _, hi[a] = cfg.band[a]
            lo[a] = cfg.band[a][0]
            dirs[a] = 0.0 if np.isnan(d) else float(d)
            # §35 Re-entry Guard：已从近期低点反弹超过阈值 -> 允许逐步回到参考权重
            if (cfg.reentry_guard > 0 and bounce is not None and ok[a]
                    and float(bounce[a].at[t]) > cfg.reentry_guard):
                dirs[a] = max(dirs[a], 0.5)
                raw[a] = max(raw[a], float(cfg.ref_weight[a] * 0.5))
                tw_min = float(cfg.ref_weight[a] * 0.5)
                lo[a] = min(lo[a], tw_min)
        # 归一化到 <=100%
        tw = {a: min(raw[a], hi[a]) for a in ASSETS}
        s = sum(tw.values())
        if s > 1.0:
            k = 1.0 / s
            tw = {a: tw[a] * k for a in ASSETS}
        # 配置带下界（仅当该资产处于确认的上行趋势、且总仓位允许时）
        if cfg.enforce_lower_band:
            need = {a: (lo[a] if (ok[a] and dirs[a] > 0.25 and tw[a] < lo[a]) else None)
                    for a in ASSETS}
            fixed = {a: v for a, v in need.items() if v is not None}
            free = [a for a in ASSETS if need[a] is None]
            total_fixed = sum(fixed.values())
            budget = max(1.0 - total_fixed, 0.0)
            free_sum = sum(tw[a] for a in free)
            if free_sum > budget and free_sum > 0:
                k = budget / free_sum
                for a in free:
                    tw[a] *= k
            for a, v in fixed.items():
                tw[a] = v
        risky = sum(tw.values())
        if risky > 1.0:
            tw = {a: tw[a] / risky for a in ASSETS}
            risky = 1.0
        rows.append({**tw, "risky_total": risky, "cash": 1.0 - risky})
    return pd.DataFrame(rows, index=bars)


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def build_strategy(fp: dict, cfg: Cfg) -> dict:
    layers = build_layer_matrices(fp["feats"], cfg)
    bars = decision_bars(fp["index"], cfg.cadence)
    bounce = None
    if cfg.reentry_guard > 0:
        bounce = {a: (fp["px"][a] / fp["px"][a].rolling(BARS_PER_DAY * 30,
                                                        min_periods=BARS_PER_DAY * 5).min() - 1.0)
                  for a in ASSETS}
    targets = compute_targets(layers, fp["px"], cfg, bars, bounce)
    return {"layers": layers, "bars": bars, "targets": targets, "cfg": cfg}


def fixed_targets(bars: pd.DatetimeIndex, weights: Dict[str, float],
                  px: pd.DataFrame) -> pd.DataFrame:
    """静态基准（§4 Buy & Hold）在同样决策 bar 上的目标权重；未上市资产权重留 Cash。"""
    rows = []
    for t in bars:
        w = {}
        for a in ASSETS:
            w[a] = 0.0 if np.isnan(px.at[t, a]) else float(weights.get(a, 0.0))
        risky = sum(w.values())
        if risky > 1.0:
            w = {a: w[a] / risky for a in ASSETS}
            risky = 1.0
        rows.append({**w, "risky_total": risky, "cash": 1.0 - risky})
    return pd.DataFrame(rows, index=bars)