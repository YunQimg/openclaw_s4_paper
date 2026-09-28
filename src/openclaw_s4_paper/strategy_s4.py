# -*- coding: utf-8 -*-
"""S4 策略层：MA200 + TSMOM 6M 资产级分档（Roadmap V2 §5）。

复现 V5 冻结口径（`scripts/v5_spot/strategies.py::build(fp, cfg, "S4")` 叠加
`scripts/v4_spot/strategies.py::targets_v3` + V3 `compute_targets`）：

  direction = 0.70 × ma_sign + 0.30 × tsmom6m      （两分量均可用时）
  scale     = bucket(direction) ∈ {0, 0.25, 0.50, 0.75, 1.0}
  raw       = ref_weight × scale
  (a) 信号层：V4 冻结带，上行趋势（scale >= 0.50）下目标不得低于带下界
  (b) 审计层：V5 冻结带，上界裁剪（本组合恒为 no-op，需留痕 0 次裁剪）
  final     = band-constrained，归一化到 <= 100%，Cash = 1 - risky_total

本模块为纯函数，不读写账户、不发邮件、不触碰交易所。

§13.2 模块边界说明（显式记录与冻结 API 清单的偏差）：
  * `compute_ma200_sign` / `compute_tsmom6m` 实现于 `indicators.py`（指标层职责），
    此处重新导出以满足 §13.2 的模块表面；数值与 `build_indicator_panel` 逐位一致。
  * `compute_targets` 的签名与 §13.2 清单不同：本实现为
    `compute_targets(indicators, cfg, listed)`。原因是 §5.4 的两层带
    （signal / audit）语义必须由 `StrategyConfig` 承载，无法仅以
    `(direction, scale, ref_weights, bands)` 表达；返回
    `(targets, scale, direction, band_audit)`，其中 targets 同时含 raw 与
    band-constrained 两支，满足 §13.2「必须同时返回」的要求。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (ASSETS, BUCKET_LEVELS, LOWER_BAND_MIN_SCALE, StrategyConfig)
from .indicators import (build_indicator_panel, compute_ma200_sign,
                         compute_tsmom6m)

# 分档档位（索引 i 对应 "d > 第 i 个阈值"）
LEVELS_BY_TIER: List[float] = list(BUCKET_LEVELS[1:])   # [0.25, 0.50, 0.75, 1.00]


# ---------------------------------------------------------------------------
# §5.3 Direction 与分档
# ---------------------------------------------------------------------------
def compute_direction(ma_sign: pd.Series, tsmom6m: pd.Series,
                      weights: Dict[str, float]) -> pd.Series:
    """按可用分量重新归一化的加权平均（§5.3）。

    分量缺失（NaN）时不参与加权，分母只累计可用分量权重。
    两个分量都不可用 -> NaN。
    """
    parts: List[Tuple[pd.Series, float]] = []
    if "ma200" in weights and weights["ma200"] > 0:
        parts.append((ma_sign.astype(float), float(weights["ma200"])))
    if "tsmom6m" in weights and weights["tsmom6m"] > 0:
        parts.append((tsmom6m.astype(float), float(weights["tsmom6m"])))

    num = None
    den = None
    for series, w in parts:
        ok = series.notna()
        contrib = series.fillna(0.0) * w * ok.astype(float)
        num = contrib if num is None else num + contrib
        d = w * ok.astype(float)
        den = d if den is None else den + d
    if num is None or den is None:
        return pd.Series(np.nan, index=ma_sign.index)
    return num / den.replace(0.0, np.nan)


def bucket_scale(direction: pd.Series, thresholds: List[float]) -> pd.Series:
    """Direction -> 分档 scale（§5.3 bucket 口径）。

    thresholds = [0.25, 0.50, 0.75]，V3 冻结口径为：

      d > 0.00 -> 0.25 ; d > 0.25 -> 0.50 ; d > 0.50 -> 0.75 ; d > 0.75 -> 1.00 ; d <= 0 -> 0

    边界值恰好等于阈值时落入**较低**档（与 V3 `direction_to_scale` 的 mask 顺序一致）。
    `d` 为 NaN（数据不足 / 未上市）-> NaN，目标权重按 0 处理。

    实现上等价于对 `[0.0] + thresholds` 逐级 mask，档位 `[0.25, 0.50, 0.75, 1.00]`。
    """
    t = [0.0] + sorted(float(x) for x in thresholds)
    levels = LEVELS_BY_TIER[:len(t)]
    d = direction.astype(float)
    s = pd.Series(0.0, index=d.index)
    for th, lvl in zip(t, levels):
        s = s.mask(d > th, lvl)
    return s.mask(d.isna(), np.nan)


def linear_scale(direction: pd.Series) -> pd.Series:
    """线性映射（`alloc_mode=linear`，仅研究用；本系统冻结为 bucket）。"""
    return direction.astype(float).clip(0.0, 1.0)


def to_scale(direction: pd.Series, cfg: StrategyConfig) -> pd.Series:
    if cfg.alloc_mode == "linear":
        return linear_scale(direction)
    return bucket_scale(direction, cfg.bucket_thresholds)


# ---------------------------------------------------------------------------
# §5.4 两层配置带
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandAudit:
    """审计层带约束留痕（§5.4(b)）。"""

    n_clipped: int                        # 被带上界裁剪的次数（本组合应为 0）
    n_out_of_band: int                    # 未裁剪但落在带外（本组合应为 0）
    per_asset_clipped: Dict[str, int]

    @property
    def is_noop(self) -> bool:
        return self.n_clipped == 0 and self.n_out_of_band == 0


def apply_signal_band(raw: pd.DataFrame, scale: pd.DataFrame,
                      cfg: StrategyConfig) -> pd.DataFrame:
    """信号层：V4 冻结带的「上行趋势下界托底」（§5.4(a)）。

    仅当该资产 `scale >= 0.50`（确认上行趋势）且目标低于带下界时，把目标托到下界。
    减仓方向（scale = 0）**不受下界约束**，目标保持 0 —— 这是策略的风险控制机制。
    """
    out = raw.copy()
    if not cfg.enforce_lower_band:
        return out
    band = cfg.signal_band_t
    for a in ASSETS:
        lo = band[a][0]
        if lo <= 0:
            continue
        uptrend = (scale[a] >= LOWER_BAND_MIN_SCALE - 1e-12).fillna(False)
        col = out[a].astype(float)
        out.loc[:, a] = np.where(uptrend & (col < lo), lo, col)
    return out


def normalize(df: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    """归一化到 <= cap（现货无杠杆），并补 cash 列。"""
    out = df.copy()
    s = out[ASSETS].sum(axis=1)
    over = s > cap
    if bool(over.any()):
        out.loc[over, ASSETS] = out.loc[over, ASSETS].div(s[over], axis=0)
    out["risky_total"] = out[ASSETS].sum(axis=1)
    out["cash"] = cap - out["risky_total"]
    return out


def apply_audit_band(banded: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, BandAudit]:
    """审计层：V5 冻结带的上界裁剪 + 越界留痕（§5.4(b)）。

    与 V5 `_apply_band` 语义一致：
      * 上限：`min(target, hi)`；
      * 下限：仅当**参考权重本身**低于带下界时才约束目标（本组合无此情形），
        否则下界不约束策略的合法减仓。
    """
    out = banded.copy()
    band = cfg.audit_band_t
    n_clipped = 0
    n_oob = 0
    per: Dict[str, int] = {a: 0 for a in ASSETS}

    for a in ASSETS:
        lo, hi = band[a]
        w_ref = float(cfg.reference_weights[a])
        col = out[a].astype(float)
        clipped = col > hi + 1e-12
        n_a = int(clipped.sum())
        if n_a:
            per[a] = n_a
            n_clipped += n_a
        col = np.minimum(col, hi)
        if w_ref < lo - 1e-12:
            below = col < lo - 1e-12
            n_b = int(below.sum())
            if n_b:
                per[a] = per[a] + n_b
                n_clipped += n_b
            col = np.maximum(col, lo)
        out.loc[:, a] = col

    n_oob = int(((out[ASSETS] < 0) | (out[ASSETS] > 1)).sum().sum())
    return out, BandAudit(n_clipped=n_clipped, n_out_of_band=n_oob,
                          per_asset_clipped=per)


# ---------------------------------------------------------------------------
# 目标权重总入口
# ---------------------------------------------------------------------------
def compute_targets(indicators: Dict[str, pd.DataFrame], cfg: StrategyConfig,
                    listed: Optional[pd.DataFrame] = None,
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BandAudit]:
    """计算全轴目标权重。

    返回 (targets, scale, direction, band_audit)：
      * targets   : raw / band-constrained / final / cash / risky_total
      * scale     : 各资产分档
      * direction : 各资产 direction score
    """
    idx = indicators[ASSETS[0]].index
    scale = pd.DataFrame(index=idx)
    direction = pd.DataFrame(index=idx)
    raw = pd.DataFrame(index=idx)

    for a in ASSETS:
        ind = indicators[a]
        d = compute_direction(ind["ma_sign"], ind["tsmom6m"],
                              cfg.direction_component_weights)
        direction[a] = d
        sc = to_scale(d, cfg)
        scale[a] = sc
        w = float(cfg.reference_weights[a])
        col = (sc.fillna(0.0) * w)
        # 未上市（价格缺失）-> 目标权重 0
        col = col.where(ind["close"].notna(), 0.0)
        raw[a] = col

    raw = raw.where(listed, 0.0) if listed is not None else raw

    signal_banded = apply_signal_band(raw, scale, cfg)
    banded, audit = apply_audit_band(signal_banded, cfg)
    final = normalize(banded)

    targets = pd.DataFrame(index=idx)
    for a in ASSETS:
        targets[f"raw_{a}"] = raw[a]
        targets[f"target_{a}"] = final[a]
    targets["risky_total"] = final["risky_total"]
    targets["cash"] = final["cash"]
    return targets, scale, direction, audit


def build_targets(close_4h: pd.DataFrame, close_daily: pd.DataFrame,
                  cfg: StrategyConfig, listed: Optional[pd.DataFrame] = None
                  ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame, BandAudit]:
    """便捷入口：从收盘价矩阵直接构造指标 + 目标权重。"""
    ind = build_indicator_panel(close_4h, close_daily, cfg.ma_length,
                               cfg.tsmom_days, cfg.tsmom_tanh_scale)
    targets, scale, direction, audit = compute_targets(ind, cfg, listed)
    return targets, ind, direction, audit


def scale_to_reference_ratio(cfg: StrategyConfig) -> Dict[str, float]:
    """参考权重比例（用于验收「缩放不改变相对比例」）。"""
    tot = sum(float(cfg.reference_weights[a]) for a in ASSETS)
    return {a: float(cfg.reference_weights[a]) / tot for a in ASSETS}


__all__ = [
    "compute_direction", "bucket_scale", "linear_scale", "to_scale",
    "apply_signal_band", "apply_audit_band", "normalize",
    "compute_targets", "build_targets", "BandAudit", "scale_to_reference_ratio",
    "compute_ma200_sign", "compute_tsmom6m",
]