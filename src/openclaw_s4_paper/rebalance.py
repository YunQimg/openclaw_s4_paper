# -*- coding: utf-8 -*-
"""调仓计划构造（Roadmap V2 §6.4 / §7 / §13.3）。

**只生成计划，不直接提交外部订单**（§13.3）。
触发规则（S4 冻结口径）：
  * 纯阈值：任一路径资产 |current_weight - target_weight| > 10pp（绝对口径）；
  * 无日历腿；
  * 必然触发：首次建仓、资产首次上市进入（L1）；
  * 最小交易阈值：|notional_delta| >= max(50 USD, equity × 0.5%)，
    低于阈值记为 NO_ACTION_THRESHOLD，不生成纸面订单。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .config import ASSETS, StrategyConfig
from .models import PaperAccount, RebalanceLeg, RebalancePlan
from .valuation import value_account

EPS = 1e-12


def min_notional_threshold(equity: float, cfg: StrategyConfig) -> float:
    """§6.4 最小交易阈值 = max(50 USD, equity × 0.5%)。"""
    return max(float(cfg.min_notional_usd),
               abs(equity) * float(cfg.min_notional_equity_pct))


def deviation(current: Dict[str, float], target: Dict[str, float]) -> float:
    """最大绝对权重偏离（§6.4 阈值口径）。"""
    return max((abs(float(current.get(a, 0.0)) - float(target.get(a, 0.0)))
                for a in ASSETS), default=0.0)


def should_rebalance(current: Dict[str, float], target: Dict[str, float],
                     cfg: StrategyConfig, *, first: bool = False,
                     listing_entry: bool = False) -> bool:
    """是否触发调仓。"""
    if first or listing_entry:
        return True
    return deviation(current, target) > float(cfg.rebalance_threshold_absolute)


def build_rebalance_plan(account: PaperAccount, target_weights: Dict[str, float],
                         prices: Dict[str, float], exec_prices: Dict[str, float],
                         cfg: StrategyConfig, as_of: datetime, *,
                         first: bool = False, listing_entry: bool = False,
                         tradable: Optional[Dict[str, bool]] = None) -> RebalancePlan:
    """构造调仓计划（先卖后买，§6.4）。

    参数
    ----
    prices      : 估值用收盘价（计算当前权重与 notional_delta）
    exec_prices : 执行价基准（该 bar 开盘价，§6.3）
    tradable    : 各资产是否可交易（未上市公司为 False）
    """
    val = value_account(account, prices, as_of)
    equity = val.equity
    tradable = tradable or {a: True for a in ASSETS}

    if equity <= 0:
        return RebalancePlan(as_of=as_of, equity=equity, legs=[],
                             weight_deltas={a: 0.0 for a in ASSETS},
                             notional_deltas={a: 0.0 for a in ASSETS})

    target = {a: float(target_weights.get(a, 0.0)) for a in ASSETS}
    current = dict(val.weights)
    weight_deltas = {a: target[a] - current[a] for a in ASSETS}
    notional_deltas = {a: weight_deltas[a] * equity for a in ASSETS}

    triggered = should_rebalance(current, target, cfg, first=first,
                                 listing_entry=listing_entry)
    if not triggered:
        return RebalancePlan(
            as_of=as_of, equity=equity, legs=[],
            weight_deltas=weight_deltas, notional_deltas=notional_deltas,
            target_error_before={a: weight_deltas[a] for a in ASSETS})

    thr = min_notional_threshold(equity, cfg)
    below: List[str] = []
    legs: List[RebalanceLeg] = []

    # ---- 先卖：目标低于当前，且满足最小交易阈值 ----
    for a in ASSETS:
        if not tradable.get(a, True):
            continue
        d = notional_deltas[a]
        if d >= -EPS:
            continue
        if abs(d) < thr:
            below.append(a)
            continue
        px = float(exec_prices.get(a, 0.0) or 0.0)
        if px <= 0:
            continue
        qty_avail = account.quantity(a)
        qty = min(-d / px, qty_avail)
        if qty <= EPS:
            continue
        reason = "listing_entry" if listing_entry and a not in current else (
            "initial" if first else "threshold")
        legs.append(RebalanceLeg(
            asset=a, side="SELL", quantity=qty, notional_delta=d,
            current_weight=current[a], target_weight=target[a],
            estimated_fill_price=px * (1.0 - cfg.slippage), reason=reason))

    # ---- 后买：目标高于当前，且满足最小交易阈值 ----
    for a in ASSETS:
        if not tradable.get(a, True):
            continue
        d = notional_deltas[a]
        if d <= EPS:
            continue
        if abs(d) < thr:
            below.append(a)
            continue
        px = float(exec_prices.get(a, 0.0) or 0.0)
        if px <= 0:
            continue
        # 数量按"目标名义额 / 有效执行价"估算，实际成交在账本层受现金约束缩放
        qty = d / (px * (1.0 + cfg.slippage))
        if qty <= EPS:
            continue
        reason = "initial" if first else ("listing_entry" if listing_entry else "threshold")
        legs.append(RebalanceLeg(
            asset=a, side="BUY", quantity=qty, notional_delta=d,
            current_weight=current[a], target_weight=target[a],
            estimated_fill_price=px * (1.0 + cfg.slippage), reason=reason))

    return RebalancePlan(
        as_of=as_of, equity=equity, legs=legs,
        weight_deltas=weight_deltas, notional_deltas=notional_deltas,
        below_threshold=sorted(set(below)),
        target_error_before={a: weight_deltas[a] for a in ASSETS})


__all__ = [
    "build_rebalance_plan", "should_rebalance", "deviation",
    "min_notional_threshold", "EPS",
]