# -*- coding: utf-8 -*-
"""估值与风险度量（Roadmap V2 §6 / §8.3 / §16）。

净值口径（§20 验收标准 9）：**现金计入总资产净值**。
  equity = Σ(position_qty × close) + cash

另提供到日 MaxDD（邮件 §8.3 要求「MaxDD to date」）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .config import ASSETS
from .models import PaperAccount


@dataclass(frozen=True)
class Valuation:
    """一次估值结果。"""

    as_of: datetime
    cash: float
    market_value: float
    equity: float
    weights: Dict[str, float]
    cash_weight: float
    unrealized_pnl: Dict[str, float]

    @property
    def risky_total(self) -> float:
        return sum(self.weights.values())


def value_account(account: PaperAccount, prices: Dict[str, float],
                  as_of: datetime) -> Valuation:
    """按给定价格对纸面账户估值。"""
    mvs: Dict[str, float] = {}
    unreal: Dict[str, float] = {}
    for a in ASSETS:
        pos = account.positions.get(a)
        px = float(prices.get(a, 0.0) or 0.0)
        qty = 0.0 if pos is None else float(pos.quantity)
        mvs[a] = qty * px
        avg = 0.0 if pos is None else float(pos.average_price)
        unreal[a] = (px - avg) * qty if qty else 0.0

    mv = sum(mvs.values())
    equity = account.cash + mv
    if equity > 0:
        weights = {a: mvs[a] / equity for a in ASSETS}
        cash_w = account.cash / equity
    else:
        weights = {a: 0.0 for a in ASSETS}
        cash_w = 0.0
    return Valuation(as_of=as_of, cash=account.cash, market_value=mv, equity=equity,
                     weights=weights, cash_weight=cash_w, unrealized_pnl=unreal)


def max_drawdown_to_date(equity_series: List[float]) -> float:
    """到日最大回撤（负数，0 表示无回撤）。"""
    peak = None
    mdd = 0.0
    for e in equity_series:
        if e is None:
            continue
        peak = e if peak is None else max(peak, e)
        if peak and peak > 0:
            mdd = min(mdd, e / peak - 1.0)
    return mdd


def equity_history_from_snapshots(rows: List[dict], prices: Dict[str, float] = None) -> List[float]:
    """从快照行提取净值序列（用于 §16 告警与邮件 MaxDD）。"""
    out: List[float] = []
    for r in rows:
        try:
            cash = float(r.get("cash") or 0.0)
        except (TypeError, ValueError):
            continue
        mv = 0.0
        if prices:
            for a in ASSETS:
                try:
                    mv += float(r.get(f"{a}_qty") or 0.0) * float(prices.get(a, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
        out.append(cash + mv)
    return out


def check_invariants(valuation: Valuation,
                     target_weights: Optional[Dict[str, float]] = None) -> dict:
    """§14.5 验收恒等式：asset_market_value + cash = equity；Σweights = 1。"""
    checks = {
        "equity_identity_ok": abs(valuation.market_value + valuation.cash
                                  - valuation.equity) <= 1e-6 * max(1.0, abs(valuation.equity)),
        "cash_non_negative": valuation.cash >= -1e-9,
        "weights_non_negative": all(w >= -1e-12 for w in valuation.weights.values()),
    }
    if target_weights is not None:
        total = sum(valuation.weights.values()) + valuation.cash_weight
        checks["weights_sum_ok"] = abs(total - 1.0) <= 1e-9
    checks["ok"] = all(checks.values())
    return checks


__all__ = [
    "Valuation", "value_account", "max_drawdown_to_date",
    "equity_history_from_snapshots", "check_invariants",
]