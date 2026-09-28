# -*- coding: utf-8 -*-
"""纸面成交模拟与账本追加（Roadmap V2 §6.3 / §6.4 / §13.4）。

**只更新本地纸面状态**，绝不调用任何交易所接口。
交易顺序（§6.4）：
  1. 先卖出
  2. 更新现金
  3. 再买入（现金不足时按比例缩放）
  4. 记录剩余未执行目标差额

成本（§6.3）：买入 fill = open × (1 + slippage)；卖出 fill = open × (1 − slippage)；
手续费 fee = notional × fee_rate，逐腿计提。

只允许现货卖出已有纸面持仓（不得做空）。
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from .config import ASSETS, StrategyConfig
from .ledger_store import append_jsonl
from .models import (LedgerEvent, LedgerResult, PaperAccount, Position,
                     RebalancePlan)
from .valuation import value_account

EPS = 1e-12


def _qty(account: PaperAccount, asset: str) -> float:
    p = account.positions.get(asset)
    return 0.0 if p is None else float(p.quantity)


def _apply_buy(account: PaperAccount, asset: str, qty: float, price: float,
               fee: float) -> None:
    """买入：增加持仓、扣现金、更新加权平均成本。"""
    pos = account.positions.get(asset)
    old_qty = 0.0 if pos is None else float(pos.quantity)
    old_avg = 0.0 if pos is None else float(pos.average_price)
    new_qty = old_qty + qty
    new_avg = ((old_qty * old_avg + qty * price) / new_qty) if new_qty > EPS else 0.0
    realized = 0.0 if pos is None else float(pos.realized_pnl)
    account.positions[asset] = Position(asset, new_qty, new_avg, realized)
    account.cash -= qty * price + fee


def _apply_sell(account: PaperAccount, asset: str, qty: float, price: float,
                fee: float) -> None:
    """卖出：减少持仓、加现金、累计已实现盈亏。"""
    pos = account.positions.get(asset)
    old_qty = 0.0 if pos is None else float(pos.quantity)
    sell_qty = min(qty, old_qty)
    if sell_qty <= EPS:
        raise ValueError(f"spot position too small to sell: {asset}={old_qty}")
    avg = 0.0 if pos is None else float(pos.average_price)
    realized = (0.0 if pos is None else float(pos.realized_pnl)) + (price - avg) * sell_qty
    left = old_qty - sell_qty
    if left <= EPS:
        account.positions.pop(asset, None)
    else:
        account.positions[asset] = Position(asset, left, avg, realized)
    account.cash += sell_qty * price - fee


def simulate_plan(account: PaperAccount, plan: RebalancePlan,
                  exec_prices: Dict[str, float], cfg: StrategyConfig,
                  *, run_id: str, input_hash: str, strategy_version: str,
                  timestamp: datetime) -> Tuple[PaperAccount, LedgerResult]:
    """模拟执行调仓计划，返回 (新账户, 结果)。

    `account` **不被就地修改**；返回的是深拷贝后的新账户，便于失败时不提交。
    """
    new = PaperAccount(
        account_id=account.account_id,
        currency=account.currency,
        initial_cash=account.initial_cash,
        cash=account.cash,
        positions={a: replace(p) for a, p in account.positions.items()},
        as_of=account.as_of,
        ledger_version=account.ledger_version,
    )

    val_before = value_account(account, exec_prices, timestamp)
    events: List[LedgerEvent] = []
    total_fee = 0.0
    total_slip = 0.0
    turnover = 0.0

    def _emit(asset: str, side: str, qty: float, price: float, fee: float,
              slip: float, cash_before: float, qty_before: float, reason: str) -> None:
        events.append(LedgerEvent(
            event_id=f"{run_id}-{len(events) + 1:03d}",
            run_id=run_id, timestamp=timestamp, event_type=reason,
            asset=asset, side=side, quantity=float(qty), price=float(price),
            notional=float(qty * price), fee=float(fee), slippage=float(slip),
            cash_before=float(cash_before), cash_after=float(new.cash),
            position_before=float(qty_before), position_after=float(_qty(new, asset)),
            strategy_version=strategy_version, input_hash=input_hash))

    # ---- 1) 先卖出 ----
    for leg in plan.legs:
        if leg.side != "SELL":
            continue
        px = float(exec_prices.get(leg.asset, 0.0) or 0.0)
        if px <= 0:
            continue
        fill = px * (1.0 - cfg.slippage)
        qty = min(float(leg.quantity), _qty(new, leg.asset))
        if qty <= EPS:
            continue
        notional = qty * fill
        fee = notional * cfg.fee
        slip = qty * px * cfg.slippage
        cash_before = new.cash
        qty_before = _qty(new, leg.asset)
        _apply_sell(new, leg.asset, qty, fill, fee)
        total_fee += fee
        total_slip += slip
        turnover += notional
        _emit(leg.asset, "SELL", qty, fill, fee, slip, cash_before, qty_before, leg.reason)

    # ---- 2) 再买入（现金不足时按比例缩放，§6.4 第 4 步）----
    buy_legs = [l for l in plan.legs if l.side == "BUY"]
    planned_out: List[float] = []
    for leg in buy_legs:
        px = float(exec_prices.get(leg.asset, 0.0) or 0.0)
        if px <= 0:
            planned_out.append(0.0)
            continue
        fill = px * (1.0 + cfg.slippage)
        notional = float(leg.quantity) * fill
        planned_out.append(notional * (1.0 + cfg.fee))

    need = sum(planned_out)
    scale = 1.0
    if need > new.cash and need > EPS:
        scale = new.cash / need

    for leg, out in zip(buy_legs, planned_out):
        if out <= EPS:
            continue
        px = float(exec_prices.get(leg.asset, 0.0) or 0.0)
        if px <= 0:
            continue
        fill = px * (1.0 + cfg.slippage)
        spend = min(out * scale, new.cash)
        if spend <= EPS:
            continue
        gross = spend / (1.0 + cfg.fee)          # 含费后可得的名义额
        qty = gross / fill
        if qty <= EPS:
            continue
        fee = gross * cfg.fee
        slip = qty * px * cfg.slippage
        cash_before = new.cash
        qty_before = _qty(new, leg.asset)
        _apply_buy(new, leg.asset, qty, fill, fee)
        total_fee += fee
        total_slip += slip
        turnover += gross
        _emit(leg.asset, "BUY", qty, fill, fee, slip, cash_before, qty_before, leg.reason)

    new.as_of = timestamp
    new.ledger_version = account.ledger_version + (1 if events else 0)

    val_after = value_account(new, exec_prices, timestamp)
    # 剩余未执行目标差额（§6.4 第 5 步）：原偏离减去本次实际达成的权重变化
    remaining = {
        a: float(plan.weight_deltas.get(a, 0.0))
        - (val_after.weights.get(a, 0.0) - val_before.weights.get(a, 0.0))
        for a in ASSETS}

    reason = plan.legs[0].reason if plan.legs else "no_action"
    result = LedgerResult(
        events=events,
        equity_before=val_before.equity, equity_after=val_after.equity,
        cash_before=account.cash, cash_after=new.cash,
        weights_before=val_before.weights, weights_after=val_after.weights,
        remaining_target_error=remaining, cash_scale=scale,
        total_fee=total_fee, total_slippage=total_slip, turnover=turnover,
        reason=reason)
    return new, result


def append_events(result: LedgerResult, ledger_path) -> List[str]:
    """把成交事件追加到账本（§13.4），返回事件 hash 列表。"""
    rows = [{
        "event_id": e.event_id, "run_id": e.run_id,
        "timestamp": e.timestamp, "event_type": e.event_type,
        "asset": e.asset, "side": e.side, "quantity": e.quantity,
        "price": e.price, "notional": e.notional, "fee": e.fee,
        "slippage": e.slippage, "cash_before": e.cash_before,
        "cash_after": e.cash_after, "position_before": e.position_before,
        "position_after": e.position_after,
        "strategy_version": e.strategy_version, "input_hash": e.input_hash,
    } for e in result.events]
    return append_jsonl(ledger_path, rows)


def append_order_log(plan: RebalancePlan, ledger_path, *, run_id: str,
                     timestamp: datetime, strategy_version: str,
                     input_hash: str) -> List[str]:
    """把调仓计划（含未触发腿）追加到 orders.jsonl（§6.2）。"""
    rows = [{
        "run_id": run_id, "timestamp": timestamp, "asset": leg.asset,
        "side": leg.side, "quantity": leg.quantity,
        "notional_delta": leg.notional_delta,
        "current_weight": leg.current_weight, "target_weight": leg.target_weight,
        "estimated_fill_price": leg.estimated_fill_price, "reason": leg.reason,
        "status": "PAPER_FILLED", "strategy_version": strategy_version,
        "input_hash": input_hash,
    } for leg in plan.legs]
    rows.append({
        "run_id": run_id, "timestamp": timestamp, "asset": "", "side": "",
        "quantity": 0.0, "notional_delta": 0.0,
        "current_weight": 0.0, "target_weight": 0.0,
        "estimated_fill_price": 0.0,
        "reason": f"plan_status={plan.status}",
        "below_threshold": ",".join(plan.below_threshold),
        "status": "PAPER_FILLED" if plan.legs else "NO_ACTION_THRESHOLD",
        "strategy_version": strategy_version, "input_hash": input_hash,
    })
    return append_jsonl(ledger_path, rows)


__all__ = ["simulate_plan", "append_events", "append_order_log"]