# -*- coding: utf-8 -*-
"""调仓触发与计划构造测试（Roadmap V2 §6.4 / §7 / §13.3 / §14.5 阈值场景）。

覆盖：纯阈值触发（10pp 绝对口径）、最大绝对偏离、最小交易阈值、
以及低于阈值时记 NO_ACTION_THRESHOLD 而不生成订单。
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from openclaw_s4_paper.config import ASSETS, StrategyConfig
from openclaw_s4_paper.models import PaperAccount, Position
from openclaw_s4_paper.rebalance import (build_rebalance_plan, deviation,
                                         min_notional_threshold, should_rebalance)

TS = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)


def _account(cash=10_000.0, **positions) -> PaperAccount:
    acct = PaperAccount(account_id="paper-s4-r9-001", cash=cash)
    for a, (qty, avg) in positions.items():
        acct.positions[a] = Position(a, qty, avg)
    return acct


def _prices(**kw) -> dict:
    base = {"BTC": 40_000.0, "ETH": 2_500.0, "SOL": 100.0, "BNB": 400.0}
    base.update(kw)
    return base


def test_should_rebalance_first_always_true():
    cfg = StrategyConfig()
    cur = {a: 0.0 for a in ASSETS}
    tgt = {a: 0.0 for a in ASSETS}
    assert should_rebalance(cur, tgt, cfg, first=True)


def test_should_rebalance_below_threshold_false():
    cfg = StrategyConfig()
    cur = {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35}
    tgt = {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.30}
    assert not should_rebalance(cur, tgt, cfg), "5pp < 10pp threshold"


def test_should_rebalance_above_threshold_true():
    cfg = StrategyConfig()
    cur = {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35}
    tgt = {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.20}
    assert should_rebalance(cur, tgt, cfg), "15pp > 10pp threshold"


def test_deviation_is_max_absolute():
    assert deviation({"BTC": 0.3, "ETH": 0.1}, {"BTC": 0.25, "ETH": 0.1}) == pytest.approx(0.05)


def test_min_notional_threshold_is_max_of_two():
    cfg = StrategyConfig()
    assert min_notional_threshold(5_000.0, cfg) == pytest.approx(50.0)
    assert min_notional_threshold(20_000.0, cfg) == pytest.approx(100.0)


def test_below_min_notional_produces_no_legs():
    """偏离超阈值但金额低于最小交易阈值 -> 记 NO_ACTION_THRESHOLD，不生成订单。"""
    cfg = StrategyConfig(min_notional_usd=10_000.0, min_notional_equity_pct=5.0)
    acct = _account(cash=10_000.0)
    plan = build_rebalance_plan(
        acct, {"BTC": 0.5, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0},
        _prices(), _prices(), cfg, TS, first=False)
    assert not plan.has_action
    assert "BTC" in plan.below_threshold
    assert plan.status == "NO_ACTION"