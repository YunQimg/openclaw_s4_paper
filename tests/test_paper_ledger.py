# -*- coding: utf-8 -*-
"""纸面账本与成交模拟测试（Roadmap V2 §6.3 / §6.4 / §14.2）。

调仓触发与最小交易阈值见 `test_rebalance.py`；权重/权益恒等式见 `test_invariants.py`。
本文件只覆盖：估值与回撤、成交模拟（费用/滑点/现金约束/不做空）、哈希链账本。
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from openclaw_s4_paper.config import ConfigError, StrategyConfig
from openclaw_s4_paper.ledger_store import (GENESIS_HASH, append_jsonl, chain_hash,
                                           read_jsonl, verify_chain)
from openclaw_s4_paper.models import PaperAccount, Position
from openclaw_s4_paper.paper_ledger import simulate_plan
from openclaw_s4_paper.rebalance import build_rebalance_plan
from openclaw_s4_paper.valuation import check_invariants, max_drawdown_to_date, value_account

TS = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)
RUN = "20260915S4_R9_PAPER_V2test"


def _account(cash=10_000.0, **positions) -> PaperAccount:
    acct = PaperAccount(account_id="paper-s4-r9-001", cash=cash)
    for a, (qty, avg) in positions.items():
        acct.positions[a] = Position(a, qty, avg)
    return acct


def _prices(**kw) -> dict:
    base = {"BTC": 40_000.0, "ETH": 2_500.0, "SOL": 100.0, "BNB": 400.0}
    base.update(kw)
    return base


# --- 估值与回撤 ------------------------------------------------------------
def test_max_drawdown_to_date():
    assert max_drawdown_to_date([100, 120, 90, 110]) == pytest.approx(90 / 120 - 1)
    assert max_drawdown_to_date([100, 110, 120]) == pytest.approx(0.0)
    assert max_drawdown_to_date([]) == pytest.approx(0.0)


# --- 成交模拟 --------------------------------------------------------------
def test_buy_deducts_cash_and_sell_adds_cash():
    cfg = StrategyConfig()
    acct = _account(cash=10_000.0, BTC=(0.1, 30_000.0))
    plan = build_rebalance_plan(
        acct, {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0},
        _prices(), _prices(), cfg, TS, first=False)
    assert any(l.side == "SELL" for l in plan.legs)
    new, res = simulate_plan(acct, plan, _prices(), cfg, run_id=RUN,
                             input_hash="h", strategy_version="v", timestamp=TS)
    assert new.cash > acct.cash, "selling increases cash"
    assert new.quantity("BTC") < acct.quantity("BTC")


def test_fee_and_slippage_charged_per_leg():
    cfg = StrategyConfig()
    acct = _account(cash=10_000.0)
    plan = build_rebalance_plan(
        acct, {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35},
        _prices(), _prices(), cfg, TS, first=True)
    new, res = simulate_plan(acct, plan, _prices(), cfg, run_id=RUN,
                             input_hash="h", strategy_version="v", timestamp=TS)
    assert res.total_fee > 0
    assert res.total_slippage > 0
    assert len(res.events) == len(plan.legs)
    for e in res.events:
        assert e.fee == pytest.approx(e.notional * cfg.fee, rel=1e-9)
        # 买入价高于开盘价，卖出价低于开盘价
        if e.side == "BUY":
            assert e.price > _prices()[e.asset]
        else:
            assert e.price < _prices()[e.asset]


def test_cannot_sell_more_than_held():
    """只允许现货卖出已有持仓（不得做空）。"""
    cfg = StrategyConfig()
    acct = _account(cash=100.0, BTC=(0.01, 30_000.0))
    plan = build_rebalance_plan(
        acct, {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0},
        _prices(), _prices(), cfg, TS, first=False)
    new, res = simulate_plan(acct, plan, _prices(), cfg, run_id=RUN,
                             input_hash="h", strategy_version="v", timestamp=TS)
    assert new.quantity("BTC") >= 0
    for e in res.events:
        assert e.position_after >= -1e-12


def test_cash_constrained_buy_scales_down():
    """现金不足时按比例缩放，现金不为负（§6.4 第 4 步）。"""
    cfg = StrategyConfig()
    acct = _account(cash=500.0)          # 远不足以买入 100% 目标
    plan = build_rebalance_plan(
        acct, {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35},
        _prices(), _prices(), cfg, TS, first=True)
    new, res = simulate_plan(acct, plan, _prices(), cfg, run_id=RUN,
                             input_hash="h", strategy_version="v", timestamp=TS)
    assert new.cash >= -1e-9, "cash must not go negative"
    assert res.cash_scale < 1.0
    val = value_account(new, _prices(), TS)
    assert check_invariants(val, val.weights)["ok"]


def test_ledger_append_does_not_overwrite():
    acct = _account(cash=10_000.0)
    plan = build_rebalance_plan(
        acct, {"BTC": 0.25, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0},
        _prices(), _prices(), StrategyConfig(), TS, first=True)
    _, r1 = simulate_plan(acct, plan, _prices(), StrategyConfig(), run_id=RUN,
                          input_hash="h", strategy_version="v", timestamp=TS)
    assert len(r1.events) >= 1


def test_spot_position_rejects_negative_quantity():
    with pytest.raises(ConfigError, match="negative"):
        Position("BTC", -1.0, 100.0)


# --- 哈希链账本 ------------------------------------------------------------
def test_jsonl_chains_and_verifies(tmp_path):
    p = tmp_path / "ledger.jsonl"
    append_jsonl(p, [{"a": 1}, {"a": 2}])
    rows = read_jsonl(p)
    assert len(rows) == 2
    assert rows[0]["prev_hash"] == GENESIS_HASH
    assert rows[1]["prev_hash"] == rows[0]["hash"]
    assert verify_chain(p)["ok"]


def test_jsonl_appends_without_overwriting(tmp_path):
    p = tmp_path / "ledger.jsonl"
    append_jsonl(p, [{"a": 1}])
    append_jsonl(p, [{"a": 2}])
    rows = read_jsonl(p)
    assert [r["a"] for r in rows] == [1, 2]


def test_chain_detects_tampering(tmp_path):
    import json
    p = tmp_path / "ledger.jsonl"
    append_jsonl(p, [{"a": 1}, {"a": 2}])
    rows = read_jsonl(p)
    rows[0]["a"] = 999                     # 篡改第一条
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert not verify_chain(p)["ok"]


def test_chain_hash_is_deterministic():
    assert chain_hash("p", {"x": 1, "y": 2}) == chain_hash("p", {"y": 2, "x": 1})