# -*- coding: utf-8 -*-
"""上市状态测试（Roadmap V2 §4.4 / §6.4 listing_rule=L1 / §14.5 场景「SOL 中途上市」）。

遵守：未上市公司不得生成买入腿；资产首次上市进入时必须触发调仓（listing_entry）。
"""
from __future__ import annotations

from datetime import datetime, timezone

from conftest import run_paper, write_market_csv

from openclaw_s4_paper.state_store import StateStore

UTC = timezone.utc


def test_listing_before_sol_keeps_cash(tmp_path):
    """SOL 上市前不得生成 SOL 买入（§4.4）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=300,
                           offsets={"SOL": 250})
    state = tmp_path / "state"
    as_of = datetime(2022, 6, 1, tzinfo=UTC)
    out = run_paper(mkt, state, as_of)
    if out["status"] == "PAPER_FILLED":
        store = StateStore(state)
        for e in store.read_events():
            assert e["asset"] != "SOL", "no SOL fill before listing"


def test_listing_entry_triggers_on_sol_first_listed(tmp_path):
    """SOL 首次上市（价格首次可用）-> 触发 listing_entry 调仓（§6.4 / L1）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=340,
                           offsets={"SOL": 250})
    state = tmp_path / "state"

    # 上市前先建仓：SOL 未上市，不得持有 SOL
    out1 = run_paper(mkt, state, datetime(2022, 6, 1, tzinfo=UTC), force_first=True)
    assert out1["status"] == "PAPER_FILLED", out1.get("error")
    assert not any(l.asset == "SOL" for l in out1["plan"].legs)
    assert StateStore(state).load_account().quantity("SOL") == 0.0

    # SOL 上市且 MA200 可用（>= 200 根 4H）后再跑一次 -> SOL 应进入且 reason=listing_entry
    out2 = run_paper(mkt, state, datetime(2022, 11, 1, tzinfo=UTC))
    assert out2["status"] == "PAPER_FILLED", out2.get("error")
    sol_legs = [l for l in out2["plan"].legs if l.asset == "SOL"]
    assert sol_legs, "SOL must enter the portfolio once listed"
    assert all(l.reason == "listing_entry" for l in sol_legs)
    assert StateStore(state).load_account().quantity("SOL") > 0.0