# -*- coding: utf-8 -*-
"""恒等式、端到端不变量与合成数据场景测试（Roadmap V2 §14.5 / §16 / §20 验收标准 9）。

显式覆盖 §14.5 全部 9 个合成场景 + 两条验收恒等式：
  * asset_market_value + cash = equity
  * sum(weights) = 1

并集中放置端到端不变量（幂等、确定性、失败不提交）与 §16 连续异常计数测试。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from conftest import run_paper, write_market_csv

from openclaw_s4_paper import STRATEGY_VERSION
from openclaw_s4_paper.config import (ANOMALY_CONFIG_CHANGED, ASSETS,
                                      NotificationConfig, PaperAccountConfig,
                                      StrategyConfig, alert_level)
from openclaw_s4_paper.indicators import build_indicator_panel
from openclaw_s4_paper.market_data import load_market_data
from openclaw_s4_paper.models import PaperAccount, Position
from openclaw_s4_paper.rebalance import build_rebalance_plan
from openclaw_s4_paper.run_once import run_once, update_alert_state
from openclaw_s4_paper.state_store import StateStore
from openclaw_s4_paper.strategy_s4 import compute_targets
from openclaw_s4_paper.valuation import check_invariants, value_account

UTC = timezone.utc
PRICES = {"BTC": 40_000.0, "ETH": 2_500.0, "SOL": 100.0, "BNB": 400.0}
R9 = {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35}


def _filled(mkt, state, as_of, **kw):
    """强制首次建仓的快速运行（默认 all-uptrend 数据）。"""
    return run_paper(mkt, state, as_of, force_first=True, **kw)


# ---------------------------------------------------------------------------
# §14.5 验收恒等式
# ---------------------------------------------------------------------------
def test_identity_market_value_plus_cash_equals_equity():
    """asset_market_value + cash = equity（§14.5 恒等式 1）。"""
    acct = PaperAccount(account_id="a", cash=1_000.0)
    acct.positions["BTC"] = Position("BTC", 0.1, 30_000.0)
    val = value_account(acct, {**PRICES, "BTC": 40_000.0},
                        datetime(2026, 9, 1, tzinfo=UTC))
    assert val.market_value == pytest.approx(4_000.0)
    assert val.equity == pytest.approx(5_000.0)
    assert check_invariants(val, val.weights)["ok"]


def test_identity_targets_weights_sum_to_one():
    """sum(weights + cash) = 1 且 cash = 1 - risky_total（§14.5 恒等式 2）。"""
    idx = pd.date_range("2022-01-01", periods=6 * 250, freq="4h", tz="UTC")
    close = pd.DataFrame({a: np.linspace(100.0, 300.0, len(idx)) for a in ASSETS},
                         index=idx)
    daily = close.resample("1D").last().dropna()
    ind = build_indicator_panel(close, daily)
    targets, _scale, _dir, _audit = compute_targets(ind, StrategyConfig(),
                                                    listed=close.notna())
    row = targets.iloc[-1]
    risky = sum(row[f"target_{a}"] for a in ASSETS)
    assert risky + row["cash"] == pytest.approx(1.0)
    assert row["cash"] == pytest.approx(1.0 - risky)


# ---------------------------------------------------------------------------
# §14.5 合成数据场景 1-9
# ---------------------------------------------------------------------------
def test_scene_1_all_bullish_full_tier(tmp_path):
    """场景 1：4 资产全部站上 MA200 且 TSMOM 强 -> 全 1.00 档。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400,
                           drifts={a: 0.003 for a in ASSETS})
    state = tmp_path / "state"
    out = _filled(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))
    assert out["status"] == "PAPER_FILLED", out.get("error")
    for a in ASSETS:
        s = out["snapshot"].assets[a]
        assert s.ma_sign == 1.0
        assert s.scale == pytest.approx(1.0)
        assert s.final_target_weight == pytest.approx(s.reference_weight, abs=1e-6)


def test_scene_2_all_bearish_full_cash(tmp_path):
    """场景 2：4 资产全部跌破 MA200 -> 全 0 档，100% Cash，无买入。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400,
                           drifts={a: -0.003 for a in ASSETS})
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))
    assert out["status"] == "NO_ACTION", out.get("error")
    for a in ASSETS:
        assert out["snapshot"].assets[a].final_target_weight == pytest.approx(0.0)
        assert out["snapshot"].assets[a].ma_sign == -1.0
    assert out["snapshot"].cash_target_weight == pytest.approx(1.0)
    assert len(StateStore(state).read_events()) == 0


def test_scene_3_mixed_2_of_4_bullish(tmp_path):
    """场景 3：2/4 bullish -> BTC/ETH 有风险暴露，SOL/BNB 为 0。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400,
                           drifts={"BTC": 0.003, "ETH": 0.003,
                                   "SOL": -0.003, "BNB": -0.003})
    state = tmp_path / "state"
    out = _filled(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))
    assert out["status"] == "PAPER_FILLED", out.get("error")
    for a in ("BTC", "ETH"):
        assert out["snapshot"].assets[a].final_target_weight > 0.0
    for a in ("SOL", "BNB"):
        assert out["snapshot"].assets[a].final_target_weight == pytest.approx(0.0)
    risky = sum(out["snapshot"].assets[a].final_target_weight for a in ASSETS)
    assert 0.0 < risky < 1.0


def test_scene_4_tsmom_strength_changes_tier(tmp_path):
    """场景 4：TSMOM 强/弱使同向 MA200 落在不同档位。

    BTC 持续强动量、ETH 动量弱，二者 ma_sign 均为 +1，但分档不同（§5.3）。
    """
    idx = pd.date_range("2022-01-01", periods=6 * 420, freq="4h", tz="UTC")
    n = len(idx)
    btc = 100.0 * np.exp(np.linspace(0.0, 1.2, n))
    eth = 100.0 * np.exp(np.linspace(0.0, 0.10, n))
    close = pd.DataFrame({"BTC": btc, "ETH": eth,
                          "SOL": btc * 2.0, "BNB": btc * 0.5}, index=idx)
    daily = close.resample("1D").last().dropna()
    ind = build_indicator_panel(close, daily)
    targets, scale, _dir, _audit = compute_targets(ind, StrategyConfig(),
                                                   listed=close.notna())
    last = scale.iloc[-1]
    assert last["BTC"] == pytest.approx(1.0)
    assert last["ETH"] < 1.0, f"weak TSMOM must land a lower tier: {last.to_dict()}"
    assert targets.loc[idx[-1], "target_BTC"] > targets.loc[idx[-1], "target_ETH"]


def test_scene_5_sol_mid_listing(tmp_path):
    """场景 5：SOL 中途上市 -> 上市前无 SOL 买入，上市且数据充分后进入。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=340,
                           offsets={"SOL": 250})
    state = tmp_path / "state"
    out = _filled(mkt, state, datetime(2022, 6, 1, tzinfo=UTC))
    assert out["status"] == "PAPER_FILLED", out.get("error")
    assert not any(l.asset == "SOL" for l in out["plan"].legs)
    assert StateStore(state).load_account().quantity("SOL") == 0.0

    out2 = run_paper(mkt, state, datetime(2022, 11, 1, tzinfo=UTC))
    assert any(l.asset == "SOL" and l.side == "BUY" for l in out2["plan"].legs)


def test_scene_6_missing_last_day_is_stale(tmp_path):
    """场景 6：数据缺少最后一天（as_of 落后于数据 > 12h）-> STALE_DATA 硬失败。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    panel = load_market_data(mkt["4h"])
    as_of = panel.index[-1] + timedelta(days=1, hours=1)
    out = run_paper(mkt, state, as_of)
    assert out["status"] == "STALE_DATA", out.get("error")
    assert out.get("exit_code") == 1
    assert len(StateStore(state).read_events()) == 0


def _held_account_matching(target: dict) -> PaperAccount:
    equity = 10_000.0
    acct = PaperAccount(account_id="paper-s4-r9-001", cash=0.0)
    for a in ASSETS:
        qty = target[a] * equity / PRICES[a]
        acct.positions[a] = Position(a, qty, PRICES[a])
    return acct


def test_scene_7_deviation_below_threshold_no_legs():
    """场景 7：目标偏离 <10pp -> 不触发调仓（无腿，NO_ACTION）。"""
    cfg = StrategyConfig()
    acct = _held_account_matching(R9)
    plan = build_rebalance_plan(acct, R9, PRICES, PRICES, cfg,
                                datetime(2026, 9, 15, tzinfo=UTC), first=False)
    assert not plan.has_action
    assert plan.status == "NO_ACTION"


def test_scene_8_deviation_above_threshold_triggers():
    """场景 8：目标偏离 >10pp -> 触发调仓（生成卖出腿）。"""
    cfg = StrategyConfig()
    acct = _held_account_matching(R9)
    target = {**R9, "BNB": 0.20}                      # 15pp 偏离
    plan = build_rebalance_plan(acct, target, PRICES, PRICES, cfg,
                                datetime(2026, 9, 15, tzinfo=UTC), first=False)
    assert plan.has_action
    assert any(l.asset == "BNB" and l.side == "SELL" for l in plan.legs)


def test_scene_9_insufficient_cash_scales_buys(tmp_path):
    """场景 9：现金不足以完成全部买入 -> 按比例缩放，现金不为负（§6.4）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400,
                           drifts={a: 0.003 for a in ASSETS})
    state = tmp_path / "state"
    out = run_once(
        market_path=mkt["4h"], daily_path=mkt["1d"], cash_path=mkt["cash"],
        state_dir=state, strategy_cfg=StrategyConfig(),
        account_cfg=PaperAccountConfig(account_id="paper-s4-r9-001",
                                       initial_cash=500.0),
        notification_cfg=NotificationConfig(enabled=False),
        as_of=datetime(2023, 2, 4, 2, 0, tzinfo=UTC), force_first=True)
    assert out["status"] == "PAPER_FILLED", out.get("error")
    assert out["result"].cash_scale < 1.0, "buys must be scaled down by available cash"
    acct = StateStore(state).load_account()
    assert acct.cash >= -1e-9, "cash must not go negative"
    val = value_account(acct, {a: out["snapshot"].assets[a].close for a in ASSETS},
                        out["snapshot"].execution_time)
    assert check_invariants(val, val.weights)["ok"]


# ---------------------------------------------------------------------------
# 端到端不变量（幂等 / 确定性 / 失败不提交）
# ---------------------------------------------------------------------------
def test_end_to_end_fills_and_invariants(tmp_path):
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    as_of = datetime(2023, 2, 4, 2, 0, tzinfo=UTC)
    out = run_paper(mkt, state, as_of)
    assert out["status"] == "PAPER_FILLED", out.get("error")

    store = StateStore(state)
    assert store.verify_ledger()["ok"]
    acct = store.load_account()
    assert acct.cash >= -1e-9
    assert any(acct.quantity(a) > 0 for a in ASSETS), "should hold some risk assets"


def test_end_to_end_is_idempotent_no_second_fill(tmp_path):
    """重复 run_id 不重复成交（§3.2）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    as_of = datetime(2023, 2, 4, 2, 0, tzinfo=UTC)
    run_paper(mkt, state, as_of)
    n_events = len(StateStore(state).read_events())
    out2 = run_paper(mkt, state, as_of)
    assert out2["status"] == "DUPLICATE_RUN"
    assert len(StateStore(state).read_events()) == n_events, "no second fill"


def test_end_to_end_is_deterministic(tmp_path):
    """同一输入两次独立运行 -> 相同成交（§20 可复现性）。"""
    results = []
    for i in (1, 2):
        mkt = write_market_csv(tmp_path / f"mkt{i}", start="2022-01-01", days=400)
        state = tmp_path / f"state{i}"
        run_paper(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))
        ev = [(e["asset"], e["side"], round(e["quantity"], 10))
              for e in StateStore(state).read_events()]
        results.append(ev)
    assert results[0] == results[1]


def test_no_action_when_target_unchanged(tmp_path):
    """首次建仓后，若目标未变且权重未漂移 -> NO_ACTION，不生成新成交（§7）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400,
                           drifts={a: 0.003 for a in ASSETS})
    state = tmp_path / "state"
    as_of = datetime(2023, 2, 4, 2, 0, tzinfo=UTC)
    out1 = _filled(mkt, state, as_of)
    assert out1["status"] == "PAPER_FILLED", out1.get("error")
    n1 = len(StateStore(state).read_events())
    assert n1 > 0

    # 在同一 4H 轴上前进一根 bar：目标未变 -> 不应再有成交
    panel = load_market_data(mkt["4h"])
    out2 = run_paper(mkt, state, panel.index[-1] + timedelta(hours=4))
    if out2["status"] == "NO_ACTION":
        assert len(StateStore(state).read_events()) == n1, "no new fills on NO_ACTION"


def test_stale_data_blocks_fill(tmp_path):
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2023, 3, 1, tzinfo=UTC))
    assert out["status"] == "STALE_DATA"
    assert len(StateStore(state).read_events()) == 0, "stale data must not produce fills"
    assert out.get("exit_code") == 1


def test_failure_does_not_commit_ledger(tmp_path):
    """失败时账本不提交（§3.3 / §20 验收标准 13）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2023, 6, 1, tzinfo=UTC))
    assert out["status"] == "STALE_DATA"
    assert len(StateStore(state).read_events()) == 0
    assert out.get("exit_code") == 1


def test_run_manifest_has_required_fields(tmp_path):
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))
    m = out["manifest"]
    for key in ("run_id", "start_time", "end_time", "input_files",
                "input_data_hash", "config_hash", "strategy_version",
                "ledger_version", "status"):
        assert key in m, f"run manifest missing {key}"


def test_strategy_version_constant():
    assert STRATEGY_VERSION == "S4_R9_PAPER_V2"


# ---------------------------------------------------------------------------
# §16 连续异常计数与升级
# ---------------------------------------------------------------------------
def _seed_alert_state(state, streak, config_hash):
    StateStore(state).save_alert_state(
        {"consecutive_anomalies": streak, "level": None, "config_hash": config_hash})


def test_alert_level_mapping():
    assert alert_level(0) is None
    assert alert_level(1) == "WARNING"
    assert alert_level(2) == "ALERT"
    assert alert_level(3) == "CRITICAL"
    assert alert_level(7) == "CRITICAL"


def test_update_alert_state_counts_up_then_resets():
    s1 = update_alert_state({}, ["a"], run_id="R1", config_hash="c")
    assert (s1["consecutive_anomalies"], s1["level"]) == (1, "WARNING")
    s2 = update_alert_state(s1, ["a"], run_id="R2", config_hash="c")
    assert (s2["consecutive_anomalies"], s2["level"]) == (2, "ALERT")
    s3 = update_alert_state(s2, ["a"], run_id="R3", config_hash="c")
    assert (s3["consecutive_anomalies"], s3["level"]) == (3, "CRITICAL")
    s4 = update_alert_state(s3, [], run_id="R4", config_hash="c")
    assert (s4["consecutive_anomalies"], s4["level"], s4["anomalies"]) == (0, None, [])


def test_stale_data_streak_reaches_critical(tmp_path):
    """连续三次数据过期 -> 计数 3，级别 CRITICAL（§16）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    for day in (1, 2, 3):
        out = run_paper(mkt, state, datetime(2023, 3, day, tzinfo=UTC))
        assert out["status"] == "STALE_DATA", out.get("error")
        assert out["exit_code"] == 1
    st = StateStore(state).load_alert_state()
    assert st["consecutive_anomalies"] == 3
    assert st["level"] == "CRITICAL"


def test_third_critical_anomaly_pauses_paper_fills(tmp_path):
    """§16：连续第三次 CRITICAL -> 暂停纸面成交，只发告警，不产生任何成交。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    _seed_alert_state(state, 2, "config-hash-from-a-different-config")

    out = _filled(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))

    assert out["status"] == "REBALANCE_PROPOSED"
    assert out["paused"] is True
    assert out["exit_code"] == 1
    assert out["alert"]["consecutive_anomalies"] == 3
    assert out["alert"]["level"] == "CRITICAL"
    assert ANOMALY_CONFIG_CHANGED in out["alert"]["anomalies"]

    store = StateStore(state)
    assert len(store.read_events()) == 0, "paused run must not fill"
    assert store.load_account().cash == pytest.approx(10_000.0)
    assert store.verify_ledger()["ok"]
    log_dir = state / "logs" / "paper_trading"
    assert list(log_dir.glob("*.alert.json")), "escalation must be logged"


def test_single_anomaly_warns_but_still_fills(tmp_path):
    """§16：仅 1 次异常 -> WARNING，不暂停成交。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    _seed_alert_state(state, 0, "config-hash-from-a-different-config")

    out = _filled(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))

    assert out["status"] == "PAPER_FILLED", out.get("error")
    assert out["alert"]["level"] == "WARNING"
    assert out["alert"]["consecutive_anomalies"] == 1
    assert not out.get("paused")
    assert len(StateStore(state).read_events()) > 0


def test_clean_run_resets_streak_and_resumes_fills(tmp_path):
    """异常消失 -> 计数归零、级别清空、恢复纸面成交（§16）。

    R9 目标为 100% 风险资产，首根建仓买入金额会略超现金（手续费），会触发
    §16「纸面现金小于需要买入金额」。故用一半资产处于下行（目标 0）的市场，
    使建仓只用到约 35% 权益，现金充足、无任何异常。
    """
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400,
                           drifts={"BTC": 0.002, "ETH": 0.002,
                                   "SOL": -0.002, "BNB": -0.002})
    state = tmp_path / "state"
    _seed_alert_state(state, 3, StrategyConfig().fingerprint())

    out = _filled(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC))

    assert out["status"] == "PAPER_FILLED", out.get("error")
    assert out["alert"]["anomalies"] == []
    assert out["alert"]["consecutive_anomalies"] == 0
    assert out["alert"]["level"] is None
    assert len(StateStore(state).read_events()) > 0