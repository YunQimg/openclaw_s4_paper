# -*- coding: utf-8 -*-
"""Milestone 6 历史回放测试（Roadmap V2 §18 M6 / §3.1 / §14.3）。

覆盖 M6 四项交付（每日状态 / 调仓邮件 / 异常记录 / 运行总结）与四项验收
（≥90 天连续 / 无账本断裂 / 无真实交易所调用 / 邮件成功率 ≥99%）。
使用合成数据缩短回放天数（验收阈值本身另行断言）。
"""
from __future__ import annotations

import csv
from pathlib import Path

from conftest import write_market_csv

from openclaw_s4_paper import cli
from openclaw_s4_paper.config import (NotificationConfig, PaperAccountConfig,
                                      StrategyConfig)
from openclaw_s4_paper.paper_replay import (EMAIL_SUCCESS_MIN, M6_MIN_DAYS,
                                            run_replay)

PKG = Path(__file__).resolve().parents[1] / "src" / "openclaw_s4_paper"


def _replay(tmp_path: Path, n_days: int = 5, **kw):
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    out = tmp_path / "m6"
    summary = run_replay(
        market_path=mkt["4h"], daily_path=mkt["1d"], cash_path=mkt["cash"],
        state_dir=out / "state", out_dir=out, strategy_cfg=StrategyConfig(),
        account_cfg=PaperAccountConfig(account_id="paper-s4-r9-001"),
        notification_cfg=NotificationConfig(enabled=True), n_days=n_days,
        source_root=PKG, **kw)
    return summary, out


def _read_csv(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- 交付 ------------------------------------------------------------------
def test_replay_writes_four_deliverables(tmp_path):
    """交付：每日状态 / 调仓邮件 / 异常记录 / 运行总结。"""
    summary, out = _replay(tmp_path)
    for name in ("m6_daily_status.csv", "m6_emails.csv", "m6_anomalies.csv",
                 "m6_summary.json", "m6_summary.md"):
        assert (out / name).exists(), f"missing deliverable {name}"
    assert summary["deliverables"]["run_summary"] == str(out / "m6_summary.json")
    assert (out / "state" / "state" / "ledger.jsonl").exists()


def test_replay_daily_status_has_one_row_per_day(tmp_path):
    summary, out = _replay(tmp_path, n_days=5)
    rows = _read_csv(out / "m6_daily_status.csv")
    assert len(rows) == 5
    assert summary["window"]["n_days"] == 5
    for r in rows:
        assert r["run_id"] and r["status"] and r["date"]
        assert r["ledger_ok"] in ("True", "true")


def test_replay_emails_recorded_with_subject(tmp_path):
    _summary, out = _replay(tmp_path, n_days=5)
    rows = _read_csv(out / "m6_emails.csv")
    assert rows, "at least the first build-out must attempt an email"
    for r in rows:
        assert r["subject"].startswith("[S4 Paper]")
        assert r["sent"] in ("True", "true")


# --- 验收 ------------------------------------------------------------------
def test_replay_no_ledger_break(tmp_path):
    summary, _out = _replay(tmp_path, n_days=5)
    assert summary["ledger"]["chain_ok"] is True
    assert summary["acceptance"]["no_ledger_break"] is True


def test_replay_no_real_exchange_calls(tmp_path):
    """出网守卫拦截次数为 0，且静态扫描无交易所能力（§2.1 / §14.3）。"""
    summary, _out = _replay(tmp_path, n_days=5)
    assert summary["network_attempts_blocked"] == 0
    assert summary["security_scan"]["ok"] is True
    assert summary["acceptance"]["no_real_exchange_calls"] is True


def test_replay_email_success_rate_ge_99(tmp_path):
    summary, _out = _replay(tmp_path, n_days=5)
    em = summary["emails"]
    assert em["attempted"] >= 1, "first build-out must trigger a rebalance email"
    assert em["sent"] == em["attempted"], "mock SMTP must deliver every attempt"
    assert em["success_rate"] >= EMAIL_SUCCESS_MIN
    assert summary["acceptance"]["email_success_rate_ge_99"] is True


def test_replay_day_threshold_and_continuity(tmp_path):
    summary, _out = _replay(tmp_path, n_days=5)
    assert M6_MIN_DAYS == 90
    assert summary["acceptance"]["days_at_least_90"] is False, "5 天 < 90 天"
    assert summary["acceptance"]["days_continuous"] is True


def test_replay_is_reproducible(tmp_path):
    """同一输入两次独立回放 -> 指纹与结果一致（§20 可复现性）。"""
    s1, _ = _replay(tmp_path / "a", n_days=5)
    s2, _ = _replay(tmp_path / "b", n_days=5)
    assert s1["replay_fingerprint"] == s2["replay_fingerprint"]
    assert s1["statuses"] == s2["statuses"]
    assert s1["ledger"] == s2["ledger"]
    assert s1["emails"] == s2["emails"]


# --- CLI 子命令 ------------------------------------------------------------
def test_replay_subcommand_smoke(tmp_path, monkeypatch, capsys):
    write_market_csv(tmp_path / "market", start="2022-01-01", days=400)
    d = cli._default_dirs()
    monkeypatch.setattr(cli, "_default_dirs",
                        lambda: {**d, "workdir": tmp_path,
                                 "market": tmp_path / "market",
                                 "config": d["config"]})
    rc = cli.main(["replay", "--days", "4", "--out-dir", str(tmp_path / "m6")])
    out = capsys.readouterr().out
    assert "[M6]" in out
    assert "chain=OK" in out
    assert rc == 1, "4 天 < 90 天 -> M6 验收未全部通过，退出码 1"
    assert (tmp_path / "m6" / "m6_summary.json").exists()