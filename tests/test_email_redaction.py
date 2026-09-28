# -*- coding: utf-8 -*-
"""邮件渲染、发送与脱敏测试（Roadmap V2 §8 / §13.5 / §14.4 / §9 禁止记录项）。"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from conftest import run_paper, write_market_csv

from openclaw_s4_paper.audit import (RedactionError, assert_clean, redact,
                                     scan_text)
from openclaw_s4_paper.config import ASSETS, NotificationConfig
from openclaw_s4_paper.email_notifier import (DISCLAIMER, SmtpConfigError,
                                              load_smtp_config,
                                              render_alert_email,
                                              render_rebalance_email,
                                              send_email)
from openclaw_s4_paper.state_store import StateStore

UTC = timezone.utc


def _filled(tmp_path, **kw):
    """跑一次会成交的运行，返回 (outcome, state_dir, mkt)。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=UTC), **kw)
    return out, state, mkt


class _FakeSmtp:
    def __init__(self) -> None:
        self.sent = []

    def send_message(self, msg):
        self.sent.append(msg)


# --- 调仓邮件 --------------------------------------------------------------
def test_email_body_contains_all_required_fields(tmp_path):
    out, state, _mkt = _filled(tmp_path, force_first=True)
    msg = render_rebalance_email(
        out["snapshot"], out["plan"], out.get("result"),
        NotificationConfig(recipient="x@y.z", sender="a@b.c"),
        mdd=-0.05, state_store=StateStore(state))
    body = msg.get_body(preferencelist=("plain",)).get_content()

    for token in ["系统：S4 + R9", "运行状态", "信号时间", "执行时间",
                  "数据最后时间", "策略版本", "run_id",
                  "市场状态", "目标组合", "raw=", "band-constrained=",
                  "Cash target", "调仓清单", "执行后", "MaxDD to date",
                  DISCLAIMER]:
        assert token in body, f"missing {token!r} in email body"

    for a in ASSETS:
        assert f"{a}:" in body
    assert "tsmom6m=" in body
    assert "scale=" in body
    assert msg["Subject"].startswith("[S4 Paper] 调仓建议与纸面成交")


def test_email_body_never_leaks_password(tmp_path):
    """密码不出现在邮件正文与日志（§14.4）。"""
    out, _state, _mkt = _filled(tmp_path)
    msg = render_rebalance_email(out["snapshot"], out["plan"], out.get("result"),
                                NotificationConfig())
    body = msg.get_body(preferencelist=("plain",)).get_content()
    for pat in ("PAPER_SMTP_PASSWORD", "api_key", "api_secret", "private_key"):
        assert pat.lower() not in body.lower()


def test_no_action_does_not_send_email(tmp_path):
    """无调仓时不发送邮件（§8.2 / §20 验收标准 11）。"""
    mkt = write_market_csv(tmp_path / "mkt", start="2022-01-01", days=400)
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2022, 3, 1, tzinfo=UTC),
                    notification_cfg=NotificationConfig(send_no_action=False),
                    notifier=object())          # 提供 notifier 但状态非调仓
    if out["status"] == "NO_ACTION":
        assert out["email_sent"] is False


# --- 异常邮件 --------------------------------------------------------------
def test_alert_email_subject_and_body():
    msg = render_alert_email(run_id="R1", status="STALE_DATA", detail="too old",
                             traceback_text="", cfg=NotificationConfig())
    assert msg["Subject"].startswith("[S4 Paper][ALERT]")
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "账本未提交" in body
    assert DISCLAIMER in body


def test_alert_email_includes_escalation_level():
    msg = render_alert_email(run_id="R1", status="STALE_DATA", detail="too old",
                             traceback_text="", cfg=NotificationConfig(),
                             level="CRITICAL", anomalies=["STALE_DATA"],
                             streak=3, paused=True)
    assert msg["Subject"].startswith("[S4 Paper][ALERT]")
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "告警级别：CRITICAL" in body
    assert "连续异常 3 次" in body
    assert "已暂停纸面成交" in body


# --- 发送 ------------------------------------------------------------------
def test_send_email_with_mock_client():
    msg = render_alert_email(run_id="R1", status="X", detail="d",
                             traceback_text="", cfg=NotificationConfig())
    client = _FakeSmtp()
    res = send_email(msg, NotificationConfig(), smtp_client=client)
    assert res.sent
    assert len(client.sent) == 1


def test_send_email_disabled_config_does_not_send():
    msg = render_alert_email(run_id="R1", status="X", detail="d",
                             traceback_text="", cfg=NotificationConfig())
    res = send_email(msg, NotificationConfig(enabled=False), smtp_client=_FakeSmtp())
    assert not res.sent
    assert "disabled" in (res.error or "")


def test_smtp_config_from_env():
    env = {"PAPER_SMTP_HOST": "h", "PAPER_SMTP_PORT": "465",
           "PAPER_SMTP_USER": "u", "PAPER_SMTP_PASSWORD": "p",
           "PAPER_SMTP_TLS": "true"}
    sc = load_smtp_config(env)
    assert sc["host"] == "h" and sc["port"] == 465 and sc["tls"] is True


def test_smtp_config_missing_raises():
    with pytest.raises(SmtpConfigError, match="missing"):
        load_smtp_config({})


# --- 脱敏与日志 ------------------------------------------------------------
def test_scan_text_detects_sensitive():
    assert "api_key" in scan_text("my api_key is here")
    assert "private_key" in scan_text("PRIVATE-KEY: abc")
    assert "smtp_password" in scan_text("PAPER_SMTP_PASSWORD=hunter2")
    assert scan_text("nothing sensitive") == []


def test_assert_clean_raises_and_redact_masks():
    with pytest.raises(RedactionError):
        assert_clean("api_secret=abcd", context="test")
    masked = redact("api_secret=abcd")
    assert "abcd" not in masked, "the secret value must be removed"
    assert "[REDACTED]" in masked
    # redact 后的文本必须能再次通过扫描（不得自触发）
    assert_clean(redact("PAPER_SMTP_PASSWORD=hunter2"), context="redacted")
    assert_clean(redact("api_key: 'sk-1234567890'"), context="redacted")
    assert_clean(redact('private_key = "abc"'), context="redacted")


def test_logs_contain_no_secrets(tmp_path):
    """每日日志目录为 logs/paper_trading/，且不得泄漏敏感信息（§9）。"""
    out, state, _mkt = _filled(tmp_path)
    log_dir = state / "logs" / "paper_trading"
    assert log_dir.exists(), "§9 log dir must be logs/paper_trading/"
    assert list(log_dir.glob("*.json")), "daily json logs must be written"
    for p in log_dir.glob("*.json"):
        text = p.read_text(encoding="utf-8")
        for pat in ("PAPER_SMTP_PASSWORD", "api_key", "api_secret", "private_key"):
            assert pat.lower() not in text.lower(), f"{pat} leaked into {p.name}"
    assert out["signal_snapshot_path"] is not None