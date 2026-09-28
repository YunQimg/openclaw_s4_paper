# -*- coding: utf-8 -*-
"""邮件通知（Roadmap V2 §8 / §13.5）。

职责：
  * 渲染调仓邮件正文（§8.3 全部字段）与异常邮件（§8.2）；
  * 发送前做敏感信息扫描，拒绝包含 SMTP 密码 / API Key / 私钥（§13.5）；
  * SMTP 凭据只从环境变量读取（§8.1），禁止写入仓库、JSON 或邮件正文。

本模块**不**包含任何交易能力。
"""
from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audit import assert_clean, json_dumps_safe
from .config import ASSETS, SMTP_ENV_VARS, NotificationConfig

DISCLAIMER = "本邮件只代表纸面交易模拟结果，系统未连接交易所，也未提交任何真实订单。"


@dataclass
class DeliveryResult:
    """邮件投递结果。"""

    sent: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class SmtpConfigError(RuntimeError):
    """SMTP 环境变量缺失或非法。"""


# ---------------------------------------------------------------------------
# SMTP 配置（只读环境变量）
# ---------------------------------------------------------------------------
def load_smtp_config(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """从环境变量读取 SMTP 配置（§8.1）。"""
    e = env if env is not None else os.environ
    missing = [k for k in ("PAPER_SMTP_HOST", "PAPER_SMTP_PORT") if not e.get(k)]
    if missing:
        raise SmtpConfigError(f"missing SMTP environment variables: {missing}")
    try:
        port = int(e["PAPER_SMTP_PORT"])
    except (TypeError, ValueError) as exc:
        raise SmtpConfigError(f"PAPER_SMTP_PORT must be an integer: {exc}") from exc
    return {
        "host": e["PAPER_SMTP_HOST"],
        "port": port,
        "user": e.get("PAPER_SMTP_USER") or None,
        "password": e.get("PAPER_SMTP_PASSWORD") or None,
        "tls": str(e.get("PAPER_SMTP_TLS", "true")).lower() in ("1", "true", "yes"),
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{float(x) * 100:.2f}%"


def _fmt_ts(ts) -> str:
    if ts is None:
        return "-"
    if isinstance(ts, str):
        return ts
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def render_rebalance_email(snapshot, plan, result, cfg: NotificationConfig,
                           mdd: Optional[float] = None,
                           state_store=None) -> EmailMessage:
    """渲染调仓邮件（§8.3）。"""
    date_str = snapshot.signal_time.astimezone(timezone.utc).strftime("%Y-%m-%d")
    subject = f"[S4 Paper] 调仓建议与纸面成交 - {date_str}"

    lines: List[str] = []
    lines.append("系统：S4 + R9")
    lines.append(f"运行状态：{snapshot.status}")
    lines.append(f"信号时间：{_fmt_ts(snapshot.signal_time)}")
    lines.append(f"执行时间：{_fmt_ts(snapshot.execution_time)}")
    lines.append(f"数据最后时间：{_fmt_ts(snapshot.data_last_time)}")
    lines.append(f"策略版本：{snapshot.strategy_version}")
    lines.append(f"run_id：{snapshot.run_id}")
    lines.append("")

    lines.append("── 市场状态（资产级）──")
    for a in ASSETS:
        s = snapshot.assets[a]
        ma = "-" if s.ma200 != s.ma200 else f"{s.ma200:,.2f}"      # NaN 检查
        ts = "-" if s.tsmom6m != s.tsmom6m else f"{s.tsmom6m:+.4f}"
        lines.append(
            f"{a}: close={s.close:,.4f} MA200={ma} ma_sign={s.ma_sign:+.0f} "
            f"tsmom6m={ts} direction={s.direction:+.4f} scale={s.scale:.2f}")
    lines.append(f"已上市资产：{', '.join(snapshot.listed_assets) or '（无）'}")
    lines.append("")

    lines.append("── 目标组合（双口径）──")
    for a in ASSETS:
        s = snapshot.assets[a]
        lines.append(f"{a}: raw={_fmt_pct(s.raw_target_weight)} "
                     f"band-constrained={_fmt_pct(s.band_constrained_target_weight)}")
    lines.append(f"Cash target：{_fmt_pct(snapshot.cash_target_weight)}")
    lines.append(f"风险资产合计：{_fmt_pct(snapshot.risky_total)}")
    lines.append("")

    lines.append("── 纸面账户 ──")
    if result is not None:
        lines.append(f"equity before：{result.equity_before:,.2f}")
    else:
        lines.append(f"equity before：{plan.equity:,.2f}")
    if state_store is not None:
        try:
            acct = state_store.load_account()
            lines.append(f"current cash：{acct.cash:,.2f}")
        except Exception:                                        # noqa: BLE001
            pass
    lines.append(f"current weights：{_weights_str(plan.weight_deltas, 'current')}")
    lines.append("")

    lines.append("── 调仓清单 ──")
    if not plan.legs:
        lines.append("（无成交腿）")
        if plan.below_threshold:
            lines.append(f"低于最小交易阈值：{', '.join(plan.below_threshold)}")
    for leg in plan.legs:
        lines.append(
            f"{leg.asset} {leg.side} qty={leg.quantity:,.6f} "
            f"est_fill={leg.estimated_fill_price:,.4f} "
            f"notional={abs(leg.notional_delta):,.2f} reason={leg.reason}")
    lines.append("")

    lines.append("── 执行后 ──")
    if result is not None:
        lines.append(f"equity after：{result.equity_after:,.2f}")
        lines.append(f"cash after：{result.cash_after:,.2f}")
        lines.append(f"fee 合计：{result.total_fee:,.4f}")
        lines.append(f"slippage 合计：{result.total_slippage:,.4f}")
        lines.append(f"turnover：{result.turnover:,.2f}")
        lines.append(f"cash scale：{result.cash_scale:.4f}")
        lines.append("weights after：")
        for a in ASSETS:
            lines.append(f"  {a}={_fmt_pct(result.weights_after.get(a, 0.0))}")
        lines.append("remaining target error：")
        for a in ASSETS:
            lines.append(f"  {a}={result.remaining_target_error.get(a, 0.0):+.6f}")
    lines.append(f"MaxDD to date：{_fmt_pct(mdd) if mdd is not None else '-'}")
    lines.append("")
    lines.append(DISCLAIMER)

    body = "\n".join(lines)
    # §13.5 发送前敏感信息扫描
    assert_clean(body, context="email body")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.recipient
    msg.set_content(body)
    return msg


def _weights_str(deltas: Dict[str, float], _label: str) -> str:
    return ", ".join(f"{a}={v:+.4f}" for a, v in deltas.items())


def render_alert_email(*, run_id: str, status: str, detail: str,
                       traceback_text: str, cfg: NotificationConfig,
                       bar_time: Optional[datetime] = None,
                       level: Optional[str] = None,
                       anomalies: Optional[List[str]] = None,
                       streak: Optional[int] = None,
                       paused: bool = False) -> EmailMessage:
    """渲染异常邮件（§8.2 / §16 告警升级）。"""
    ts = bar_time or datetime.now(timezone.utc)
    date_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
    subject = f"[S4 Paper][ALERT] 运行失败或数据过期 - {date_str}"

    body = "\n".join([
        "系统：S4 + R9",
        f"运行状态：{status}",
        f"run_id：{run_id}",
        f"数据最后时间：{_fmt_ts(bar_time)}",
        f"告警级别：{level or '(none)'}（连续异常 {streak or 0} 次）",
        "",
        "── 异常项 ──",
        "\n".join(anomalies) if anomalies else "(none)",
        "",
        "── 错误详情 ──",
        detail,
        "",
        "── traceback ──",
        traceback_text or "(none)",
        "",
        ("账本未提交：本次运行不产生任何纸面成交（连续三次 CRITICAL，已暂停纸面成交）。"
         if paused else
         "账本未提交：本次运行不产生任何纸面成交。"),
        "",
        DISCLAIMER,
    ])
    assert_clean(body, context="alert email body")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.sender
    msg["To"] = cfg.recipient
    msg.set_content(body)
    return msg


# ---------------------------------------------------------------------------
# 发送
# ---------------------------------------------------------------------------
def send_email(message: EmailMessage, cfg: NotificationConfig,
               env: Optional[Dict[str, str]] = None,
               smtp_client=None) -> DeliveryResult:
    """发送邮件。SMTP 凭据只来自环境变量（§8.1）。

    `smtp_client` 供测试注入 mock；为 None 时按环境变量真实投递。
    """
    try:
        # 发送前再次确认正文与主题不含敏感信息
        assert_clean(message.get("Subject", ""), context="email subject")
        payload = message.get_body(preferencelist=("plain",))
        if payload is not None:
            assert_clean(payload.get_content(), context="email body")
    except Exception as e:                                       # noqa: BLE001
        return DeliveryResult(False, None, f"redaction check failed: {e}")

    if not cfg.enabled:
        return DeliveryResult(False, None, "notification disabled in config")

    try:
        if smtp_client is not None:
            return _send_with(smtp_client, message)

        sc = load_smtp_config(env)
        if sc["tls"]:
            client = smtplib.SMTP_SSL(sc["host"], sc["port"],
                                      context=ssl.create_default_context(), timeout=30)
        else:
            client = smtplib.SMTP(sc["host"], sc["port"], timeout=30)
        try:
            if not sc["tls"]:
                client.starttls(context=ssl.create_default_context())
            if sc["user"] and sc["password"]:
                client.login(sc["user"], sc["password"])
            return _send_with(client, message)
        finally:
            try:
                client.quit()
            except Exception:                                    # noqa: BLE001
                pass
    except Exception as e:                                       # noqa: BLE001
        return DeliveryResult(False, None, f"{type(e).__name__}: {e}")


def _send_with(client, message: EmailMessage) -> DeliveryResult:
    client.send_message(message)
    msg_id = message.get("Message-ID")
    if not msg_id:
        import hashlib
        msg_id = hashlib.sha256(
            (message.get("Subject", "") + message.get("To", "")).encode()
        ).hexdigest()[:16]
    return DeliveryResult(True, msg_id, None)


# ---------------------------------------------------------------------------
# 附件（§8.4）
# ---------------------------------------------------------------------------
def attach_csv(message: EmailMessage, csv_text: str, filename: str) -> EmailMessage:
    message.add_attachment(csv_text.encode("utf-8"),
                           maintype="text", subtype="csv", filename=filename)
    return message


def attach_text(message: EmailMessage, text: str, filename: str) -> EmailMessage:
    message.add_attachment(text.encode("utf-8"),
                           maintype="text", subtype="plain", filename=filename)
    return message


def write_email_artifacts(out_dir: Path, run_id: str, snapshot, plan,
                          result) -> Dict[str, Path]:
    """写出 §8.4 附件（paper_rebalance_*.csv / paper_snapshot_*.json）。"""
    import csv
    import io
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = snapshot.signal_time.astimezone(timezone.utc).strftime("%Y%m%d")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["asset", "side", "quantity", "estimated_fill_price",
                "notional", "current_weight", "target_weight", "reason"])
    for leg in plan.legs:
        w.writerow([leg.asset, leg.side, f"{leg.quantity:.10f}",
                    f"{leg.estimated_fill_price:.10f}",
                    f"{abs(leg.notional_delta):.2f}",
                    f"{leg.current_weight:.6f}", f"{leg.target_weight:.6f}",
                    leg.reason])
    p_csv = out_dir / f"paper_rebalance_{date_str}.csv"
    p_csv.write_text(buf.getvalue(), encoding="utf-8")

    payload = {
        "run_id": run_id, "status": snapshot.status, "strategy_version": snapshot.strategy_version,
        "signal_time": _fmt_ts(snapshot.signal_time),
        "execution_time": _fmt_ts(snapshot.execution_time),
        "targets": snapshot.target_weights,
        "cash_target": snapshot.cash_target_weight,
        "equity_before": None if result is None else result.equity_before,
        "equity_after": None if result is None else result.equity_after,
    }
    p_json = out_dir / f"paper_snapshot_{date_str}.json"
    p_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": p_csv, "json": p_json}


__all__ = [
    "DeliveryResult", "SmtpConfigError", "DISCLAIMER", "load_smtp_config",
    "render_rebalance_email", "render_alert_email", "send_email",
    "attach_csv", "attach_text", "write_email_artifacts",
]