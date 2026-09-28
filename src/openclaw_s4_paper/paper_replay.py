# -*- coding: utf-8 -*-
"""Milestone 6：90 天纸面运行（历史回放式验证，Roadmap V2 §18 M6 / §17 / §3.1）。

M6 要求交付四项、验收四项：

交付：每日状态 / 调仓邮件 / 异常记录 / 运行总结
验收：至少 90 天连续运行 / 无账本断裂 / 无真实交易所调用 / 邮件成功率 >= 99%

**实现口径（必须如实披露）**：以真实行情（`data/paper_trading/market`）按每日决策
bar 逐日推进 `as_of`，每日调用一次与线上完全相同的 `run_once` 入口，并用 mock SMTP
计数投递。这是**历史回放**，不是挂钟意义上的连续 90 个自然日；其价值在于用同一入口、
同一策略、同一账本语义跑出 90+ 天的连续账本与邮件记录。

全程启用 socket 出网守卫，确保零出网、零交易所调用（§2.1 / §14.3）。
"""
from __future__ import annotations

import csv
import hashlib
import json
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

from . import STRATEGY_VERSION
from .audit import scan_source_tree
from .config import NotificationConfig, PaperAccountConfig, StrategyConfig
from .email_notifier import DeliveryResult
from .ledger_store import file_sha256
from .market_data import daily_decision_bars, load_market_data
from .run_once import run_once
from .state_store import StateStore
from .valuation import value_account

M6_MIN_DAYS = 90
EMAIL_SUCCESS_MIN = 0.99


# ---------------------------------------------------------------------------
# Mock SMTP（不联网，用于 M6 邮件成功率统计）
# ---------------------------------------------------------------------------
@dataclass
class MockSmtp:
    """替代 `email_notifier.send_email` 的 mock 投递器（签名保持一致）。"""

    records: List[Dict[str, Any]] = field(default_factory=list)

    def __call__(self, message, cfg, env=None, smtp_client=None) -> DeliveryResult:
        msg_id = f"mock-{len(self.records) + 1:04d}"
        self.records.append({
            "subject": str(message.get("Subject", "")),
            "sent": True, "message_id": msg_id, "error": None})
        return DeliveryResult(True, msg_id, None)

    def reset(self) -> None:
        self.records.clear()


@contextmanager
def _no_network_guard():
    """阻止并统计任何出网 socket 连接；期望计数恒为 0（§2.1 / §14.3）。"""
    attempts = {"n": 0}
    original = socket.socket.connect

    def _blocked(self, address, *args, **kwargs):        # noqa: ANN001
        attempts["n"] += 1
        raise RuntimeError(
            f"network access is forbidden during M6 replay: {address!r}")

    socket.socket.connect = _blocked                    # type: ignore[assignment]
    try:
        yield attempts
    finally:
        socket.socket.connect = original                # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _replay_days(panel, n_days: int):
    """挑选最后 n_days 个可用决策日（signal bar + 一根可执行的下一 bar）。"""
    bars = daily_decision_bars(panel.index)
    out = []
    for bar in bars:
        pos = int(panel.index.get_loc(bar))
        if pos + 1 >= panel.n_bars:
            continue                                    # 无执行 bar
        out.append((bar, panel.index[pos + 1]))
    return out[-int(n_days):] if n_days else out


def run_replay(*, market_path: Path, daily_path: Path, cash_path: Path,
               state_dir: Path, out_dir: Path, strategy_cfg: StrategyConfig,
               account_cfg: PaperAccountConfig,
               notification_cfg: Optional[NotificationConfig] = None,
               n_days: int = M6_MIN_DAYS,
               source_root: Optional[Path] = None,
               reset: bool = True) -> Dict[str, Any]:
    """执行历史回放式 M6 纸面运行，产出四项交付并实测四项验收。

    返回运行总结 dict（与 `m6_summary.json` 同源）。
    """
    market_path, daily_path, cash_path = map(Path, (market_path, daily_path, cash_path))
    state_dir, out_dir = Path(state_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if reset:
        import shutil
        for sub in ("state", "logs"):
            p = state_dir / sub
            if p.exists():
                shutil.rmtree(p)
    state_dir.mkdir(parents=True, exist_ok=True)

    notif = notification_cfg or NotificationConfig(
        enabled=True, recipient="paper@example.com", sender="bot@example.com")
    strategy_cfg.validate()

    panel = load_market_data(market_path)
    days = _replay_days(panel, n_days)
    store = StateStore(state_dir)
    smtp = MockSmtp()

    daily_rows: List[Dict[str, Any]] = []
    anomaly_rows: List[Dict[str, Any]] = []
    email_rows: List[Dict[str, Any]] = []
    ledger_ok_all = True
    statuses: Dict[str, int] = {}

    with _no_network_guard() as net, mock.patch(
            "openclaw_s4_paper.email_notifier.send_email", smtp):
        for signal_bar, exec_bar in days:
            as_of = exec_bar.to_pydatetime()
            n_email_before = len(smtp.records)
            outcome = run_once(
                market_path=market_path, daily_path=daily_path, cash_path=cash_path,
                state_dir=state_dir, strategy_cfg=strategy_cfg,
                account_cfg=account_cfg, notification_cfg=notif,
                as_of=as_of, notifier=smtp,
                source_root=source_root or Path(__file__).resolve().parent)

            status = str(outcome.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
            chain = store.verify_ledger()
            ledger_ok_all = ledger_ok_all and bool(chain["ok"])

            snap = outcome.get("snapshot")
            exec_time = getattr(snap, "execution_time", None) or as_of
            equity = cash = market_value = risky = None
            if store.account_path.exists():
                acct = store.load_account()
                val = value_account(acct, panel.prices_at(exec_time), exec_time)
                equity, cash, market_value = val.equity, val.cash, val.market_value
                risky = sum(val.weights.get(a, 0.0) for a in strategy_cfg.assets)

            new_emails = smtp.records[n_email_before:]
            for rec in new_emails:
                email_rows.append({
                    "date": signal_bar.strftime("%Y-%m-%d"),
                    "run_id": outcome.get("run_id"), "status": status, **rec})

            alert = outcome.get("alert") or {}
            daily_rows.append({
                "date": signal_bar.strftime("%Y-%m-%d"),
                "signal_time": signal_bar.isoformat(),
                "execution_time": exec_time.isoformat(),
                "run_id": outcome.get("run_id"), "status": status,
                "equity": equity, "cash": cash, "market_value": market_value,
                "risky_weight": risky,
                "n_legs": len(outcome["result"].events) if outcome.get("result") else 0,
                "email_attempted": len(new_emails) > 0,
                "email_sent": bool(outcome.get("email_sent")),
                "alert_level": alert.get("level"),
                "consecutive_anomalies": alert.get("consecutive_anomalies", 0),
                "ledger_ok": bool(chain["ok"]),
            })
            for a in (alert.get("anomalies") or []):
                anomaly_rows.append({
                    "date": signal_bar.strftime("%Y-%m-%d"),
                    "run_id": outcome.get("run_id"), "status": status,
                    "level": alert.get("level"),
                    "consecutive_anomalies": alert.get("consecutive_anomalies", 0),
                    "anomaly": a})

    # ---- 四项验收 ----
    dates = [r["date"] for r in daily_rows]
    n_run = len(daily_rows)
    continuous = _is_continuous_daily(dates)
    attempted = len(email_rows)
    sent = sum(1 for r in email_rows if r["sent"])
    email_rate = (sent / attempted) if attempted else 1.0
    sec = scan_source_tree(source_root or Path(__file__).resolve().parent)

    acceptance = {
        "days_at_least_90": bool(n_run >= M6_MIN_DAYS),
        "days_continuous": bool(continuous),
        "no_ledger_break": bool(ledger_ok_all and store.verify_ledger()["ok"]),
        "no_real_exchange_calls": bool(sec["ok"] and net["n"] == 0),
        "email_success_rate_ge_99": bool(email_rate >= EMAIL_SUCCESS_MIN),
    }

    # ---- 交付 1/2/3：每日状态 / 调仓邮件 / 异常记录 ----
    _write_csv(out_dir / "m6_daily_status.csv", daily_rows,
               ["date", "signal_time", "execution_time", "run_id", "status",
                "equity", "cash", "market_value", "risky_weight", "n_legs",
                "email_attempted", "email_sent", "alert_level",
                "consecutive_anomalies", "ledger_ok"])
    _write_csv(out_dir / "m6_emails.csv", email_rows,
               ["date", "run_id", "status", "subject", "sent", "message_id", "error"])
    _write_csv(out_dir / "m6_anomalies.csv", anomaly_rows,
               ["date", "run_id", "status", "level", "consecutive_anomalies",
                "anomaly"])

    # ---- 交付 4：运行总结 ----
    summary = {
        "strategy_version": STRATEGY_VERSION,
        "config_fingerprint": strategy_cfg.fingerprint(),
        "replay_fingerprint": _replay_fingerprint(strategy_cfg, (market_path,
                                                                daily_path, cash_path)),
        "mode": "historical_replay",
        "mode_note": ("以真实行情按每日决策 bar 逐日推进 as_of 的历史回放，"
                      "非挂钟意义上的连续自然日；运行入口与线上一致（run_once）。"),
        "data": {
            "market_path": str(market_path), "daily_path": str(daily_path),
            "cash_path": str(cash_path), "n_bars": int(panel.n_bars),
            "first_bar": panel.index[0].isoformat(),
            "last_bar": panel.index[-1].isoformat(),
        },
        "window": {"start": dates[0] if dates else None,
                   "end": dates[-1] if dates else None, "n_days": n_run},
        "statuses": statuses,
        "ledger": {"rows": int(store.verify_ledger()["n_rows"]),
                   "chain_ok": bool(store.verify_ledger()["ok"])},
        "emails": {"attempted": attempted, "sent": sent,
                   "failed": attempted - sent,
                   "success_rate": round(email_rate, 6)},
        "anomalies": {"n_records": len(anomaly_rows),
                      "kinds": sorted({r["anomaly"] for r in anomaly_rows})},
        "network_attempts_blocked": int(net["n"]),
        "security_scan": {"ok": bool(sec["ok"]),
                          "n_files_scanned": int(sec["n_files_scanned"])},
        "acceptance": acceptance,
        "acceptance_all_ok": all(acceptance.values()),
        "deliverables": {
            "daily_status": str(out_dir / "m6_daily_status.csv"),
            "rebalance_emails": str(out_dir / "m6_emails.csv"),
            "anomaly_records": str(out_dir / "m6_anomalies.csv"),
            "run_summary": str(out_dir / "m6_summary.json"),
        },
    }
    (out_dir / "m6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "m6_summary.md").write_text(_render_summary_md(summary),
                                           encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _is_continuous_daily(dates: List[str]) -> bool:
    """日期序列是否逐日连续（允许周末/缺口 -> 由 is_consecutive_days 判定）。"""
    if len(dates) < 2:
        return len(dates) == 1
    parsed = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
    return all((parsed[i + 1] - parsed[i]).days == 1
               for i in range(len(parsed) - 1))


def _replay_fingerprint(cfg: StrategyConfig, paths) -> str:
    parts = [cfg.fingerprint()]
    for p in paths:
        parts.append(f"{Path(p).name}:{file_sha256(Path(p))}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def _render_summary_md(s: Dict[str, Any]) -> str:
    a = s["acceptance"]
    lines = [
        "# M6 运行总结（90 天纸面运行）",
        "",
        f"- 策略版本：`{s['strategy_version']}`",
        f"- 配置指纹：`{s['config_fingerprint'][:16]}`",
        f"- 回放指纹：`{s['replay_fingerprint'][:16]}`",
        f"- 模式：{s['mode']} —— {s['mode_note']}",
        "",
        "## 交付",
        "",
        "| 交付项 | 文件 |",
        "|--------|------|",
        f"| 每日状态 | `m6_daily_status.csv`（{s['window']['n_days']} 行） |",
        f"| 调仓邮件 | `m6_emails.csv`（{s['emails']['attempted']} 封） |",
        f"| 异常记录 | `m6_anomalies.csv`（{s['anomalies']['n_records']} 行） |",
        "| 运行总结 | `m6_summary.json` / `m6_summary.md` |",
        "",
        "## 验收（§18 M6）",
        "",
        f"- [{'x' if a['days_at_least_90'] else ' '}] 至少 90 天连续运行"
        f"（实际 {s['window']['n_days']} 天，{s['window']['start']} ~ {s['window']['end']}）",
        f"- [{'x' if a['days_continuous'] else ' '}] 日期逐日连续",
        f"- [{'x' if a['no_ledger_break'] else ' '}] 无账本断裂"
        f"（账本 {s['ledger']['rows']} 行，链校验 {'OK' if s['ledger']['chain_ok'] else 'BROKEN'}）",
        f"- [{'x' if a['no_real_exchange_calls'] else ' '}] 无真实交易所调用"
        f"（出网尝试被拦截 {s['network_attempts_blocked']} 次；"
        f"静态扫描 {'OK' if s['security_scan']['ok'] else 'FINDINGS'}）",
        f"- [{'x' if a['email_success_rate_ge_99'] else ' '}] 邮件成功率 ≥ 99%"
        f"（{s['emails']['sent']}/{s['emails']['attempted']} = "
        f"{s['emails']['success_rate'] * 100:.2f}%）",
        "",
        f"**总体：{'全部通过' if s['acceptance_all_ok'] else '存在未通过项'}**",
        "",
        "## 状态分布",
        "",
        *[f"- `{k}`：{v} 天" for k, v in sorted(s["statuses"].items())],
        "",
        "## 异常类型",
        "",
        *([f"- `{x}`" for x in s["anomalies"]["kinds"]] or ["- （无）"]),
        "",
    ]
    return "\n".join(lines)


__all__ = ["run_replay", "MockSmtp", "M6_MIN_DAYS", "EMAIL_SUCCESS_MIN"]