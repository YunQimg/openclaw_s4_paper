# -*- coding: utf-8 -*-
"""单次运行编排（Roadmap V2 §11.3 主任务伪流程 / §3.2 幂等 / §3.3 失败策略）。

流程：
  load config -> validate -> load market data -> validate freshness/schema
  -> load paper ledger -> compute S4 targets -> compute current weights
  -> compute rebalance deltas -> simulate sells -> simulate buys
  -> validate invariants -> append ledger events -> write snapshot
  -> write run manifest -> send email

任何一步失败：账本不提交、发送异常邮件、退出非零状态（§11.3）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import STRATEGY_VERSION
from .audit import (assert_clean, build_run_manifest, json_dumps_safe,
                    scan_source_tree)
from .config import (ASSETS, ANOMALY_BAND_CLIPPED, ANOMALY_CASH_INSUFFICIENT,
                     ANOMALY_CONFIG_CHANGED, ANOMALY_EMAIL_FAILED,
                     ANOMALY_MA_INSUFFICIENT, ANOMALY_STATUSES,
                     ANOMALY_TSMOM_INSUFFICIENT, ALERT_CRITICAL, alert_level,
                     NotificationConfig, PaperAccountConfig, StrategyConfig,
                     ConfigError)
from .ledger_store import file_sha256, verify_chain, write_snapshot
from .market_data import (MarketPanel, freshness, load_daily_close,
                          load_market_data, next_execution_bar,
                          validate_market_data)
from .models import (AssetSignal, PaperAccount, SignalSnapshot)
from .paper_ledger import append_events, append_order_log, simulate_plan
from .rebalance import build_rebalance_plan, deviation
from .state_store import StateStore
from .strategy_s4 import build_targets
from .valuation import check_invariants, max_drawdown_to_date, value_account


class RunError(RuntimeError):
    """运行失败（触发异常邮件 + 非零退出）。"""

    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__(message)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def make_run_id(as_of: datetime, strategy_version: str, input_hash: str) -> str:
    """§3.2 run_id = YYYYMMDD + strategy_version + input_hash_prefix。"""
    return f"{as_of.strftime('%Y%m%d')}{strategy_version}{input_hash[:8]}"


def compute_input_hash(market_path: Path, daily_path: Path, cash_path: Path) -> str:
    """输入数据联合指纹（§9 input data hash）。"""
    parts = []
    for p in (market_path, daily_path, cash_path):
        parts.append(f"{Path(p).name}:{file_sha256(Path(p))}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _prices_from_panel(panel: MarketPanel, at, *, field: str = "close"
                       ) -> Dict[str, float]:
    if field == "open":
        return panel.opens_at(at)
    return panel.prices_at(at)


def _output_paths(store: StateStore, bar_time: datetime) -> Dict[str, Any]:
    """§11.2 主任务输出路径映射。

      signal_snapshot_path     -> logs/paper_trading/<date>.signal.json
      rebalance_proposal_path  -> state/orders.jsonl（调仓计划日志）
      paper_ledger_update_path -> state/ledger.jsonl（成交账本）
      email_status_path        -> logs/paper_trading/<date>.email.json（未写则为 None）
    """
    d = bar_time.astimezone(timezone.utc).strftime("%Y-%m-%d")
    email_log = store.log_dir / f"{d}.email.json"
    return {
        "signal_snapshot_path": str(store.log_dir / f"{d}.signal.json"),
        "rebalance_proposal_path": str(store.orders_path),
        "paper_ledger_update_path": str(store.ledger_path),
        "email_status_path": str(email_log) if email_log.exists() else None,
    }


# ---------------------------------------------------------------------------
# §16 连续异常计数与升级
# ---------------------------------------------------------------------------
def update_alert_state(prior: Dict[str, Any], anomalies: List[str], *,
                       run_id: str, config_hash: str,
                       now: Optional[datetime] = None) -> Dict[str, Any]:
    """更新连续异常计数与告警级别（§16）。

    本次**无**异常 -> 计数归零、级别清空（恢复正常）；
    本次**有**异常 -> 计数 +1，级别按 1 次 WARNING / 2 次 ALERT / >=3 次 CRITICAL。
    """
    streak = (int(prior.get("consecutive_anomalies") or 0) + 1) if anomalies else 0
    ts = now or datetime.now(timezone.utc)
    return {
        "consecutive_anomalies": streak,
        "level": alert_level(streak),
        "anomalies": list(anomalies),
        "run_id": run_id,
        "config_hash": config_hash,
        "updated_at": ts.astimezone(timezone.utc).isoformat(),
    }


def _email_configured(notifier, notification_cfg) -> bool:
    return (notifier is not None and notification_cfg is not None
            and bool(notification_cfg.enabled))


def _send_alert(notifier, notification_cfg, *, run_id: str, status: str,
                anomalies: List[str], alert: Dict[str, Any], detail: str,
                bar_time, store: StateStore, paused: bool = False
                ) -> Dict[str, Any]:
    """发送 §16 高优先级异常邮件（带升级级别与连续次数）。

    无论是否配置邮件，都把升级留痕写入 `<date>.alert.json`（§9 / §16 可审计）。
    """
    email_status = {"sent": False, "message_id": None, "attempted": False}
    if _email_configured(notifier, notification_cfg):
        from .email_notifier import render_alert_email, send_email

        msg = render_alert_email(
            run_id=run_id, status=status, detail=detail, traceback_text="",
            cfg=notification_cfg, bar_time=bar_time, level=alert.get("level"),
            anomalies=anomalies, streak=alert.get("consecutive_anomalies"),
            paused=paused)
        res = send_email(msg, notification_cfg)
        email_status = {"sent": bool(res.sent), "message_id": res.message_id,
                        "attempted": True}

    payload = {"run_id": run_id, "status": status, "level": alert.get("level"),
               "consecutive_anomalies": alert.get("consecutive_anomalies"),
               "anomalies": anomalies, "paused": paused,
               "sent": bool(email_status["sent"]),
               "message_id": email_status["message_id"]}
    assert_clean(json_dumps_safe(payload), context="alert log")
    store.write_daily_log(bar_time, "alert", payload)
    return email_status


# ---------------------------------------------------------------------------
# 主运行
# ---------------------------------------------------------------------------
def run_once(*, market_path: Path, daily_path: Path, cash_path: Path, state_dir: Path,
             strategy_cfg: StrategyConfig, account_cfg: PaperAccountConfig,
             notification_cfg: Optional[NotificationConfig] = None,
             as_of: Optional[datetime] = None, force_first: bool = False,
             notifier=None, source_root: Optional[Path] = None) -> Dict[str, Any]:
    """执行一次纸面调仓运行，返回运行摘要 dict。"""
    started = datetime.now(timezone.utc)
    as_of = as_of or started
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    strategy_cfg.validate()
    store = StateStore(state_dir)

    input_hash = compute_input_hash(market_path, daily_path, cash_path)
    config_hash = strategy_cfg.fingerprint({"input_hash": input_hash,
                                            "strategy_version": STRATEGY_VERSION})
    input_files = {
        "market_data_path": str(market_path),
        "daily_data_path": str(daily_path),
        "cash_rate_path": str(cash_path),
    }

    account = (store.load_account() if store.account_path.exists()
               else store.init_account(account_id=account_cfg.account_id,
                                       currency=account_cfg.currency,
                                       initial_cash=account_cfg.initial_cash))

    # ---- §16 连续异常状态（上一轮）----
    prior_alert = store.load_alert_state()

    last_bar_hint = as_of
    run_id = make_run_id(as_of, STRATEGY_VERSION, input_hash)
    # 策略配置指纹（只含配置本身；不含每次数据更新都会变的 input_hash）
    pure_config_hash = strategy_cfg.fingerprint()

    # ---- 幂等（§3.2）----
    prior = store.find_run(run_id)
    if prior is not None and (prior.get("status") in ("PAPER_FILLED", "NO_ACTION")
                              or prior.get("paused")):
        content_mismatch = (prior.get("input_data_hash") not in (None, input_hash)
                            or prior.get("config_hash") not in (None, config_hash))
        anomalies = (["duplicate_run_content_mismatch"] if content_mismatch else [])
        alert = update_alert_state(prior_alert, anomalies, run_id=run_id,
                                   config_hash=pure_config_hash, now=started)
        store.save_alert_state(alert)
        manifest = build_run_manifest(
            run_id=run_id, status="DUPLICATE_RUN",
            start_time=started.isoformat(), end_time=datetime.now(timezone.utc).isoformat(),
            input_files=input_files, input_hash=input_hash, config_hash=config_hash,
            strategy_version=STRATEGY_VERSION, ledger_version=account.ledger_version,
            extra={"duplicate_of": prior.get("status"), "alert": alert,
                   "note": "duplicate run skipped; no second fill generated"})
        store.write_run_manifest(run_id, manifest)
        return {"run_id": run_id, "status": "DUPLICATE_RUN", "email_sent": False,
                "manifest": manifest, "alert": alert,
                **_output_paths(store, as_of)}

    # ---- §16 配置指纹变化（只比策略配置本身，不含每次变动的 input_hash）----
    anomalies: List[str] = []
    if prior_alert.get("config_hash") and prior_alert["config_hash"] != pure_config_hash:
        anomalies.append(ANOMALY_CONFIG_CHANGED)

    email_status: Dict[str, Any] = {"sent": False, "message_id": None,
                                    "attempted": False}
    try:
        # ---- 账本完整性（§16 / §17.2）----
        chain = verify_chain(store.ledger_path)
        if not chain["ok"]:
            raise RunError("INVARIANT_FAILED",
                           f"ledger hash chain broken at row {chain['broken_at']}")

        # ---- 市场数据 ----
        panel = load_market_data(market_path)
        vr = validate_market_data(panel)
        if not vr.ok:
            raise RunError("INSUFFICIENT_DATA",
                           "market data validation failed: " + "; ".join(vr.errors))

        fx = freshness(panel, as_of)
        if not fx["is_fresh"]:
            raise RunError("STALE_DATA",
                           f"market data is {fx['age_hours']:.1f}h old "
                           f"(limit {fx['max_hours']}h)")

        signal_time = fx["data_last_time"]
        exec_time = next_execution_bar(panel, signal_time, strategy_cfg.execution_lag_bars)
        last_bar_hint = exec_time

        daily = load_daily_close(daily_path)

        # ---- S4 目标权重 ----
        targets, indicators, direction, band_audit = build_targets(
            panel.close, daily, strategy_cfg, listed=panel.close.notna())

        if not bool(panel.close.loc[signal_time].notna().any()):
            raise RunError("INSUFFICIENT_DATA",
                           f"no listed asset with valid price at {signal_time}")

        # 决策 bar 必须是当日首根（daily cadence）；目标在决策 bar 上取值
        tgt_row = targets.loc[signal_time]
        raw_target = {a: float(tgt_row.get(f"raw_{a}", 0.0)) for a in ASSETS}
        final_target = {a: float(tgt_row.get(f"target_{a}", 0.0)) for a in ASSETS}

        listed = panel.listed_assets(signal_time)
        for a in ASSETS:
            if a not in listed:
                raw_target[a] = 0.0
                final_target[a] = 0.0

        risky_total = sum(final_target.values())
        cash_target = 1.0 - risky_total

        # ---- 信号快照 ----
        signals: Dict[str, AssetSignal] = {}
        for a in ASSETS:
            ind = indicators[a]
            row = ind.loc[signal_time]
            scale = (float(final_target[a] / float(strategy_cfg.reference_weights[a]))
                     if strategy_cfg.reference_weights[a] > 0 else 0.0)
            signals[a] = AssetSignal(
                asset=a,
                close=float(row.get("close") or 0.0),
                ma200=float(row.get("ma200") or float("nan")),
                ma_sign=float(row.get("ma_sign") or 0.0),
                tsmom6m=float(row.get("tsmom6m") or float("nan")),
                direction=float(direction.at[signal_time, a]),
                scale=float(scale),
                reference_weight=float(strategy_cfg.reference_weights[a]),
                raw_target_weight=raw_target[a],
                band_constrained_target_weight=final_target[a],
                final_target_weight=final_target[a],
            )

        # ---- §16 高危异常检测（数据不足 / 审计层裁剪）----
        for a in listed:
            ind_row = indicators[a].loc[signal_time]
            if pd.isna(ind_row.get("ma200")):
                anomalies.append(f"{ANOMALY_MA_INSUFFICIENT}:{a}")
            if pd.isna(ind_row.get("tsmom6m")):
                anomalies.append(f"{ANOMALY_TSMOM_INSUFFICIENT}:{a}")
        if int(band_audit.n_clipped) > 0:
            anomalies.append(ANOMALY_BAND_CLIPPED)

        # ---- 调仓计划（执行价 = 执行 bar 开盘价，§6.3）----
        close_px = _prices_from_panel(panel, signal_time)
        exec_px = _prices_from_panel(panel, exec_time, field="open")
        tradable = {a: (a in listed) for a in ASSETS}

        is_first = (not account.positions) and abs(
            account.cash - account.initial_cash) < 1e-9
        listing_entry = any(
            not _had_position_before(store, a) and final_target[a] > 0
            and a in listed for a in ASSETS)

        plan = build_rebalance_plan(
            account, final_target, close_px, exec_px, strategy_cfg,
            as_of=exec_time, first=bool(force_first or is_first),
            listing_entry=bool(listing_entry), tradable=tradable)

        # 目标偏离不足阈值 -> NO_ACTION（§7）
        if not plan.has_action:
            val = value_account(account, close_px, signal_time)
            status = "NO_ACTION"
            snapshot = SignalSnapshot(
                as_of=as_of, signal_time=signal_time, execution_time=exec_time,
                data_last_time=fx["data_last_time"], strategy_version=STRATEGY_VERSION,
                run_id=run_id, input_hash=input_hash, config_hash=config_hash,
                assets=signals, listed_assets=listed, risky_total=risky_total,
                cash_target_weight=cash_target, status=status)
            _write_no_action_artifacts(store, snapshot, plan, val)
            email_status = _maybe_notify(notifier, notification_cfg, snapshot, plan,
                                         None, status, store)
            if email_status.get("attempted") and not email_status.get("sent"):
                anomalies.append(ANOMALY_EMAIL_FAILED)
            alert = update_alert_state(prior_alert, anomalies, run_id=run_id,
                                       config_hash=pure_config_hash, now=started)
            store.save_alert_state(alert)
            if anomalies:
                _send_alert(notifier, notification_cfg, run_id=run_id, status=status,
                            anomalies=anomalies, alert=alert,
                            detail="本次运行无调仓，但检测到 §16 高危异常",
                            bar_time=signal_time, store=store)
            manifest = build_run_manifest(
                run_id=run_id, status=status, start_time=started.isoformat(),
                end_time=datetime.now(timezone.utc).isoformat(),
                input_files=input_files, input_hash=input_hash,
                config_hash=config_hash, strategy_version=STRATEGY_VERSION,
                ledger_version=account.ledger_version,
                email_message_id=email_status.get("message_id"),
                extra={"band_audit": asdict(band_audit), "alert": alert,
                       "max_target_deviation": deviation(
                           val.weights, final_target)})
            store.write_run_manifest(run_id, manifest)
            return {"run_id": run_id, "status": status,
                    "email_sent": bool(email_status.get("sent")),
                    "manifest": manifest, "plan": plan, "snapshot": snapshot,
                    "alert": alert, **_output_paths(store, signal_time)}

        # ---- 模拟成交（不就地修改账户）----
        new_account, result = simulate_plan(
            account, plan, exec_px, strategy_cfg, run_id=run_id,
            input_hash=input_hash, strategy_version=STRATEGY_VERSION,
            timestamp=exec_time)

        # ---- 恒等式校验，失败则不提交（§3.3 / §14.5）----
        val_after = value_account(new_account, exec_px, exec_time)
        inv = check_invariants(val_after, final_target)
        if not inv["ok"]:
            raise RunError("INVARIANT_FAILED", f"ledger invariants violated: {inv}")

        if result.cash_scale < 1.0 - 1e-12:
            anomalies.append(ANOMALY_CASH_INSUFFICIENT)

        # ---- §16 成交前定级：连续三次 CRITICAL 则暂停纸面成交，只发告警 ----
        pending = update_alert_state(prior_alert, anomalies, run_id=run_id,
                                     config_hash=pure_config_hash, now=started)
        if pending["consecutive_anomalies"] >= ALERT_CRITICAL:
            store.save_alert_state(pending)
            snapshot = SignalSnapshot(
                as_of=as_of, signal_time=signal_time, execution_time=exec_time,
                data_last_time=fx["data_last_time"],
                strategy_version=STRATEGY_VERSION, run_id=run_id,
                input_hash=input_hash, config_hash=config_hash, assets=signals,
                listed_assets=listed, risky_total=risky_total,
                cash_target_weight=cash_target, status="REBALANCE_PROPOSED")
            return _paused(store, run_id=run_id, snapshot=snapshot, plan=plan,
                           anomalies=anomalies, alert=pending, started=started,
                           input_files=input_files, input_hash=input_hash,
                           config_hash=config_hash,
                           ledger_version=account.ledger_version,
                           notifier=notifier, notification_cfg=notification_cfg)

        # ---- 提交：先账本，后账户（§17.2）----
        append_events(result, store.ledger_path)
        append_order_log(plan, store.orders_path, run_id=run_id,
                         timestamp=exec_time, strategy_version=STRATEGY_VERSION,
                         input_hash=input_hash)
        store.save_account(new_account)
        write_snapshot(new_account, store.snapshots_path,
                       extra={"run_id": run_id, "equity": val_after.equity,
                              "execution_time": exec_time.isoformat()})

        eq_hist = [e.get("equity") for e in _read_snapshot_rows(store)]
        mdd = max_drawdown_to_date([float(e) for e in eq_hist if e not in (None, "")])

        status = "PAPER_FILLED"
        snapshot = SignalSnapshot(
            as_of=as_of, signal_time=signal_time, execution_time=exec_time,
            data_last_time=fx["data_last_time"], strategy_version=STRATEGY_VERSION,
            run_id=run_id, input_hash=input_hash, config_hash=config_hash,
            assets=signals, listed_assets=listed, risky_total=risky_total,
            cash_target_weight=cash_target, status=status)

        _write_signal_log(store, snapshot, plan, result, band_audit, mdd)
        email_status = _maybe_notify(notifier, notification_cfg, snapshot, plan,
                                     result, status, store, mdd=mdd)
        if email_status.get("attempted") and not email_status.get("sent"):
            anomalies.append(ANOMALY_EMAIL_FAILED)

        alert = update_alert_state(prior_alert, anomalies, run_id=run_id,
                                   config_hash=pure_config_hash, now=started)
        store.save_alert_state(alert)
        if anomalies:
            _send_alert(notifier, notification_cfg, run_id=run_id, status=status,
                        anomalies=anomalies, alert=alert,
                        detail="本次运行已完成纸面成交，但检测到 §16 高危异常",
                        bar_time=signal_time, store=store)

        manifest = build_run_manifest(
            run_id=run_id, status=status, start_time=started.isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
            input_files=input_files, input_hash=input_hash,
            config_hash=config_hash, strategy_version=STRATEGY_VERSION,
            ledger_version=new_account.ledger_version,
            email_message_id=email_status.get("message_id"),
            extra={"band_audit": asdict(band_audit), "alert": alert,
                   "n_legs": len(result.events),
                   "cash_scale": result.cash_scale,
                   "max_dd_to_date": mdd,
                   "equity_after": result.equity_after,
                   "email_sent": bool(email_status.get("sent"))})
        store.write_run_manifest(run_id, manifest)
        return {"run_id": run_id, "status": status,
                "email_sent": bool(email_status.get("sent")),
                "manifest": manifest, "plan": plan, "result": result,
                "snapshot": snapshot, "account": new_account, "alert": alert,
                **_output_paths(store, signal_time)}

    except RunError as e:
        return _fail(store, run_id, e.status, str(e), started, input_files,
                     input_hash, config_hash, account, as_of, notifier,
                     notification_cfg, traceback.format_exc(), last_bar_hint,
                     anomalies, prior_alert, pure_config_hash)
    except Exception as e:                                      # noqa: BLE001
        return _fail(store, run_id, "EXECUTION_ERROR", str(e), started, input_files,
                     input_hash, config_hash, account, as_of, notifier,
                     notification_cfg, traceback.format_exc(), last_bar_hint,
                     anomalies, prior_alert, pure_config_hash)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _read_snapshot_rows(store: StateStore) -> list:
    import csv
    p = store.snapshots_path
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _had_position_before(store: StateStore, asset: str) -> bool:
    for ev in store.read_events():
        if ev.get("asset") == asset and float(ev.get("position_after") or 0.0) > 0:
            return True
    return False


def _write_signal_log(store: StateStore, snapshot: SignalSnapshot, plan, result,
                      band_audit, mdd: float) -> None:
    payload = {
        "run_id": snapshot.run_id,
        "signal_time": snapshot.signal_time,
        "execution_time": snapshot.execution_time,
        "status": snapshot.status,
        "listed_assets": snapshot.listed_assets,
        "risky_total": snapshot.risky_total,
        "cash_target_weight": snapshot.cash_target_weight,
        "band_audit": asdict(band_audit),
        "max_dd_to_date": mdd,
        "assets": {a: {
            "close": s.close, "ma200": s.ma200, "ma_sign": s.ma_sign,
            "tsmom6m": s.tsmom6m, "direction": s.direction, "scale": s.scale,
            "reference_weight": s.reference_weight,
            "raw_target_weight": s.raw_target_weight,
            "band_constrained_target_weight": s.band_constrained_target_weight,
            "final_target_weight": s.final_target_weight,
        } for a, s in snapshot.assets.items()},
        "legs": [asdict_rebalance_leg(l) for l in plan.legs],
        "below_threshold": plan.below_threshold,
        "cash_scale": result.cash_scale if result else None,
        "equity_before": result.equity_before if result else None,
        "equity_after": result.equity_after if result else None,
    }
    assert_clean(json_dumps_safe(payload), context="signal log")
    store.write_daily_log(snapshot.signal_time, "signal", payload)


def asdict_rebalance_leg(leg) -> dict:
    from dataclasses import asdict as _asdict
    return _asdict(leg)


def _write_no_action_artifacts(store: StateStore, snapshot: SignalSnapshot, plan,
                               val) -> None:
    payload = {
        "run_id": snapshot.run_id,
        "signal_time": snapshot.signal_time,
        "status": snapshot.status,
        "reason": "NO_ACTION" if not plan.below_threshold else "NO_ACTION_THRESHOLD",
        "below_threshold": plan.below_threshold,
        "max_deviation": deviation(val.weights, snapshot.target_weights),
        "threshold": 0.10,
        "assets": {a: {"final_target_weight": s.final_target_weight,
                       "current_weight": val.weights.get(a, 0.0)}
                   for a, s in snapshot.assets.items()},
        "note": "no-action status written to run log only; no email sent",
    }
    assert_clean(json_dumps_safe(payload), context="no-action log")
    store.write_daily_log(snapshot.signal_time, "signal", payload)


def _maybe_notify(notifier, notification_cfg, snapshot, plan, result, status,
                  store, mdd: Optional[float] = None) -> Dict[str, Any]:
    """按 §8.2 决定是否发邮件。无调仓 -> 不发（除非配置 send_no_action）。

    返回 `attempted` 以区分「按规则跳过」与「尝试发送但失败」（§16 邮件失败异常）。
    """
    if not _email_configured(notifier, notification_cfg):
        return {"sent": False, "message_id": None, "attempted": False}
    from .email_notifier import render_rebalance_email, send_email

    if status == "NO_ACTION" and not notification_cfg.send_no_action:
        return {"sent": False, "message_id": None, "attempted": False}

    msg = render_rebalance_email(snapshot, plan, result, notification_cfg,
                                 mdd=mdd, state_store=store)
    res = send_email(msg, notification_cfg)
    payload = {"run_id": snapshot.run_id, "status": status,
               "sent": bool(res.sent), "message_id": res.message_id,
               "error": res.error}
    store.write_daily_log(snapshot.signal_time, "email", payload)
    return {"sent": bool(res.sent), "message_id": res.message_id,
            "attempted": True, "error": res.error}


def _paused(store: StateStore, *, run_id: str, snapshot: SignalSnapshot, plan,
            anomalies: List[str], alert: Dict[str, Any], started: datetime,
            input_files, input_hash: str, config_hash: str, ledger_version: int,
            notifier, notification_cfg) -> Dict[str, Any]:
    """§16：连续三次 CRITICAL -> 暂停纸面成交，不提交账本，只发送告警邮件。

    状态复用 §7 的 REBALANCE_PROPOSED（已建议、未成交），并用 `paused=true` 显式标记，
    不引入 §7 之外的新状态。
    """
    email_status = _send_alert(
        notifier, notification_cfg, run_id=run_id,
        status="REBALANCE_PROPOSED", anomalies=anomalies, alert=alert,
        detail="连续三次 CRITICAL：已暂停纸面成交，本次不产生任何纸面成交",
        bar_time=snapshot.signal_time, store=store, paused=True)
    manifest = build_run_manifest(
        run_id=run_id, status="REBALANCE_PROPOSED",
        start_time=started.isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        input_files=input_files, input_hash=input_hash, config_hash=config_hash,
        strategy_version=STRATEGY_VERSION, ledger_version=ledger_version,
        email_message_id=email_status.get("message_id"),
        extra={"alert": alert, "paused": True, "n_legs_proposed": len(plan.legs),
               "note": "paper fills paused after 3 consecutive CRITICAL anomalies"})
    store.write_run_manifest(run_id, manifest)
    return {"run_id": run_id, "status": "REBALANCE_PROPOSED", "paused": True,
            "email_sent": bool(email_status.get("sent")), "manifest": manifest,
            "plan": plan, "snapshot": snapshot, "alert": alert, "exit_code": 1,
            **_output_paths(store, snapshot.signal_time)}


def _fail(store: StateStore, run_id: str, status: str, message: str, started,
          input_files, input_hash, config_hash, account, as_of, notifier,
          notification_cfg, tb: str, last_bar_hint,
          anomalies: Optional[List[str]] = None,
          prior_alert: Optional[Dict[str, Any]] = None,
          pure_config_hash: str = "") -> Dict[str, Any]:
    """失败路径：不更新持仓、不发送成功邮件、发送异常邮件、非零退出（§3.3）。"""
    store.append_error_log(as_of, f"[{run_id}] {status}: {message}\n{tb}")

    codes = list(anomalies or [])
    if status in ANOMALY_STATUSES and status not in codes:
        codes.append(status)
    alert = update_alert_state(prior_alert or {}, codes, run_id=run_id,
                               config_hash=pure_config_hash,
                               now=datetime.now(timezone.utc))
    store.save_alert_state(alert)

    email_status = {"sent": False, "message_id": None, "attempted": False}
    paused = alert["consecutive_anomalies"] >= ALERT_CRITICAL

    if _email_configured(notifier, notification_cfg):
        try:
            from .email_notifier import (render_alert_email, send_email)
            msg = render_alert_email(
                run_id=run_id, status=status, detail=message, traceback_text=tb,
                cfg=notification_cfg, bar_time=last_bar_hint,
                level=alert.get("level"), anomalies=codes,
                streak=alert.get("consecutive_anomalies"), paused=paused)
            res = send_email(msg, notification_cfg)
            email_status = {"sent": bool(res.sent), "message_id": res.message_id,
                            "attempted": True}
        except Exception:                                       # noqa: BLE001
            pass
    store.write_daily_log(last_bar_hint, "alert", {
        "run_id": run_id, "status": status, "level": alert.get("level"),
        "consecutive_anomalies": alert.get("consecutive_anomalies"),
        "anomalies": codes, "paused": paused,
        "sent": bool(email_status.get("sent")),
        "message_id": email_status.get("message_id")})

    manifest = build_run_manifest(
        run_id=run_id, status=status, start_time=started.isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(), input_files=input_files,
        input_hash=input_hash, config_hash=config_hash,
        strategy_version=STRATEGY_VERSION,
        ledger_version=account.ledger_version if account else 0,
        email_message_id=email_status.get("message_id"), exception=tb,
        extra={"error": message, "ledger_committed": False, "alert": alert,
               "email_sent": bool(email_status.get("sent"))})
    store.write_run_manifest(run_id, manifest)
    return {"run_id": run_id, "status": status, "error": message,
            "email_sent": bool(email_status.get("sent")), "manifest": manifest,
            "alert": alert, "exit_code": 1,
            **_output_paths(store, as_of)}


# ---------------------------------------------------------------------------
# §11.1 主任务入口（Milestone 1 验收：python -m openclaw_s4_paper.run_once --help）
# ---------------------------------------------------------------------------
def _build_arg_parser():
    """§11.1 主任务输入参数（外加自有运行开关）。"""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m openclaw_s4_paper.run_once",
        description="执行一次 S4 + R9 纸面调仓（仅纸面，不连接交易所）。"
                    "输入契约见 Roadmap V2 §11.1，输出见 §11.2。")
    ap.add_argument("--as-of", default=None,
                    help="决策时刻（ISO8601 UTC，缺省为当前时刻）")
    ap.add_argument("--market-data-path", default=None,
                    help="4H OHLCV csv；1d.csv 与 cash_rate_dff.csv 从同目录读取")
    ap.add_argument("--paper-account-path", default=None,
                    help="纸面账户配置 json（account_id / currency / initial_cash）")
    ap.add_argument("--strategy-config-path", default=None,
                    help="策略配置 json（§12 strategy.json）")
    ap.add_argument("--notification-config-path", default=None,
                    help="通知配置 json（缺省 config/notification.json，不存在则不发送）")
    ap.add_argument("--state-dir", default=None,
                    help="状态根目录（StateStore；缺省为包根）")
    ap.add_argument("--send-mail", action="store_true",
                    help="真实发送 SMTP 邮件（缺省 dry-run，不投递）")
    ap.add_argument("--force-first", action="store_true",
                    help="强制首次建仓（即使账户非 pristine）")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """`python -m openclaw_s4_paper.run_once` 入口。返回进程退出码。"""
    from .cli import _default_dirs, _resolve_dir
    from .config import NotificationConfig, PaperAccountConfig, StrategyConfig

    args = _build_arg_parser().parse_args(argv)
    d = _default_dirs()

    market_path = (Path(args.market_data_path) if args.market_data_path
                   else d["market"] / "4h.csv")
    market_dir = market_path.parent
    state_dir = _resolve_dir(args.state_dir, d["workdir"])
    strat_path = _resolve_dir(args.strategy_config_path,
                              d["config"] / "strategy.json")
    acct_path = _resolve_dir(args.paper_account_path,
                             d["config"] / "paper_account.example.json")
    notif_path = (Path(args.notification_config_path)
                  if args.notification_config_path
                  else d["config"] / "notification.json")
    if not notif_path.exists():
        notif_path = None

    try:
        strat = StrategyConfig.load(strat_path)
        acct_cfg = PaperAccountConfig.load(acct_path)
    except ConfigError as e:
        print(f"[FATAL] config rejected: {e}", file=sys.stderr)
        return 2
    notif = NotificationConfig.load(notif_path) if notif_path else None

    as_of = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
             if args.as_of else None)

    notifier = None
    if args.send_mail:
        from .email_notifier import send_email as _real_send

        class _RealNotifier:
            def send(self, message, cfg):
                return _real_send(message, cfg)

        notifier = _RealNotifier()

    outcome = run_once(
        market_path=market_path, daily_path=market_dir / "1d.csv",
        cash_path=market_dir / "cash_rate_dff.csv", state_dir=state_dir,
        strategy_cfg=strat, account_cfg=acct_cfg, notification_cfg=notif,
        as_of=as_of, force_first=bool(args.force_first), notifier=notifier,
        source_root=Path(__file__).resolve().parent)

    print(f"[{outcome.get('status')}] run_id={outcome.get('run_id')}")
    for key in ("signal_snapshot_path", "rebalance_proposal_path",
                "paper_ledger_update_path", "email_status_path"):
        print(f"  {key}={outcome.get(key)}")
    if outcome.get("error"):
        print(f"  error: {outcome['error']}", file=sys.stderr)
    return int(outcome.get("exit_code", 0))


__all__ = ["run_once", "make_run_id", "compute_input_hash", "update_alert_state",
           "RunError", "main"]


if __name__ == "__main__":
    raise SystemExit(main())