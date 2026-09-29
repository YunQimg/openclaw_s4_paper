# -*- coding: utf-8 -*-
"""命令行入口（Roadmap V2 §10 / §17.1）。

子命令：
  init      初始化纸面账户
  signal    每日信号检查（计算目标与调仓建议，执行纸面成交）
  settle    纸面收盘结算（按收盘价更新净值与回撤快照）
  health    健康检查（数据新鲜度、账本完整性、策略版本）
  download  下载市场数据（直连不可达时走代理）
  replay    历史回放式 90 天纸面运行（§18 Milestone 6）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import STRATEGY_VERSION, __version__
from .config import (DEFAULT_PROXY, NotificationConfig, PaperAccountConfig,
                     StrategyConfig, ConfigError)
from .state_store import StateStore


def _repo_root() -> Path:
    """仓库根。本包为 src-layout：`<repo>/src/openclaw_s4_paper/`，故 parents[2] 即仓库根。"""
    return Path(__file__).resolve().parents[2]


def _default_dirs() -> dict:
    """默认目录：仓库根下放 config 与 state（StateStore 会再在其内部建 state/ 与
    logs/ 子目录），市场数据放在仓库 data/ 下。"""
    root = _repo_root()
    return {
        "config": root / "config",
        "workdir": root,
        "market": root / "data" / "paper_trading" / "market",
    }


# ---------------------------------------------------------------------------
def _resolve_dir(value: str | None, default: Path) -> Path:
    """CLI 参数为 None 时回落到默认目录（避免 Path(None) 崩溃）。"""
    return Path(value) if value else Path(default)


def cmd_init(args: argparse.Namespace) -> int:
    d = _default_dirs()
    acct_cfg = PaperAccountConfig.load(_resolve_dir(
        args.account_config, d["config"] / "paper_account.example.json"))
    store = StateStore(_resolve_dir(args.state_dir, d["workdir"]))
    acct = store.init_account(account_id=acct_cfg.account_id,
                              currency=acct_cfg.currency,
                              initial_cash=acct_cfg.initial_cash,
                              force=bool(args.force))
    print(f"[ok] paper account initialized: {store.account_path}")
    print(f"     account_id={acct.account_id} cash={acct.cash:,.2f} {acct.currency}")
    return 0


def _load_cfg(args: argparse.Namespace):
    d = _default_dirs()
    cfg_path = _resolve_dir(args.strategy_config, d["config"] / "strategy.json")
    if args.notification_config:
        notif_path = Path(args.notification_config)
    else:
        cand = d["config"] / "notification.json"
        notif_path = cand if cand.exists() else None
    strat = StrategyConfig.load(cfg_path)
    notif = NotificationConfig.load(notif_path) if notif_path else None
    acct = PaperAccountConfig.load(_resolve_dir(
        args.account_config, d["config"] / "paper_account.example.json"))
    return strat, notif, acct


def _paths(args: argparse.Namespace):
    d = _default_dirs()
    md = _resolve_dir(args.market_dir, d["market"])
    return {
        "market": md / "4h.csv",
        "daily": md / "1d.csv",
        "cash": md / "cash_rate_dff.csv",
        "state": _resolve_dir(args.state_dir, d["workdir"]),
    }


def _make_notifier(args: argparse.Namespace):
    """默认 dry-run：不真实发信，除非显式 --send-mail。"""
    if not args.send_mail:
        return None
    from .email_notifier import send_email as _real_send

    class _RealNotifier:
        def send(self, message, cfg):
            return _real_send(message, cfg)

    return _RealNotifier()


def cmd_signal(args: argparse.Namespace) -> int:
    from .run_once import run_once
    try:
        strat, notif, acct_cfg = _load_cfg(args)
    except ConfigError as e:
        print(f"[FATAL] config rejected: {e}", file=sys.stderr)
        return 2

    p = _paths(args)
    as_of = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
             if args.as_of else datetime.now(timezone.utc))

    notifier = _make_notifier(args)
    outcome = run_once(
        market_path=p["market"], daily_path=p["daily"], cash_path=p["cash"],
        state_dir=p["state"], strategy_cfg=strat, account_cfg=acct_cfg,
        notification_cfg=notif, as_of=as_of,
        force_first=bool(args.force_first), notifier=notifier,
        source_root=Path(__file__).resolve().parent)

    status = outcome.get("status")
    print(f"[{status}] run_id={outcome.get('run_id')}")
    if outcome.get("error"):
        print(f"  error: {outcome['error']}", file=sys.stderr)
    print(f"  email_sent={outcome.get('email_sent')}")
    return int(outcome.get("exit_code", 0))


def cmd_settle(args: argparse.Namespace) -> int:
    """按收盘价更新未实现盈亏、净值与回撤快照（§3.1 Task B）。"""
    from .ledger_store import write_snapshot
    from .market_data import (freshness, load_market_data, validate_market_data)
    from .valuation import max_drawdown_to_date, value_account

    strat, _, _ = _load_cfg(args)
    p = _paths(args)
    store = StateStore(p["state"])
    if not store.account_path.exists():
        print("[FATAL] paper account not initialized; run `init` first", file=sys.stderr)
        return 2

    as_of = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
             if args.as_of else datetime.now(timezone.utc))
    panel = load_market_data(p["market"])
    vr = validate_market_data(panel)
    if not vr.ok:
        print(f"[FATAL] market data invalid: {vr.errors}", file=sys.stderr)
        return 2
    fx = freshness(panel, as_of)
    at = fx["data_last_time"]

    acct = store.load_account()
    val = value_account(acct, panel.prices_at(at), at)
    write_snapshot(acct, store.snapshots_path,
                   extra={"run_id": f"settle-{at.strftime('%Y%m%d')}",
                          "equity": val.equity,
                          "execution_time": at.isoformat(),
                          "kind": "settlement"})

    rows = _read_snapshot_equity(store)
    mdd = max_drawdown_to_date(rows)
    payload = {"as_of": as_of, "data_last_time": at, "equity": val.equity,
               "cash": val.cash, "market_value": val.market_value,
               "weights": val.weights, "cash_weight": val.cash_weight,
               "max_dd_to_date": mdd, "is_fresh": fx["is_fresh"]}
    store.write_daily_log(at, "run", payload)
    print(f"[settled] as_of={at} equity={val.equity:,.2f} MaxDD={mdd * 100:.2f}%")
    return 0 if fx["is_fresh"] else 1


def _read_snapshot_equity(store: StateStore):
    import csv
    p = store.snapshots_path
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        out = []
        for r in csv.DictReader(f):
            try:
                out.append(float(r.get("equity") or 0.0))
            except (TypeError, ValueError):
                continue
        return out


def cmd_health(args: argparse.Namespace) -> int:
    """数据新鲜度 + 账本完整性 + 策略版本（§3.1 Task C）。"""
    from .audit import scan_source_tree
    from .market_data import (freshness, load_market_data, validate_market_data)

    try:
        strat, _, _ = _load_cfg(args)
    except ConfigError as e:
        print(f"[FATAL] config rejected: {e}", file=sys.stderr)
        return 2

    p = _paths(args)
    store = StateStore(p["state"])
    problems = []

    ref_now = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
               if getattr(args, "as_of", None) else datetime.now(timezone.utc))

    print(f"strategy_version : {STRATEGY_VERSION}")
    print(f"config_hash      : {strat.fingerprint()[:16]}")

    chain = store.verify_ledger()
    print(f"ledger chain     : {'OK' if chain['ok'] else 'BROKEN'} "
          f"({chain['n_rows']} rows)")
    if not chain["ok"]:
        problems.append(f"ledger chain broken at row {chain['broken_at']}")

    if p["market"].exists():
        panel = load_market_data(p["market"])
        vr = validate_market_data(panel)
        fx = freshness(panel, ref_now)
        print(f"market data      : {panel.n_bars} bars, last={fx['data_last_time']}")
        print(f"freshness        : {fx['status']} (age {fx['age_hours']:.1f}h)")
        if not vr.ok:
            problems.append(f"market data invalid: {vr.errors}")
        if not fx["is_fresh"]:
            problems.append(f"data stale: {fx['age_hours']:.1f}h")
    else:
        problems.append(f"market data missing: {p['market']}")
        print(f"market data      : MISSING ({p['market']})")

    sec = scan_source_tree(Path(__file__).resolve().parent)
    print(f"security scan    : {'OK' if sec['ok'] else 'FINDINGS'} "
          f"({sec['n_files_scanned']} files)")
    if not sec["ok"]:
        problems.append(f"security findings: {sec['findings'][:3]}")

    payload = {"strategy_version": STRATEGY_VERSION,
               "config_hash": strat.fingerprint(),
               "ledger": chain, "problems": problems,
               "security_scan": sec}
    store.write_daily_log(datetime.now(timezone.utc), "run", payload)

    if problems:
        print("\n[HEALTH] problems detected:")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("\n[HEALTH] all checks passed")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """下载市场数据（直连可达则不用代理，否则走代理）。"""
    from .download_data import download_market_data, resolve_proxies, FetcherUnavailable

    d = _default_dirs()
    out = Path(args.out_dir) if args.out_dir else d["market"]
    try:
        proxies = resolve_proxies(args.proxy)
    except FetcherUnavailable as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2
    print(f"proxy: {proxies.get('https') or 'direct (no proxy)'}")
    print(f"out  : {out}")
    try:
        paths = download_market_data(out, args.proxy, args.assets, args.start)
    except Exception as e:                                       # noqa: BLE001
        print(f"[FATAL] download failed: {e}", file=sys.stderr)
        return 1
    for k, v in paths.items():
        print(f"  [ok] {k}: {v} ({v.stat().st_size:,} bytes)")
    return 0


# ---------------------------------------------------------------------------
def cmd_replay(args: argparse.Namespace) -> int:
    """§18 Milestone 6：历史回放式 90 天纸面运行（mock SMTP，不联网）。"""
    from .paper_replay import run_replay

    try:
        strat, notif, acct_cfg = _load_cfg(args)
    except ConfigError as e:
        print(f"[FATAL] config rejected: {e}", file=sys.stderr)
        return 2

    # 回放需要统计邮件投递 -> 用 mock SMTP，因此强制启用通知配置（不真实发信）
    if notif is None:
        notif = NotificationConfig(enabled=True)

    p = _paths(args)
    d = _default_dirs()
    out_dir = Path(args.out_dir) if args.out_dir else d["workdir"] / "reports" / "m6"
    # StateStore 会在 root 下自建 state/ 与 logs/；root 直接取 out_dir，
    # 避免出现 reports/m6/state/state/ 双重嵌套（与缺口 F 的目录约定一致）。
    state_dir = Path(args.state_dir) if args.state_dir else out_dir

    summary = run_replay(
        market_path=p["market"], daily_path=p["daily"], cash_path=p["cash"],
        state_dir=state_dir, out_dir=out_dir, strategy_cfg=strat,
        account_cfg=acct_cfg, notification_cfg=notif, n_days=int(args.days),
        source_root=Path(__file__).resolve().parent)

    acc, win, em = summary["acceptance"], summary["window"], summary["emails"]
    print(f"[M6] mode={summary['mode']} days={win['n_days']} "
          f"({win['start']} ~ {win['end']})")
    print(f"  ledger: {summary['ledger']['rows']} rows, "
          f"chain={'OK' if summary['ledger']['chain_ok'] else 'BROKEN'}")
    print(f"  emails: {em['sent']}/{em['attempted']} = {em['success_rate'] * 100:.2f}%")
    print(f"  network attempts blocked: {summary['network_attempts_blocked']}")
    for k, v in acc.items():
        print(f"  [{'x' if v else ' '}] {k}")
    for name, path in summary["deliverables"].items():
        print(f"  {name}: {path}")
    return 0 if summary["acceptance_all_ok"] else 1


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="openclaw_s4_paper",
        description="S4 + R9 spot paper-trading system (paper only, no exchange access)")
    ap.add_argument("--version", action="version",
                    version=f"openclaw_s4_paper {__version__} ({STRATEGY_VERSION})")
    sub = ap.add_subparsers(dest="command", required=True)

    def _common(p):
        p.add_argument("--strategy-config", default=None)
        p.add_argument("--notification-config", default=None)
        p.add_argument("--account-config", default=None)
        p.add_argument("--state-dir", default=None)
        p.add_argument("--market-dir", default=None)
        p.add_argument("--as-of", default=None,
                       help="ISO8601 UTC, e.g. 2026-09-20T00:00:00+00:00")

    p_init = sub.add_parser("init", help="initialize the paper account")
    _common(p_init)
    p_init.add_argument("--force", action="store_true", help="reset an existing account")
    p_init.set_defaults(func=cmd_init)

    p_sig = sub.add_parser("signal", help="daily signal check + paper fills")
    _common(p_sig)
    p_sig.add_argument("--force-first", action="store_true",
                       help="force initial build-out even if account is not pristine")
    p_sig.add_argument("--send-mail", action="store_true",
                       help="actually send email (default: dry-run, no SMTP)")
    p_sig.set_defaults(func=cmd_signal)

    p_set = sub.add_parser("settle", help="paper settlement at close")
    _common(p_set)
    p_set.set_defaults(func=cmd_settle)

    p_h = sub.add_parser("health", help="health check")
    _common(p_h)
    p_h.set_defaults(func=cmd_health)

    p_d = sub.add_parser("download", help="download market data (auto proxy)")
    p_d.add_argument("--out-dir", default=None)
    p_d.add_argument("--proxy", default=None,
                     help=f"force this proxy (default: auto-detect; fallback {DEFAULT_PROXY})")
    p_d.add_argument("--assets", nargs="*", default=None)
    p_d.add_argument("--start", default="2017-01-01")
    p_d.set_defaults(func=cmd_download)

    p_r = sub.add_parser("replay", help="Milestone 6: historical 90-day paper replay")
    _common(p_r)
    p_r.add_argument("--days", type=int, default=90,
                     help="回放天数（默认 90，M6 验收下限）")
    p_r.add_argument("--out-dir", default=None,
                     help="交付与总结输出目录（默认 openclaw_s4_paper/reports/m6）")
    p_r.set_defaults(func=cmd_replay)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130
    except ConfigError as e:
        print(f"[FATAL] config error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())