# -*- coding: utf-8 -*-
"""CLI 层测试（Roadmap V2 §10 / §14.3）。

重点覆盖：默认目录解析不得因 None 参数崩溃、策略配置非法时拒绝启动、
以及副本目录结构不会出现重复嵌套。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from conftest import write_market_csv

from openclaw_s4_paper import cli
from openclaw_s4_paper.state_store import StateStore


def test_default_dirs_are_repo_rooted():
    d = cli._default_dirs()
    assert d["config"].name == "config"
    assert d["config"].parent.name == "openclaw_s4_paper"
    assert d["market"].name == "market"


def test_resolve_dir_handles_none():
    """None 参数必须回落到默认目录（历史上 Path(None) 会崩溃）。"""
    default = Path("C:/tmp/x")
    assert cli._resolve_dir(None, default) == default
    assert cli._resolve_dir("", default) == default
    assert cli._resolve_dir("D:/custom", default) == Path("D:/custom")


def test_init_with_no_args_uses_defaults(tmp_path, monkeypatch, capsys):
    """`init` 不带任何参数必须成功（不得 Path(None) 崩溃）。"""
    d = cli._default_dirs()
    monkeypatch.setattr(cli, "_default_dirs",
                        lambda: {**d, "workdir": tmp_path,
                                 "config": d["config"]})
    rc = cli.main(["init", "--force"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "paper account initialized" in out
    # 状态目录应恰好是 <tmp>/state，不能是 <tmp>/state/state
    assert (tmp_path / "state" / "paper_account.json").exists()
    assert not (tmp_path / "state" / "state").exists(), "must not double-nest state/"


def test_state_store_layout(tmp_path):
    store = StateStore(tmp_path)
    assert store.account_path == tmp_path / "state" / "paper_account.json"
    assert store.ledger_path == tmp_path / "state" / "ledger.jsonl"
    assert store.snapshots_path == tmp_path / "state" / "snapshots" / "snapshots.csv"
    # §9 日志目录固定为 logs/paper_trading/
    assert store.log_dir == tmp_path / "logs" / "paper_trading"


def test_signal_with_no_args_uses_defaults(tmp_path, monkeypatch):
    """`signal` 不带参数必须走默认目录且不崩溃。"""
    mkt = write_market_csv(tmp_path / "market", start="2022-01-01", days=400)
    d = cli._default_dirs()
    monkeypatch.setattr(cli, "_default_dirs",
                        lambda: {**d, "workdir": tmp_path, "market": tmp_path / "market"})
    as_of = "2023-02-04T02:00:00+00:00"
    rc = cli.main(["signal", "--as-of", as_of])
    assert rc == 0
    store = StateStore(tmp_path)
    assert store.verify_ledger()["ok"]


def test_invalid_strategy_config_exits_2(tmp_path, monkeypatch, capsys):
    """策略配置非法 -> 退出码 2（§12 启动拒绝项）。"""
    import json
    bad = tmp_path / "strategy.json"
    bad.write_text(json.dumps({"strategy_id": "WRONG_ID"}), encoding="utf-8")
    d = cli._default_dirs()
    monkeypatch.setattr(cli, "_default_dirs",
                        lambda: {**d, "workdir": tmp_path})
    rc = cli.main(["health", "--strategy-config", str(bad)])
    assert rc == 2
    assert "rejected" in capsys.readouterr().err.lower()


def test_download_without_proxy_fails_hard(monkeypatch, capsys):
    """显式传入空代理 -> 下载硬失败（不允许静默直连）。"""
    monkeypatch.delenv("PAPER_DISABLE_NETWORK_FETCHES", raising=False)
    monkeypatch.setenv("PAPER_PROXY", "")
    rc = cli.main(["download", "--proxy", ""])
    assert rc == 2
    assert "proxy" in capsys.readouterr().err.lower()


def test_download_disabled_network_fails(monkeypatch):
    monkeypatch.setenv("PAPER_DISABLE_NETWORK_FETCHES", "1")
    from openclaw_s4_paper.download_data import FetcherUnavailable, resolve_proxies
    with pytest.raises(FetcherUnavailable, match="disabled"):
        resolve_proxies()


def test_resolve_proxies_uses_env_then_default(monkeypatch):
    from openclaw_s4_paper.config import DEFAULT_PROXY
    from openclaw_s4_paper.download_data import resolve_proxies

    monkeypatch.delenv("PAPER_DISABLE_NETWORK_FETCHES", raising=False)
    monkeypatch.setenv("PAPER_PROXY", "http://127.0.0.1:9999")
    assert resolve_proxies()["https"] == "http://127.0.0.1:9999"
    assert resolve_proxies("http://explicit:1")["https"] == "http://explicit:1"

    monkeypatch.delenv("PAPER_PROXY", raising=False)
    assert resolve_proxies()["https"] == DEFAULT_PROXY


def test_health_passes_with_fresh_data(tmp_path, monkeypatch, capsys):
    """数据新鲜时 health 应返回 0。"""
    mkt = write_market_csv(tmp_path / "market", start="2022-01-01", days=400)
    from openclaw_s4_paper.market_data import load_market_data
    panel = load_market_data(mkt["4h"])
    # 最后一个完整 bar 之后 1 小时（< 12h 新鲜度阈值）
    fresh_as_of = (panel.index[-1] + __import__("pandas").Timedelta(hours=1)).isoformat()

    d = cli._default_dirs()
    monkeypatch.setattr(cli, "_default_dirs",
                        lambda: {**d, "workdir": tmp_path, "market": tmp_path / "market"})
    StateStore(tmp_path).init_account(account_id="a", currency="USD",
                                      initial_cash=10_000.0, force=True)
    rc = cli.main(["health", "--as-of", fresh_as_of])
    out = capsys.readouterr().out
    assert "ledger chain     : OK" in out
    assert "security scan    : OK" in out
    assert rc == 0, out


# ---------------------------------------------------------------------------
# §18 M1 验收命令：python -m openclaw_s4_paper.run_once --help
# ---------------------------------------------------------------------------
def test_run_once_main_help_lists_section_11_1_inputs(capsys):
    """§11.1 主任务入口必须暴露五个输入参数并给出 --help。"""
    from openclaw_s4_paper.run_once import main
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--as-of", "--market-data-path", "--paper-account-path",
                 "--strategy-config-path", "--notification-config-path"):
        assert flag in out, f"missing §11.1 input flag {flag}"


def test_run_once_module_help_via_subprocess():
    """§18 M1 验收命令字面量：`python -m openclaw_s4_paper.run_once --help` 退出码 0。"""
    import os
    import subprocess
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(
        [sys.executable, "-B", "-m", "openclaw_s4_paper.run_once", "--help"],
        capture_output=True, text=True, env=env, cwd=str(src.parent))
    assert r.returncode == 0, r.stderr
    assert "--as-of" in r.stdout


# ---------------------------------------------------------------------------
# §11.2 主任务输出：四个路径键
# ---------------------------------------------------------------------------
def test_run_once_output_has_section_11_2_paths(tmp_path):
    """§11.2 输出必须包含 4 个路径键（signal/rebalance/ledger/email）。"""
    from conftest import run_paper
    mkt = write_market_csv(tmp_path / "market", start="2022-01-01", days=400)
    state = tmp_path / "state"
    out = run_paper(mkt, state, datetime(2023, 2, 4, 2, 0, tzinfo=timezone.utc),
                    force_first=True)
    for key in ("signal_snapshot_path", "rebalance_proposal_path",
                "paper_ledger_update_path", "email_status_path"):
        assert key in out, f"§11.2 output missing {key}"

    assert out["signal_snapshot_path"], "signal snapshot path must be present"
    assert out["rebalance_proposal_path"].endswith("orders.jsonl")
    assert out["paper_ledger_update_path"].endswith("ledger.jsonl")
    # 未启用邮件 -> email_status_path 允许为 None
    assert out["email_status_path"] is None