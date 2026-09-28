# -*- coding: utf-8 -*-
"""本地状态存储：纸面账户、账本、运行清单、日志（Roadmap V2 §6.1 / §6.2 / §9）。

目录布局（§10）：
  state/paper_account.json        当前账户（覆盖式，唯一非追加文件之一）
  state/ledger.jsonl              成交事件账本（追加式，哈希链）
  state/orders.jsonl              调仓计划日志（追加式，哈希链）
  state/snapshots/snapshots.csv   逐日快照（追加式）
  state/runs/<run_id>.json        运行清单（幂等判定的依据）
  logs/paper_trading/<date>.*.json / .log   每日日志（§9）
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ledger_store import append_jsonl, read_jsonl, verify_chain
from .models import PaperAccount, Position


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


class StateStore:
    """纸面交易状态读写（不含任何交易能力）。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.state_dir = self.root / "state"
        # §9 日志目录固定为 logs/paper_trading/
        self.log_dir = self.root / "logs" / "paper_trading"
        self.snap_dir = self.state_dir / "snapshots"
        self.run_dir = self.state_dir / "runs"
        for d in (self.state_dir, self.log_dir, self.snap_dir, self.run_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- 路径 --------------------------------------------------------------
    @property
    def account_path(self) -> Path:
        return self.state_dir / "paper_account.json"

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / "ledger.jsonl"

    @property
    def orders_path(self) -> Path:
        return self.state_dir / "orders.jsonl"

    @property
    def snapshots_path(self) -> Path:
        return self.snap_dir / "snapshots.csv"

    def run_manifest_path(self, run_id: str) -> Path:
        return self.run_dir / f"{run_id}.json"

    # -- 告警状态（§16 连续异常计数）---------------------------------------
    @property
    def alert_state_path(self) -> Path:
        return self.state_dir / "alert_state.json"

    def load_alert_state(self) -> Dict[str, Any]:
        """读取连续异常计数；文件缺失时返回空状态（视为 0 次）。"""
        if not self.alert_state_path.exists():
            return {}
        try:
            return json.loads(self.alert_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_alert_state(self, payload: Dict[str, Any]) -> Path:
        p = self.alert_state_path
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=_json_default,
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return p

    # -- 账户 --------------------------------------------------------------
    def init_account(self, *, account_id: str, currency: str,
                     initial_cash: float, force: bool = False) -> PaperAccount:
        if self.account_path.exists() and not force:
            return self.load_account()
        acct = PaperAccount(account_id=account_id, currency=currency,
                            initial_cash=float(initial_cash), cash=float(initial_cash))
        self.save_account(acct)
        return acct

    def load_account(self) -> PaperAccount:
        if not self.account_path.exists():
            raise FileNotFoundError(
                f"paper account not initialized: {self.account_path}")
        d = json.loads(self.account_path.read_text(encoding="utf-8"))
        positions = {}
        for a, p in (d.get("positions") or {}).items():
            positions[a] = Position(a, float(p.get("quantity", 0.0)),
                                    float(p.get("average_price", 0.0)),
                                    float(p.get("realized_pnl", 0.0)))
        as_of = d.get("as_of")
        if as_of:
            as_of = datetime.fromisoformat(as_of)
        return PaperAccount(
            account_id=d.get("account_id", "paper-s4-r9-001"),
            currency=d.get("currency", "USD"),
            initial_cash=float(d.get("initial_cash", 10_000.0)),
            cash=float(d.get("cash", 0.0)),
            positions=positions, as_of=as_of,
            ledger_version=int(d.get("ledger_version", 1)))

    def save_account(self, account: PaperAccount) -> None:
        """覆盖式保存（仅在账本已成功提交后调用，§17.2）。"""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.account_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(asdict(account), default=_json_default,
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        tmp.replace(self.account_path)          # 原子替换，避免半写状态

    # -- 账本 --------------------------------------------------------------
    def verify_ledger(self) -> dict:
        return verify_chain(self.ledger_path)

    def read_events(self) -> List[dict]:
        return read_jsonl(self.ledger_path)

    # -- 运行清单（幂等，§3.2）---------------------------------------------
    def find_run(self, run_id: str) -> Optional[dict]:
        p = self.run_manifest_path(run_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def write_run_manifest(self, run_id: str, manifest: Dict[str, Any]) -> Path:
        p = self.run_manifest_path(run_id)
        p.write_text(json.dumps(manifest, default=_json_default,
                                ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    # -- 日志（§9）---------------------------------------------------------
    def write_daily_log(self, date: datetime, kind: str, payload: Dict[str, Any]) -> Path:
        d = date.astimezone(timezone.utc).strftime("%Y-%m-%d")
        p = self.log_dir / f"{d}.{kind}.json"
        p.write_text(json.dumps(payload, default=_json_default,
                                ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def append_error_log(self, date: datetime, text: str) -> Path:
        d = date.astimezone(timezone.utc).strftime("%Y-%m-%d")
        p = self.log_dir / f"{d}.error.log"
        with p.open("a", encoding="utf-8", newline="\n") as f:
            f.write(text.rstrip() + "\n")
        return p


__all__ = ["StateStore"]