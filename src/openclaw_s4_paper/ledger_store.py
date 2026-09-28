# -*- coding: utf-8 -*-
"""追加式 JSONL 账本与快照存储（Roadmap V2 §6.2 / §17.3）。

设计约束：
  * **追加而非覆盖**：`append_*` 只以 `"a"` 模式打开文件；
  * 每行一个 JSON 对象，带 `prev_hash` / `hash` 形成哈希链（§16 账本 hash 连续性）；
  * 快照（snapshots.csv）是唯一的覆盖式文件，用于断点恢复（§17.2）。
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GENESIS_HASH = "0" * 64


class LedgerIntegrityError(RuntimeError):
    """账本哈希链断裂（§16 告警项）。"""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def canonical_json(payload: Dict[str, Any]) -> str:
    """稳定序列化（键排序 + 无空格），用于哈希计算。"""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=_json_default, ensure_ascii=False)


def chain_hash(prev_hash: str, payload: Dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(payload)).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> List[str]:
    """追加记录并维护哈希链，返回新增记录的 hash 列表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(path)
    prev = existing[-1].get("hash", GENESIS_HASH) if existing else GENESIS_HASH

    hashes: List[str] = []
    lines: List[str] = []
    for rec in records:
        payload = {k: v for k, v in rec.items() if k not in ("prev_hash", "hash")}
        h = chain_hash(prev, payload)
        row = {**payload, "prev_hash": prev, "hash": h}
        lines.append(json.dumps(row, default=_json_default, ensure_ascii=False))
        hashes.append(h)
        prev = h
    if lines:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return hashes


def verify_chain(path: Path) -> dict:
    """校验账本哈希链连续性（§17.2：不连续 -> 拒绝提交）。"""
    path = Path(path)
    rows = read_jsonl(path)
    if not rows:
        return {"ok": True, "n_rows": 0, "broken_at": None}
    prev = GENESIS_HASH
    for i, row in enumerate(rows):
        h = row.get("hash")
        payload = {k: v for k, v in row.items() if k not in ("prev_hash", "hash")}
        expect = chain_hash(prev, payload)
        if row.get("prev_hash") != prev or h != expect:
            return {"ok": False, "n_rows": len(rows), "broken_at": i}
        prev = h
    return {"ok": True, "n_rows": len(rows), "broken_at": None}


def write_snapshot(account: Any, path: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    """写入账户快照（覆盖式，§6.2 snapshots.csv / §17.2 断点恢复）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(account) if is_dataclass(account) else dict(account)
    positions = d.pop("positions", {}) or {}
    row: Dict[str, Any] = {
        "account_id": d.get("account_id"),
        "currency": d.get("currency"),
        "initial_cash": d.get("initial_cash"),
        "cash": d.get("cash"),
        "as_of": _json_default(d["as_of"]) if d.get("as_of") else "",
        "ledger_version": d.get("ledger_version"),
    }
    for a in ("BTC", "ETH", "SOL", "BNB"):
        pos = positions.get(a)
        row[f"{a}_qty"] = 0.0 if pos is None else float(
            pos.get("quantity", 0.0) if isinstance(pos, dict) else pos.quantity)
    if extra:
        row.update(extra)

    is_new = not path.exists()
    with path.open("a" if is_new else "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def read_latest_snapshot(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def file_sha256(path: Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


__all__ = [
    "GENESIS_HASH", "LedgerIntegrityError", "canonical_json", "chain_hash",
    "read_jsonl", "append_jsonl", "verify_chain", "write_snapshot",
    "read_latest_snapshot", "file_sha256",
]