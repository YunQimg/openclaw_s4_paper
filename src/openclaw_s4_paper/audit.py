# -*- coding: utf-8 -*-
"""审计与安全扫描（Roadmap V2 §9 / §13.5 / §14.3）。

两类职责：
  1. **敏感信息扫描**：邮件正文、日志、运行清单在写出前必须通过扫描，
     拒绝包含 SMTP 密码 / API Key / 私钥 / 助记词（§9 禁止记录项）。
  2. **安全静态扫描**：确保生产代码不出现交易所下单能力（§14.3）。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# §13.5 / §14.3 敏感与禁用模式
SENSITIVE_PATTERNS: Dict[str, str] = {
    "api_key": r"api[_-]?key",
    "api_secret": r"api[_-]?secret",
    "private_key": r"private[_-]?key",
    "seed_phrase": r"seed[_-]?phrase",
    "smtp_password": r"PAPER_SMTP_PASSWORD|\bsmtp[_-]?pass(word)?\b",
    "bearer_token": r"bearer\s+[A-Za-z0-9\-._~+/]{16,}",
}

# §14.3 生产代码不得出现的交易所能力
FORBIDDEN_CAPABILITIES: List[str] = [
    "place_order", "cancel_order", "get_account_balance",
    "get_open_orders", "withdraw", "transfer",
]
FORBIDDEN_IMPORTS: List[str] = [
    "ccxt", "binance.client", "binance.spot", "okx", "coinbase",
    "nautilus_trader", "web3",
]

EXCHANGE_URL_PATTERN = re.compile(
    r"https?://[^\s\"']*(api\.binance\.com|api\.okx\.com|api\.coinbase\.com)"
    r"[^\s\"']*/(order|account|withdraw|trade)", re.I)


class RedactionError(ValueError):
    """文本包含禁止出现的敏感信息。"""


def scan_text(text: str, *, context: str = "text") -> List[str]:
    """扫描文本，返回命中的敏感模式名列表。"""
    hits: List[str] = []
    for name, pat in SENSITIVE_PATTERNS.items():
        if re.search(pat, text, re.I):
            hits.append(name)
    return hits


def assert_clean(text: str, *, context: str = "text") -> None:
    """不通过则抛 RedactionError（邮件发送前 / 日志写出前调用）。"""
    hits = scan_text(text, context=context)
    if hits:
        raise RedactionError(
            f"{context} contains forbidden sensitive material: {hits}")


# 形如 `key = value` / `key: value` / `key="value"` 的凭据赋值
_ASSIGNMENT = r"(?P<eq>\s*[:=]\s*)(?P<val>\"[^\"]*\"|'[^']*'|\S+)"


# 掩码标签使用序号而非模式名：若标签里带回模式名，掩码后的文本会再次命中扫描，
# 导致 `assert_clean(redact(x))` 无法通过（自触发循环）。
REDACT_LABEL = "[REDACTED]"


def redact(text: str) -> str:
    """把命中的敏感模式（含其后的凭据值）替换为掩码。

    仅替换模式名不够：`api_secret=abcd` 里真正敏感的是 `abcd`。
    因此对「模式名 + 赋值」组合整体掩码，再兜底掩码裸模式名。
    """
    out = text
    for _name, pat in SENSITIVE_PATTERNS.items():
        out = re.sub(rf"{pat}{_ASSIGNMENT}", REDACT_LABEL, out, flags=re.I)
        out = re.sub(pat, REDACT_LABEL, out, flags=re.I)
    return out


def scan_source_tree(root: Path, *, skip_dirs: Iterable[str] = (
        "__pycache__", ".git", "tests", "test")) -> dict:
    """§14.3 静态扫描：确认生产代码无交易所依赖与下单能力。

    测试目录被跳过（§14.3 允许在测试名称中出现这些字符串）。
    """
    root = Path(root)
    skip = set(skip_dirs)
    findings: List[dict] = []
    n_files = 0

    for p in sorted(root.rglob("*.py")):
        if any(part in skip for part in p.parts):
            continue
        n_files += 1
        text = p.read_text(encoding="utf-8", errors="ignore")
        for imp in FORBIDDEN_IMPORTS:
            for m in re.finditer(rf"^\s*(?:from|import)\s+{re.escape(imp)}\b",
                                 text, re.M):
                findings.append({"file": str(p), "kind": "forbidden_import",
                                 "match": imp, "line": text[:m.start()].count("\n") + 1})
        for cap in FORBIDDEN_CAPABILITIES:
            for m in re.finditer(rf"\b{re.escape(cap)}\s*\(", text):
                line_no = text[:m.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1] if line_no <= len(
                    text.splitlines()) else ""
                # 允许在注释/字符串中提及（§14.3）
                if line.strip().startswith("#") or '"""' in line or "'''" in line:
                    continue
                findings.append({"file": str(p), "kind": "forbidden_call",
                                 "match": cap, "line": line_no})
        for m in EXCHANGE_URL_PATTERN.finditer(text):
            findings.append({"file": str(p), "kind": "exchange_order_url",
                             "match": m.group(0)[:80],
                             "line": text[:m.start()].count("\n") + 1})

    return {"ok": not findings, "n_files_scanned": n_files, "findings": findings}


def build_run_manifest(*, run_id: str, status: str, start_time: str, end_time: str,
                       input_files: Dict[str, str], input_hash: str,
                       config_hash: str, strategy_version: str,
                       ledger_version: int, email_message_id: Optional[str] = None,
                       exception: Optional[str] = None,
                       extra: Optional[dict] = None) -> dict:
    """§9 日志必须记录字段的清单。"""
    manifest = {
        "run_id": run_id,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "input_files": input_files,
        "input_data_hash": input_hash,
        "config_hash": config_hash,
        "strategy_version": strategy_version,
        "ledger_version": ledger_version,
        "email_message_id": email_message_id,
        "exception_traceback": exception,
    }
    if extra:
        manifest.update(extra)
    # 写出前做敏感信息扫描（§9 禁止记录 SMTP password / API key）
    assert_clean(json_dumps_safe(manifest), context="run manifest")
    return manifest


def json_dumps_safe(obj) -> str:
    import json
    from dataclasses import asdict, is_dataclass
    from datetime import datetime, timezone

    def default(o):
        if isinstance(o, datetime):
            return o.astimezone(timezone.utc).isoformat()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    return json.dumps(obj, default=default, ensure_ascii=False, sort_keys=True)


__all__ = [
    "SENSITIVE_PATTERNS", "FORBIDDEN_CAPABILITIES", "FORBIDDEN_IMPORTS",
    "RedactionError", "scan_text", "assert_clean", "redact", "REDACT_LABEL",
    "scan_source_tree", "build_run_manifest", "json_dumps_safe",
]