# -*- coding: utf-8 -*-
"""安全边界测试（Roadmap V2 §13.1 / §14.3）。

确保生产代码不存在交易所客户端依赖、下单/撤单/提现能力、交易所下单 URL，
也不存在读取交易所密钥的路径；并验证 paper_only=false 时启动即被拒绝。
允许在测试名称与错误信息中出现这些字符串（§14.3）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclaw_s4_paper.audit import scan_source_tree
from openclaw_s4_paper.config import ConfigError, StrategyConfig

PKG = Path(__file__).resolve().parents[1] / "src" / "openclaw_s4_paper"


def test_production_code_has_no_exchange_capability():
    """§14.3 静态扫描：生产代码无交易所依赖与下单能力。"""
    res = scan_source_tree(PKG)
    assert res["ok"], f"security findings: {res['findings']}"
    assert res["n_files_scanned"] >= 10


def test_market_data_module_has_no_exchange_functions():
    """§13.1 market_data 不得定义 submit_order / load_exchange_account。

    只检查函数定义（AST），因为模块 docstring 会**合法地**把这两个名字列为禁止项。
    """
    import ast

    tree = ast.parse((PKG / "market_data.py").read_text(encoding="utf-8"))
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "submit_order" not in defined
    assert "load_exchange_account" not in defined


def test_paper_only_false_rejected_at_startup(tmp_path):
    """§14.3：paper_only=false 时程序拒绝启动。"""
    cfg = StrategyConfig()
    bad = cfg.to_dict()
    bad["paper_only"] = False
    p = tmp_path / "strategy.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match="paper_only"):
        StrategyConfig.load(p)