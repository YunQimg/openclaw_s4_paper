# -*- coding: utf-8 -*-
"""V3 现货多策略系统 —— 共享装载与运行工具"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from scripts.v3_spot import engine as E
from scripts.v3_spot import indicators as I
from scripts.v3_spot import strategy as S

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "v3_spot"

# 回测起点：确保 BTC/ETH/BNB 的 MA200(4H) 与 TSMOM6M 均已成立；
# TSMOM12M 在 2018-11 前不参与（自动按可用分量归一化）；SOL 2020-12 起可交易。
START = "2018-06-01"
SOL_START = "2020-12-15"


@lru_cache(maxsize=4)
def load_panel():
    return I.build_feature_panel(I.load_panel(DATA_DIR))


def get_fp(start: str | None = START, end: str | None = None) -> dict:
    fp = load_panel()
    return I.slice_panel(fp, start=start, end=end)


def cash_rate(idx) -> pd.Series:
    return E.load_cash_rate(idx, DATA_DIR)


def run_strategy(fp: dict, cfg: S.Cfg, label: str = "") -> dict:
    st = S.build_strategy(fp, cfg)
    res = E.run_backtest(fp, st["targets"], cfg, cash_rate(fp["index"]))
    res.update({"label": label, "cfg": cfg, "targets": st["targets"],
                "layers": st["layers"], "bars": st["bars"]})
    return res


def run_benchmark(fp: dict, cfg: S.Cfg, weights=None) -> dict:
    w = dict(weights or S.REF_WEIGHT)
    return E.benchmark_buy_hold(fp, w, cfg, cash_rate(fp["index"]))


def run_benchmark_calendar(fp: dict, cfg: S.Cfg, freq: str, weights=None) -> dict:
    """日历再平衡的 Buy & Hold 基准（freq: "M" 月 / "Q" 季 / "2Q" 半年）。"""
    w = dict(weights or S.REF_WEIGHT)
    return E.benchmark_calendar(fp, w, cfg, freq, cash_rate(fp["index"]))


def sol_available_index(fp: dict) -> pd.DatetimeIndex:
    return fp["index"][fp["px"]["SOL"].notna()]