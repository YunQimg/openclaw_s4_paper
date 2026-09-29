# -*- coding: utf-8 -*-
"""V4 现货多资产系统 —— 数据装载、上市日口径与质量审计（Roadmap §3）

复用 V3 的 4H 面板与指标层（scripts/v3_spot/indicators.py），
在其上构建 V4 需要的：公共时间轴、可交易掩码、上市过渡事件、现金利率。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.v3_spot import indicators as V3I
from scripts.v4_spot.config import ASSETS, START

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "v3_spot"


@lru_cache(maxsize=8)
def _full_panel(data_dir: str = str(DATA_DIR)) -> dict:
    return V3I.build_feature_panel(V3I.load_panel(Path(data_dir)))


def get_panel(start: str = START, end: str | None = None) -> dict:
    """返回 V4 面板：index / px / feats / tradable（可交易掩码）。"""
    fp = _full_panel()
    idx = fp["index"]
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end, tz="UTC")]
    px = fp["px"].reindex(idx)
    feats = {a: f.reindex(idx) for a, f in fp["feats"].items()}
    tradable = px.notna()
    return {"index": idx, "px": px, "feats": feats, "tradable": tradable,
            "assets": list(ASSETS)}


def load_cash_rate(idx: pd.DatetimeIndex, data_dir: Path = DATA_DIR) -> pd.Series:
    """DFF 年化利率 -> 每根 4H bar 的现金收益率。

    严格 point-in-time：日频利率按 *前一日* 已知值生效（shift(1)），
    再按 365 日复利折算到 6 根 4H bar。负利率截断为 0（现货现金不付息）。
    """
    f = Path(data_dir) / "cash_rate_dff.csv"
    if not f.exists():
        return pd.Series(0.0, index=idx)
    df = pd.read_csv(f, parse_dates=["timestamp"])
    if df.empty:
        return pd.Series(0.0, index=idx)
    s = df.set_index("timestamp")["rate_pct"].sort_index() / 100.0
    days = pd.DatetimeIndex(idx.normalize().unique())
    daily = s.reindex(days).ffill().shift(1).reindex(idx.normalize())
    daily.index = idx
    r_year = daily.clip(lower=0.0).fillna(0.0)
    r_bar = (1.0 + r_year) ** (1.0 / 365.0) - 1.0
    return (r_bar / 6.0).fillna(0.0)


def cash_rate_series(idx: pd.DatetimeIndex, mode: str) -> pd.Series:
    """按 Cash_0 / Cash_Rate 模式返回现金每 bar 收益率。"""
    if mode == "Cash_Rate":
        return load_cash_rate(idx)
    return pd.Series(0.0, index=idx)


# ---------------------------------------------------------------------------
# 审计：数据质量 / 上市过渡 / 权重恒等式
# ---------------------------------------------------------------------------
def data_quality(fp: dict) -> pd.DataFrame:
    """§17 v4_data_quality：每个资产的首末 bar、缺失率、重复与 OHLC 合法性。"""
    idx = fp["index"]
    rows: List[dict] = []
    for a in ASSETS:
        px = fp["px"][a]
        f = fp["feats"][a]
        n_total = len(idx)
        n_missing = int(px.isna().sum())
        first = px.first_valid_index()
        last = px.last_valid_index()
        bad_ohlc = 0
        if {"high", "low", "close"} <= set(f.columns):
            hi, lo, cl = f["high"], f["low"], f["close"]
            bad_ohlc = int(((hi < lo) | (hi < cl) | (lo > cl)).fillna(False).sum())
        rows.append({
            "asset": a,
            "bars_total": n_total,
            "bars_listed": int(n_total - n_missing),
            "bars_unlisted": n_missing,
            "first_listed_bar": first,
            "last_listed_bar": last,
            "listing_share": round((n_total - n_missing) / n_total, 6),
            "dup_timestamps": int(idx.duplicated().sum()),
            "bad_ohlc_rows": bad_ohlc,
            "nan_close_listed": int(px.dropna().isna().sum()),
        })
    return pd.DataFrame(rows)


def listing_transition(fp: dict) -> pd.DataFrame:
    """§17 v4_listing_transition：未上市 -> 已上市 的过渡事件与 Cash Reserve 释放。"""
    tradable = fp["tradable"]
    rows: List[dict] = []
    for a in ASSETS:
        m = tradable[a].to_numpy()
        first_i = int(np.argmax(m)) if m.any() else -1
        ev = {
            "asset": a,
            "listed_ever": bool(m.any()),
            "first_listed_bar": fp["index"][first_i] if first_i >= 0 else pd.NaT,
            "cash_reserve_weight": None,
        }
        rows.append(ev)
    return pd.DataFrame(rows)


def weight_invariants(weights: pd.DataFrame, tol: float = 1e-6) -> dict:
    """§19.1 权重恒等式审计：sum(asset)+cash=1、所有权重 >= 0。"""
    w = weights[ASSETS]
    tot = w.sum(axis=1) + weights["cash"]
    worst = float((tot - 1.0).abs().max()) if len(tot) else 0.0
    neg_asset = int((w < -tol).to_numpy().sum())
    neg_cash = int((weights["cash"] < -tol).sum())
    return {
        "max_abs_sum_error": worst,
        "min_sum": float(tot.min()) if len(tot) else np.nan,
        "max_sum": float(tot.max()) if len(tot) else np.nan,
        "negative_asset_bars": neg_asset,
        "negative_cash_bars": neg_cash,
        "passed": bool(worst <= 1e-6 and neg_asset == 0 and neg_cash == 0),
    }