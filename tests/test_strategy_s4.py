# -*- coding: utf-8 -*-
"""策略单元测试（Roadmap V2 §13.2 / §14.1 / §20 验收标准 1）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openclaw_s4_paper.config import (ASSETS, SIGNAL_BAND, StrategyConfig,
                                      ConfigError)
from openclaw_s4_paper.indicators import (build_asset_indicators, compute_ma200_sign,
                                          compute_tsmom6m, ma, ma_sign,
                                          map_daily_to_4h, tsmom_daily)
from openclaw_s4_paper.market_data import (daily_decision_bars, load_daily_close,
                                           load_market_data)
from openclaw_s4_paper.strategy_s4 import (apply_audit_band, apply_signal_band,
                                           bucket_scale, build_targets,
                                           compute_direction, normalize)


# --- MA200 -----------------------------------------------------------------
def test_ma200_uses_full_window_and_is_nan_before():
    n = 250
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series(np.arange(1.0, n + 1), index=idx)
    m = ma(close, 200)
    assert m.iloc[:199].isna().all(), "MA200 must be NaN before 200 bars"
    assert not np.isnan(m.iloc[199])
    assert m.iloc[199] == pytest.approx(close.iloc[:200].mean())


@pytest.mark.parametrize("relation,expected", [("above", 1.0), ("below", -1.0)])
def test_ma_sign_above_below(relation, expected):
    n = 260
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    base = np.full(n, 100.0)
    if relation == "above":
        base[-1] = 200.0
    else:
        base[-1] = 50.0
    close = pd.Series(base, index=idx)
    s = ma_sign(close, 200)
    assert s.iloc[-1] == expected


def test_ma_sign_zero_when_equal():
    n = 210
    idx = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series(np.full(n, 100.0), index=idx)
    s = ma_sign(close, 200)
    assert s.iloc[-1] == 0.0, "close == MA200 -> sign 0"


def test_ma_sign_zero_before_enough_bars():
    idx = pd.date_range("2022-01-01", periods=150, freq="4h", tz="UTC")
    close = pd.Series(np.linspace(80, 200, 150), index=idx)
    s = ma_sign(close, 200)
    assert (s == 0.0).all(), "insufficient bars -> sign 0"


# --- TSMOM 6M --------------------------------------------------------------
def test_tsmom_formula_matches_tanh_of_momentum():
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 400.0, 300), index=idx)
    ts = tsmom_daily(close, 126, 0.5)
    raw = close / close.shift(126) - 1.0
    expect = np.tanh(raw / 0.5)
    pd.testing.assert_series_equal(ts, expect, check_names=False)


def test_tsmom_is_point_in_time_shifted():
    """4H bar t 上的 tsmom 只能依赖 <= t-1 的日线（§5.2 无未来泄漏）。"""
    idx4h = pd.date_range("2022-01-01", periods=6 * 30, freq="4h", tz="UTC")
    close_4h = pd.Series(np.linspace(100, 200, len(idx4h)), index=idx4h)
    daily = close_4h.resample("1D").last().dropna()
    ts_daily = tsmom_daily(daily, 126, 0.5)
    mapped = map_daily_to_4h(ts_daily, idx4h)

    # 手工构造：应等于 ts_daily.shift(1).ffill()
    manual = ts_daily.shift(1).reindex(idx4h, method="ffill")
    pd.testing.assert_series_equal(mapped, manual, check_names=False)

    # 若把某一日线值改掉，当日 4H 映射值不得随之改变
    day = daily.index[20]
    bumped = daily.copy()
    bumped.loc[day] = bumped.loc[day] * 10.0
    mapped2 = map_daily_to_4h(tsmom_daily(bumped, 126, 0.5), idx4h)
    same_day_bars = idx4h[(idx4h >= day) & (idx4h < day + pd.Timedelta(days=1))]
    assert mapped.reindex(same_day_bars).fillna(-999).equals(
        mapped2.reindex(same_day_bars).fillna(-999)), \
        "same-day daily change must not leak into that day's 4H bars"


# --- Direction -------------------------------------------------------------
def test_direction_weighting_two_components():
    idx = pd.date_range("2022-01-01", periods=5, freq="4h", tz="UTC")
    sign = pd.Series(1.0, index=idx)
    ts = pd.Series(1.0, index=idx)
    d = compute_direction(sign, ts, {"ma200": 0.35, "tsmom6m": 0.15})
    assert np.allclose(d, 1.0)

    sign0 = pd.Series(0.0, index=idx)
    d2 = compute_direction(sign0, ts, {"ma200": 0.35, "tsmom6m": 0.15})
    assert np.allclose(d2, 0.30), "0.15/0.50 = 0.30 when ma_sign=0"


def test_direction_renormalizes_when_component_missing():
    idx = pd.date_range("2022-01-01", periods=5, freq="4h", tz="UTC")
    sign = pd.Series(1.0, index=idx)
    ts = pd.Series(np.nan, index=idx)
    d = compute_direction(sign, ts, {"ma200": 0.35, "tsmom6m": 0.15})
    assert np.allclose(d, 1.0), "missing tsmom -> direction == ma_sign"


def test_direction_nan_when_all_components_missing():
    idx = pd.date_range("2022-01-01", periods=3, freq="4h", tz="UTC")
    nan = pd.Series(np.nan, index=idx)
    d = compute_direction(nan, nan, {"ma200": 0.35, "tsmom6m": 0.15})
    assert d.isna().all()


# --- Bucketing -------------------------------------------------------------
@pytest.mark.parametrize("d,expected", [
    (1.00, 1.00), (0.80, 1.00), (0.76, 1.00),
    (0.75, 0.75), (0.60, 0.75), (0.51, 0.75),
    (0.50, 0.50), (0.30, 0.50), (0.26, 0.50),
    (0.25, 0.25), (0.10, 0.25), (0.01, 0.25),
    (0.00, 0.00), (-0.40, 0.00), (-1.00, 0.00),
])
def test_bucket_boundaries_land_in_lower_tier(d, expected):
    s = pd.Series([d])
    assert bucket_scale(s, [0.25, 0.50, 0.75]).iloc[0] == pytest.approx(expected)


def test_bucket_nan_stays_nan():
    s = pd.Series([np.nan])
    assert np.isnan(bucket_scale(s, [0.25, 0.50, 0.75]).iloc[0])


def test_ma_sign_negative_always_zero_scale():
    """ma_sign = -1 -> direction <= -0.40 -> scale 恒为 0（无中间档）。"""
    idx = pd.date_range("2022-01-01", periods=4, freq="4h", tz="UTC")
    for ts_val in (-1.0, -0.5, 0.0, 0.5, 1.0):
        d = compute_direction(pd.Series(-1.0, index=idx),
                              pd.Series(ts_val, index=idx),
                              {"ma200": 0.35, "tsmom6m": 0.15})
        assert (bucket_scale(d, [0.25, 0.50, 0.75]) == 0.0).all()


def test_ma_sign_positive_buckets_are_half_or_higher():
    """ma_sign = +1 -> scale ∈ {0.50, 0.75, 1.00}。"""
    idx = pd.date_range("2022-01-01", periods=4, freq="4h", tz="UTC")
    for ts_val in (1.0, 0.5, 0.0, -0.5, -1.0):
        d = compute_direction(pd.Series(1.0, index=idx),
                              pd.Series(ts_val, index=idx),
                              {"ma200": 0.35, "tsmom6m": 0.15})
        s = bucket_scale(d, [0.25, 0.50, 0.75])
        assert s.isin([0.50, 0.75, 1.00]).all(), f"tsmom={ts_val} -> {s.tolist()}"


# --- 两层配置带 ------------------------------------------------------------
def test_signal_band_floor_applies_to_sol_bnb_in_uptrend():
    """SOL/BNB 上行且分档 >= 0.50 时托至 30% / 35%（§5.4(a)）。"""
    cfg = StrategyConfig()
    idx = pd.date_range("2022-01-01", periods=2, freq="4h", tz="UTC")
    raw = pd.DataFrame({"BTC": 0.125, "ETH": 0.05, "SOL": 0.15, "BNB": 0.175},
                       index=idx)
    scale = pd.DataFrame({"BTC": 0.50, "ETH": 0.50, "SOL": 0.50, "BNB": 0.50},
                         index=idx)
    out = apply_signal_band(raw, scale, cfg)
    assert out["SOL"].iloc[0] == pytest.approx(0.30)
    assert out["BNB"].iloc[0] == pytest.approx(0.35)
    assert out["BTC"].iloc[0] == pytest.approx(0.125), "BTC 下界 5% 不触发"
    assert out["ETH"].iloc[0] == pytest.approx(0.05), "ETH == 下界不触发"


def test_signal_band_does_not_lift_reduction_to_zero():
    """减仓（scale = 0）不受下界约束，目标保持 0（风险控制不得被摧毁）。"""
    cfg = StrategyConfig()
    idx = pd.date_range("2022-01-01", periods=1, freq="4h", tz="UTC")
    raw = pd.DataFrame({"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0}, index=idx)
    scale = pd.DataFrame({"BTC": 0.0, "ETH": 0.0, "SOL": 0.0, "BNB": 0.0}, index=idx)
    out = apply_signal_band(raw, scale, cfg)
    assert (out[ASSETS].iloc[0] == 0.0).all()


def test_audit_band_is_noop_for_r9():
    """R9 各资产距 V5 带上界 >= 5pp -> 0 次裁剪（§5.4(b)）。"""
    cfg = StrategyConfig()
    idx = pd.date_range("2022-01-01", periods=4, freq="4h", tz="UTC")
    banded = pd.DataFrame({"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35},
                          index=idx)
    out, audit = apply_audit_band(banded, cfg)
    assert audit.n_clipped == 0
    assert audit.n_out_of_band == 0
    assert audit.is_noop


def test_audit_band_clips_above_upper():
    cfg = StrategyConfig()
    idx = pd.date_range("2022-01-01", periods=1, freq="4h", tz="UTC")
    banded = pd.DataFrame({"BTC": 0.50, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35},
                          index=idx)
    out, audit = apply_audit_band(banded, cfg)
    assert out["BTC"].iloc[0] == pytest.approx(0.30), "clipped to V5 upper 30%"
    assert audit.n_clipped == 1


# --- 归一化与恒等式 --------------------------------------------------------
def test_normalize_scales_and_sets_cash():
    idx = pd.date_range("2022-01-01", periods=2, freq="4h", tz="UTC")
    df = pd.DataFrame({"BTC": 0.6, "ETH": 0.3, "SOL": 0.3, "BNB": 0.3}, index=idx)
    out = normalize(df)
    assert out[ASSETS].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert out["cash"].iloc[0] == pytest.approx(0.0)


def test_normalize_preserves_relative_ratios():
    """同档位下 R9 的相对比例不被缩放改变。"""
    cfg = StrategyConfig()
    idx = pd.date_range("2022-01-01", periods=1, freq="4h", tz="UTC")
    # 全资产 scale = 1.0 -> 目标应等于参考权重
    df = pd.DataFrame({a: cfg.reference_weights[a] for a in ASSETS}, index=idx)
    out = normalize(df)
    for a in ASSETS:
        assert out[a].iloc[0] == pytest.approx(cfg.reference_weights[a])


# --- §13.2 缺口 H：compute_ma200_sign / compute_tsmom6m --------------------
def test_compute_ma200_sign_matches_low_level():
    """§13.2 `compute_ma200_sign` 与 ma / ma_sign 逐位一致。"""
    idx = pd.date_range("2022-01-01", periods=260, freq="4h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 300.0, len(idx)), index=idx)
    out = compute_ma200_sign(close, length=200)
    assert list(out.columns) == ["ma200", "ma_sign"]
    pd.testing.assert_series_equal(out["ma200"], ma(close, 200), check_names=False)
    pd.testing.assert_series_equal(out["ma_sign"], ma_sign(close, 200), check_names=False)


def test_compute_tsmom6m_matches_low_level():
    """§13.2 `compute_tsmom6m` 与 tsmom_daily 逐位一致（日线轴）。"""
    idx = pd.date_range("2022-01-01", periods=300, freq="1D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 400.0, len(idx)), index=idx)
    out = compute_tsmom6m(close, days=126, tanh_scale=0.5)
    assert list(out.columns) == ["tsmom6m"]
    pd.testing.assert_series_equal(out["tsmom6m"], tsmom_daily(close, 126, 0.5),
                                   check_names=False)


# --- 冻结口径一致性（Roadmap V2 §20 验收标准 1 / Milestone 2）----------------
REPO = Path(__file__).resolve().parents[1]
V3_DATA = REPO / "data" / "v3_spot"

pytestmark_frozen = pytest.mark.skipif(
    not (V3_DATA / "BTC_4h.csv").exists(),
    reason="frozen V3 dataset (data/v3_spot) not available")


def _write_paper_inputs(tmp: Path) -> dict:
    """把 v3_spot 的 4H/1D 转成纸面系统输入格式。"""
    rows4h, rows1d = [], []
    for a in ASSETS:
        for src, bucket in ((f"{a}_4h.csv", rows4h), (f"{a}_1d.csv", rows1d)):
            df = pd.read_csv(V3_DATA / src, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["asset"] = a
            bucket.append(df[["timestamp", "asset", "open", "high",
                              "low", "close", "volume"]])
    tmp.mkdir(parents=True, exist_ok=True)
    p4h, p1d = tmp / "4h.csv", tmp / "1d.csv"
    pd.concat(rows4h, ignore_index=True).sort_values(
        ["timestamp", "asset"]).to_csv(p4h, index=False)
    pd.concat(rows1d, ignore_index=True).sort_values(
        ["timestamp", "asset"]).to_csv(p1d, index=False)
    import shutil
    pcash = tmp / "cash_rate_dff.csv"
    shutil.copy(V3_DATA / "cash_rate_dff.csv", pcash)
    return {"4h": p4h, "1d": p1d, "cash": pcash}


def _frozen_v5_targets():
    """用 V5 冻结模块生成 S4 + R9 oracle 目标权重。"""
    sys.path.insert(0, str(REPO))
    from scripts.v5_spot import data as V5D, strategies as V5S
    from scripts.v5_spot.config import V5Cfg

    cfg = V5Cfg(ref="R9", cadence="daily", rebal_threshold=0.10,
                threshold_basis="absolute", cash_return="Cash_Rate",
                listing_rule="L1", fee=0.0010, slippage=0.0002)
    fp = V5D.get_panel(start="2018-06-01")
    tv, spec = V5S.build(fp, cfg, "S4")
    return tv, spec


@pytestmark_frozen
def test_s4_r9_matches_frozen_v5_bit_for_bit(tmp_path):
    """Milestone 2 验收：目标权重与 V5 冻结口径逐位一致。"""
    p = _write_paper_inputs(tmp_path / "market")
    panel = load_market_data(p["4h"])
    daily = load_daily_close(p["1d"])
    cfg = StrategyConfig()
    targets, ind, direction, audit = build_targets(
        panel.close, daily, cfg, listed=panel.close.notna())

    start = pd.Timestamp("2018-06-01", tz="UTC")
    mine = targets[targets.index >= start]
    mine_bars = daily_decision_bars(mine.index)

    tv5, spec = _frozen_v5_targets()
    assert spec.mode == "threshold", "S4 frozen spec must be pure threshold"
    assert spec.threshold == pytest.approx(0.10)
    assert spec.freq is None, "no calendar leg"

    common = tv5.index.intersection(mine_bars)
    assert len(common) > 3000, f"expected ~3034 decision bars, got {len(common)}"

    for a in ASSETS:
        m = mine.loc[common, f"target_{a}"].to_numpy(dtype=float)
        t = tv5.loc[common, a].to_numpy(dtype=float)
        assert np.abs(m - t).max() == pytest.approx(0.0, abs=1e-12), \
            f"{a} target weights diverge from frozen V5"

    m = mine.loc[common, "cash"].to_numpy(dtype=float)
    t = tv5.loc[common, "cash"].to_numpy(dtype=float)
    assert np.abs(m - t).max() == pytest.approx(0.0, abs=1e-12)


@pytestmark_frozen
def test_s4_r9_target_levels_match_roadmap(tmp_path):
    """§5.4(c) 档位表：BTC/ETH 三档，SOL/BNB 仅 0 与参考权重。"""
    p = _write_paper_inputs(tmp_path / "market")
    panel = load_market_data(p["4h"])
    daily = load_daily_close(p["1d"])
    targets, _, _, _ = build_targets(
        panel.close, daily, StrategyConfig(), listed=panel.close.notna())
    start = pd.Timestamp("2018-06-01", tz="UTC")
    mine = targets[targets.index >= start]
    bars = daily_decision_bars(mine.index)

    def levels(a):
        return sorted(set(np.round(mine.loc[bars, f"target_{a}"].to_numpy(), 6)))

    assert levels("BTC") == pytest.approx([0.0, 0.125, 0.1875, 0.25])
    assert levels("ETH") == pytest.approx([0.0, 0.05, 0.075, 0.10])
    assert levels("SOL") == pytest.approx([0.0, 0.30])
    assert levels("BNB") == pytest.approx([0.0, 0.35])


@pytestmark_frozen
def test_audit_band_is_zero_clip_on_full_history(tmp_path):
    """§5.4(b)：V5 审计带在本组合全历史上 0 次裁剪。"""
    p = _write_paper_inputs(tmp_path / "market")
    panel = load_market_data(p["4h"])
    daily = load_daily_close(p["1d"])
    _, _, _, audit = build_targets(
        panel.close, daily, StrategyConfig(), listed=panel.close.notna())
    assert audit.n_clipped == 0
    assert audit.n_out_of_band == 0
    assert audit.is_noop


# --- 配置校验 --------------------------------------------------------------
def test_config_validate_rejects_bad_reference_weights():
    cfg = StrategyConfig()
    cfg.reference_weights = {"BTC": 0.9, "ETH": 0.05, "SOL": 0.03, "BNB": 0.02}
    with pytest.raises(ConfigError, match="outside audit band|only"):
        cfg.validate()


def test_config_validate_rejects_paper_only_false():
    cfg = StrategyConfig()
    cfg.paper_only = False
    with pytest.raises(ConfigError, match="paper_only"):
        cfg.validate()


def test_config_validate_rejects_exchange_enabled():
    cfg = StrategyConfig()
    cfg.exchange_trading_enabled = True
    with pytest.raises(ConfigError, match="exchange_trading_enabled"):
        cfg.validate()


def test_config_validate_rejects_calendar_rebalance():
    cfg = StrategyConfig()
    cfg.rebalance_mode = "calendar"
    with pytest.raises(ConfigError, match="threshold"):
        cfg.validate()


def test_config_validate_rejects_out_of_order_buckets():
    cfg = StrategyConfig()
    cfg.bucket_thresholds = [0.50, 0.25, 0.75]
    with pytest.raises(ConfigError, match="ascending"):
        cfg.validate()


def test_config_validate_rejects_bad_band():
    cfg = StrategyConfig()
    cfg.signal_band = {**cfg.signal_band, "BTC": [0.30, 0.10]}
    with pytest.raises(ConfigError, match=r"lo < hi"):
        cfg.validate()


def test_indicator_panel_shape():
    idx = pd.date_range("2022-01-01", periods=6 * 60, freq="4h", tz="UTC")
    close = pd.DataFrame({a: np.linspace(100, 150, len(idx)) for a in ASSETS}, index=idx)
    daily = close.resample("1D").last().dropna()
    from openclaw_s4_paper.indicators import build_indicator_panel
    ind = build_indicator_panel(close, daily)
    for a in ASSETS:
        assert set(["close", "ma200", "ma_sign", "tsmom6m"]) <= set(ind[a].columns)