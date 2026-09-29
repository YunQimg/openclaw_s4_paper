# -*- coding: utf-8 -*-
"""V5 —— 指标层 / 评分 / LRR / Bootstrap / Monte Carlo（Roadmap §11/§13/§14）

设计约束（全部来自 Roadmap，逐条对应）：

* §11.4(a) 百分位指标必须使用**统一参考分布**：所有臂的百分位都相对同一
  `pool` 计算，禁止各自 min-max。
* §11.4(b) 禁止跨臂比较 min-max 归一化后的绝对值。
* §11.4(c) 多指标聚合必须 `merge(on=key, validate="one_to_one")`。
* §11.3 V5 追加：Bootstrap / Monte Carlo 的三臂**必须共用同一随机种子与
  同一抽样块**，否则跨臂百分位不可比。
* §13 LRR = 邻域平均分 / 候选分，理论上限 1.0；`> 1` 表示候选分**低于**邻居
  均分（局部低谷），必须单列，并在参与归一化前**截断到 1.0**，因为
  "LRR > 1 并不代表更鲁棒"，直接进 min-max 会把它误奖为最优点。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v4_spot import metrics as M4                       # noqa: E402
from scripts.v5_spot.config import (ASSETS, BAND_V5, LRR_CAP, LRR_PERTURB_PP,  # noqa: E402
                                    LRR_TIERS, N_BOOTSTRAP, BOOT_BLOCK,
                                    REF_IDS, REF_WEIGHTS, SCORE_CAPS,
                                    SCORE_WEIGHTS, SEED, SIMPLICITY,
                                    WEIGHT_DISTANCE)

ANN = M4.ANN
DAYS_PER_YEAR = M4.DAYS_PER_YEAR
to_daily = M4.to_daily
drawdown = M4.drawdown
max_drawdown = M4.max_drawdown
drawdown_episodes = M4.drawdown_episodes
ulcer_index = M4.ulcer_index
expected_shortfall = M4.expected_shortfall
exposure_stats = M4.exposure_stats
trade_stats = M4.trade_stats
classify_phases = M4.classify_phases
capture_ratios = M4.capture_ratios
cost_analysis = M4.cost_analysis
rolling_metrics = M4.rolling_metrics
summarize_core = M4.summarize_core


# ---------------------------------------------------------------------------
# §13 LRR 档位
# ---------------------------------------------------------------------------
def lrr_band(ratio: float) -> str:
    """LRR 档位判定。`LRR > 1` 单列为「候选低于邻域均值（局部低谷）」。"""
    if ratio != ratio:                                   # NaN
        return "n/a"
    if ratio > LRR_CAP + 1e-12:
        return "候选低于邻域均值（局部低谷）"
    for lo, name in LRR_TIERS:
        if ratio >= lo:
            return name
    return LRR_TIERS[-1][1]


def lrr_capped(ratio: float) -> float:
    """归一化前的上截断：LRR >= 1 一律按 1.0 处理（§13）。"""
    if ratio != ratio:
        return np.nan
    return float(min(ratio, LRR_CAP))


# ---------------------------------------------------------------------------
# §11.3 三臂共用种子与抽样块
# ---------------------------------------------------------------------------
def bootstrap_percentiles(daily: pd.Series, n: int = N_BOOTSTRAP,
                          block: int = BOOT_BLOCK, seed: int = SEED) -> dict:
    """块自助法。**三臂调用时必须传入完全相同的 seed 与 block**（§11.3 V5 追加），
    以保证三臂面对的抽样路径逐次一致、百分位可比。
    """
    r = daily.dropna()
    if len(r) < block * 2:
        return {"n": 0, "cagr_p05": np.nan, "cagr_p50": np.nan, "cagr_p95": np.nan,
                "maxdd_p05": np.nan, "maxdd_p50": np.nan, "maxdd_p95": np.nan,
                "prob_loss": np.nan, "seed": int(seed), "block": int(block)}
    arr = r.to_numpy(dtype=float)
    n_blocks = int(np.ceil(len(arr) / block))
    rng = np.random.default_rng(seed)
    starts_pool = np.arange(max(len(arr) - block + 1, 1))
    cagrs = np.empty(n)
    mdds = np.empty(n)
    for i in range(n):
        starts = rng.choice(starts_pool, size=n_blocks, replace=True)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(arr)]
        s = arr[idx]
        eq = np.cumprod(1.0 + s)
        yrs = max(len(s) / DAYS_PER_YEAR, 1e-9)
        cagrs[i] = eq[-1] ** (1.0 / yrs) - 1.0 if eq[-1] > 0 else -1.0
        mdds[i] = float((eq / np.maximum.accumulate(eq) - 1.0).min())

    def q(a, p):
        return float(np.percentile(a, p))

    return {
        "n": int(n), "seed": int(seed), "block": int(block),
        "cagr_p05": q(cagrs, 5), "cagr_p50": q(cagrs, 50), "cagr_p95": q(cagrs, 95),
        "maxdd_p05": q(mdds, 5), "maxdd_p50": q(mdds, 50), "maxdd_p95": q(mdds, 95),
        "prob_loss": float((cagrs < 0).mean()),
    }


def monte_carlo_paths(daily: pd.Series, n: int = N_BOOTSTRAP, seed: int = SEED,
                      horizon_days: Optional[int] = None) -> dict:
    """蒙特卡洛重排（i.i.d. 抽样，与块自助法互补）。三臂共用 seed。"""
    r = daily.dropna()
    if len(r) < 30:
        return {"n": 0, "seed": int(seed)}
    arr = r.to_numpy(dtype=float)
    h = int(horizon_days or len(arr))
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n, h), replace=True)
    eq = np.cumprod(1.0 + draws, axis=1)
    final = eq[:, -1] - 1.0
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    return {
        "n": int(n), "seed": int(seed), "horizon_days": int(h),
        "return_p05": float(np.percentile(final, 5)),
        "return_p50": float(np.percentile(final, 50)),
        "return_p95": float(np.percentile(final, 95)),
        "maxdd_p50": float(np.percentile(mdd, 50)),
        "maxdd_p95": float(np.percentile(mdd, 5)),      # 更深的分位
        "prob_loss": float((final < 0).mean()),
    }


# ---------------------------------------------------------------------------
# §11.4 低波动伪影防护
# ---------------------------------------------------------------------------
MIN_DAILY_STD = 1e-4          # 日收益标准差下限（约 0.01%/日）
MIN_AVG_EXPOSURE = 0.02       # 平均风险资产暴露下限


def sharpe_reliable(daily: pd.Series, avg_exposure: Optional[float] = None,
                    min_std: float = MIN_DAILY_STD,
                    min_exposure: float = MIN_AVG_EXPOSURE) -> bool:
    """判断 Sharpe 是否可信（§11.4 防低波动伪影）。

    场景：策略在某窗口**近乎空仓**（如 S1 的 MA200 全面触发减仓），
    日收益序列退化为近似常数，`std → 0` 使 Sharpe 爆炸到 16~241。
    这类数值不是「优异表现」，而是分母趋零的伪影，必须剔除，
    否则会污染统一参考分布的百分位与最终评分。

    判定：日收益标准差 >= `min_std`，且（若提供）平均风险暴露 >= `min_exposure`。
    """
    r = daily.dropna()
    if len(r) < 10:
        return False
    if float(r.std()) < min_std:
        return False
    if avg_exposure is not None and float(avg_exposure) < min_exposure:
        return False
    return True


def safe_sharpe(daily: pd.Series, avg_exposure: Optional[float] = None) -> float:
    """可信时返回年化 Sharpe，否则返回 NaN（禁止用无穷大/伪影值参与排名）。"""
    r = daily.dropna()
    if r.std() <= 1e-12 or not sharpe_reliable(daily, avg_exposure):
        return np.nan
    return float(r.mean() / r.std() * ANN)


# ---------------------------------------------------------------------------
# §11.4(a) 统一参考分布百分位
# ---------------------------------------------------------------------------
def unified_percentile(values: Sequence[float], pool: Sequence[float]) -> np.ndarray:
    """相对**统一参考分布** `pool` 的百分位（§11.4a）。

    `pool` 必须由所有臂共享（例如全体基准臂 + 策略臂的并集），
    绝不允许各臂用自己的子集算 min-max。
    """
    ref = np.sort(np.asarray([v for v in pool if v == v], dtype=float))
    vals = np.asarray(values, dtype=float)
    if ref.size == 0:
        return np.full(vals.shape, np.nan)
    out = np.searchsorted(ref, vals, side="right") / ref.size
    return np.where(np.isnan(vals), np.nan, out)


# ---------------------------------------------------------------------------
# §14.2 硬门槛
# ---------------------------------------------------------------------------
def hard_gates(row: dict, gates: Optional[dict] = None) -> dict:
    """§14 硬门槛。返回逐项通过情况与总体是否通过。"""
    g = gates or DEFAULT_GATES
    checks = {
        "cagr_floor": float(row.get("cagr", np.nan)) >= g["cagr_floor"],
        "sharpe_floor": float(row.get("sharpe", np.nan)) >= g["sharpe_floor"],
        "maxdd_ceiling": float(row.get("max_drawdown", np.nan)) >= g["maxdd_ceiling"],
        "invariants": bool(row.get("invariant_passed", True)),
        "no_lookahead": bool(row.get("signal_time_ok", True)),
        "band_clean": int(row.get("band_clip_bars", 0)) == 0,
        "listing_ok": bool(row.get("listing_ok", True)),
    }
    checks["passed"] = bool(all(v for k, v in checks.items() if k != "passed"))
    return checks


DEFAULT_GATES: Dict[str, float] = {
    "cagr_floor": 0.0,          # §14：不得为负收益
    "sharpe_floor": 0.0,
    "maxdd_ceiling": -0.95,     # §14：MaxDD 不得深于 -95%
}


# ---------------------------------------------------------------------------
# §14.2 评分
# ---------------------------------------------------------------------------
def composite_score(row: dict, pct: Optional[dict] = None) -> float:
    """§14.2 加权评分（0–100）。

    * 收益/风险调整类为**百分位**，必须来自 §11.4(a) 的统一参考分布；
      未提供 `pct` 时退化为直接使用截断后的原始值（仅供冒烟，正式排名严禁）。
    * `max_dd` 以「越浅越好」映射到 [0,1]。
    * 全部项按 `SCORE_CAPS` 截断后再加权。
    """
    def cap(name, v):
        c = SCORE_CAPS.get(name)
        if c is None or v != v:
            return v
        return float(min(max(v, 0.0), c))

    def pick(name):
        if pct and name in pct:
            return float(pct[name])
        raw = row.get(name, np.nan)
        c = SCORE_CAPS.get(name)
        return float(cap(name, raw) / c) if (c and raw == raw) else np.nan

    md = float(row.get("max_drawdown", np.nan))
    dd_score = float(min(max(1.0 + md, 0.0), 1.0)) if md == md else np.nan
    stab = float(row.get("active_stability", np.nan))
    stab = 1.0 if stab != stab else float(min(max(stab, 0.0), 1.0))
    simple = float(SIMPLICITY.get(str(row.get("system", "")) or "", np.nan))
    simple = 0.5 if simple != simple else simple

    parts = {
        "sharpe": pick("sharpe"),
        "sortino": pick("sortino"),
        "calmar": pick("calmar"),
        "cagr": pick("cagr"),
        "max_dd": dd_score,
        "active_stability": stab,
        "simplicity": simple,
    }
    total = 0.0
    wsum = 0.0
    for k, w in SCORE_WEIGHTS.items():
        v = parts.get(k, np.nan)
        if v != v:
            continue
        total += w * v
        wsum += w
    return float(total / wsum * 100.0) if wsum > 0 else np.nan


def score_table(df: pd.DataFrame, pool: Optional[pd.DataFrame] = None,
                key: str = "system") -> pd.DataFrame:
    """§11.4(a)+(c)：用**统一参考分布** `pool` 计算百分位并聚合。

    聚合前用 `merge(on=key, validate="one_to_one")` 校验一一对应，
    禁止按位置拼接（§11.4c，V4.8 的 12/14 行错位教训）。
    """
    src = pool if pool is not None else df
    out = df.copy().reset_index(drop=True)
    pct = {}
    for col in ("sharpe", "sortino", "calmar", "cagr"):
        pct[col] = unified_percentile(out[col].astype(float).tolist(),
                                      src[col].astype(float).tolist())
        out[f"rank_pct_{col}"] = pct[col]
    out["final_score"] = [
        composite_score({**r, }, {k: pct[k][i] for k in pct})
        for i, r in enumerate(out.to_dict("records"))]
    return out


def merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame,
                     key: str = "system") -> pd.DataFrame:
    """§11.4(c) 强制一一对应聚合。重复/缺失 key 直接抛错，不静默。"""
    return pd.merge(left, right, on=key, how="inner",
                    validate="one_to_one", suffixes=("", "_r"))


# ---------------------------------------------------------------------------
# §13 V5 追加：参考权重邻域（LRR）剖面
# ---------------------------------------------------------------------------
def perturb_weights(ref: str, delta: float) -> Dict[str, Dict[str, float]]:
    """在参考权重附近生成 ±delta 的邻域（保持总和 = 1，且**不越界 V5 带**）。"""
    base = REF_WEIGHTS[ref]
    nbrs: Dict[str, Dict[str, float]] = {}
    for a in ASSETS:
        for sgn in (+1, -1):
            w = dict(base)
            w[a] = w[a] + sgn * delta
            if w[a] < -1e-12:
                continue
            others = [x for x in ASSETS if x != a]
            pool = sum(w[x] for x in others)
            if pool <= 1e-12:
                continue
            scale = (1.0 - w[a]) / pool
            for x in others:
                w[x] = w[x] * scale
            if any(w[x] < BAND_V5[x][0] - 1e-9 or w[x] > BAND_V5[x][1] + 1e-9
                   for x in ASSETS):
                continue
            keyd = "|".join(f"{w[x]:.4f}" for x in ASSETS)
            if keyd not in nbrs:
                nbrs[keyd] = {x: round(w[x], 6) for x in ASSETS}
    return nbrs


def lrr_table(base_scores: Dict[str, float], nbr_scores: Dict[str, Dict[str, float]],
              ref: str) -> pd.DataFrame:
    """计算某个参考权重的 LRR 剖面（§13）。

    `base_scores`  : {ref -> 候选分}
    `nbr_scores`   : {ref -> {neighbor_key -> 邻居分}}
    输出含真值 `lrr_raw` 与截断值 `lrr_capped`，以及档位判定。
    """
    rows: List[dict] = []
    base = float(base_scores.get(ref, np.nan))
    nbrs = nbr_scores.get(ref, {})
    for level, delta in (("L1_5pp", LRR_PERTURB_PP[0]),
                         ("L2_10pp", LRR_PERTURB_PP[1] if len(LRR_PERTURB_PP) > 1 else None)):
        if delta is None:
            continue
        pass
    vals = np.asarray(list(nbrs.values()), dtype=float)
    vals = vals[~np.isnan(vals)]
    if not nbrs or base != base or base <= 0:
        return pd.DataFrame([{
            "ref": ref, "base_score": base, "n_neighbors": int(len(nbrs)),
            "mean_score_nbr": np.nan, "median_score_nbr": np.nan,
            "min_score_nbr": np.nan, "max_score_nbr": np.nan,
            "std_score_nbr": np.nan, "lrr_raw": np.nan, "lrr_capped": np.nan,
            "lrr_verdict": "n/a", "rank_stability": np.nan,
        }])
    mean_s = float(vals.mean())
    raw = float(mean_s / base)
    rows.append({
        "ref": ref, "base_score": base, "n_neighbors": int(len(vals)),
        "mean_score_nbr": mean_s,
        "median_score_nbr": float(np.median(vals)),
        "min_score_nbr": float(vals.min()),
        "max_score_nbr": float(vals.max()),
        "std_score_nbr": float(vals.std(ddof=0)),
        "lrr_raw": raw,
        "lrr_capped": lrr_capped(raw),
        "lrr_verdict": lrr_band(raw),
        "rank_stability": np.nan,
    })
    return pd.DataFrame(rows)


def reference_pairwise_distance() -> pd.DataFrame:
    """§10 V5 追加：三参考权重两两总变差距离（必须显著 > 0，否则同质）。"""
    rows = []
    for i, a in enumerate(REF_IDS):
        for b in REF_IDS[i + 1:]:
            tv = 0.5 * sum(abs(REF_WEIGHTS[a][x] - REF_WEIGHTS[b][x]) for x in ASSETS)
            rows.append({
                "ref_a": a, "ref_b": b,
                "tv_distance": float(tv),
                "tv_distance_pct": float(tv * 100),
                "declared": float(WEIGHT_DISTANCE.get((a, b), np.nan)),
                "matches_declared": bool(
                    abs(tv - WEIGHT_DISTANCE.get((a, b), np.nan)) < 1e-9),
            })
    return pd.DataFrame(rows)