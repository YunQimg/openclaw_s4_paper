# -*- coding: utf-8 -*-
"""V5 —— 统一组合执行引擎（Roadmap §2/§3/§5/§6/§19）

执行语义**完全继承** V4 `scripts.v4_spot.engine.simulate`（同一成本逐腿计提、
同一现金复利、同一上市进入、同一 signal→t+1 执行、同一卖出先于买入的现金约束），
仅追加 V5 需要的**配置带事件审计**（Roadmap §2.2.2 / §2.3 / §19.2）：

* `band_events`：显式记录目标权重被带上限/下界裁剪的每一根 bar；
* 若在 V5-Band 下出现任何裁剪，说明 §2.2.2 的带宽计算有误，V5 必须停止后续排名
  （Roadmap §5.3 追加验收）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v4_spot import engine as V4E                     # noqa: E402
from scripts.v4_spot.config import RebalanceSpec              # noqa: E402
from scripts.v5_spot.config import ASSETS, V5Cfg              # noqa: E402

EPS = 1e-9
period_boundary = V4E.period_boundary


def _to_v4_cfg(cfg: V5Cfg):
    """把 V5Cfg 投影到 V4Cfg（执行层唯一入口），保证口径零漂移。"""
    from scripts.v4_spot.config import V4Cfg
    return V4Cfg(
        fee=cfg.fee, slippage=cfg.slippage, cash_return=cfg.cash_return,
        listing_rule=cfg.listing_rule, band_mode=cfg.band_mode,
        band_override=cfg.band,
        ref_weight=cfg.ref_weight, initial=cfg.initial,
        exec_lag_bars=cfg.exec_lag_bars, cadence=cfg.cadence,
        rebal_threshold=cfg.rebal_threshold, threshold_basis=cfg.threshold_basis,
        ma_len=cfg.ma_len, tsmom_days=cfg.tsmom_days, tsmom_days2=cfg.tsmom_days2,
        risk_multiplier=cfg.risk_multiplier, breadth_floor=cfg.breadth_floor,
    )


def band_audit(targets: pd.DataFrame, band: Dict[str, tuple], ref: str,
               group: str, tradable: pd.DataFrame,
               ref_weight: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """逐 bar 检查目标权重与配置带的关系（Roadmap §2.2.2 / §2.3 / §19.2）。

    **审计口径必须与 `strategies._apply_band` 的约束口径完全一致**，
    否则会把策略的合法减仓误判为越界（实测 S3 曾因此产生 5501 个假事件）。

    下界语义（关键）
    ----------------
    配置带约束的是「相对参考权重的**主动偏离**」，不是禁止资产离场。
    减仓（`w < w_ref`）是降低风险，**不受带下界约束**；只有当参考权重本身
    低于带下界时（V4 带下 R13 的 ETH/BNB、R9 的 BTC/SOL/BNB），下界才生效。

    因此下界越界判定为：

        ``below_lo = (w < lo - eps) AND (w_ref >= lo) AND (w >= w_ref)``

    即：仅在「参考权重在带内」（下界本不该约束）且「目标不低于参考权重」
    （属增持方向）时，低于下界才算异常。

    情形分类（§2.3「三组命名不得混用」）
    -----------------------------------
    * ``violation``（严格越界）：``w > hi``，或上述 ``below_lo``。
      这是真正被带约束掉、需要留痕的事件。
    * ``on_boundary``（恰好压界）：``|w - hi| <= eps`` 或 ``|w - lo| <= eps``，
      权重落在边界上、未越界但无缓冲。§19.2 要求计入 ``n_bars_out_of_band``。
    * 带内：两者皆无。

    ``n_bars_clipped`` 只统计 violation；``n_bars_out_of_band`` = violation +
    on_boundary（§19.2 验收口径）。V5-Band 网关用前者。

    只有「已上市」资产的权重才参与判断：未上市资产的权重会被引擎的
    listing plan 归零并留在 Cash Reserve，不属于配置带语义。
    """
    idx = targets.index
    base = targets.reindex(idx)[ASSETS].ffill().fillna(0.0)
    trad = tradable.reindex(idx).fillna(False).to_numpy(dtype=bool)
    rw = ref_weight or {}
    rows: List[dict] = []
    for j, a in enumerate(ASSETS):
        w = base[a].to_numpy(dtype=float)
        ok = trad[:, j]
        lo, hi = band[a]
        w_ref = float(rw.get(a, np.nan))
        # 严格越界：上界无条件生效
        up = ok & (w > hi + EPS)
        # 下界：仅在「参考权重在带内」且「属增持方向」时生效
        if w_ref == w_ref and w_ref >= lo - EPS:
            dn = ok & (w < lo - EPS) & (w >= w_ref - EPS)
        else:
            # 参考权重本身低于带下界（§2.3 裁剪情形）：低于下界即为事件
            dn = ok & (w < lo - EPS)
        # 恰好压界 = 未越界但无缓冲
        touch_hi = ok & ~up & (np.abs(w - hi) <= EPS)
        touch_lo = ok & ~dn & (np.abs(w - lo) <= EPS)
        n_up, n_dn = int(up.sum()), int(dn.sum())
        n_th, n_tl = int(touch_hi.sum()), int(touch_lo.sum())
        n_clip = n_up + n_dn
        n_oob = n_clip + n_th + n_tl
        if n_oob == 0:
            continue
        ev = up | dn | touch_hi | touch_lo
        rows.append({
            "ref": ref, "band_group": group, "asset": a,
            "weight_when_clipped": float(w[ev][0]),
            "weight_when_out_of_band": float(w[ev][0]),
            "ref_weight": w_ref,
            "band_lo": lo, "band_hi": hi,
            "n_bars_above_hi": n_up, "n_bars_below_lo": n_dn,
            "n_bars_clipped": n_clip,
            "n_bars_on_hi": n_th, "n_bars_on_lo": n_tl,
            "n_bars_on_boundary": n_th + n_tl,
            "n_bars_out_of_band": n_oob,
            "max_excess_above_hi": float((w[up] - hi).max()) if n_up else 0.0,
            "max_deficit_below_lo": float((lo - w[dn]).max()) if n_dn else 0.0,
            "first_clip_bar": idx[ev][0],
        })
    return pd.DataFrame(rows)


def simulate(fp: dict, targets: pd.DataFrame, cfg: V5Cfg, spec: RebalanceSpec,
             cash_rate: Optional[pd.Series] = None, label: str = "") -> dict:
    """V5 事件驱动组合回测 = V4 执行 + 配置带事件审计。"""
    v4 = _to_v4_cfg(cfg)
    res = V4E.simulate(fp, targets, v4, spec, cash_rate=cash_rate, label=label)
    ba = band_audit(targets, cfg.band, cfg.ref, cfg.band_group, fp["tradable"],
                    ref_weight=cfg.ref_weight)
    res["band_events"] = ba
    res["band_clip_bars"] = int(ba["n_bars_clipped"].sum()) if len(ba) else 0
    res["ref"] = cfg.ref
    # 组合恒等式（§2.1）：每根 bar sum(asset)+cash = 1
    w = res["weights"]
    tot = w[ASSETS].sum(axis=1) + w["cash"]
    res["invariant_max_err"] = float((tot - 1.0).abs().max()) if len(tot) else 0.0
    res["invariant_min_asset"] = float(w[ASSETS].min().min()) if len(w) else 0.0
    res["invariant_min_cash"] = float(w["cash"].min()) if len(w) else 0.0
    res["invariant_passed"] = bool(
        res["invariant_max_err"] <= 1e-6
        and res["invariant_min_asset"] >= -1e-6
        and res["invariant_min_cash"] >= -1e-6)
    # §19.6 信号时序：signal_time <= execution_time，且 lag>0 信号必须严格早于执行。
    #
    # V4 引擎在 index[0] 的「首次建仓」上令 signal_time == execution_time
    # （engine.py 中 `sig_t = idx[i - lag] if (lag > 0 and i - lag >= 0 and not first)
    # else idx[i]`）。首根 bar 之前不存在任何信息，t 与 t+1 无从区分，属**定义上
    # 的例外**，不是前视偏差；除此之外每一腿都必须严格 signal < execution。
    tr = res.get("trades", pd.DataFrame())
    if len(tr):
        sig = pd.to_datetime(tr["signal_time"], utc=True)
        exe = pd.to_datetime(tr["execution_time"], utc=True)
        exec_first = pd.Timestamp(fp["index"][0])
        is_first_bar = exe == exec_first
        strictly_later = bool((sig[~is_first_bar] < exe[~is_first_bar]).all())
        first_bar_ok = bool((sig[is_first_bar] == exe[is_first_bar]).all())
        res["signal_time_ok"] = bool(strictly_later and first_bar_ok)
        res["signal_first_bar_exempt"] = int(is_first_bar.sum())
        res["signal_strict_violations"] = int((sig[~is_first_bar] >= exe[~is_first_bar]).sum())
        res["n_distinct_signal_times"] = int(sig.nunique())
    else:
        res["signal_time_ok"] = True
        res["signal_first_bar_exempt"] = 0
        res["signal_strict_violations"] = 0
        res["n_distinct_signal_times"] = 0
    res["has_price_time"] = bool("price_time" in tr.columns) if len(tr) else True
    return res


def band_stop_check(band_events: pd.DataFrame, band_group: str) -> dict:
    """V5 关键闸门（§5.3 追加 / §19.2）：V5-Band 下不得出现任何裁剪。"""
    total = int(band_events["n_bars_clipped"].sum()) if len(band_events) else 0
    return {
        "band_group": band_group,
        "clip_events": total,
        "halt_required": bool(band_group == "V5-Band" and total > 0),
    }