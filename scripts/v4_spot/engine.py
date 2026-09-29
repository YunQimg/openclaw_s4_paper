# -*- coding: utf-8 -*-
"""V4 现货多资产系统 —— 统一组合执行引擎（Roadmap §2/§3/§5/§6/§19）

设计原则
--------
* 基准与策略共用完全相同的执行、成本、现金与上市口径（§6 公平性）。
* 现货约束：无杠杆、无做空、`sum(asset) + cash = 1`。
* 成本逐腿计提，买入腿 fee/slippage 与卖出腿分别记录（§6）。
* 上市前不可交易资产的权重留在 Cash Reserve；上市后按 L1/L2/L3 规则进入（§5.1）。
* 信号在 bar t 生成，默认于 t+1 执行（§3.3），交易日志保留三种时间。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.v4_spot.config import ASSETS, LISTING_L2_DAYS, RebalanceSpec, V4Cfg

EPS = 1e-9
MONTHS = {"M": 1, "Q": 3, "2Q": 6, "Y": 12}
REASON_LABEL = {"initial": "initial", "calendar": "calendar", "threshold": "threshold",
                "hybrid": "calendar+threshold", "listing": "listing_entry"}


def period_boundary(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """日历周期边界布尔数组（首根 bar 为 True）。

    支持频率：D / W / 2W / M / Q / 2Q / Y。
    * D   —— 每个 bar 都是边界（除首根外的每一根都触发）
    * W   —— 周（周一为一周起点，按 1970-01-05 起的周序号划分）
    * 2W  —— 双周（周序号 // 2）
    原有 M / Q / 2Q / Y 分支未改动，行为与 V4/V4.5 完全一致（向后兼容）。
    """
    if freq == "D":
        b = np.zeros(len(idx), dtype=bool)
        if len(idx):
            b[0] = True
            b[1:] = True
        return b
    if freq in ("W", "2W"):
        days = np.asarray((idx.normalize()
                           - pd.Timestamp("1970-01-05", tz="UTC")).days, dtype=np.int64)
        wk = days // 7
        if freq == "2W":
            wk = wk // 2
        b = np.zeros(len(idx), dtype=bool)
        if len(idx):
            b[0] = True
            b[1:] = wk[1:] != wk[:-1]
        return b
    span = MONTHS[freq]
    ym = idx.year.to_numpy() * 12 + (idx.month.to_numpy() - 1)
    key = ym // span
    b = np.zeros(len(idx), dtype=bool)
    if len(idx):
        b[0] = True
        b[1:] = key[1:] != key[:-1]
    return b


def _listing_plan(fp: dict, cfg: V4Cfg, spec: RebalanceSpec
                  ) -> tuple[np.ndarray, np.ndarray]:
    """返回 (ramp, force_entry)。

    ramp[i, a]      : 该资产在 bar i 允许进入的目标权重比例（L1=1 / L2=线性 / L3=0->1）
    force_entry[i]  : 该 bar 必须执行一次再平衡（用于上市进入，不受 spec.mode 限制）
    """
    idx = fp["index"]
    tradable = fp["tradable"]
    n = len(idx)
    ramp = np.zeros((n, len(ASSETS)))
    force = np.zeros(n, dtype=bool)
    l3_freq = spec.freq if spec.mode in ("calendar", "calendar_thresh") else "Y"
    bnd = period_boundary(idx, l3_freq)

    for j, a in enumerate(ASSETS):
        m = tradable[a].to_numpy(dtype=bool)
        if not m.any():
            continue
        first = int(np.argmax(m))
        ramp[first:, j] = 1.0
        if cfg.listing_rule == "L2":
            for i in range(first, min(first + LISTING_L2_DAYS, n)):
                ramp[i, j] = (i - first + 1) / float(LISTING_L2_DAYS)
            force[first:min(first + LISTING_L2_DAYS, n)] = True
        elif cfg.listing_rule == "L3":
            later = np.where(bnd[first + 1:])[0]
            if len(later):
                b = first + 1 + int(later[0])
                ramp[first:b, j] = 0.0
                force[b] = True
            # 若之后没有日历边界（如 never 模式下的 B0），该资产保持 Cash
        else:                                          # L1
            ramp[first:, j] = 1.0
            force[first] = True
    force[0] = force[0] or True                        # 首根 bar 必须初始建仓
    return ramp, force


def simulate(fp: dict, targets: pd.DataFrame, cfg: V4Cfg, spec: RebalanceSpec,
             cash_rate: Optional[pd.Series] = None, label: str = "") -> dict:
    """事件驱动组合回测。

    targets : 决策 bar -> 目标权重（仅需 ASSETS 列；未上市资产填 0 或 NaN）
    spec    : 再平衡触发规则
    """
    idx = fp["index"]
    P = fp["px"].to_numpy(dtype=float)
    n = len(idx)
    col = {a: j for j, a in enumerate(ASSETS)}
    tradable = fp["tradable"].to_numpy(dtype=bool)

    base = targets.reindex(idx).ffill()[ASSETS].to_numpy(dtype=float)
    base = np.nan_to_num(base, nan=0.0)
    base = np.clip(base, 0.0, None)
    s = base.sum(axis=1, keepdims=True)
    over = (s > 1.0).ravel()
    if over.any():
        base[over] = base[over] / s[over]                   # §2.1 超出部分转 Cash

    lag = max(int(cfg.exec_lag_bars), 0)
    eff = base.copy()
    if lag > 0:
        eff[lag:] = base[:-lag]

    ramp, force = _listing_plan(fp, cfg, spec)
    T = eff * ramp * tradable.astype(float)

    cr = (cash_rate.reindex(idx).fillna(0.0).to_numpy(dtype=float)
          if cash_rate is not None else np.zeros(n))
    do_cash = cfg.cash_return == "Cash_Rate"
    c_fee, c_slip = cfg.fee, cfg.slippage

    bnd = period_boundary(idx, spec.freq) if spec.mode in ("calendar", "calendar_thresh") else None
    use_thr = spec.mode in ("threshold", "calendar_thresh")
    thr = spec.threshold if spec.threshold is not None else 1.0
    relative = spec.basis == "relative"

    shares = np.zeros(len(ASSETS))
    cash = float(cfg.initial)
    trades: List[dict] = []
    events: List[dict] = []
    eq = np.zeros(n)
    w_hist = np.zeros((n, len(ASSETS)))
    cash_w = np.zeros(n)
    tgt_hist = np.zeros((n, len(ASSETS)))
    turnover_total = 0.0
    cash_interest_total = 0.0
    cash_constrained_bars = 0

    def value(i: int, cash_v: float, sh: np.ndarray) -> float:
        row = P[i]
        v = cash_v
        for j in range(len(ASSETS)):
            q = sh[j]
            if q and not np.isnan(row[j]):
                v += q * row[j]
        return v

    def do_rebalance(i: int, tv: np.ndarray, reason: str, first: bool) -> None:
        nonlocal cash, turnover_total, cash_constrained_bars
        row = P[i]
        pv = value(i, cash, shares)
        if pv <= 0:
            return
        w_before = np.zeros(len(ASSETS))
        for j in range(len(ASSETS)):
            if not np.isnan(row[j]):
                w_before[j] = shares[j] * row[j] / pv
        legs: List[dict] = []
        tnotional = 0.0

        # ---- 先卖 ----
        for j in range(len(ASSETS)):
            p = row[j]
            if np.isnan(p) or p <= 0 or not tradable[i, j]:
                continue
            cur_val = shares[j] * p
            tgt_val = tv[j] * pv
            d = tgt_val - cur_val
            if d >= -EPS:
                continue
            qty = -d / p
            exec_p = p * (1.0 - c_slip)
            gross = qty * exec_p
            fee = gross * c_fee
            shares[j] -= qty
            cash += gross - fee
            turnover_total += (-d) / pv
            tnotional += -d
            legs.append({"asset": ASSETS[j], "side": "sell", "qty": qty,
                         "price": exec_p, "ref_price": p, "notional": -d,
                         "fee": fee, "slippage": (-d) * c_slip,
                         "cost_total": fee + (-d) * c_slip, "reason": reason})
        # ---- 后买（现金约束下按比例缩放）----
        plan = []
        need_total = 0.0
        for j in range(len(ASSETS)):
            p = row[j]
            if np.isnan(p) or p <= 0 or not tradable[i, j]:
                continue
            cur_val = shares[j] * p
            d = tv[j] * pv - cur_val
            if d <= EPS:
                continue
            gross = d * (1.0 + c_slip)
            out = gross * (1.0 + c_fee)
            plan.append((j, p, d, gross, out))
            need_total += out
        scale = 1.0
        if need_total > cash:
            scale = cash / need_total if need_total > 0 else 0.0
            cash_constrained_bars += 1
        for j, p, d, gross, out in plan:
            d2 = d * scale
            if d2 <= EPS:
                continue
            exec_p = p * (1.0 + c_slip)
            spend = min(d2 * (1.0 + c_slip) * (1.0 + c_fee), cash)
            if spend <= EPS:
                continue
            g = spend / (1.0 + c_fee)          # 成交名义额（exec_p 口径）
            qty = g / exec_p
            fee = g * c_fee
            not_ref = qty * p                  # 参考价口径名义额
            shares[j] += qty
            cash -= spend
            turnover_total += not_ref / pv
            tnotional += not_ref
            legs.append({"asset": ASSETS[j], "side": "buy", "qty": qty,
                         "price": exec_p, "ref_price": p, "notional": not_ref,
                         "fee": fee, "slippage": not_ref * c_slip,
                         "cost_total": fee + not_ref * c_slip, "reason": reason})
        if not legs:
            return
        sig_t = idx[i - lag] if (lag > 0 and i - lag >= 0 and not first) else idx[i]
        for lg in legs:
            trades.append({"asset": lg["asset"], "side": lg["side"],
                           "signal_time": sig_t, "execution_time": idx[i],
                           "price_time": idx[i], "price": lg["price"],
                           "ref_price": lg["ref_price"], "notional": lg["notional"],
                           "fee": lg["fee"], "slippage": lg["slippage"],
                           "cost_total": lg["cost_total"], "reason": lg["reason"]})
        w_after = np.zeros(len(ASSETS))
        pv2 = value(i, cash, shares)
        for j in range(len(ASSETS)):
            if not np.isnan(row[j]) and pv2 > 0:
                w_after[j] = shares[j] * row[j] / pv2
        ev = {"time": idx[i], "reason": REASON_LABEL.get(reason, reason),
              "turnover": tnotional / pv, "cost_total": float(sum(x["cost_total"] for x in legs)),
              "n_legs": len(legs), "label": label,
              "cash_scale": scale, "signalled_at": sig_t}
        ev.update({f"w_before_{a}": float(w_before[j]) for j, a in enumerate(ASSETS)})
        ev.update({f"w_target_{a}": float(tv[j]) for j, a in enumerate(ASSETS)})
        ev.update({f"w_after_{a}": float(w_after[j]) for j, a in enumerate(ASSETS)})
        events.append(ev)

    # ---- 初始建仓 ----
    do_rebalance(0, T[0], "initial", first=True)

    for i in range(n):
        if i > 0:
            if do_cash and cash > 0 and cr[i]:
                interest = cash * cr[i]
                cash += interest
                cash_interest_total += interest
            tv = T[i]
            need = False
            reason = "calendar"
            if force[i]:
                need, reason = True, "listing"
            if not need and bnd is not None and bnd[i]:
                need, reason = True, "calendar"
            if not need and use_thr:
                pv = value(i, cash, shares)
                if pv > 0:
                    dev = 0.0
                    for j in range(len(ASSETS)):
                        p = P[i, j]
                        if np.isnan(p):
                            continue
                        cur = shares[j] * p / pv
                        if relative:
                            denom = tv[j] if tv[j] > 1e-6 else np.nan
                            dv = abs(cur - tv[j]) / denom if denom == denom else 0.0
                        else:
                            dv = abs(cur - tv[j])
                        dev = max(dev, dv)
                    if dev > thr:
                        need = True
                        reason = "hybrid" if spec.mode == "calendar_thresh" else "threshold"
            if need:
                do_rebalance(i, tv, reason, first=False)
        pv = value(i, cash, shares)
        eq[i] = pv
        row = P[i]
        for j in range(len(ASSETS)):
            if not np.isnan(row[j]) and pv > 0:
                w_hist[i, j] = shares[j] * row[j] / pv
        cash_w[i] = cash / pv if pv > 0 else 0.0
        tgt_hist[i] = T[i]

    eqs = pd.Series(eq, index=idx)
    weights = pd.DataFrame(w_hist, index=idx, columns=ASSETS)
    weights["cash"] = cash_w
    weights["risky_total"] = weights[ASSETS].sum(axis=1)
    desired = pd.DataFrame(tgt_hist, index=idx, columns=ASSETS)
    desired["cash"] = 1.0 - desired[ASSETS].sum(axis=1)
    years = max((idx[-1] - idx[0]).days, 1) / 365.25
    tr = pd.DataFrame(trades)
    ev_df = pd.DataFrame(events)
    return {
        "label": label,
        "equity": eqs,
        "weights": weights,
        "desired": desired,
        "trades": tr,
        "events": ev_df,
        "turnover_total": float(turnover_total),
        "turnover_annual": float(turnover_total / years) if years > 0 else 0.0,
        "cash_interest_total": float(cash_interest_total),
        "cash_constrained_bars": int(cash_constrained_bars),
        "n_trades": int(len(tr)),
        "n_rebalances": int(len(ev_df)),
        "spec": spec.key(),
        "cfg_tag": cfg.tag(),
    }