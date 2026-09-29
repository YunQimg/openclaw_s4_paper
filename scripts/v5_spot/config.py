# -*- coding: utf-8 -*-
"""V5 —— 冻结配置与常量（Roadmap §2/§4/§6/§7/§13/§19）

V5 与 V4 的关系
---------------
* 执行口径（成本、现金、上市进入、信号时序）**完全继承** V4，未做任何改动。
* 新增一等公民：三个参考权重 `R13 / R3 / R9`（Roadmap §2.2.1）。
* 新增 V5 默认配置带（Roadmap §2.2.2），以 `band_override` 注入，
  不改动 `scripts/v4_spot/config.py` 的 `REF_WEIGHT` / `BAND`（Roadmap 附录 B.3）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASSETS: List[str] = ["BTC", "ETH", "SOL", "BNB"]

# ---------------------------------------------------------------------------
# §2.2.1 三个参考权重（V5 一等公民，地位对等）
# ---------------------------------------------------------------------------
REF_WEIGHTS: Dict[str, Dict[str, float]] = {
    "R13": {"BTC": 0.15, "ETH": 0.15, "SOL": 0.35, "BNB": 0.35},   # 均衡
    "R3":  {"BTC": 0.15, "ETH": 0.10, "SOL": 0.35, "BNB": 0.40},   # 收益（= V4 基线）
    "R9":  {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35},   # 保守
}
REF_IDS: List[str] = ["R13", "R3", "R9"]
REF_LABEL: Dict[str, str] = {
    "R13": "15/15/35/35", "R3": "15/10/35/40", "R9": "25/10/30/35"}
REF_ROLE: Dict[str, str] = {
    "R13": "均衡中枢（V4.8 Overall 第 1）",
    "R3": "收益中枢（V4.8 CAGR/跨版本最优，= V4 基线）",
    "R9": "保守中枢（V4.8 OOS 衰减最小）",
}

# ---------------------------------------------------------------------------
# §2.2.2 配置带
# ---------------------------------------------------------------------------
# V5 默认带 = 三参考权重包络再统一外扩 5pp
BAND_V5: Dict[str, Tuple[float, float]] = {
    "BTC": (0.10, 0.30), "ETH": (0.05, 0.20),
    "SOL": (0.25, 0.40), "BNB": (0.30, 0.45),
}
# V4 默认带（冻结历史口径，仅供 V4-Band-Raw / V4-Band-Strict 对照）
BAND_V4: Dict[str, Tuple[float, float]] = {
    "BTC": (0.05, 0.25), "ETH": (0.05, 0.15),
    "SOL": (0.30, 0.40), "BNB": (0.35, 0.45),
}
NO_BAND: Dict[str, Tuple[float, float]] = {a: (0.0, 1.0) for a in ASSETS}

# §2.2.2 包络与间距校验（供 §19.2 验收）
ENVELOPE: Dict[str, Tuple[float, float]] = {
    "BTC": (0.15, 0.25), "ETH": (0.10, 0.15),
    "SOL": (0.30, 0.35), "BNB": (0.35, 0.40),
}
BAND_OVEREXPAND_PP = 0.05
REQUIRED_BOUNDARY_GAP_PP = 0.05      # 每项资产距任一约束边界至少 5pp

# §2.3 三组命名（不得混用排名）
BAND_GROUPS: Dict[str, str] = {
    "V5-Band": "V5 默认带（V5 主口径）",
    "V4-Band-Raw": "V4 默认带，超界允许越界（记录越界事件）",
    "V4-Band-Strict": "V4 默认带，超界裁剪到边界",
    "No-Band": "仅受总风险资产权重约束",
}

# §3.1 样本区间
START = "2018-06-01"

# §6 公平交易成本矩阵（fee, slippage）—— 与 V4 完全一致
COST_MATRIX: List[Tuple[float, float]] = [
    (0.0005, 0.0000),
    (0.0010, 0.0002),
    (0.0010, 0.0005),
    (0.0015, 0.0005),
]
DEFAULT_COST: Tuple[float, float] = (0.0010, 0.0002)

# §3.2 / §5.1 上市进入规则
LISTING_RULES: List[str] = ["L1", "L2", "L3"]
DEFAULT_LISTING_RULE = "L1"
LISTING_L2_DAYS = 10

# §5.2 现金模式
CASH_MODES: List[str] = ["Cash_0", "Cash_Rate"]
PRIMARY_CASH = "Cash_Rate"

# §4 再平衡基准定义
CAL_FREQS: Dict[str, str] = {"Monthly": "M", "Quarterly": "Q",
                             "Semiannual": "2Q", "Annual": "Y"}
THRESHOLDS: List[float] = [0.05, 0.10, 0.15, 0.20]
HYBRIDS: List[Tuple[str, float]] = [("Y", 0.10), ("2Q", 0.10), ("Q", 0.10)]

BENCHMARK_IDS: List[str] = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]

# §7 策略候选
STRATEGY_IDS: List[str] = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
RISK_MULTIPLIERS: List[float] = [0.0, 0.25, 0.50, 0.75]

# §13 参数敏感性
MA_LENS: List[int] = [150, 200, 250]
TSMOM_LENS: List[int] = [63, 126, 252]
REBAL_FREQS_SENS: List[str] = ["M", "Q", "2Q", "Y"]

# §13 V5 追加：带内权重扰动（LRR）
LRR_PERTURB_PP: List[float] = [0.05, 0.10]
# LRR 档位（§13：>1 单列，且参与归一化前截断到 1.0）
LRR_TIERS: List[Tuple[float, str]] = [
    (0.98, "非常平坦"), (0.95, "较稳健"), (0.90, "中等"), (0.00, "存在明显尖峰风险")]
LRR_CAP = 1.0

# §12 Walk Forward 窗口（Train/Validate → Test）
WF_WINDOWS: List[Tuple[str, str, str, str]] = [
    ("2018-06-01", "2021-12-31", "2022-12-31", "2023-12-31"),
    ("2019-01-01", "2022-12-31", "2023-12-31", "2024-12-31"),
    ("2020-01-01", "2023-12-31", "2024-12-31", "2025-12-31"),
    ("2021-01-01", "2024-12-31", "2025-12-31", "2026-09-20"),
]

# §11.3 Bootstrap / Monte Carlo
N_BOOTSTRAP = 10_000
BOOT_BLOCK = 20
# 三臂必须使用同一随机种子与同一抽样块（§11.3 V5 追加）
SEED = 20260920

# §14.2 评分权重
SCORE_WEIGHTS: Dict[str, float] = {
    "sharpe": 0.25, "sortino": 0.20, "calmar": 0.20, "cagr": 0.15,
    "max_dd": 0.10, "active_stability": 0.05, "simplicity": 0.05,
}
SCORE_CAPS: Dict[str, float] = {
    "sharpe": 2.0, "sortino": 3.0, "calmar": 1.5, "cagr": 0.5}
SIMPLICITY: Dict[str, float] = {"S0": 1.00, "S1": 0.90, "S2": 0.80, "S3": 0.85,
                                "S4": 0.70, "S5": 0.60, "S6": 0.50, "S7": 0.30}

# §10 V5 追加：参考权重两两总变差距离
WEIGHT_DISTANCE: Dict[Tuple[str, str], float] = {
    ("R13", "R3"): 0.05, ("R13", "R9"): 0.10, ("R3", "R9"): 0.10}


# ---------------------------------------------------------------------------
def envelope_of(refs: Optional[List[str]] = None) -> Dict[str, Tuple[float, float]]:
    """参考权重集合的资产包络（min/max）。"""
    ids = refs or REF_IDS
    out: Dict[str, Tuple[float, float]] = {}
    for a in ASSETS:
        vals = [REF_WEIGHTS[r][a] for r in ids]
        out[a] = (min(vals), max(vals))
    return out


def band_boundary_gap(weights: Dict[str, float],
                      band: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """各资产距配置带边界的最近距离（正=带内，负=越界）。"""
    return {a: min(weights[a] - band[a][0], band[a][1] - weights[a]) for a in ASSETS}


def audit_band_compat(band: Dict[str, Tuple[float, float]],
                      refs: Optional[List[str]] = None,
                      need_gap: float = 0.0) -> "object":
    """§19.2 参考权重 × 配置带 相容性审计（返回 DataFrame）。"""
    import pandas as pd
    ids = refs or REF_IDS
    rows = []
    for r in ids:
        gap = band_boundary_gap(REF_WEIGHTS[r], band)
        for a in ASSETS:
            w = REF_WEIGHTS[r][a]
            rows.append({
                "ref": r, "asset": a, "weight": w,
                "band_lo": band[a][0], "band_hi": band[a][1],
                "gap": gap[a],
                "on_boundary": bool(abs(gap[a]) < 1e-9),
                "out_of_band": bool(gap[a] < 0),
                "clipped": bool(w < band[a][0] or w > band[a][1]),
                "gap_ok": bool(gap[a] >= need_gap - 1e-12),
            })
    return pd.DataFrame(rows)


def clip_to_band(weights: Dict[str, float],
                 band: Dict[str, Tuple[float, float]]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """V4-Band-Strict：把参考权重裁剪到带边界，返回 (裁剪后权重, 各资产是否被裁剪)。"""
    out, hit = {}, {}
    for a in ASSETS:
        lo, hi = band[a]
        w = float(weights.get(a, 0.0))
        w2 = min(max(w, lo), hi)
        out[a] = w2
        hit[a] = bool(abs(w2 - w) > 1e-12)
    return out, hit


# ---------------------------------------------------------------------------
@dataclass
class V5Cfg:
    """V5 统一配置对象：三臂 + 基准 + 策略共用同一份执行/成本/现金口径。"""

    # 参考权重（§2.2.1）
    ref: str = "R3"                                  # R13 | R3 | R9
    # 成本（§6）
    fee: float = DEFAULT_COST[0]
    slippage: float = DEFAULT_COST[1]
    # 现金（§5.2）
    cash_return: str = PRIMARY_CASH
    # 上市进入（§5.1）
    listing_rule: str = DEFAULT_LISTING_RULE
    # 配置带（§2.2.3 / §2.3）
    band_mode: str = "Band"                          # Band | No-Band
    band_group: str = "V5-Band"                      # V5-Band|V4-Band-Raw|V4-Band-Strict
    band_override: Optional[Dict[str, Tuple[float, float]]] = None
    # 执行（§3.3）
    initial: float = 10_000.0
    exec_lag_bars: int = 1
    cadence: str = "daily"                           # 4h | daily | weekly
    rebal_threshold: float = 0.10
    threshold_basis: str = "absolute"                # absolute | relative
    # 策略模块参数（§7，继承 V4）
    ma_len: int = 200
    tsmom_days: int = 126
    tsmom_days2: int = 252
    risk_multiplier: float = 0.0
    breadth_floor: float = 0.25
    # §9.2/§13 研究用：临时替换参考权重（资产剔除、LRR 邻域扰动）。
    # 仅影响 `ref_weight`，**不改变 `ref` 身份**，便于结果按臂归档。
    ref_weight_override: Optional[Dict[str, float]] = None

    @property
    def cost_per_leg(self) -> float:
        return self.fee + self.slippage

    @property
    def ref_weight(self) -> Dict[str, float]:
        if self.ref_weight_override is not None:
            return dict(self.ref_weight_override)
        return dict(REF_WEIGHTS[self.ref])

    @property
    def band(self) -> Dict[str, Tuple[float, float]]:
        if self.band_mode == "No-Band":
            return dict(NO_BAND)
        if self.band_override is not None:
            return dict(self.band_override)
        if self.band_group == "V4-Band-Raw" or self.band_group == "V4-Band-Strict":
            return dict(BAND_V4)
        return dict(BAND_V5)

    @property
    def enforce_band(self) -> bool:
        """是否对目标权重执行带上限/下限裁剪。

        V4-Band-Raw 显式声明「允许越界」（Roadmap §2.3），故仅作记录、不裁剪。
        """
        if self.band_mode == "No-Band":
            return False
        return self.band_group != "V4-Band-Raw"

    def tag(self) -> str:
        return (f"{self.ref}|{self.band_group}|{self.cash_return}|{self.listing_rule}"
                f"|fee={self.fee:.4f}|slip={self.slippage:.4f}|cad={self.cadence}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ref_weight"] = self.ref_weight
        d["band"] = self.band
        d["cost_per_leg"] = self.cost_per_leg
        d["enforce_band"] = self.enforce_band
        return d


@dataclass(frozen=True)
class RebalanceSpec:
    """再平衡触发规则（与参考权重正交）。"""

    mode: str = "calendar"                           # never|calendar|threshold|calendar_thresh
    freq: Optional[str] = "Y"                        # M | Q | 2Q | Y
    threshold: Optional[float] = None
    basis: str = "absolute"

    def key(self) -> str:
        if self.mode == "never":
            return "never"
        if self.mode == "calendar":
            return f"cal:{self.freq}"
        if self.mode == "threshold":
            return f"thr:{self.basis}:{self.threshold:.2f}"
        return f"hyb:{self.freq}+{self.basis}:{self.threshold:.2f}"


def config_fingerprint(cfg: V5Cfg, extra: Optional[dict] = None) -> str:
    """§19.8 可复现性：配置 + V5 源码 + 数据的联合指纹。"""
    import scripts.v5_spot.benchmarks as B
    import scripts.v5_spot.data as D
    import scripts.v5_spot.engine as E
    import scripts.v5_spot.strategies as S
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    for mod in (E, S, B, D):
        blob += Path(mod.__file__).read_text(encoding="utf-8")
    for f in sorted((Path(__file__).resolve().parents[2] / "data" / "v3_spot").glob("*.csv")):
        blob += f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}"
    if extra:
        blob += json.dumps(extra, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()