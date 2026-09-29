# -*- coding: utf-8 -*-
"""V4 现货多资产系统 —— 冻结配置与常量（Roadmap §2/§4/§6/§7/§13）

本模块只包含"申报过的"规则常量与配置对象，不含任何计算逻辑。
所有硬约束（无杠杆/无做空/权重和=100%）在此声明，供引擎与报告引用。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASSETS: List[str] = ["BTC", "ETH", "SOL", "BNB"]

# §2.2 参考权重与配置带（V4 当前默认口径：B-Balanced 15/10/35/40）
REF_WEIGHT: Dict[str, float] = {"BTC": 0.15, "ETH": 0.10, "SOL": 0.35, "BNB": 0.40}
BAND: Dict[str, Tuple[float, float]] = {
    "BTC": (0.05, 0.25), "ETH": (0.05, 0.15),
    "SOL": (0.30, 0.40), "BNB": (0.35, 0.45),
}
NO_BAND: Dict[str, Tuple[float, float]] = {a: (0.0, 1.0) for a in ASSETS}

# 历史锚点（V4.0~V4.3 已发布口径）：冻结，不得随默认权重变化。
# V4.1/V4.3 的 REF45 与 Strict Band、以及对照臂均引用这两个常量。
LEGACY_REF45: Dict[str, float] = {"BTC": 0.45, "ETH": 0.20, "SOL": 0.15, "BNB": 0.20}
LEGACY_BAND: Dict[str, Tuple[float, float]] = {
    "BTC": (0.35, 0.55), "ETH": (0.15, 0.25),
    "SOL": (0.10, 0.20), "BNB": (0.15, 0.25),
}

# §3.1 样本区间
START = "2018-06-01"

# §6 公平交易成本矩阵（fee, slippage）
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
LISTING_L2_DAYS = 10                 # L2：上市后按未来 N 个交易日分批补足

# §5.2 现金模式
CASH_MODES: List[str] = ["Cash_0", "Cash_Rate"]

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

# 随机种子（§11.3 Bootstrap / §19.6 可复现性）
SEED = 20260920


@dataclass
class V4Cfg:
    """V4 统一配置对象：基准与策略共用同一份执行/成本/现金口径。"""

    # 成本（§6）
    fee: float = DEFAULT_COST[0]
    slippage: float = DEFAULT_COST[1]
    # 现金（§5.2）
    cash_return: str = "Cash_0"          # Cash_0 | Cash_Rate
    # 上市进入（§5.1）
    listing_rule: str = DEFAULT_LISTING_RULE
    # 配置带（§2.2）
    band_mode: str = "Band"              # Band | No-Band
    band_override: Optional[Dict[str, Tuple[float, float]]] = None   # 显式带覆盖（对照臂用）
    ref_weight: Dict[str, float] = field(default_factory=lambda: dict(REF_WEIGHT))
    # 执行
    initial: float = 10_000.0
    exec_lag_bars: int = 1               # §3.3 信号 -> 下一 bar 执行
    # 再平衡触发
    cadence: str = "daily"               # 4h | daily | weekly（策略决策频率）
    rebal_threshold: float = 0.10
    threshold_basis: str = "absolute"    # absolute | relative
    # 策略模块参数（供复用 V3 指标层）
    ma_len: int = 200
    tsmom_days: int = 126
    tsmom_days2: int = 252
    risk_multiplier: float = 0.0         # S1：跌破 MA200 后的目标权重乘数
    breadth_floor: float = 0.25          # S2：组合级出场下限

    @property
    def cost_per_leg(self) -> float:
        return self.fee + self.slippage

    @property
    def band(self) -> Dict[str, Tuple[float, float]]:
        if self.band_override is not None:
            return dict(self.band_override)
        return dict(BAND) if self.band_mode == "Band" else dict(NO_BAND)

    def tag(self) -> str:
        """唯一运行标识（用于文件与审计）。"""
        return (f"{self.band_mode}|{self.cash_return}|{self.listing_rule}"
                f"|fee={self.fee:.4f}|slip={self.slippage:.4f}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band"] = self.band
        d["cost_per_leg"] = self.cost_per_leg
        return d


@dataclass(frozen=True)
class RebalanceSpec:
    """再平衡触发规则（与目标权重解耦）。

    mode:
      never            —— 仅初始建仓 + 上市进入（B0）
      calendar         —— 日历周期触发（B1-B4）
      threshold        —— 偏离阈值触发（B5）
      calendar_thresh  —— 日历 + 阈值混合（B6）
    """

    mode: str = "calendar"
    freq: Optional[str] = "Y"            # M | Q | 2Q | Y
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


def config_fingerprint(cfg: V4Cfg, extra: Optional[dict] = None) -> str:
    """§19.6 可复现性：配置 + 策略源码 + 数据的联合指纹。"""
    import scripts.v4_spot.benchmarks as B
    import scripts.v4_spot.data as D
    import scripts.v4_spot.engine as E
    import scripts.v4_spot.strategies as S
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    for mod in (E, S, B, D):
        blob += Path(mod.__file__).read_text(encoding="utf-8")
    for f in sorted((Path(__file__).resolve().parents[2] / "data" / "v3_spot").glob("*.csv")):
        blob += f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}"
    if extra:
        blob += json.dumps(extra, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()