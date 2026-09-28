# -*- coding: utf-8 -*-
"""配置模型与冻结常量（Roadmap V2 §5.4 / §12）。

本模块只包含"申报过的"规则常量与配置对象，不含任何计算逻辑。
所有硬约束（paper_only / 无杠杆 / 权重和=100%）在此声明，供策略层、账本与
审计模块引用。

启动时 `Config.validate()` 必须拒绝任何违反 Roadmap §12 的配置组合。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASSETS: List[str] = ["BTC", "ETH", "SOL", "BNB"]

STRATEGY_ID = "S4_R9_PAPER_V2"

# ---------------------------------------------------------------------------
# §12 参考权重 R9（V5 冻结口径，不允许在线调参）
# ---------------------------------------------------------------------------
REF_WEIGHTS: Dict[str, float] = {"BTC": 0.25, "ETH": 0.10, "SOL": 0.30, "BNB": 0.35}

# ---------------------------------------------------------------------------
# §5.4(a) 信号层：V4 冻结带（供上行趋势下界托底）。V5 回测中 S4 的信号层
# 强制使用这一套带，**不得**用审计带替代，否则会改变信号本身（差 5pp）。
# ---------------------------------------------------------------------------
SIGNAL_BAND: Dict[str, Tuple[float, float]] = {
    "BTC": (0.05, 0.25), "ETH": (0.05, 0.15),
    "SOL": (0.30, 0.40), "BNB": (0.35, 0.45),
}

# ---------------------------------------------------------------------------
# §5.4(b) 审计层：V5 冻结带（越界防护与裁剪留痕）。本组合下恒为 no-op。
# ---------------------------------------------------------------------------
AUDIT_BAND: Dict[str, Tuple[float, float]] = {
    "BTC": (0.10, 0.30), "ETH": (0.05, 0.20),
    "SOL": (0.25, 0.40), "BNB": (0.30, 0.45),
}

REQUIRED_BOUNDARY_GAP_PP = 0.05          # §12：参考权重距审计带任一边界 >= 5pp

# ---------------------------------------------------------------------------
# §5.3 S4 方向与分档（V5 冻结口径）
# ---------------------------------------------------------------------------
MODULES: List[str] = ["ma200", "tsmom6m"]
DIRECTION_WEIGHTS: Dict[str, float] = {"ma200": 0.35, "tsmom6m": 0.15}
BUCKET_THRESHOLDS: List[float] = [0.25, 0.50, 0.75]
BUCKET_LEVELS: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]
MA_LENGTH = 200
TSMOM_DAYS = 126
TSMOM_TANH_SCALE = 0.5

# 信号层下界托底的最低分档（V4 `enforce_lower_band` 的 `dirs > 0.25` 条件）
LOWER_BAND_MIN_SCALE = 0.50

# ---------------------------------------------------------------------------
# §6.3 / §6.4 执行口径
# ---------------------------------------------------------------------------
FEE = 0.0010
SLIPPAGE = 0.0002
CASH_RETURN = "Cash_Rate"
LISTING_RULE = "L1"
REBALANCE_MODE = "threshold"
REBAL_THRESHOLD_ABSOLUTE = 0.10
DECISION_CADENCE = "daily"
DECISION_TIMEFRAME = "4H"
EXEC_LAG_BARS = 1

MIN_NOTIONAL_USD = 50.0                  # §6.4 最小交易阈值
MIN_NOTIONAL_EQUITY_PCT = 0.005

FRESHNESS_MAX_HOURS = 12                 # §4.3 数据新鲜度阈值
BARS_PER_DAY = 6                         # 4H bar / day

# §7 调仓决策状态
STATUSES: List[str] = [
    "NO_ACTION", "REBALANCE_PROPOSED", "PAPER_FILLED", "STALE_DATA",
    "INSUFFICIENT_DATA", "INVARIANT_FAILED", "DUPLICATE_RUN", "EXECUTION_ERROR",
]

# §16 连续异常计数
ALERT_WARNING = 1
ALERT_ALERT = 2
ALERT_CRITICAL = 3

# §16 高危异常编码（进入连续计数与升级）
ANOMALY_CONFIG_CHANGED = "config_hash_changed"
ANOMALY_BAND_CLIPPED = "audit_band_nonzero_clip"
ANOMALY_CASH_INSUFFICIENT = "cash_insufficient_for_buys"
ANOMALY_EMAIL_FAILED = "email_send_failed"
ANOMALY_MA_INSUFFICIENT = "ma200_insufficient"
ANOMALY_TSMOM_INSUFFICIENT = "tsmom6m_insufficient"
ANOMALY_WEIGHTS_NOT_100 = "weights_not_100pct"
ANOMALY_NEGATIVE_CASH = "negative_cash"
ANOMALY_LEDGER_BROKEN = "ledger_chain_broken"
ANOMALY_STATUSES: Tuple[str, ...] = (
    "STALE_DATA", "INSUFFICIENT_DATA", "INVARIANT_FAILED", "EXECUTION_ERROR")

ALERT_LEVEL_NAMES: Dict[int, str] = {
    ALERT_WARNING: "WARNING", ALERT_ALERT: "ALERT", ALERT_CRITICAL: "CRITICAL"}


def alert_level(streak: int) -> Optional[str]:
    """连续异常次数 -> 告警级别（§16）。0 次表示本次运行正常。"""
    if streak <= 0:
        return None
    if streak >= ALERT_CRITICAL:
        return ALERT_LEVEL_NAMES[ALERT_CRITICAL]
    return ALERT_LEVEL_NAMES[streak]


class ConfigError(ValueError):
    """配置非法（Roadmap §12 启动拒绝项）。"""


def band_boundary_gap(weights: Dict[str, float],
                      band: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """各资产距配置带边界的最近距离（正=带内，负=越界）。"""
    return {a: min(weights[a] - band[a][0], band[a][1] - weights[a]) for a in ASSETS}


# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """策略配置（对应 config/strategy.json，Roadmap §12）。"""

    strategy_id: str = STRATEGY_ID
    assets: List[str] = field(default_factory=lambda: list(ASSETS))
    reference_weights: Dict[str, float] = field(default_factory=lambda: dict(REF_WEIGHTS))
    modules: List[str] = field(default_factory=lambda: list(MODULES))
    ma_length: int = MA_LENGTH
    tsmom_days: int = TSMOM_DAYS
    tsmom_tanh_scale: float = TSMOM_TANH_SCALE
    direction_component_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DIRECTION_WEIGHTS))
    alloc_mode: str = "bucket"
    bucket_thresholds: List[float] = field(default_factory=lambda: list(BUCKET_THRESHOLDS))
    quality_layer: bool = False
    risk_layer: bool = False
    enforce_lower_band: bool = True
    signal_band: Dict[str, List[float]] = field(
        default_factory=lambda: {a: list(v) for a, v in SIGNAL_BAND.items()})
    audit_band: Dict[str, List[float]] = field(
        default_factory=lambda: {a: list(v) for a, v in AUDIT_BAND.items()})
    rebalance_mode: str = REBALANCE_MODE
    rebalance_threshold_absolute: float = REBAL_THRESHOLD_ABSOLUTE
    decision_cadence: str = DECISION_CADENCE
    listing_rule: str = LISTING_RULE
    fee: float = FEE
    slippage: float = SLIPPAGE
    cash_return: str = CASH_RETURN
    decision_timeframe: str = DECISION_TIMEFRAME
    execution_lag_bars: int = EXEC_LAG_BARS
    min_notional_usd: float = MIN_NOTIONAL_USD
    min_notional_equity_pct: float = MIN_NOTIONAL_EQUITY_PCT
    paper_only: bool = True
    exchange_trading_enabled: bool = False

    # -- 派生属性 ---------------------------------------------------------
    @property
    def cost_per_leg(self) -> float:
        return self.fee + self.slippage

    @property
    def signal_band_t(self) -> Dict[str, Tuple[float, float]]:
        return {a: (float(v[0]), float(v[1])) for a, v in self.signal_band.items()}

    @property
    def audit_band_t(self) -> Dict[str, Tuple[float, float]]:
        return {a: (float(v[0]), float(v[1])) for a, v in self.audit_band.items()}

    # -- 校验 -------------------------------------------------------------
    def validate(self) -> None:
        """按 Roadmap §12 逐条拒绝非法配置。"""
        if self.paper_only is not True:
            raise ConfigError("paper_only must be true; this system never trades live")
        if self.exchange_trading_enabled is not False:
            raise ConfigError("exchange_trading_enabled must be false")
        if self.strategy_id != STRATEGY_ID:
            raise ConfigError(f"strategy_id must be {STRATEGY_ID!r}, got {self.strategy_id!r}")
        if sorted(self.assets) != sorted(ASSETS):
            raise ConfigError(f"assets must be exactly {ASSETS}")

        total = sum(float(self.reference_weights.get(a, 0.0)) for a in ASSETS)
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"reference_weights must sum to 1, got {total!r}")

        audit = self.audit_band_t
        for a in ASSETS:
            w = float(self.reference_weights[a])
            lo, hi = audit[a]
            if w < lo - 1e-9 or w > hi + 1e-9:
                raise ConfigError(f"reference weight {a}={w} outside audit band {audit[a]}")
            gap = min(w - lo, hi - w)
            if gap < REQUIRED_BOUNDARY_GAP_PP - 1e-9:
                raise ConfigError(
                    f"{a} weight {w} only {gap:.4f} from audit band boundary "
                    f"(requires >= {REQUIRED_BOUNDARY_GAP_PP})")

        if self.ma_length <= 0:
            raise ConfigError("ma_length must be positive")
        if self.tsmom_days <= 0:
            raise ConfigError("tsmom_days must be positive")
        if self.tsmom_tanh_scale <= 0:
            raise ConfigError("tsmom_tanh_scale must be positive")

        th = [float(x) for x in self.bucket_thresholds]
        if any(b <= a for a, b in zip(th, th[1:])):
            raise ConfigError(f"bucket_thresholds must be strictly ascending, got {th}")
        if self.alloc_mode not in ("bucket", "linear"):
            raise ConfigError(f"unsupported alloc_mode {self.alloc_mode!r}")

        for name, band in (("signal_band", self.signal_band_t), ("audit_band", audit)):
            for a in ASSETS:
                lo, hi = band[a]
                if lo >= hi:
                    raise ConfigError(f"{name}[{a}] requires lo < hi, got ({lo}, {hi})")

        if self.rebalance_mode != "threshold":
            raise ConfigError(
                "S4 frozen spec requires rebalance_mode='threshold' (no calendar leg)")
        if self.decision_cadence != "daily":
            raise ConfigError("S4 frozen spec requires decision_cadence='daily'")
        if self.listing_rule != "L1":
            raise ConfigError("paper system fixes listing_rule='L1'")
        if self.execution_lag_bars < 0:
            raise ConfigError("execution_lag_bars must be >= 0")
        if self.fee < 0 or self.slippage < 0:
            raise ConfigError("fee and slippage cannot be negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyConfig":
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown strategy config keys: {unknown}")
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "StrategyConfig":
        cfg = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        cfg.validate()
        return cfg

    def fingerprint(self, extra: Optional[dict] = None) -> str:
        """配置指纹（§9 日志必须记录 config hash）。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        if extra:
            blob += json.dumps(extra, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class NotificationConfig:
    """邮件通知配置（对应 config/notification.json，Roadmap §8.1）。

    SMTP 凭据**不得**出现在本对象中，只能来自环境变量或 OpenClaw Secret Store。
    """

    enabled: bool = True
    recipient: str = "user@example.com"
    sender: str = "paper-system@example.com"
    timezone: str = "Asia/Shanghai"
    send_no_action: bool = False
    attach_csv: bool = True

    @classmethod
    def load(cls, path: Path) -> "NotificationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown notification config keys: {unknown}")
        return cls(**data)


@dataclass
class PaperAccountConfig:
    """纸面账户配置（对应 config/paper_account.json，Roadmap §6.1）。"""

    account_id: str = "paper-s4-r9-001"
    currency: str = "USD"
    initial_cash: float = 10_000.0

    @classmethod
    def load(cls, path: Path) -> "PaperAccountConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown paper account config keys: {unknown}")
        return cls(**data)


# ---------------------------------------------------------------------------
# SMTP 环境变量名（§8.1）—— 只读环境变量，禁止写入仓库与日志
# ---------------------------------------------------------------------------
SMTP_ENV_VARS: List[str] = [
    "PAPER_SMTP_HOST", "PAPER_SMTP_PORT", "PAPER_SMTP_USER",
    "PAPER_SMTP_PASSWORD", "PAPER_SMTP_TLS",
]

# 代理（所有下载操作必须走代理）
DEFAULT_PROXY = "http://127.0.0.1:7897"