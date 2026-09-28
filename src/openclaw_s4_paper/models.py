# -*- coding: utf-8 -*-
"""领域数据模型（Roadmap V2 §4 / §5 / §6 / §7）。

全部为 frozen dataclass，带不变式校验；不含任何 I/O 或交易能力。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .config import ASSETS, STATUSES, ConfigError

# ---------------------------------------------------------------------------
# 市场数据
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """单根 4H OHLCV。"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.close <= 0:
            raise ConfigError(f"close must be > 0, got {self.close!r}")
        if self.high < max(self.open, self.close) - 1e-12:
            raise ConfigError(
                f"high ({self.high}) must be >= max(open, close) at {self.timestamp}")
        if self.low > min(self.open, self.close) + 1e-12:
            raise ConfigError(
                f"low ({self.low}) must be <= min(open, close) at {self.timestamp}")


@dataclass(frozen=True)
class ValidationResult:
    """市场数据校验结果（§4.2）。"""

    ok: bool
    n_bars: int
    n_assets: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ConfigError("market data validation failed: " + "; ".join(self.errors))


@dataclass(frozen=True)
class AssetSignal:
    """单资产信号快照（§5.5 要求保存的全部字段）。"""

    asset: str
    close: float
    ma200: float
    ma_sign: float
    tsmom6m: float
    direction: float
    scale: float
    reference_weight: float
    raw_target_weight: float
    band_constrained_target_weight: float
    final_target_weight: float

    @property
    def traded(self) -> bool:
        return self.final_target_weight > 0.0


@dataclass(frozen=True)
class SignalSnapshot:
    """一次运行的完整信号快照（§9 信号日志）。"""

    as_of: datetime
    signal_time: datetime
    execution_time: datetime
    data_last_time: datetime
    strategy_version: str
    run_id: str
    input_hash: str
    config_hash: str
    assets: Dict[str, AssetSignal]
    listed_assets: List[str]
    risky_total: float
    cash_target_weight: float
    status: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ConfigError(f"unknown status {self.status!r}")

    @property
    def target_weights(self) -> Dict[str, float]:
        return {a: s.final_target_weight for a, s in self.assets.items()}

    @property
    def raw_target_weights(self) -> Dict[str, float]:
        return {a: s.raw_target_weight for a, s in self.assets.items()}


# ---------------------------------------------------------------------------
# 纸面账户
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """现货多头纸面持仓。"""

    asset: str
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity < -1e-12:
            raise ConfigError(
                f"spot position cannot be negative (no shorting): {self.asset}={self.quantity}")


@dataclass
class PaperAccount:
    """本地纸面账户状态（§6.1）。"""

    account_id: str
    currency: str = "USD"
    initial_cash: float = 10_000.0
    cash: float = 10_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    as_of: Optional[datetime] = None
    ledger_version: int = 1

    def __post_init__(self) -> None:
        if self.cash < -1e-9:
            raise ConfigError(f"cash cannot be negative, got {self.cash!r}")

    def quantity(self, asset: str) -> float:
        p = self.positions.get(asset)
        return 0.0 if p is None else p.quantity

    def market_value(self, prices: Dict[str, float]) -> float:
        """风险资产市值（缺失价格的资产按 0 计）。"""
        return sum(self.quantity(a) * float(prices.get(a, 0.0) or 0.0) for a in ASSETS)

    def equity(self, prices: Dict[str, float]) -> float:
        """净值 = 风险资产市值 + 现金（§20 验收标准 9）。"""
        return self.cash + self.market_value(prices)

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {a: 0.0 for a in ASSETS}
        return {a: self.quantity(a) * float(prices.get(a, 0.0) or 0.0) / eq for a in ASSETS}


@dataclass(frozen=True)
class LedgerEvent:
    """追加式账本事件（§6.2 字段清单）。"""

    event_id: str
    run_id: str
    timestamp: datetime
    event_type: str
    asset: str
    side: str
    quantity: float
    price: float
    notional: float
    fee: float
    slippage: float
    cash_before: float
    cash_after: float
    position_before: float
    position_after: float
    strategy_version: str
    input_hash: str


@dataclass(frozen=True)
class LedgerResult:
    """一次纸面成交模拟的结果。"""

    events: List[LedgerEvent]
    equity_before: float
    equity_after: float
    cash_before: float
    cash_after: float
    weights_before: Dict[str, float]
    weights_after: Dict[str, float]
    remaining_target_error: Dict[str, float]
    cash_scale: float
    total_fee: float
    total_slippage: float
    turnover: float
    reason: str


# ---------------------------------------------------------------------------
# 调仓计划
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RebalanceLeg:
    """单条调仓腿（先卖后买，§6.4 交易顺序）。"""

    asset: str
    side: str                       # BUY | SELL
    quantity: float
    notional_delta: float
    current_weight: float
    target_weight: float
    estimated_fill_price: float
    reason: str


@dataclass(frozen=True)
class RebalancePlan:
    """调仓计划（§7：只生成计划，不直接提交外部订单）。"""

    as_of: datetime
    equity: float
    legs: List[RebalanceLeg]
    weight_deltas: Dict[str, float]
    notional_deltas: Dict[str, float]
    below_threshold: List[str] = field(default_factory=list)
    target_error_before: Dict[str, float] = field(default_factory=dict)

    @property
    def has_action(self) -> bool:
        return bool(self.legs)

    @property
    def status(self) -> str:
        return "REBALANCE_PROPOSED" if self.legs else "NO_ACTION"