# -*- coding: utf-8 -*-
"""OpenClaw S4 + R9 现货纸面交易系统。

本项目是纸面交易系统：
  * 不连接交易所；
  * 不提交真实订单；
  * 不读取真实账户余额或持仓；
  * 所有调仓都是本地模拟并通过邮件通知。

策略固定为 S4 + R9（Roadmap V2 §1）：
  S4 = MA200 方向 × TSMOM 6M 资产级分档
  R9 = BTC 25% / ETH 10% / SOL 30% / BNB 35%
"""

__version__ = "2.0.0"
STRATEGY_VERSION = "S4_R9_PAPER_V2"

__all__ = ["__version__", "STRATEGY_VERSION"]