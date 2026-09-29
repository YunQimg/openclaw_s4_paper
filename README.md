# OpenClaw S4 + R9 现货纸面交易系统

> **本项目是纸面交易系统。**
> **本项目不连接交易所。**
> **本项目不提交真实订单。**
> **本项目不读取真实账户余额或持仓。**
> **所有调仓都是本地模拟并通过邮件通知。**

策略固定为 `S4 + R9`：

```text
S4 = MA200 方向 × TSMOM 6M 资产级分档（bucket）
R9 = BTC 25% / ETH 10% / SOL 30% / BNB 35%
```

系统与 V5 回测冻结口径**逐位一致**（已在 3034 个日频决策 bar 上验证
`max|diff| = 0`），不引入任何 S4 或 R9 专属的额外调参。

Roadmap：`docs/OpenClaw S4 R9 纸面交易系统 Roadmap V2.md`

---

## 1. 安装

```bash
pip install -e openclaw_s4_paper
# 或仅运行时依赖
pip install pandas numpy requests
```

要求 Python >= 3.10。

## 2. 配置环境变量

SMTP 凭据**只能**来自环境变量或 OpenClaw Secret Store（禁止写入仓库/配置/日志）：

```bash
export PAPER_SMTP_HOST=smtp.example.com
export PAPER_SMTP_PORT=465
export PAPER_SMTP_USER=paper-system@example.com
export PAPER_SMTP_PASSWORD=********
export PAPER_SMTP_TLS=true

# 下载代理（直连不通 Google 时使用）
export PAPER_PROXY=http://127.0.0.1:7897
```

配置文件：

```text
config/strategy.json                 策略与执行口径（含 §12 冻结字段）
config/notification.json             邮件收件人/发件人（SMTP 密码不在此文件）
config/paper_account.json            账户 id / 币种 / 初始现金
```

## 3. 准备市场数据

```bash
# 自动判定代理（直连可达 Google 则不用代理）；也可用 --proxy 强制指定
python -m openclaw_s4_paper.cli download --out-dir <市场数据目录>
```

输出：

```text
4h.csv              BTC/ETH/SOL/BNB 现货 4H OHLCV（timestamp,asset,open,high,low,close,volume）
1d.csv              1D OHLCV（TSMOM 6M 使用）
cash_rate_dff.csv   FRED DFF 有效联邦基金利率（Cash_Rate 模式）
```

数据契约（§4.2）：UTC 时间戳、close > 0、high ≥ max(open, close)、
low ≤ min(open, close)、无重复 asset/timestamp、时间升序。

## 4. 初始化纸面账户

```bash
python -m openclaw_s4_paper.cli init
```

生成 `state/paper_account.json`：

```json
{
  "account_id": "paper-s4-r9-001",
  "currency": "USD",
  "initial_cash": 10000.0
}
```

## 5. 手动运行一次

```bash
# 信号检查 + 纸面成交（默认不发信，dry-run）
python -m openclaw_s4_paper.cli signal

# 指定历史时刻（回放/补算）
python -m openclaw_s4_paper.cli signal --as-of 2026-09-15T06:00:00+00:00

# 真正发信
python -m openclaw_s4_paper.cli signal --send-mail

# 收盘结算
python -m openclaw_s4_paper.cli settle

# 健康检查
python -m openclaw_s4_paper.cli health
```

## 6. OpenClaw 调度配置

见 `openclaw/tasks.yaml`（三个任务 + 一个数据刷新任务）、
`openclaw/permissions.yaml`（最小权限 + 明确拒绝项）、
`openclaw/runbook.md`（运维与故障处置）。

```text
UTC 00:00  s4_r9_market_data_refresh
UTC 00:10  s4_r9_signal_check
UTC 00:20  s4_r9_paper_settlement
UTC 01:00  s4_r9_health_check
```

## 7. 邮件示例

有调仓时（主题 `[S4 Paper] 调仓建议与纸面成交 - YYYY-MM-DD`）：

```text
系统：S4 + R9
运行状态：PAPER_FILLED
信号时间：2026-09-15 00:00 UTC
执行时间：2026-09-15 08:00 UTC
数据最后时间：2026-09-15 00:00 UTC
策略版本：S4_R9_PAPER_V2
run_id：20260915S4_R9_PAPER_V2925e39a7

── 市场状态（资产级）──
BTC: close=75,812.0000 MA200=76,945.39 ma_sign=+1 tsmom6m=+0.1234 direction=+0.6740 scale=0.75
ETH: close=2,474.7849 MA200=... ma_sign=+1 tsmom6m=+0.2345 direction=+0.7448 scale=0.75
SOL: close=100.4601 ... scale=1.00
BNB: close=715.2330 ... scale=1.00
已上市资产：BTC, ETH, SOL, BNB

── 目标组合（双口径）──
BTC: raw=18.00% band-constrained=18.75%
...
Cash target：7.50%
风险资产合计：92.50%

── 调仓清单 ──
BTC BUY qty=0.024368 est_fill=76945.3860 notional=1875.00 reason=initial
...

── 执行后 ──
equity after：9,989.05
MaxDD to date：-0.11%

本邮件只代表纸面交易模拟结果，系统未连接交易所，也未提交任何真实订单。
```

无调仓时**不发送邮件**，仅写入 `logs/<date>.signal.json`。
异常时主题为 `[S4 Paper][ALERT] 运行失败或数据过期 - YYYY-MM-DD`。

## 8. 故障恢复

```text
账本已提交、邮件失败  -> 只重试邮件，绝不重跑 signal
账本未提交（任何失败）-> 从 state/snapshots/snapshots.csv 最后一行恢复账户
```

详见 `openclaw/runbook.md` §4/§5。

## 9. 备份与恢复

每日备份并保留至少 90 天：

```text
state/paper_account.json
state/ledger.jsonl
state/orders.jsonl
state/snapshots/snapshots.csv
state/runs/
logs/
```

## 10. 安全边界

**禁止出现**：`place_order` / `cancel_order` / `get_account_balance` /
`get_open_orders` / `withdraw` / `transfer`，以及任何交易所 SDK 依赖。

`python -m openclaw_s4_paper.cli health` 会对生产代码做静态扫描
（`audit.scan_source_tree`），发现上述能力即判失败。

**明确不提供**：交易所 API Key 配置教程、任何下单/撤单/查余额能力。

## 11. 测试

```bash
cd openclaw_s4_paper
python -m pytest tests -q
```

测试覆盖：策略分档与带语义、TSMOM point-in-time、纸面账本与幂等、
邮件渲染与脱敏、安全静态扫描、合成数据端到端。

## 12. 模块清单

```text
config.py           配置模型 + §12 校验 + 两层带常量
models.py           frozen dataclass 领域模型（含不变式）
market_data.py      装载/校验/新鲜度/执行 bar
indicators.py       MA200 与 TSMOM 6M（point-in-time）
strategy_s4.py      S4 方向、分档、两层带、目标权重
rebalance.py        调仓计划（阈值触发、先卖后买、最小交易阈值）
paper_ledger.py     纸面成交模拟 + 账本追加
valuation.py        估值、MaxDD、恒等式校验
ledger_store.py     追加式 JSONL + 哈希链 + 快照
state_store.py      账户/账本/运行清单/日志读写
audit.py            敏感信息扫描 + 交易所能力静态扫描 + 运行清单
email_notifier.py   邮件渲染（§8.3）+ 脱敏 + 发送
run_once.py         主流程编排（幂等、失败策略、原子提交）
download_data.py    市场数据下载（自动判定代理）
cli.py              命令行入口（init/signal/settle/health/download）
```