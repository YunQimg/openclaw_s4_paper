# OpenClaw S4 + R9 纸面交易系统 — 部署说明

> 本说明用于把系统从零部署到 OpenClaw 运行时并在生产调度下持续运行。
> 完整规格见 `docs/OpenClaw S4 R9 纸面交易系统 Roadmap V2.md`。

---

## 0. 性质与红线

本系统是**纸面交易系统**：

- 不连接交易所
- 不提交真实订单
- 不读取真实账户余额或持仓
- 所有调仓都是本地模拟并通过邮件通知

生产代码禁止出现 `place_order` / `cancel_order` / `get_account_balance` /
`get_open_orders` / `withdraw` / `transfer`，以及任何交易所 SDK 依赖。
`health` 任务会做静态扫描（`audit.scan_source_tree`），发现上述能力即判失败。

---

## 1. 前置条件

```text
Python >= 3.10
运行时依赖：pandas / numpy / requests
可用的 HTTP 代理（仅当直连不通 Google 时需要；默认 http://127.0.0.1:7897）
可发信的 SMTP 账号（凭据仅存环境变量 / OpenClaw Secret Store）
OpenClaw 运行时（用于调度 tasks.yaml）
```

---

## 2. 安装

```bash
pip install -e openclaw_s4_paper
# 或仅安装运行时依赖
pip install pandas numpy requests
```

---

## 3. 配置环境变量（凭据绝不落盘）

SMTP 凭据**只能**来自环境变量或 OpenClaw Secret Store，禁止写入仓库 / JSON 配置 / 日志。

```bash
export PAPER_SMTP_HOST=smtp.example.com
export PAPER_SMTP_PORT=465
export PAPER_SMTP_USER=paper-system@example.com
export PAPER_SMTP_PASSWORD=********
export PAPER_SMTP_TLS=true

# 下载代理（直连不通 Google 时使用）
export PAPER_PROXY=http://127.0.0.1:7897
```

---

## 4. 配置文件

位于 `openclaw_s4_paper/config/`：

```text
strategy.json              策略与执行口径（R9 参考权重 + §12 冻结字段）
notification.json          收件人 / 发件人（SMTP 密码不在此文件）
paper_account.json         账户 id / 币种 / 初始现金
```

可从同目录示例起步：`notification.example.json`、`paper_account.example.json`。

---

## 5. 准备市场数据

```bash
# 自动判定代理（直连可达 Google 则不用代理）；也可用 --proxy 强制指定
python -m openclaw_s4_paper.cli download \
  --out-dir data/paper_trading/market
```

产出：

```text
4h.csv              BTC/ETH/SOL/BNB 现货 4H OHLCV（timestamp,asset,open,high,low,close,volume）
1d.csv              1D OHLCV（TSMOM 6M 使用）
cash_rate_dff.csv   FRED DFF 有效联邦基金利率（Cash_Rate 模式）
```

数据契约（§4.2）：UTC 时间戳、close > 0、high ≥ max(open, close)、
low ≤ min(open, close)、无重复 asset/timestamp、时间升序。
详见 `data/README.md`。可通过 `PAPER_DISABLE_NETWORK_FETCHES=1` 强制禁网。

---

## 6. 初始化纸面账户

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

---

## 7. 试运行验证（部署必做）

```bash
# 信号检查 + 纸面成交（默认 dry-run，不真实发信）
python -m openclaw_s4_paper.cli signal --as-of 2026-09-15T06:00:00+00:00

# 收盘结算
python -m openclaw_s4_paper.cli settle

# 健康检查（数据新鲜度 / 账本完整性 / 策略版本 / 源码静态扫描）
python -m openclaw_s4_paper.cli health
```

期望：`signal` 产生纸面成交并写账本（状态 `PAPER_FILLED` 或 `NO_ACTION`）；
`health` 全部通过。

---

## 8. OpenClaw 调度部署

在 OpenClaw 注册 `openclaw/tasks.yaml`，按序执行（**顺序不可颠倒**，
结算必须在信号之后，否则净值快照会缺少当次成交）：

| UTC 时间 | 任务 | 作用 |
| --- | --- | --- |
| 00:00 | `s4_r9_market_data_refresh` | 下载 4H/1D/现金利率（直连不可达时走代理） |
| 00:10 | `s4_r9_signal_check` | 计算信号 + 模拟成交 + 发调仓邮件 |
| 00:20 | `s4_r9_paper_settlement` | 收盘结算，更新净值与回撤 |
| 01:00 | `s4_r9_health_check` | 健康检查 |

任务已内置 `retry`（2–3 次 + 退避）与 `on_failure: send_alert_email`。

权限按 `openclaw/permissions.yaml` 授予：

```text
允许：workspace 读写（config / state / logs）+ 市场数据读写
      + 邮件发送（收件人受 config/notification.json 约束）
      + 出网白名单（api.binance.us / data-api.binance.vision / fred.stlouisfed.org
        / www.google.com 连通性探针）
拒绝：交易所凭据 / 交易所账户读取 / 下单 / 撤单 / 提现 / 转账
      / 浏览器自动化 / 云钱包 / 无关目录的 shell 访问
```

密钥只从 OpenClaw Secret Store 或环境变量读取，且禁止落仓库 / 日志 / 邮件正文。

---

## 9. 运行产物与监控

```text
state/paper_account.json        当前账户（覆盖式）
state/ledger.jsonl              成交事件账本（追加式，哈希链）
state/orders.jsonl              调仓计划日志（追加式，哈希链）
state/snapshots/snapshots.csv   逐日快照（追加式）
state/runs/<run_id>.json        运行清单（幂等判定依据）
logs/paper_trading/<date>.*.json   每日日志（§9）
```

- 无调仓：**不发送邮件**，仅写入 `logs/paper_trading/<date>.signal.json`
- 运行状态码：

```text
PAPER_FILLED       有纸面成交
NO_ACTION          无调仓（目标未变或低于阈值）
STALE_DATA         数据过期（最后完整 4H bar 距执行时间 > 12h）
DUPLICATE_RUN      同一 (as_of, 策略版本, 输入哈希) 重复运行，正常保护
INSUFFICIENT_DATA  MA200 < 200 根 bar 或 TSMOM 6M 日线不足 126 交易日
INVARIANT_FAILED   账本哈希链断裂，拒绝提交
```

邮件主题：

```text
有调仓：[S4 Paper] 调仓建议与纸面成交 - YYYY-MM-DD
异常：  [S4 Paper][ALERT] 运行失败或数据过期 - YYYY-MM-DD
```

---

## 10. 备份（每日，保留 ≥ 90 天）

```text
state/paper_account.json
state/ledger.jsonl
state/orders.jsonl
state/snapshots/snapshots.csv
state/runs/
logs/
```

---

## 11. 故障恢复

详见 `openclaw/runbook.md` §4 / §5。速查：

```text
STALE_DATA         -> 检查 refresh 与直连/代理，补数后重跑 signal
INSUFFICIENT_DATA  -> 确认数据起点足够早（MA200≈34 天；TSMOM 6M≈126 交易日）
INVARIANT_FAILED   -> 禁止手工编辑 ledger；用 snapshots.csv 末行重建账户
DUPLICATE_RUN      -> 正常保护，无需处置
邮件发送失败       -> 只重试邮件，绝不重跑 signal
账本未提交         -> 从上一个完整 snapshot 恢复账户后重跑
```

---

## 12. 部署后验收

```bash
cd openclaw_s4_paper
python -m pytest tests -q                                  # 期望 127 passed
python -B -m openclaw_s4_paper.cli replay --days 90        # M6 四项验收，exit 0
```

`replay` 为历史回放式 90 天验证（真实行情按每日决策 bar 逐日推进 `as_of`，
mock SMTP、socket 出网守卫），产物写入 `reports/m6/`：

```text
m6_daily_status.csv   每日状态
m6_emails.csv         调仓邮件
m6_anomalies.csv      异常记录
m6_summary.json/.md   运行总结
```

验收四项：≥90 天连续运行、无账本断裂、无真实交易所调用、邮件成功率 ≥99%。

---

## 13. 版本升级

```text
1. 修改 config/strategy.json（必须同步更新 strategy_id）
2. 运行 health 确认新配置通过 §12 校验
3. 归档旧 state/ 目录，初始化新账户
4. 新版本首个交易日人工核对：参考权重与分档结果符合预期
```