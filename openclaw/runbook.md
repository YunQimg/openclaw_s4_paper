# OpenClaw 运行手册（S4 + R9 纸面交易系统）

本手册只描述**调度、权限与运维流程**，不保存任何凭据。

---

## 1. 系统身份

```text
策略：S4 + R9
策略版本：S4_R9_PAPER_V2
账户：paper-s4-r9-001
性质：纸面交易（不连接交易所，不提交真实订单）
```

## 2. 每日运行顺序

```text
UTC 00:00  s4_r9_market_data_refresh   下载市场数据（走代理）
UTC 00:10  s4_r9_signal_check          计算信号 + 模拟成交 + 发调仓邮件
UTC 00:20  s4_r9_paper_settlement      收盘结算，更新净值与回撤
UTC 01:00  s4_r9_health_check          健康检查
```

顺序不可颠倒：结算必须在信号之后，否则净值快照会缺少当次成交。

## 3. 首次部署检查清单

```text
[ ] 依赖安装        pip install -e openclaw_s4_paper
[ ] 策略配置        config/strategy.json 存在且 reference_weights = R9
[ ] 通知配置        config/notification.json 填写 recipient / sender
[ ] SMTP 环境变量   在 Secret Store 中配置 5 个 PAPER_SMTP_* 变量
[ ] 代理可用        PAPER_PROXY 指向可用代理（默认 http://127.0.0.1:7897）
[ ] 市场数据        data/paper_trading/market/{4h,1d,cash_rate_dff}.csv
[ ] 账户初始化      python -m openclaw_s4_paper.cli init
[ ] 试运行          python -m openclaw_s4_paper.cli signal --as-of <过去时刻>（不带 --send-mail）
[ ] 健康检查        python -m openclaw_s4_paper.cli health
```

## 4. 常见故障处置

### 4.1 STALE_DATA（数据超过 12 小时）

```text
现象：signal 任务状态 STALE_DATA，收到 [ALERT] 邮件，账本未提交
处置：
  1. 检查 market_data_refresh 任务是否成功
  2. 检查代理是否可用（下载必须走代理，无代理会硬失败）
  3. 手动补数：python -m openclaw_s4_paper.cli download --out-dir <市场数据目录>
  4. 数据恢复后重跑 signal
```

### 4.2 INSUFFICIENT_DATA

```text
现象：MA200 不足 200 根 bar，或 TSMOΜ 6M 日线不足 126 个交易日
处置：确认市场数据起始时间足够早（4H 需要约 34 天，6M 动量需要约 126 个交易日）
```

### 4.3 INVARIANT_FAILED（账本哈希链断裂）

```text
现象：账本 verify_chain 失败，运行拒绝提交
处置：
  1. 禁止手工编辑 ledger.jsonl
  2. 从最近一次完整 snapshot 恢复账户：
       用 snapshots.csv 最后一行重建 paper_account.json
  3. 若断裂由硬件/磁盘引起，把 ledger.jsonl 备份后人工归档，重建新账本
```

### 4.4 DUPLICATE_RUN

```text
现象：同一 (as_of, strategy_version, input_hash) 重复运行
说明：这是**正常保护**，不会二次成交。无需处置。
```

### 4.5 邮件发送失败

```text
现象：账本已提交，但 email.json 记录 sent=false
处置：**只重试邮件，绝不重跑 signal**（重跑会被 DUPLICATE_RUN 拦下，
      但仍应从 email.json 取回内容手工补发或触发独立邮件重试任务）
```

## 5. 断点恢复（§17.2）

```text
账本已提交、邮件失败  -> 只重试邮件
账本未提交（任何失败）-> 从上一个完整 snapshot 恢复，重新运行
```

恢复步骤：

```text
1. 读取 state/snapshots/snapshots.csv 最后一行
2. 用该行现金与各资产数量重建 state/paper_account.json
3. 确认 state/ledger.jsonl 的 verify_chain 通过
4. 重新运行 signal 任务
```

## 6. 备份

每日备份（保留至少 90 天逐日状态 + 全部交易事件 + 全部邮件摘要）：

```text
state/paper_account.json
state/ledger.jsonl
state/orders.jsonl
state/snapshots/snapshots.csv
state/runs/
logs/
```

## 7. 安全边界（任何情况下不得突破）

```text
不得把 exchange API Key / Secret 写入代码、配置或日志
不得为系统增加下单、撤单、查余额能力
不得把 SMTP 密码写入仓库、JSON 配置或邮件正文
不得授予浏览器自动化或无关目录的 shell 访问
```

`health` 任务内含生产代码静态扫描（`audit.scan_source_tree`），
一旦发现 `place_order` / `cancel_order` / `withdraw` 等能力或交易所 SDK 依赖，
健康检查会失败并列为 problem。

## 8. 版本升级

```text
1. 修改 config/strategy.json（必须同步更新 strategy_id）
2. 运行 health 确认新配置通过 §12 校验
3. 归档旧 state/ 目录，初始化新账户
4. 新版本首个交易日的人工核对：确认参考权重与分档结果符合预期
```