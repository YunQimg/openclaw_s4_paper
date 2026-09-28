# data/ —— 市场数据说明（Roadmap V2 §4 / §9 / §10 / §13.1）

本目录只保存**只读的市场数据**与说明，不保存任何凭据、密钥或账户信息。
纸面交易系统在运行时**只读取**这里的数据，绝不向交易所发起请求（§14.3）。

## 文件位置

默认市场数据目录为仓库根下的：

```text
data/paper_trading/market/
├─ 4h.csv              # 4 小时 K 线（BTC/ETH/SOL/BNB）
├─ 1d.csv              # 日线 K 线（用于 TSMOM 6M 与 MA200 面板）
└─ cash_rate_dff.csv   # FRED DFF 有效联邦基金利率（现金收益 Cash_Rate）
```

（本 `openclaw_s4_paper/data/` 目录仅存放本说明文件；实际 csv 位于上面的仓库路径。）

## 列契约

### `4h.csv` / `1d.csv`

| 列 | 类型 | 说明 |
|----|------|------|
| `timestamp` | ISO-8601 UTC | bar 的开始时间（UTC，带 `Z`/`+00:00`） |
| `asset` | str | 3 字母资产代码，取值 `BTC` / `ETH` / `SOL` / `BNB` |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价（不变式：`low <= min(open, close)`） |
| `close` | float | 收盘价 |
| `volume` | float | 成交量 |

- 多资产按 `timestamp, asset` 排序后合并到同一文件。
- `4h.csv` 用于信号与执行价格；`1d.csv` 用于 TSMOM 6M 与日频决策 bar。
- 未上市资产在上市前的行应缺失（或价格为 `NaN`），由上市规则（`L1`）处理。

### `cash_rate_dff.csv`

| 列 | 类型 | 说明 |
|----|------|------|
| `timestamp` | ISO-8601 UTC | 日期 |
| `rate_pct` | float | 年化利率，单位为**百分数**（如 `4.33` 表示 4.33%） |

现金收益按 `rate_pct / 100` 折算为日收益，并使用 `shift(1)` 保证时点可见性（无未来泄漏）。

## 生成方式

必须通过显式下载命令生成，网络请求**强制走代理**（§4 / §13.1）：

```powershell
$env:PYTHONPATH = "src"
python -B -m openclaw_s4_paper.cli download            # 使用默认代理 http://127.0.0.1:7897
python -B -m openclaw_s4_paper.cli download --proxy http://127.0.0.1:7897
```

- 代理来源优先级：`--proxy` 参数 > `PAPER_PROXY` 环境变量 > `DEFAULT_PROXY`（`http://127.0.0.1:7897`）。
- 代理缺失或为空时抛 `FetcherUnavailable` 并**硬失败**，不允许静默直连。
- 设置 `PAPER_DISABLE_NETWORK_FETCHES=1` 可在 CI / 离线环境禁用一切网络获取。

## 合成数据（测试用）

单元测试与冒烟测试使用**确定性合成数据**（`tests/conftest.py` 的 `write_market_csv`），
不依赖网络、不依赖 `random`，保证可复现：

- 关闭漂移：`drifts={a: 0.0}`；全上行：`drifts={a: 0.003}`；全下行：`drifts={a: -0.003}`。
- 模拟中途上市：`offsets={"SOL": 250}`（SOL 前 250 个交易日无数据）。

## 不可变约定

- 已落盘的历史数据文件视为**不可变**，重新下载前应先备份。
- 运行清单（`logs/paper_trading/*.json`）会记录 `input_files` 与 `input_data_hash`，
  用于事后复现与审计（§9）。