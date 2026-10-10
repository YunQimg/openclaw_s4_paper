# OpenClaw S4 + R9 纸面交易系统 — Task State

- 规格：`docs/OpenClaw S4 R9 纸面交易系统 Roadmap V2.md`
- 策略版本：`S4_R9_PAPER_V2`
- 状态：缺口 A–J 全部修复；Milestone 6 已完成并实测通过；M7 长周期优化提案已登记并**已执行**（S4 再平衡阈值 0.10→0.20，见 §5）；重新冻结完成（新指纹见 §5.7）
- 全量测试：**131 passed / 0 failed**（`python -B -m pytest tests -p no:cacheprovider`）

---

## 1. 缺口修复记录（A–H）

| 缺口 | 规格依据 | 现象 | 修复 | 证据 |
| --- | --- | --- | --- | --- |
| A | §16 | 缺「连续异常计数 + 自动暂停」机制 | 补齐 1 次 WARNING / 连续 2 次 ALERT / 连续 3 次 CRITICAL→暂停纸面成交；`_paused` 返回 `alert`；补测试 import；调整干净运行测试场景 | `tests/test_invariants.py` 新增 7 项 §16 测试（当轮 101 全绿） |
| B | §18 M1 / §11.1 | `python -m openclaw_s4_paper.run_once --help` 空跑失败（无 `__main__`） | `run_once.py` 增 `_build_arg_parser()` / `main(argv)` / `__main__` 入口，暴露 §11.1 五个输入 flag | `test_cli.py::test_run_once_module_help_via_subprocess`、`::test_run_once_main_help_lists_section_11_1_inputs` |
| C | §11.2 | 主任务输出缺 4 个路径键 | `run_once._output_paths()` 返回 `signal_snapshot_path` / `rebalance_proposal_path` / `paper_ledger_update_path` / `email_status_path` | `test_cli.py::test_run_once_output_has_section_11_2_paths` |
| D | §9 | 日志目录偏差 | `StateStore.log_dir` 固定为 `logs/paper_trading/` | `reports/m6/logs/paper_trading/*.json` 实测存在 |
| E | §10 | 测试目录契约（7 个文件名）+ 缺 `data/README.md` | 测试按 §10 重组：`test_end_to_end.py` 用例分发迁入、`test_frozen_parity.py` 并入 `test_strategy_s4.py`，删除被清空的旧文件；新增 `data/README.md` | 下节目录对照 |
| G | §14.5 | 合成数据 9 个场景未全覆盖 | 9 场景（含 `test_no_action_when_target_unchanged` 等）在 §10 文件下全覆盖 | `tests/test_invariants.py`、`tests/test_rebalance.py` 等 |
| H | §13.2 | 缺 `compute_ma200_sign` / `compute_tsmom6m` | `indicators.py` 新增两函数 | `test_strategy_s4.py::test_compute_ma200_sign_matches_low_level`、`::test_compute_tsmom6m_matches_low_level` |
| F | §10 | `state/state/` 双重嵌套残留 | 清理双重嵌套目录；`replay` 子命令 `state_dir` 默认取 `out_dir` 本身（`StateStore` 自建 `state/`+`logs/`），避免复现 | `reports/m6/{state,logs}` 实测无嵌套 |

### §10 测试文件对照

§10 要求的 7 个文件名全部存在：
`test_strategy_s4.py` / `test_listing_state.py` / `test_paper_ledger.py` /
`test_rebalance.py` / `test_invariants.py` / `test_email_redaction.py` /
`test_no_exchange_access.py`。

偏差（如实披露）：另有 `test_cli.py`（B/C 的 CLI 契约）与 `test_replay.py`（M6 验收），
二者为规格新增能力所需的验收载体，超出 §10 建议清单。

---

## 2. Milestone 6（§18）— 90 天纸面运行

**实现口径**：历史回放式验证。以真实行情 `data/paper_trading/market` 按每日决策 bar
逐日推进 `as_of`，每日调用与线上完全相同的 `run_once`，mock SMTP 计数投递，全程
socket 出网守卫。**这是历史回放，不是挂钟意义上的连续 90 个自然日**（已在
`m6_summary.json` 的 `mode_note` 中披露）。

代码产物：
- `src/openclaw_s4_paper/paper_replay.py`（`run_replay` / `MockSmtp` / `_no_network_guard` 等）
- `src/openclaw_s4_paper/cli.py` 新增 `replay` 子命令
- `tests/test_replay.py`（9 项测试）

### 2.1 四项交付（`reports/m6/`）

| 交付项 | 文件 |
| --- | --- |
| 每日状态 | `m6_daily_status.csv`（90 行） |
| 调仓邮件 | `m6_emails.csv`（21 封） |
| 异常记录 | `m6_anomalies.csv`（0 行） |
| 运行总结 | `m6_summary.json` / `m6_summary.md` |

### 2.2 四项验收（实测）

窗口 `2026-06-29 ~ 2026-09-26`，90 天：

- [x] 至少 90 天连续运行（`days_at_least_90` + `days_continuous`）
- [x] 无账本断裂（`ledger 31 行`，哈希链 `OK`）
- [x] 无真实交易所调用（出网尝试被拦截 `0` 次；源码静态扫描 `ok`，17 文件）
- [x] 邮件成功率 ≥ 99%（`21/21 = 100.00%`）

`acceptance_all_ok = true`，命令 exit code `0`。

关键指纹（**M6 运行时口径，阈值 0.10**；M7 变更后的现行指纹见 §5.7）：
- `config_fingerprint = d4785a79c5bae8e83036d856e3f9c6a66751f9e2ec21253aea5cafec8596cc77`
- `replay_fingerprint = 3e7b986190a08cadea9d12fa4abab11edc2caad0f0623dcd8a31c28a692fbef0`

### 2.3 复现命令

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"; $env:PYTHONPATH="src"
python -B -m pytest tests -p no:cacheprovider            # 131 passed
python -B -m openclaw_s4_paper.cli replay --days 90      # M6 四项验收, exit 0
```

---

## 3. 仍存在的偏差 / 未覆盖项

- 测试文件数超出 §10 建议清单（`test_cli.py`、`test_replay.py`），见 §1。
- M6 为历史回放口径，非挂钟连续自然日，已在总结中显式披露。
- 纸面交易真实挂钟运行（§3.1 调度 Task A/B/C 的连续生产化）不在本次范围。

---

## 4. 生产调度修正（缺口 I / J，第三轮）

| 缺口 | 规格依据 | 现象 | 修复 | 证据 |
| --- | --- | --- | --- | --- |
| I | §9 / §10 | `tasks.yaml` 传 `--state-dir state`，因 `StateStore` 自建 `state/`，落盘成 `state/state/ledger.jsonl` | 三个任务统一改 `--state-dir .`，落回 `state/` + `logs/paper_trading/` | `state_store.py` 定义 `state_dir = root/"state"` |
| J | §6.4 / §3.1 | 文档写 UTC 00:10 运行，实测 exit 1：`no execution bar available … exceeds panel end` | `signal --as-of` 固定当日 04:00:01Z；调度后移至 refresh 04:01Z → signal 04:15Z → settle 04:30Z → health 05:00Z | `market_data.py` `next_execution_bar` |

说明：日频决策 bar = 当日首根 4H bar（00:00Z），04:00Z 才收盘，执行 bar 取其后一根（04:00Z，用其开盘价）。
故 refresh 与 signal 都必须排在 04:00Z 之后；`--as-of 04:00:01Z` 与 M6 回放的 `as_of = exec_bar`
解析出的决策/执行 bar 逐位一致，无前视。文档已同步：`openclaw/runbook.md` §2/§4.6、
`DEPLOYMENT.md` §8、`README.md` §6、`op_log.md` 第三轮。

---

## 5. M7 长周期回测（**已执行**）

**口径**：向量回测，窗口 `2017-08-17 ~ 2026-09-20`（9.1 年，4H 面板 `data/v3_spot`），
R9 臂，费率 10bp + 滑点 2bp，现金 `Cash_Rate`，上市规则 `L1`。
入口：`scripts/v5_spot`（`get_panel → strategies.build(S4) → engine.simulate`）。
**本轮回测实验阶段未改动任何线上冻结参数**（`config/strategy.json` 原样）；
最终修改方案的落笔与重新冻结见 §5.7。

### 5.1 全周期结果

| 组合 | CAGR | Sharpe | Calmar | MaxDD |
| --- | --- | --- | --- | --- |
| **S4（MA200+TSMOM6M）** | 95.6% | **1.70** | **1.87** | **-51.1%** |
| B0 从不再平衡（纯持有 R9） | 74.8% | 1.08 | 0.87 | -85.9% |
| B4 年度再平衡 | 107.0% | 1.33 | 1.34 | -79.6% |
| B5 阈值 10%（与 S4 同触发） | 86.8% | 1.25 | 1.07 | -81.5% |

分年度：牛市让渡上行、熊市/震荡大幅超额（2018 +22.0pp、2022 +47.7pp、2025 +35.5pp、2026 +26.0pp）；
各年 S4 回撤均显著小于 B0。

### 5.2 换手与成本

- 成交 **1,561** 腿；同资产 **≤3 日反手 642** 次（BNB 199 / BTC 181 / SOL 155 / ETH 107）
- **成本占毛利 8.38%**（`cost_pct_of_gross_profit`）；平均风险敞口 44.5%

### 5.3 敏感性

| 阈值 | CAGR | Sharpe | Calmar | ≤3日反手 | 成本/毛利 |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 92.6% | 1.68 | 1.80 | 860 | 8.95% |
| **0.10（当前）** | 95.6% | 1.70 | 1.87 | 642 | 8.38% |
| **0.20** | **101.6%** | **1.72** | **1.92** | 488 | 7.91% |
| 0.30 | 95.6% | 1.65 | 1.63 | 220 | 5.53% |
| never（不再平衡） | 36.9% | 0.84 | 0.53 | 0 | 0.01% |

调仓节奏 `weekly`：Calmar 1.42 / 反手 6 / 成本 5.16%（弱于 `daily`）。

### 5.4 结论（修正 M6 短窗判断）

M6 的 90 天窗口显示"择时无贡献"，属**单边牛市下的保险费**，非策略缺陷；
长周期上 S4 显著提升风险调整后收益（Calmar 1.87 vs 0.87）。

- **首选优化**：再平衡阈值 **0.10 → 0.15~0.20**（0.20 全指标最优，0.10→0.30 为平缓平台，抗过拟合）
- **实证否决**：取消再平衡（Calmar 0.53）、改周频（Calmar 1.42）、低阈值 0.05

### 5.5 交付与证据

- `reports/m7/m7_recommendations.md`（建议清单）
- `reports/m7/m7_backtest_2017plus.csv`、`m7_yearly.csv`、`m7_equity.csv`、`m7_threshold_sensitivity.csv`

### 5.6 待决策（已决）

1. 阈值 → **0.20**（WF 4/4 窗口选中，全成本档位占优）
2. 接受指纹变更、V5 解冻 —— **已批准执行（2026-10-10）**
3. 主目标函数 → **Calmar**

### 5.7 最终修改方案（回测决定，**已执行**）

唯一采纳：**S4 再平衡阈值 0.10 → 0.20**（`absolute`）。
其余建议：2 项否决、2 项无法在冻结引擎内验证、1 项未验证。

关键数字（0.10 → 0.20）：CAGR 95.6%→101.6%、Calmar 1.872→1.923、Sharpe 1.699→1.716、
≤3 日反手 637→487、成本/毛利 8.38%→7.91%；代价 **MaxDD -51.1%→-52.9%**。

编辑点（3 处，须同步）：
1. `scripts/v4_spot/strategies.py:32` `DEFAULT_SPECS["S4"]` 0.10 → 0.20
2. `config/strategy.json:33` `rebalance_threshold_absolute` 0.10 → 0.20
3. `src/openclaw_s4_paper/config.py:68` `REBAL_THRESHOLD_ABSOLUTE` 0.10 → 0.20

护栏：`0 < thr < max(ref_weight)`（R9 下 < 0.35）；0.35 档已实测失效
（17 腿 / avg_risky 0.019 / CAGR 8.2%）。

否决：M7-C2 滞回（任一 delta 的 Calmar 均 < 基线 1.9227）、
M7-C3 `risk_multiplier`（S4 走 V3 路径不消费该参数，四档结果逐位相同）。
挂起：M7-C5 漂移带（需新写信号层）、M7-B1 最小交易额（冻结引擎不支持）、M7-B2/B3（生产健壮性）。

#### 执行记录（2026-10-10）

- 3 处编辑点已落笔（均 0.10 → 0.20）：`scripts/v4_spot/strategies.py:32`、
  `config/strategy.json:33`、`src/openclaw_s4_paper/config.py:68`。
  引擎消费 `RebalanceSpec.threshold`（非 `cfg.rebal_threshold`），故 3 处即完整集合。
- 附带修正（非参数项，随本轮一并落地）：
  - `src/openclaw_s4_paper/run_once.py` 无动作日志的 `threshold` 字段由硬编码 `0.10`
    改为读取 `strategy_cfg.rebalance_threshold_absolute`（消除与实际触发口径不一致）。
  - `src/openclaw_s4_paper/rebalance.py` 模块 docstring 与 3 处测试注释同步为 20pp。
- 冻结口径校验：`test_strategy_s4.py::test_s4_r9_matches_frozen_v5_bit_for_bit`
  目标权重逐位一致（`max|diff| = 0`，>3000 决策 bar）通过。
- 全量测试：**131 passed / 0 failed**（历史文档记 127，为漏记，非本轮新增）。
- 重新冻结指纹（`config/strategy.json` 现行口径，阈值 0.20）：
  - `config_fingerprint = d653f7a96201a640a54f3f18881f20c44d7bb903aad11848bc1228d583f1ca0a`（原 `d4785a79…`）
  - `replay_fingerprint = 9c9d114f120df132aa8ae1c2103db1435cb4899ab863a6810aa6ba3391ee409d`（原 `3e7b9861…`）
  （`M6 90 天回放交付物 reports/m6/*` 仍记录旧指纹，属历史产物，未重跑。）

完整方案：`reports/m7/m7_final_plan.md`；证据：`reports/m7/m7_exp1..5_*.csv`。