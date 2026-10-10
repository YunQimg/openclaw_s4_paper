# OpenClaw S4 + R9 纸面交易系统 — Operation Log

按时间倒序记录操作、发现、修复与验证。

---

## 第四轮：M7 长周期优化落地（阈值 0.10 → 0.20，重新冻结）

### 1. 参数变更（用户批准执行）
- **决定**：唯一采纳 S4 再平衡阈值 `0.10 → 0.20`（`absolute` 口径）。依据长周期回测
  （`reports/m7/m7_final_plan.md`，窗口 2017-08-17 ~ 2026-09-20）：
  CAGR 95.6%→101.6%、Calmar 1.872→1.923、Sharpe 1.699→1.716、
  ≤3 日反手 637→487、成本/毛利 8.38%→7.91%；代价 **MaxDD -51.1%→-52.9%**。
  回撤微升为「最深段换位」（全局最深由 2021-23 熊转至 2018 熊，同段 2021-23 仅差 0.78pp），
  非同一段变深；用户已接受该代价。
- **落笔（3 处同步）**：
  - `scripts/v4_spot/strategies.py:32` `DEFAULT_SPECS["S4"]` 0.10 → 0.20（S3/S5/S6/S7 保持 0.10）
  - `config/strategy.json:33` `rebalance_threshold_absolute` 0.10 → 0.20
  - `src/openclaw_s4_paper/config.py:68` `REBAL_THRESHOLD_ABSOLUTE` 0.10 → 0.20
- **引擎口径确认**：`scripts/v4_spot/engine.py` 消费 `RebalanceSpec.threshold`（非 `cfg.rebal_threshold`），
  故上述 3 处即完整集合。

### 2. 附带修正（非参数项）
- `run_once.py` 无动作日志 `threshold` 字段：硬编码 `0.10` → `strategy_cfg.rebalance_threshold_absolute`
  （`_write_no_action_artifacts` 增 `strategy_cfg` 入参），消除日志与实际触发口径不一致。
- `rebalance.py` 模块 docstring、`test_rebalance.py`（含文件 docstring）、
  `test_invariants.py` 场景 7/8、`test_strategy_s4.py` 冻结断言的阈值文案/幅度同步为 20pp / 0.20。

### 3. 验证与重新冻结
- 冻结红线：`test_strategy_s4.py::test_s4_r9_matches_frozen_v5_bit_for_bit`
  目标权重逐位一致（`max|diff| = 0`，>3000 决策 bar）通过 —— parity 机制未坏。
- 全量回归：`python -B -m pytest tests -p no:cacheprovider` → **131 passed / 0 failed**
  （历史文档记 127 为漏记，非本轮新增）。
- 指纹刷新：`config_fingerprint d4785a79… → d653f7a9…`；
  `replay_fingerprint 3e7b9861… → 9c9d114f…`（旧值已用当前代码 + 旧参数复算校验一致）。
- 护栏记录：阈值须满足 `0 < thr < max(ref_weight)`（R9 下 < 0.35）；实测 0.35 档失效
  （17 腿 / avg_risky 0.019 / CAGR 8.2%，建仓永不触发）。
- 未重跑：`reports/m6/*` 90 天回放交付物仍记录旧指纹，属历史产物，本轮未重跑。

---

## 第三轮：生产调度两处真实缺陷（挂钟口径，已实测）

### 1. 缺口 I — `--state-dir state` 复发「缺口 F」双重嵌套
- **现象**：`tasks.yaml` 三个任务都传 `--state-dir state`。因 `StateStore(root)` 会在 root 下
  自建 `state/` 与 `logs/paper_trading/`（`state_store.py`），实际落盘成 `state/state/ledger.jsonl`，
  §9 声明的 `state/ledger.jsonl` 根本不写入。
- **修复**：三个任务统一改为 `--state-dir .`（等价于省略该 flag，缺省回落仓库根的
  `_paths()` 默认值），落盘恢复为 `state/` + `logs/paper_trading/`，与 §9 一致。
- **证据**：`StateStore(root/state)` 的 `state_dir = root/state/state`（源码路径定义）。

### 2. 缺口 J — 文档写的 UTC 00:10 运行必然失败（无执行 bar）
- **现象**：实测（当日 10:18Z）跑 `signal` 报
  `EXECUTION_ERROR: no execution bar available … exceeds panel end`，exit 1。
- **根因**：日频决策 bar = 当日首根 4H bar（00:00Z），要到 **04:00Z 才收盘**；
  `run_once` 取 `signal_time = 最后一个完整 bar`，再要求 `next_execution_bar(+1)` 存在
  （`market_data.py`）。在 00:xx 运行，面板止于前一日 20:00Z bar，+1 根不存在 → 直接失败。
- **修复**：
  - `signal` 的 `--as-of` 固定为「当日 04:00:01Z」（决策 bar 收盘后 1 秒，无前视；
    与 M6 回放的 `as_of = exec_bar` 解析出的决策/执行 bar 完全一致）。
  - 调度整体后移到 04:00Z 之后：refresh 04:01Z → signal 04:15Z → settle 04:30Z → health 05:00Z。
  - **关键联动**：`--as-of 04:00:01Z` 能成立的前提是面板已含 04:00Z 执行 bar；下载器
    `fetch_klines` 以 `end_ms=now` 下载、会带回正在形成的 bar，故 `market_data_refresh`
    必须同样排在 04:00Z 之后（原 00:00Z 太早，会取到未收盘的决策 bar 且缺执行 bar）。
- **同步文档**：`openclaw/runbook.md` §2/§4.6、`DEPLOYMENT.md` §8、`README.md` §6。

---

## 第二轮：缺口 B/C/D/E/G/H 与 Milestone 6

### 1. 缺口 B/C — run_once CLI 与 §11.2 输出契约
- **现象**：`python -m openclaw_s4_paper.run_once --help` 空跑失败；主任务返回值缺 4 个路径键。
- **修复**：`run_once.py` 增 `_build_arg_parser` / `main` / `__main__`（暴露 §11.1 五个输入 flag）；
  `_output_paths()` 补齐 `signal_snapshot_path` / `rebalance_proposal_path` /
  `paper_ledger_update_path` / `email_status_path`。
- **验证**：`test_cli.py` 三例（module help via subprocess、main help flag 列表、输出四路径）。

### 2. 缺口 D — 日志目录
- **现象**：日志未落在 `logs/paper_trading/`。
- **修复**：`StateStore.log_dir = root/logs/paper_trading`。
- **验证**：`reports/m6/logs/paper_trading/` 实测存在 111 个日志文件。

### 3. 缺口 H — §13.2 指标函数缺失
- **现象**：缺 `compute_ma200_sign` / `compute_tsmom6m`。
- **修复**：`indicators.py` 新增两函数（委托底层 `ma`/`ma_sign`/`tsmom_daily`，保持逐位一致）。
- **验证**：`test_strategy_s4.py` 新增两例，`pd.testing.assert_series_equal` 逐位一致。

### 4. 缺口 E/G — §10 测试重组 + §14.5 九场景
- **现象**：测试文件命名与 §10 不一致；缺 `data/README.md`；§14.5 合成数据场景未全覆盖。
- **修复**：`test_end_to_end.py` 用例分发迁入 §10 目标文件、`test_frozen_parity.py` 并入
  `test_strategy_s4.py`，删除被清空的旧文件；补 `test_invariants.py::test_no_action_when_target_unchanged`；
  新增 `data/README.md`（列契约、生成方式、代理优先级、不可变约定）。
- **验证**：§10 七个文件名齐备（另有两项偏差，见 task_state.md §1）。

### 5. 缺口 F 回归修复 — replay 状态目录双嵌套
- **现象**：首版 `replay` 默认 `state_dir = out_dir/"state"`，因 `StateStore` 自建 `state/`+`logs/`，
  产生 `reports/m6/state/state/` 双重嵌套（即缺口 F 的模式复现）。
- **修复**：`cmd_replay` 默认 `state_dir = out_dir`，使落盘为 `reports/m6/state/` 与 `reports/m6/logs/`。
- **验证**：删除旧输出目录后重跑，实测无 `state/state/` 嵌套。

### 6. Milestone 6 — 90 天纸面运行
- **实现**：新增 `paper_replay.py`（历史回放：逐日推进 `as_of` 调 `run_once`，mock SMTP，socket 出网守卫）；
  `cli.py` 增 `replay` 子命令；新增 `tests/test_replay.py`（9 例）。
- **首轮验证**：`tests/test_replay.py` 9 passed；全量 127 passed。
- **真实回放**：`python -B -m openclaw_s4_paper.cli replay --days 90`
  - 窗口 `2026-06-29 ~ 2026-09-26`，90 天；账本 31 行链 OK；邮件 21/21=100.00%；出网拦截 0 次。
  - 四项交付写入 `reports/m6/`；五项验收全 `[x]`，exit 0。

### 7. 清理
- 删除 `openclaw_s4_paper/logs/`、`state/logs/`、`state_test/`、`.pytest_cache/`、`_pytest_out.txt`
  （经用户确认）；保留 `state/` 主体与 `reports/m6` 交付物。

### 8. 全量回归
- `python -B -m pytest tests -p no:cacheprovider` → **127 passed / 0 failed**。

---

## 第一轮（承接上一窗口）

### 缺口 A — §16 连续异常计数与自动暂停
- 实现 1 次 WARNING / 连续 2 次 ALERT / 连续 3 次 CRITICAL→暂停纸面成交。
- 补测试 import（`ANOMALY_CONFIG_CHANGED`、`alert_level`、`update_alert_state`）；
  `_paused` 补 `alert` 字段供上游/测试读取；调整干净运行测试场景确保无异常。
- 新增 7 项 §16 测试，当轮总数 101 全绿。

### 缺口 F — `state/state/` 残留
- 删除 `openclaw_s4_paper/state/state/` 双重嵌套目录，保留父级 `state/`。

### 已知提示
- `cash_insufficient_for_buys` 异常在 R9 策略下可能重复出现，若判据过严可调整阈值。