# OpenClaw S4 + R9 纸面交易系统 — Operation Log

按时间倒序记录操作、发现、修复与验证。

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