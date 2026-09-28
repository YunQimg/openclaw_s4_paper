# OpenClaw S4 + R9 纸面交易系统 — Task State

- 规格：`docs/OpenClaw S4 R9 纸面交易系统 Roadmap V2.md`
- 策略版本：`S4_R9_PAPER_V2`
- 状态：缺口 A–H 全部修复；Milestone 6 已完成并实测通过
- 全量测试：**127 passed / 0 failed**（`python -B -m pytest tests -p no:cacheprovider`）

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

关键指纹：
- `config_fingerprint = d4785a79c5bae8e83036d856e3f9c6a66751f9e2ec21253aea5cafec8596cc77`
- `replay_fingerprint = 3e7b986190a08cadea9d12fa4abab11edc2caad0f0623dcd8a31c28a692fbef0`

### 2.3 复现命令

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"; $env:PYTHONPATH="src"
python -B -m pytest tests -p no:cacheprovider            # 127 passed
python -B -m openclaw_s4_paper.cli replay --days 90      # M6 四项验收, exit 0
```

---

## 3. 仍存在的偏差 / 未覆盖项

- 测试文件数超出 §10 建议清单（`test_cli.py`、`test_replay.py`），见 §1。
- M6 为历史回放口径，非挂钟连续自然日，已在总结中显式披露。
- 纸面交易真实挂钟运行（§3.1 调度 Task A/B/C 的连续生产化）不在本次范围。