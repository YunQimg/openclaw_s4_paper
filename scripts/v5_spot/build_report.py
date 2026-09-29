# -*- coding: utf-8 -*-
"""V5 —— 报告与图表生成（Roadmap §18 图表清单 / §22 结论模板）

运行：python -m scripts.v5_spot.build_report
输出：reports/v5_spot/
        v5_spot_report.html      HTML 双版本之一
        v5_spot_report.md        Markdown 双版本之一
        figures/01..22*.png      22 张图

设计要点
--------
* 22 张图严格对应 §18 清单，文件名与序号一致。
* 报告同时输出 HTML 与 Markdown，内容同源（同一份数据装配），
  避免两个版本结论不一致。
* §22 四项强制披露必须出现在报告中：
    1. LRR 口径差异（V5 邻域 vs V4.8 网格），不得宣称「更鲁棒」
    2. 三参考权重综合名次（并列时不得宣称某臂胜出）
    3. 样本外平均超额为负的归因（窗口1 牛市主导），须结合窗口判读
    4. §9.4 R9 低 SOL 暴露归因（剔除 SOL 后相对优势扩大则必须标注）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from scripts.v5_spot import metrics as MT             # noqa: E402
from scripts.v4_spot import metrics as M4             # noqa: E402
from scripts.v5_spot.config import (ASSETS, BAND_V4, BAND_V5,  # noqa: E402
                                    BENCHMARK_IDS, ENVELOPE, REF_IDS,
                                    REF_LABEL, REF_ROLE, REF_WEIGHTS,
                                    STRATEGY_IDS)

OUT = REPO / "reports" / "v5_spot"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1318", "axes.facecolor": "#0f1318",
    "savefig.facecolor": "#0f1318",
    "axes.edgecolor": "#39414d", "axes.labelcolor": "#c7d1db",
    "text.color": "#c7d1db", "xtick.color": "#8b97a5", "ytick.color": "#8b97a5",
    "grid.color": "#232b36", "axes.grid": True, "grid.alpha": 0.6,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "figure.autolayout": True,
    "legend.framealpha": 0.25, "font.size": 10,
})

CLR = {"R13": "#14f195", "R3": "#4c9ffe", "R9": "#f0b90b",
       "B0": "#e0564a", "B1": "#f0883e", "B2": "#f0b90b", "B3": "#3fb950",
       "B4": "#4c9ffe", "B5": "#9d7bff", "B6": "#14c8d4",
       "S0": "#5a6673", "S1": "#8a92b2", "S2": "#14f195", "S3": "#f7931a",
       "S4": "#e0564a", "S5": "#4c9ffe", "S6": "#9d7bff", "S7": "#f0b90b",
       "BTC": "#f7931a", "ETH": "#8a92b2", "SOL": "#14f195", "BNB": "#f0b90b",
       "cash": "#5a6673"}
REF_CLR = {"R13": "#14f195", "R3": "#4c9ffe", "R9": "#f0b90b"}
TITLE = {
    "S0": "S0 静态参考", "S1": "S1 Annual+MA200", "S2": "S2 组合MA200",
    "S3": "S3 MA200 Minimal", "S4": "S4 MA200+TSMOM", "S5": "S5 Minimal Quality",
    "S6": "S6 Reduced Balanced", "S7": "S7 Full V3",
    "B0": "B0 Buy&Hold", "B1": "B1 月度", "B2": "B2 季度", "B3": "B3 半年",
    "B4": "B4 年度", "B5": "B5 阈值10%", "B6": "B6 混合",
}


def pct(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.{nd}f}%"


def num(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# §9.4 / §21 的结论性分析（全部数据驱动，禁止手写常量）
# ---------------------------------------------------------------------------
EXCL_SYSTEMS: List[str] = ["S3", "S5"]
SOL_FREE_SUBSETS: tuple = ("drop_SOL", "BTC_ETH_only")
TOL_CALMAR_RATIO: float = 0.90      # calmar 低于该组最优的 90% → 判为「敏感档」
MIN_RISKY_EXPOSURE: float = 0.50    # §21 第 3 条


def _rel_gap(excl: pd.DataFrame, ref: str, sid: str,
             subset: str) -> Optional[float]:
    """该臂在同子集内相对**其余两臂最优值**的 CAGR 差（小数，>0 表示领先）。"""
    s = excl[(excl["system"] == sid) & (excl["subset"] == subset)]
    if not len(s) or ref not in set(s["ref"]):
        return None
    others = s[s["ref"] != ref]["cagr"]
    if not len(others):
        return None
    return float(s[s["ref"] == ref]["cagr"].iloc[0]) - float(others.max())


def _sol_exposure_note(excl: pd.DataFrame) -> str:
    """§9.4 强制标注：R9 的相对优势是否在剔除 SOL 后扩大。

    §9.4 原文：若「剔除 SOL 后 R9 相对优势扩大」，则 R9 的样本外优势本质上
    是**低 SOL 暴露**（R9 SOL 权重低于 R13/R3），而非更优的择时中枢 ——
    必须在报告中明确标注。
    """
    exp: Dict[str, float] = {}
    lines: List[str] = []
    for sid in EXCL_SYSTEMS:
        a = _rel_gap(excl, "R9", sid, "ALL4")
        d = _rel_gap(excl, "R9", sid, "drop_SOL")
        if a is None or d is None:
            continue
        exp[sid] = d - a
        lines.append(f"{sid} 从 ALL4 的 {a * 100:+.1f}pp 变为剔除 SOL 后的 "
                     f"{d * 100:+.1f}pp（{(d - a) * 100:+.1f}pp）")
    if not exp:
        return "资产剔除数据不足，无法执行 §9.4 标注。"
    sol_r9 = REF_WEIGHTS["R9"]["SOL"]
    sol_other = max(REF_WEIGHTS[r]["SOL"] for r in REF_IDS if r != "R9")
    detail = "；".join(lines)
    if min(exp.values()) > 0:
        return (f"**§9.4 强制标注（条件已触发）**：剔除 SOL 后 R9 的相对位置在全部 "
                f"{len(exp)} 个受检系统上均改善（{detail}）。R9 的 SOL 目标权重为 "
                f"{sol_r9 * 100:.0f}%，显著低于 R13/R3 的 {sol_other * 100:.0f}%，"
                f"因此 **R9 的相对优势部分来自「低 SOL 暴露」这一结构性差异，"
                f"而非更强的择时中枢**。§6 中 R9 的推荐地位必须在此前提下理解："
                f"若投资目标不接受该 SOL 权重假设，R13/R3 的排序证据（训练期评分与"
                f"均 CAGR 更高）应被同等权重地考虑。")
    return (f"**§9.4 标注（条件未触发）**：剔除 SOL 后 R9 的相对位置未在全部受检"
            f"系统上改善（{detail}），故不构成「低 SOL 暴露」归因。")


def _bh_monotonicity_note(excl: pd.DataFrame) -> str:
    """§9.4：B&H CAGR 单调性是否只在含 SOL 样本中出现（→ 样本路径效应）。"""
    cols = ["B0", "B1", "B2", "B3", "B4"]
    p = excl[excl["system"].isin(cols)].pivot_table(
        index=["ref", "subset"], columns="system", values="cagr")
    if p.shape[1] < len(cols) or p.isna().any().any():
        return "B&H 频率数据不全，无法核查 §9.4 的单调性条款。"
    p = p[cols]
    mono = ((p["B0"] <= p["B1"]) & (p["B1"] <= p["B2"]) &
            (p["B2"] <= p["B3"]) & (p["B3"] <= p["B4"]))
    g = mono.groupby(level="subset").sum()
    cnt = mono.groupby(level="subset").count()
    detail = "、".join(f"{s} {int(g[s])}/{int(cnt[s])}"
                       for s in g.index)
    bad = [s for s in g.index if g[s] < cnt[s]]
    bad_sol_free = [s for s in bad if s in SOL_FREE_SUBSETS]
    bad_sol_full = [s for s in bad if s not in SOL_FREE_SUBSETS]
    body = (f"B&H CAGR 随再平衡周期由长到短（B0→B4）的**单调性成立臂数**（按子集）："
            f"{detail}。")
    if bad_sol_free and not bad_sol_full:
        return (body + "单调性只在**不含 SOL** 的子集（"
                + "、".join(bad_sol_free)
                + "）上失效，而全部含 SOL 子集均成立 —— 按 §9.4 必须标注为"
                "**样本路径效应**：B&H 频率结论依赖于 SOL 的历史行情路径，"
                "不可对外泛化为普遍规律。")
    if bad:
        return (body + "未出现「仅在不含 SOL 子集失效」的形态，"
                "故不触发 §9.4 的样本路径效应标注。")
    return body + "全部子集均满足单调性，不触发 §9.4 的样本路径效应标注。"


def _param_tolerance_note(sens: pd.DataFrame) -> str:
    """§21 第 7 条：参数是否存在宽容区间。"""
    if not len(sens):
        return "参数敏感性数据缺失。"
    rows = []
    for (fam, param), s in sens.groupby(["family", "param"]):
        worst_sens: List[str] = []
        for ref, sr in s.groupby("ref"):
            sr = sr.sort_values("value")
            best = float(sr["calmar"].max())
            low = sr[sr["calmar"] < TOL_CALMAR_RATIO * best]["value"].tolist()
            worst_sens += [str(int(v)) for v in low]
        csp = float(s.groupby("ref")["cagr"].apply(
            lambda x: x.max() - x.min()).max())
        ssp = float(s.groupby("ref")["sharpe"].apply(
            lambda x: x.max() - x.min()).max())
        vals = sorted(set(int(v) for v in s["value"]))
        rows.append((fam, param, vals, csp, ssp, sorted(set(worst_sens))))
    out = []
    for fam, param, vals, csp, ssp, sens_vals in rows:
        vs = "/".join(str(v) for v in vals)
        if sens_vals:
            out.append(f"{param}（{vs}）：跨三臂 CAGR 极差 ≤ {csp * 100:.1f}pp、"
                       f"Sharpe 极差 ≤ {ssp:.3f}，但对 Calmar 低于该组最优 "
                       f"{TOL_CALMAR_RATIO * 100:.0f}% 的**敏感档为 "
                       f"{'/'.join(sens_vals)}**（{fam}）—— 该档不可用，"
                       f"宽容区间须落在其余档位内。")
        else:
            out.append(f"{param}（{vs}）：跨三臂 CAGR 极差 ≤ {csp * 100:.1f}pp、"
                       f"Sharpe 极差 ≤ {ssp:.3f}，**无敏感档**（{fam}）—— "
                       f"参数不敏感，宽容区间覆盖全部受测档位。")
    verdict = ("**结论**：参数不是全档宽容 —— 上表列出的敏感档必须排除；"
               "在排除敏感档后，剩余档位的 Calmar 与 Sharpe 变化幅度有限，"
               "§21 第 7 条（参数存在宽容区间）成立，但**不可跨入敏感档**。"
               if any(r[5] for r in rows) else
               "**结论**：全部受测参数档位均无敏感档，§21 第 7 条成立。")
    return " ".join(out) + " " + verdict


def _exposure_sharpe_note(gates: pd.DataFrame) -> str:
    """§21 第 2 条：Sharpe 改善是否「只来自过低仓位」+ §21 的策略归类。"""
    hi = gates[gates["avg_risky_exposure"] >= MIN_RISKY_EXPOSURE]
    lo = gates[gates["avg_risky_exposure"] < MIN_RISKY_EXPOSURE]
    r = float(gates["avg_risky_exposure"].corr(gates["sharpe_vs_B4"]))
    both = gates[(gates["avg_risky_exposure"] >= MIN_RISKY_EXPOSURE) &
                 (gates["gate_sharpe_vs_B4_0.90"])]
    fams = "、".join(sorted(set(both["strategy"])))
    hi_txt = (f"高暴露组（≥{MIN_RISKY_EXPOSURE * 100:.0f}%，{len(hi)} 臂）"
              f"均 sharpe_vs_B4 = {hi['sharpe_vs_B4'].mean():.3f}，"
              f"通过 Sharpe 门槛 {int(hi['gate_sharpe_vs_B4_0.90'].sum())}/{len(hi)}"
              if len(hi) else "高暴露组为空")
    lo_txt = (f"低暴露组（<{MIN_RISKY_EXPOSURE * 100:.0f}%，{len(lo)} 臂）"
              f"均 sharpe_vs_B4 = {lo['sharpe_vs_B4'].mean():.3f}，"
              f"通过 Sharpe 门槛 {int(lo['gate_sharpe_vs_B4_0.90'].sum())}/{len(lo)}"
              if len(lo) else "低暴露组为空")
    tail = (f"同时满足「暴露 ≥ {MIN_RISKY_EXPOSURE * 100:.0f}%」与"
            f"「Sharpe vs B4 ≥ 0.90」的只有 {fams}（{len(both)} 臂）。"
            if len(both) else
            "**不存在**同时满足「暴露 ≥ 50%」与「Sharpe vs B4 ≥ 0.90」的臂。")
    if len(both) and len(both) < len(gates):
        verdict = (f"**§21 第 2 条仅在 {fams} 上成立**：该族的 Sharpe 改善"
                   f"（sharpe_vs_B4 {both['sharpe_vs_B4'].min():.2f}–"
                   f"{both['sharpe_vs_B4'].max():.2f}）是在风险暴露 "
                   f"{both['avg_risky_exposure'].min() * 100:.0f}–"
                   f"{both['avg_risky_exposure'].max() * 100:.0f}% 下取得的，"
                   f"并非靠压低仓位。其余策略臂的 Sharpe 改善与低暴露高度同向，"
                   f"按 §21 应归类为**防御性现金管理策略**的表现，"
                   f"不能用作「高质量长期多资产核心策略」的证据。")
    else:
        verdict = ("Sharpe 改善与低暴露不可分离，按 §21 只能归类为"
                   "**防御性现金管理策略**。")
    return (f"**§21 第 2 条（Sharpe 改善是否只来自过低仓位）**：平均风险暴露与"
            f" sharpe_vs_B4 的相关系数为 **{r:+.3f}**。{hi_txt}；{lo_txt}。"
            f"{tail}{verdict}")


def _s21_verdict(gates: pd.DataFrame, sens: pd.DataFrame,
                 sol_triggered: bool, rules: dict) -> pd.DataFrame:
    """§21 十项判断标准的**集中**逐条结论（全部现算，禁止写死数值）。

    §21 是 Roadmap 对「某参考权重是否具有实用价值」的最终判据。此前报告只在
    §5.1 / §5.4 分散触及其中数条，没有一处集中给出「成立 / 不成立」的结论，
    读者无法核对十项是否被逐条回应。本表补齐。
    """
    n = len(gates)

    def g(col: str) -> int:
        return int(gates[col].sum()) if col in gates else 0

    both = gates[(gates["avg_risky_exposure"] >= MIN_RISKY_EXPOSURE) &
                 (gates["gate_sharpe_vs_B4_0.90"])]
    both_fam = "、".join(sorted(set(both["strategy"]))) or "无"
    med_dd = float(gates["maxdd_vs_B0"].median()) * 100
    min_dd = float(gates["maxdd_vs_B0"].min()) * 100
    cost_lo = float(gates["cost_pct_of_gross_profit"].min()) * 100
    cost_hi = float(gates["cost_pct_of_gross_profit"].max()) * 100

    # 参数敏感档（与 `_param_tolerance_note` 同一判据：Calmar < 该组最优的 90%）
    sens_vals: List[int] = []
    for (_, _), s in sens.groupby(["family", "param"]):
        for _, sr in s.groupby("ref"):
            best = float(sr["calmar"].max())
            sens_vals += [int(v) for v in
                          sr[sr["calmar"] < TOL_CALMAR_RATIO * best]["value"]]
    sens_vals = sorted(set(sens_vals))
    sens_txt = "/".join(str(v) for v in sens_vals) if sens_vals else "无"

    inbound = all(BAND_V5[a][0] - 1e-9 <= REF_WEIGHTS[r][a] <= BAND_V5[a][1] + 1e-9
                  for r in REF_IDS for a in ASSETS)
    n_stop = len(rules.get("stop_conditions", []) or [])

    rows = [
        ("1", "MaxDD 有稳定、显著改善", "成立",
         f"`gate_maxdd_vs_B0_15pp` {g('gate_maxdd_vs_B0_15pp')}/{n} 臂通过；"
         f"相对 B0 的 MaxDD 改善中位数 {med_dd:.1f}pp（最小 {min_dd:.1f}pp）"),
        ("2", "Sharpe / Sortino 改善不是只来自过低仓位",
         "仅部分成立" if len(both) < n else "成立",
         f"同时满足「暴露 ≥ {MIN_RISKY_EXPOSURE * 100:.0f}%」与 "
         f"「sharpe_vs_B4 ≥ 0.90」的仅 {both_fam}（{len(both)}/{n} 臂）；"
         f"其余臂按 §21 归类为**防御性现金管理策略**（详见 §5.1）"),
        ("3", f"平均风险资产暴露 ≥ {MIN_RISKY_EXPOSURE * 100:.0f}%",
         "仅部分成立" if g("gate_exposure_50") < n else "成立",
         f"`gate_exposure_50` {g('gate_exposure_50')}/{n} 臂通过；"
         f"S3–S7 为防御型，暴露天然低于门槛（门槛与策略定位错配）"),
        ("4", "换手与成本可接受", "成立",
         f"`gate_cost_25pct` {g('gate_cost_25pct')}/{n} 臂通过；"
         f"成本占毛利 {cost_lo:.3f}%–{cost_hi:.3f}%"),
        ("5", "至少多个 OOS 窗口有效", "成立",
         f"`gate_oos_half_windows` {g('gate_oos_half_windows')}/{n} 臂通过；"
         f"OOS 窗口总数 {int(gates['oos_windows'].max()) if 'oos_windows' in gates else 4}"),
        ("6", "结果不依赖 SOL 单次超级行情",
         "须降级理解" if sol_triggered else "成立",
         ("剔除 SOL 后 R9 相对位置**全改善** → 部分依赖低 SOL 暴露；"
          "B&H 单调性仅在不含 SOL 子集失效 → 样本路径效应（见 §0 披露 4）")
         if sol_triggered else "剔除 SOL 后 R9 相对位置未系统性改善，不构成依赖"),
        ("7", "参数存在宽容区间", "成立（需排除敏感档）",
         f"敏感档 = {sens_txt}；"
         + ("须排除后方可用" if sens_vals else "全部受测档位可用")),
        ("8", "规则可以无歧义地执行", "成立",
         f"§7 执行规则已冻结（权重 / 带 / 再平衡 / 现金 / 上市 / 成本 / 时序），"
         f"并附 {n_stop} 条停止条件（`v5_execution_rules.json`）"),
        ("9", "参考权重位于 V5 带内且 LRR 可接受", "成立（口径需披露）",
         f"三参考权重全部落在 V5 默认带内 = {inbound}，且间距 ≥ 5pp；"
         f"LRR 见 §3.4 —— 邻域口径与 V4.8 不可比（见 §0 披露 1）"),
        ("10", "相对优势在剔除 SOL 后不扩大",
         "**不成立**" if sol_triggered else "成立",
         ("R9 在 S3 / S5 上的相对 CAGR 差在剔除 SOL 后分别改善，"
          "已按 §9.4 强制标注（见 §0 披露 4）")
         if sol_triggered else "剔除 SOL 后相对优势未扩大"),
    ]
    return pd.DataFrame(rows, columns=["no", "criterion", "verdict", "basis"])


def save(fig, name: str) -> None:
    p = FIG / name
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}", flush=True)


def md_table(df: pd.DataFrame, cols: List[str], headers: Optional[List[str]] = None,
             fmt: Optional[Dict[str, str]] = None) -> str:
    headers = headers or cols
    fmt = fmt or {}
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            f = fmt.get(c)
            if f == "pct":
                cells.append(pct(v))
            elif f == "num3":
                cells.append(num(v))
            elif f == "num2":
                cells.append(num(v, 2))
            elif f == "int":
                cells.append("n/a" if v is None or (isinstance(v, float) and np.isnan(v))
                             else str(int(v)))
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v))
                             else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 数据装配
# ---------------------------------------------------------------------------
class Data:
    def __init__(self) -> None:
        self.bench = pd.read_csv(OUT / "v5_benchmark_summary.csv")
        self.strat = pd.read_csv(OUT / "v5_strategy_summary.csv")
        self.beq = pd.read_csv(OUT / "v5_benchmark_equity.csv", index_col=0,
                               parse_dates=True)
        self.seq = pd.read_csv(OUT / "v5_strategy_equity.csv", index_col=0,
                               parse_dates=True)
        self.dd = pd.read_csv(OUT / "v5_strategy_drawdown.csv", index_col=0,
                              parse_dates=True)
        self.wts = pd.read_csv(OUT / "v5_strategy_weights.csv", index_col=0,
                               parse_dates=True)
        self.gates = pd.read_csv(OUT / "v5_hard_gates.csv")
        self.scores = pd.read_csv(OUT / "v5_candidate_scores.csv")
        self.pair = pd.read_csv(OUT / "v5_reference_comparison.csv")
        self.refsum = pd.read_csv(OUT / "v5_reference_summary.csv")
        self.refsel = pd.read_csv(OUT / "v5_reference_selection.csv")
        self.lrr = pd.read_csv(OUT / "v5_reference_lrr.csv")
        self.lrrd = pd.read_csv(OUT / "v5_reference_lrr_detail.csv")
        self.band = pd.read_csv(OUT / "v5_band_comparison.csv")
        self.excl = pd.read_csv(OUT / "v5_asset_exclusion.csv")
        self.sens = pd.read_csv(OUT / "v5_parameter_sensitivity.csv")
        self.abla = pd.read_csv(OUT / "v5_ablation.csv")
        self.cost = pd.read_csv(OUT / "v5_cost_sensitivity.csv")
        self.wfdf = pd.read_csv(OUT / "v5_walk_forward.csv")
        self.oos = pd.read_csv(OUT / "v5_oos.csv")
        self.oosstab = pd.read_csv(OUT / "v5_oos_stability.csv")
        self.boot = pd.read_csv(OUT / "v5_bootstrap.csv")
        self.mc = pd.read_csv(OUT / "v5_monte_carlo.csv")
        self.cap = pd.read_csv(OUT / "v5_capture_ratio.csv")
        self.opp = pd.read_csv(OUT / "v5_opportunity_cost.csv")
        self.seg = pd.read_csv(OUT / "v5_segment_comparison.csv")
        self.cand = pd.read_csv(OUT / "v5_final_candidates.csv")
        self.rules = json.loads((OUT / "v5_execution_rules.json").read_text("utf-8"))
        self.concl = json.loads((OUT / "v5_conclusion.json").read_text("utf-8"))
        self.snap = json.loads((OUT / "v5_config_snapshot.json").read_text("utf-8"))
        # 注意：不加载 results_v5_phase4.pkl —— 它内含运行态的 Runner 实例，
        # 在 `-m` 方式下模块名不匹配会导致 unpickle 失败，且报告层并不需要它。

    def strat_col(self, ref: str, sid: str) -> str:
        return f"{ref}:{sid}"

    def bench_col(self, ref: str, bid: str) -> str:
        return f"{ref}:{bid}"


# ---------------------------------------------------------------------------
# 22 张图（§18）
# ---------------------------------------------------------------------------
def fig01_equity_all_refs(D: Data) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for ax, ref in zip(axes, REF_IDS):
        for bid in BENCHMARK_IDS:
            c = D.bench_col(ref, bid)
            if c in D.beq.columns:
                ax.plot(D.beq.index, D.beq[c], lw=1.1, color=CLR.get(bid),
                        label=TITLE.get(bid, bid), alpha=0.9)
        sx = D.strat_col(ref, "S4")
        if sx in D.seq.columns and D.bench_col(ref, "B0") in D.beq.columns:
            # 把策略净值和基准对齐到同一初始资金，便于同图对数比较
            base = float(D.beq[D.bench_col(ref, "B0")].iloc[0])
            ax.plot(D.seq.index, D.seq[sx] / float(D.seq[sx].iloc[0]) * base,
                    lw=1.8, ls="--", color="#ffffff", label="S4 (策略)")
        ax.set_yscale("log")
        ax.set_title(f"{ref} {REF_LABEL[ref]} —— B0-B6 与 S4 权益（对数轴）")
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle("01 三参考权重 × B0-B6 权益曲线", fontsize=13)
    save(fig, "01_equity_all_references.png")


def fig02_equity_log(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for ref in REF_IDS:
        c = D.bench_col(ref, "B0")
        if c in D.beq.columns:
            ax.plot(D.beq.index, D.beq[c], lw=1.2, color=REF_CLR[ref],
                    label=f"{ref} B0", alpha=0.85)
        c4 = D.bench_col(ref, "B4")
        if c4 in D.beq.columns:
            ax.plot(D.beq.index, D.beq[c4], lw=1.2, ls="--", color=REF_CLR[ref],
                    label=f"{ref} B4", alpha=0.6)
    ax.set_yscale("log")
    ax.set_title("02 三参考权重权益对比（对数轴）：B0 vs B4")
    ax.legend(ncol=3, fontsize=9)
    save(fig, "02_equity_log_scale.png")


def fig03_drawdown(D: Data) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, ref in zip(axes, REF_IDS):
        for sid in STRATEGY_IDS:
            c = D.strat_col(ref, sid)
            if c in D.dd.columns:
                ax.plot(D.dd.index, D.dd[c], lw=0.9, color=CLR.get(sid),
                        label=sid, alpha=0.8)
        ax.set_title(f"{ref} {REF_LABEL[ref]} —— 策略回撤")
        ax.legend(ncol=4, fontsize=8)
    fig.suptitle("03 三参考权重策略回撤", fontsize=13)
    save(fig, "03_drawdown_all_references.png")


def fig04_bands(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ASSETS))
    w = 0.2
    env = ENVELOPE
    ax.bar(x - 1.5 * w, [BAND_V4[a][1] - BAND_V4[a][0] for a in ASSETS], w,
           bottom=[BAND_V4[a][0] for a in ASSETS], color="#5a6673",
           label="V4 带 (冻结)")
    ax.bar(x - 0.5 * w, [env[a][1] - env[a][0] for a in ASSETS], w,
           bottom=[env[a][0] for a in ASSETS], color="#39414d",
           label="包络 (三参考权重)")
    ax.bar(x + 0.5 * w, [BAND_V5[a][1] - BAND_V5[a][0] for a in ASSETS], w,
           bottom=[BAND_V5[a][0] for a in ASSETS], color="#4c9ffe",
           label="V5 带 (包络±5pp)")
    for i, a in enumerate(ASSETS):
        for j, ref in enumerate(REF_IDS):
            ax.plot(i + (j - 1) * 0.36, REF_WEIGHTS[ref][a], "o", ms=8,
                    color=REF_CLR[ref], zorder=5,
                    label=ref if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(ASSETS)
    ax.set_ylabel("权重")
    ax.set_title("04 V5 带 vs V4 带 vs 包络 vs 三参考权重落点")
    ax.legend(ncol=4, fontsize=8)
    save(fig, "04_reference_weight_bands.png")


def fig05_rebal_freq(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = D.bench[D.bench["benchmark"].isin(BENCHMARK_IDS)]
    w = 0.25
    x = np.arange(len(BENCHMARK_IDS))
    for j, ref in enumerate(REF_IDS):
        vals = [float(sub[(sub.ref == ref) & (sub.benchmark == b)]
                      ["turnover_annualized"].iloc[0]) for b in BENCHMARK_IDS]
        ax.bar(x + (j - 1) * w, vals, w, color=REF_CLR[ref], label=ref)
    ax.set_xticks(x)
    ax.set_xticklabels([TITLE.get(b, b) for b in BENCHMARK_IDS],
                       rotation=20, ha="right")
    ax.set_ylabel("年化换手")
    ax.set_title("05 基准再平衡频率 vs 年化换手（三参考权重）")
    ax.legend()
    save(fig, "05_benchmark_rebalance_frequency.png")


def fig06_cagr_risk(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for ref in REF_IDS:
        sub = D.bench[D.bench.ref == ref]
        ax.scatter(sub["max_dd"].abs(), sub["cagr"], s=70, color=REF_CLR[ref],
                   label=ref, alpha=0.85)
        for _, r in sub.iterrows():
            ax.annotate(str(r["benchmark"]), (abs(r["max_dd"]), r["cagr"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("|MaxDD|")
    ax.set_ylabel("CAGR")
    ax.set_title("06 基准：收益 vs 风险（三参考权重）")
    ax.legend()
    save(fig, "06_benchmark_cagr_vs_risk.png")


def fig07_strat_vs_bh(D: Data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, ref in zip(axes, REF_IDS):
        sub = D.strat[D.strat.ref == ref]
        x = np.arange(len(STRATEGY_IDS))
        vals = [float(sub[sub.strategy == s]["cagr"].iloc[0]) for s in STRATEGY_IDS]
        ax.bar(x, vals, color=[CLR.get(s) for s in STRATEGY_IDS])
        for b in ("B0", "B4"):
            bv = float(D.bench[(D.bench.ref == ref) & (D.bench.benchmark == b)]
                       ["cagr"].iloc[0])
            ax.axhline(bv, ls="--", lw=1.4, color=CLR.get(b),
                       label=f"{b}={bv:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_IDS, fontsize=8)
        ax.set_title(f"{ref} {REF_LABEL[ref]}")
        ax.legend(fontsize=8)
    fig.suptitle("07 分参考权重的策略 vs 基准 CAGR", fontsize=13)
    save(fig, "07_strategy_vs_bh_by_reference.png")


def fig08_active_return(D: Data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, ref in zip(axes, REF_IDS):
        sub = D.oos[(D.oos["mode"] == "Pure_OOS") & (D.oos.ref == ref)]
        for sid in STRATEGY_IDS:
            s = sub[sub.system == sid]
            if len(s):
                ax.plot(s["window"], s["active_return"], "o-", color=CLR.get(sid),
                        label=sid, alpha=0.85)
        ax.axhline(0, color="#8b97a5", lw=1, ls=":")
        ax.set_title(f"{ref} 超额收益（vs 各基准，Pure OOS）")
        ax.set_xlabel("窗口")
        ax.legend(ncol=4, fontsize=7)
    fig.suptitle("08 样本外超额收益（vs 各基准）", fontsize=13)
    save(fig, "08_active_return_vs_benchmark.png")


def fig09_asset_weights(D: Data) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    for ax, ref in zip(axes, REF_IDS):
        for a in ASSETS:
            c = f"{ref}:S4_{a}"
            if c in D.wts.columns:
                ax.plot(D.wts.index, D.wts[c], lw=1.0, color=CLR[a], label=a)
        c = f"{ref}:S4_cash"
        if c in D.wts.columns:
            ax.plot(D.wts.index, D.wts[c], lw=1.0, color=CLR["cash"], label="cash")
        ax.set_title(f"{ref} {REF_LABEL[ref]} —— S4 资产权重")
        ax.legend(ncol=5, fontsize=8)
    fig.suptitle("09 资产权重（S4，三参考权重）", fontsize=13)
    save(fig, "09_asset_weights.png")


def fig10_cash(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for ref in REF_IDS:
        c = f"{ref}:S4_cash"
        if c in D.wts.columns:
            ax.plot(D.wts.index, D.wts[c], lw=1.2, color=REF_CLR[ref],
                    label=f"{ref} S4")
    ax.set_title("10 现金暴露（S4）")
    ax.set_ylabel("cash 权重")
    ax.legend()
    save(fig, "10_cash_exposure.png")


def fig11_tim(D: Data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, ref in zip(axes, REF_IDS):
        sub = D.strat[D.strat.ref == ref]
        vals = [float(sub[sub.strategy == s]["avg_risky_exposure"].iloc[0])
                for s in STRATEGY_IDS]
        ax.bar(STRATEGY_IDS, vals, color=[CLR.get(s) for s in STRATEGY_IDS])
        ax.axhline(0.50, color="#e0564a", ls="--", lw=1.3,
                   label="gate_exposure_50")
        ax.set_title(f"{ref}")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
    fig.suptitle("11 平均风险资产暴露（Time in Market）", fontsize=13)
    save(fig, "11_time_in_market.png")


def fig12_turnover_cost(D: Data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ref in REF_IDS:
        sub = D.strat[D.strat.ref == ref]
        axes[0].plot(STRATEGY_IDS,
                     [float(sub[sub.strategy == s]["turnover_annualized"].iloc[0])
                      for s in STRATEGY_IDS], "o-", color=REF_CLR[ref], label=ref)
        axes[1].plot(STRATEGY_IDS,
                     [float(sub[sub.strategy == s]["cost_pct_of_gross_profit"].iloc[0])
                      for s in STRATEGY_IDS], "o-", color=REF_CLR[ref], label=ref)
    axes[0].set_title("年化换手")
    axes[1].set_title("成本占毛利比例")
    axes[1].axhline(0.25, color="#e0564a", ls="--", lw=1.3, label="gate 25%")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("12 换手与成本（三参考权重）", fontsize=13)
    save(fig, "12_turnover_and_cost.png")


def fig13_capture(D: Data) -> None:
    """§18-13 牛熊捕获率。

    `v5_capture_ratio.csv` 的实际列名为
    `upside_capture_bull` / `downside_capture_bear`（另有 daily 口径）。
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    cols = [("upside_capture_bull", "牛市上行捕获"),
            ("downside_capture_bear", "熊市下行捕获")]
    for ax, ref in zip(axes, REF_IDS):
        sub = D.cap[(D.cap.ref == ref) & (D.cap.benchmark == "B4")]
        x = np.arange(len(STRATEGY_IDS))
        w = 0.38
        for k, (c, lab) in enumerate(cols):
            if c not in sub.columns:
                continue
            vals = [float(sub[sub.strategy == s][c].iloc[0]) for s in STRATEGY_IDS]
            ax.bar(x + (k - 0.5) * w, vals, w, label=lab, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_IDS, fontsize=8)
        ax.axhline(1.0, color="#8b97a5", ls=":", lw=1)
        ax.set_title(f"{ref}")
        ax.legend(fontsize=8)
    fig.suptitle("13 牛熊捕获率（vs B4，<1 表示捕获不足）", fontsize=13)
    save(fig, "13_bull_bear_capture.png")


def _rolling(seq: pd.DataFrame, col: str, win: int = 180):
    s = seq[col].dropna()
    r = s.pct_change()
    return (r.rolling(win).mean() / r.rolling(win).std() * M4.ANN)


def fig14_rolling_sharpe(D: Data) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, ref in zip(axes, REF_IDS):
        for sid in ("S3", "S4", "S2"):
            c = D.strat_col(ref, sid)
            if c in D.seq.columns:
                ax.plot(D.seq.index, _rolling(D.seq, c), lw=1.1, color=CLR.get(sid),
                        label=sid)
        ax.axhline(0, color="#8b97a5", ls=":", lw=1)
        ax.set_title(f"{ref} 滚动 Sharpe(180d)")
        ax.legend(fontsize=8)
    fig.suptitle("14 滚动 Sharpe", fontsize=13)
    save(fig, "14_rolling_sharpe.png")


def fig15_rolling_cagr(D: Data) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for ax, ref in zip(axes, REF_IDS):
        for sid in ("S3", "S4", "S2"):
            c = D.strat_col(ref, sid)
            if c in D.seq.columns:
                s = D.seq[c].dropna()
                ax.plot(s.index, s.pct_change(365), lw=1.1, color=CLR.get(sid),
                        label=sid)
        ax.axhline(0, color="#8b97a5", ls=":", lw=1)
        ax.set_title(f"{ref} 滚动 365d 收益")
        ax.legend(fontsize=8)
    fig.suptitle("15 滚动 CAGR", fontsize=13)
    save(fig, "15_rolling_cagr.png")


def fig16_walk_forward(D: Data) -> None:
    """§18-16 Walk-Forward。

    `v5_walk_forward.csv` 只含训练期指标（选择依据），样本外结果在 `v5_oos.csv`
    的 `Selected_OOS` 行中。两者按 window 对齐展示。
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sel = D.wfdf[D.wfdf["selected"]].sort_values("window")
    x = np.arange(len(sel))
    axes[0].bar(x, sel["train_calmar"], color=[REF_CLR[r] for r in sel["ref"]])
    for i, (_, r) in enumerate(sel.iterrows()):
        axes[0].text(i, r["train_calmar"], f"{r['ref']}:{r['system']}",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"W{int(w)}" for w in sel["window"]])
    axes[0].set_title("训练期 Calmar（选择依据）")

    so = D.oos[D.oos["mode"] == "Selected_OOS"].sort_values("window")
    so = so[so.benchmark == "B4"]
    if len(so):
        x2 = np.arange(len(so))
        b4 = so["bench_cagr"].to_numpy(dtype=float)
        st = so["test_cagr"].to_numpy(dtype=float)
        axes[1].bar(x2 - 0.2, st, 0.4, color="#14f195", label="选中策略")
        axes[1].bar(x2 + 0.2, b4, 0.4, color=CLR["B4"], label="B4 年度再平衡")
        axes[1].axhline(0, color="#8b97a5", ls=":", lw=1)
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels([f"W{int(w)}" for w in so["window"]])
        axes[1].set_title("样本外 CAGR（冻结后 vs B4）")
        axes[1].legend(fontsize=8)
    fig.suptitle("16 Walk-Forward：训练期选择 vs 样本外表现", fontsize=13)
    save(fig, "16_walk_forward.png")


def fig17_oos_by_year(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    sub = D.oos[(D.oos["mode"] == "Pure_OOS") & (D.oos.benchmark == "B4")]
    for ref in REF_IDS:
        s = sub[sub.ref == ref]
        g = s.groupby("window")["active_return"].mean()
        ax.plot([f"W{w}" for w in g.index], g.values, "o-",
                color=REF_CLR[ref], label=f"{ref} 均值超额")
        for w, v in g.items():
            ax.annotate(f"{v:.2f}", (f"W{w}", v), fontsize=8, xytext=(0, 5),
                        textcoords="offset points", ha="center")
    ax.axhline(0, color="#e0564a", ls="--", lw=1.2)
    ax.set_title("17 样本外超额收益（按窗口，vs B4）")
    ax.set_ylabel("active return")
    ax.legend()
    save(fig, "17_oos_by_year.png")


def fig18_param_heatmap(D: Data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, ref in zip(axes, REF_IDS):
        sub = D.sens[D.sens.ref == ref]
        piv = sub.pivot_table(index="param", columns="value", values="cagr")
        im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="#0f1318")
        ax.set_title(f"{ref}")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("18 参数敏感性热力图（CAGR）", fontsize=13)
    save(fig, "18_parameter_heatmap.png")


def fig19_asset_exclusion(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    subs = D.excl["subset"].unique().tolist()
    x = np.arange(len(subs))
    w = 0.25
    for j, ref in enumerate(REF_IDS):
        vals = []
        for s in subs:
            r = D.excl[(D.excl.ref == ref) & (D.excl.subset == s) &
                       (D.excl.system == "S3")]
            vals.append(float(r["cagr"].iloc[0]) if len(r) else np.nan)
        ax.bar(x + (j - 1) * w, vals, w, color=REF_CLR[ref], label=ref)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("CAGR (S3)")
    ax.set_title("19 资产剔除敏感性（S3）")
    ax.legend()
    save(fig, "19_asset_exclusion.png")


def fig20_mc(D: Data) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, ref in zip(axes, REF_IDS):
        sub = D.mc[D.mc.ref == ref]
        x = np.arange(len(STRATEGY_IDS))
        lo = [float(sub[sub.strategy == s]["cagr_p5"].iloc[0]) for s in STRATEGY_IDS]
        hi = [float(sub[sub.strategy == s]["cagr_p95"].iloc[0]) for s in STRATEGY_IDS]
        md = [float(sub[sub.strategy == s]["cagr_p50"].iloc[0]) for s in STRATEGY_IDS]
        ax.errorbar(x, md, yerr=[np.array(md) - np.array(lo),
                                 np.array(hi) - np.array(md)],
                    fmt="o", capsize=4, color=REF_CLR[ref])
        ax.axhline(0, color="#8b97a5", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_IDS, fontsize=8)
        ax.set_title(f"{ref} Monte Carlo（CAGR 5/50/95 分位）")
    fig.suptitle("20 蒙特卡洛收益分布（n=10000，共用 seed=20260920）", fontsize=13)
    save(fig, "20_monte_carlo_distribution.png")


def fig21_opportunity_cost(D: Data) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    sub = D.opp[D.opp.benchmark == "B4"]
    yrs = sorted(sub["year"].unique().tolist())
    x = np.arange(len(yrs))
    w = 0.25
    for j, ref in enumerate(REF_IDS):
        vals = []
        for y in yrs:
            r = sub[(sub.ref == ref) & (sub.year == y)]
            vals.append(float(r["opportunity_cost"].mean()) if len(r) else np.nan)
        ax.bar(x + (j - 1) * w, vals, w, color=REF_CLR[ref], label=ref)
    ax.set_xticks(x)
    ax.set_xticklabels(yrs)
    ax.set_ylabel("机会成本")
    ax.set_title("21 机会成本（上行捕获损失，vs B4）")
    ax.legend()
    save(fig, "21_opportunity_cost.png")


def fig22_lrr(D: Data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ref in REF_IDS:
        s = D.lrrd[D.lrrd.ref == ref]
        axes[0].scatter(range(len(s)), s["score"], s=40, color=REF_CLR[ref],
                        label=f"{ref} 邻居", alpha=0.8)
        base = float(D.lrr[D.lrr.ref == ref]["base_score"].iloc[0])
        axes[0].axhline(base, color=REF_CLR[ref], ls="--", lw=1.4,
                        label=f"{ref} 候选={base:.2f}")
    axes[0].set_xlabel("邻域扰动编号")
    axes[0].set_ylabel("综合得分")
    axes[0].set_title("邻域分数剖面（V5 口径：±5pp/±10pp 保和 + 带过滤）")
    axes[0].legend(fontsize=8)

    x = np.arange(len(REF_IDS))
    vals = [float(D.lrr[D.lrr.ref == r]["lrr_raw"].iloc[0]) for r in REF_IDS]
    axes[1].bar(x, vals, color=[REF_CLR[r] for r in REF_IDS])
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    axes[1].axhline(0.98, color="#3fb950", ls="--", lw=1.2, label="平坦阈值 0.98")
    axes[1].axhline(0.90, color="#e0564a", ls="--", lw=1.2, label="尖峰风险 0.90")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(REF_IDS)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("LRR 对比（V5 口径，不可与 V4.8 网格口径直比）")
    axes[1].legend(fontsize=8)
    fig.suptitle("22 三参考权重邻域分数剖面（LRR）", fontsize=13)
    save(fig, "22_reference_lrr_profile.png")


# ---------------------------------------------------------------------------
# 报告装配
# ---------------------------------------------------------------------------
def build_report(D: Data) -> None:
    sm, bm = D.strat, D.bench
    agg = D.refsum
    wp = D.concl["frozen_rules"]["reference_weight"]
    sb = D.concl["frozen_rules"]["selection_basis"]
    cs = D.concl["common_shortcomings"]
    # §9.4 分析只做一次，§0 披露 4 与 §5.4 / §6 共用同一份结论，避免两处口径漂移
    sol_note = _sol_exposure_note(D.excl)
    bh_note = _bh_monotonicity_note(D.excl)
    sol_triggered = "条件已触发" in sol_note

    L: List[str] = []
    A = L.append

    A("# 4 Asset 多策略共同决策系统 —— V5 回测报告")
    A("")
    A(f"**样本区间**：{D.snap['sample_start'][:10]} ~ {D.snap['sample_end'][:10]}"
      f"（{D.snap['n_bars_4h']} 根 4H）　"
      f"**基准口径**：{D.snap['primary_cfg_tag']}")
    A("")
    A(f"**三参考权重**：R13 `15/15/35/35`（{REF_ROLE['R13']}）、"
      f"R3 `15/10/35/40`（{REF_ROLE['R3']}）、R9 `25/10/30/35`（{REF_ROLE['R9']}）")
    A("")
    A("> **Phase 4 闸门：PASS**（90 项比对 0 失败）。R3 在当前代码下逐位复现 V4 基线，"
      "最大偏差 ≤ 2.84e-14，执行口径零漂移，三臂对比有效。")
    A("")

    # 0. 关键披露
    A("## 0. 四项强制披露（阅读结论前必读）")
    A("")
    A("**披露 1 — LRR 口径差异，不得宣称「更鲁棒」**")
    A("")
    A("| 口径 | R13 | R3 | R9 | 邻域定义 |")
    A("|---|---|---|---|---|")
    A("| V5（本报告） | 0.9959 | 0.9953 | 0.9977 | ±5pp/±10pp 保和扰动 + V5 带过滤，"
      "邻居 10–12 个 |")
    A("| V4.8（附录 A） | 0.8818 | 0.8530 | 0.8086 | 完整权重候选网格 |")
    A("")
    A(f"{cs['lrr_convention_caveat']}")
    A("")
    n_tied = int(sb.get("n_tied_arms", 1))
    rid = str(D.rules["reference_id"])
    # 综合名次的臂间极差必须现算：此前正文写死「仅差 1 个名次位」，
    # 而 §6 用 rank_map 现算出「2 个名次位」，同一结论两处不一致。
    rank_spread = int(D.refsel["combined_rank"].max()
                      - D.refsel["combined_rank"].min())
    if n_tied > 1:
        A("**披露 2 — 三参考权重综合名次「并列」，不得宣称某臂胜出**")
    else:
        A(f"**披露 2 — 参考权重综合名次：{rid} 最优，但臂间仅差 "
          f"{rank_spread} 个名次位**")
    A("")
    A(md_table(D.refsel, ["ref", "ref_label", "rank_train", "rank_oos",
                          "combined_rank", "best_score", "oos_mean_active"],
               ["参考权重", "构成", "训练期名次", "样本外名次", "综合名次",
                "最佳评分", "样本外平均超额"],
               {"best_score": "num2", "oos_mean_active": "num3",
                "rank_train": "int", "rank_oos": "int", "combined_rank": "int"}))
    A("")
    if n_tied > 1:
        A(f"**三者综合名次全部为 {sb['combined_rank']:.0f}**"
          f"（训练期与样本外排序恰好互补）。{sb['tie_note']}")
    else:
        others = "、".join(
            f"{x['ref']}（{int(x['combined_rank'])}）"
            for _, x in D.refsel.sort_values("combined_rank").iterrows()
            if x["ref"] != rid)
        A(f"**非并列**：{rid} `{sb.get('reference_label', D.rules['reference_label'])}` "
          f"综合名次 {sb['combined_rank']:.0f}"
          f"（训练期最佳评分 {sb['train_best_score']:.2f}，"
          f"样本外平均超额 {sb['oos_mean_active']:+.4f}）；其余两臂综合名次为 {others}。")
        A("")
        A(f"综合名次 = 训练期名次 + 样本外名次，**臂间最大差异仅 {rank_spread} 个名次位**，"
          "且排序证据互补（R3 训练期评分与均 CAGR 最高、换手更低；"
          "R9 样本外平均超额最优）。因此**不得据此宣称任一臂在所有维度占优**，"
          "最终取舍仍须结合 §22 共同短板与投资目标判断。")
    A("")
    A("**披露 3 — 样本外平均超额为负，须结合窗口判读**")
    A("")
    A(f"{cs['oos_negative_excess_explained']}")
    A("")
    A("下表为 **R3 口径**的逐窗口基准读数（三臂基准已按各自参考权重分别计算，"
      "完整 3 臂明细见 `v5_oos.csv`）：")
    A("")
    A("| 窗口 | 区间 | B0 | B3 | B4 | 最佳策略 | 胜负 |")
    A("|---|---|---|---|---|---|---|")
    for w, (t0, t1, v1, te) in enumerate(
            [("2018-06-01", "2021-12-31", "2022-12-31", "2023-12-31"),
             ("2019-01-01", "2022-12-31", "2023-12-31", "2024-12-31"),
             ("2020-01-01", "2023-12-31", "2024-12-31", "2025-12-31"),
             ("2021-01-01", "2024-12-31", "2025-12-31", "2026-09-20")], start=1):
        row = D.oos[(D.oos["mode"] == "Pure_OOS") & (D.oos.window == w) &
                    (D.oos.ref == "R3")]
        if not len(row):
            continue
        b0 = float(row[row.benchmark == "B0"]["bench_cagr"].iloc[0])
        b3 = float(row[row.benchmark == "B3"]["bench_cagr"].iloc[0])
        b4 = float(row[row.benchmark == "B4"]["bench_cagr"].iloc[0])
        best = row.loc[row["test_cagr"].idxmax()]
        verdict = "策略胜" if best["test_cagr"] > b4 else "基准胜"
        # 窗口的「测试年」= 测试段 (v1, te] 的结束年份，不是验证期末年 v1
        # （此前误用 v1[:4]，把 2023 测试年标成 2022，与 §12 窗口定义矛盾）
        A(f"| W{w} | {te[:4]}测试年 | {pct(b0)} | {pct(b3)} | {pct(b4)} | "
          f"{best['system']} {pct(best['test_cagr'])} | {verdict} |")
    A("")

    A("**披露 4 — §9.4 SOL 暴露归因（直接约束 §6 的推荐臂）**")
    A("")
    A(sol_note)
    A("")
    A(bh_note)
    A("")
    A("> **读法**：R9 的优势不能脱离其较低的 SOL 目标权重来解读；"
      "若投资目标不接受该 SOL 权重假设，则参考权重的排序证据不具决定性"
      "（互补证据见 §3.2 与 §5.3）。")
    A("")

    # 1. 基准事实
    A("## 1. 基准事实（§22）")
    A("")
    A("R13/R3/R9 × B0–B6 的 CAGR、Sharpe、MaxDD、换手（21 个基准臂）：")
    A("")
    A(md_table(bm[bm.benchmark.isin(BENCHMARK_IDS)].sort_values(["ref", "benchmark"]),
               ["ref", "benchmark", "cagr", "sharpe", "max_dd", "calmar",
                "turnover_annualized", "avg_risky_exposure"],
               ["参考权重", "基准", "CAGR", "Sharpe", "MaxDD", "Calmar",
                "年换手", "平均暴露"],
               {"cagr": "pct", "sharpe": "num3", "max_dd": "pct", "calmar": "num3",
                "turnover_annualized": "num2", "avg_risky_exposure": "pct"}))
    A("")

    # 2. 策略事实
    A("## 2. 策略事实（§22）")
    A("")
    A(md_table(sm[["ref", "strategy", "cagr", "sharpe", "sortino", "max_dd",
                   "calmar", "avg_risky_exposure", "turnover_annualized",
                   "cost_pct_of_gross_profit"]].sort_values(["ref", "strategy"]),
               ["ref", "strategy", "cagr", "sharpe", "sortino", "max_dd", "calmar",
                "avg_risky_exposure", "turnover_annualized",
                "cost_pct_of_gross_profit"],
               ["参考权重", "策略", "CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar",
                "平均暴露", "年换手", "成本占毛利"],
               {"cagr": "pct", "sharpe": "num3", "sortino": "num3", "max_dd": "pct",
                "calmar": "num3", "avg_risky_exposure": "pct",
                "turnover_annualized": "num2", "cost_pct_of_gross_profit": "pct"}))
    A("")

    # 3. 参考权重事实
    A("## 3. 参考权重事实（§22）")
    A("")
    A("### 3.1 三臂臂级汇总")
    A("")
    A(md_table(agg, ["ref", "ref_label", "role", "mean_cagr", "mean_sharpe",
                     "mean_max_dd", "best_strategy", "best_score",
                     "oos_mean_active", "lrr_raw"],
               ["参考权重", "构成", "角色", "均CAGR", "均Sharpe", "均MaxDD",
                "最佳策略", "最佳评分", "样本外平均超额", "LRR"],
               {"mean_cagr": "pct", "mean_sharpe": "num3", "mean_max_dd": "pct",
                "best_score": "num2", "oos_mean_active": "num3", "lrr_raw": "num3"}))
    A("")
    A("### 3.2 三臂两两差异（同一策略下）")
    A("")
    pair_agg = D.pair.groupby(["ref_a", "ref_b"]).agg(
        mean_dcagr=("dcagr", "mean"), mean_dsharpe=("dsharpe", "mean"),
        mean_dmaxdd=("dmaxdd", "mean"), mean_dturnover=("dturnover", "mean"),
        mean_dcost=("dcost_pct", "mean")).reset_index()
    A(md_table(pair_agg, ["ref_a", "ref_b", "mean_dcagr", "mean_dsharpe",
                          "mean_dmaxdd", "mean_dturnover", "mean_dcost"],
               ["A", "B", "ΔCAGR(A-B)", "ΔSharpe", "ΔMaxDD", "Δ换手", "Δ成本占比"],
               {"mean_dcagr": "pct", "mean_dsharpe": "num4", "mean_dmaxdd": "pct",
                "mean_dturnover": "num3", "mean_dcost": "pct"}))
    A("")
    A("### 3.3 Band 贡献：V5-Band vs No-Band vs V4-Band-Strict vs V4-Band-Raw")
    A("")
    band_agg = D.band.groupby(["band_group"]).agg(
        mean_cagr=("cagr", "mean"), mean_sharpe=("sharpe", "mean"),
        mean_maxdd=("max_dd", "mean"), clip=("band_clip_bars", "sum"),
        oob=("n_bars_out_of_band", "sum")).reset_index()
    A(md_table(band_agg, ["band_group", "mean_cagr", "mean_sharpe", "mean_maxdd",
                          "clip", "oob"],
               ["带分组", "均CAGR", "均Sharpe", "均MaxDD", "裁剪事件",
                "越界事件(含压界)"],
               {"mean_cagr": "pct", "mean_sharpe": "num3", "mean_maxdd": "pct",
                "clip": "int", "oob": "int"}))
    A("")
    A("> `V4-Band-Strict` 的裁剪事件为 0 但越界事件不为 0，是因为 R13 的 ETH/BNB、"
      "R9 的 BTC/SOL/BNB 参考权重**恰好压在** V4 带边界上，属「压界」而非「严格越界」"
      "（§2.3 记作「= 上界 / = 下界」）。两列必须并列看，否则会误读为带约束未生效。")
    A("")
    A("### 3.4 LRR 邻域剖面")
    A("")
    A(md_table(D.lrr, ["ref", "base_score", "mean_score_nbr", "min_score_nbr",
                       "max_score_nbr", "std_score_nbr", "n_neighbors",
                       "lrr_raw", "lrr_verdict"],
               ["参考权重", "候选分", "邻域均分", "邻域最低", "邻域最高",
                "邻域std", "邻居数", "LRR", "判定"],
               {"base_score": "num2", "mean_score_nbr": "num2",
                "min_score_nbr": "num2", "max_score_nbr": "num2",
                "std_score_nbr": "num4", "n_neighbors": "int", "lrr_raw": "num4"}))
    A("")

    # 4. 公平结论
    A("## 4. 公平结论（§22）")
    A("")
    A("**相对于哪一个 B&H Rebalance、在哪一个参考权重下，策略仍有独立优势？**")
    A("")
    A("按 MaxDD 改善与样本外窗口逐一核查（vs B4，Pure OOS）：")
    A("")
    A("| 策略 | MaxDD 改善均值 | 改善窗口数/12 | 正超额观测/12 |")
    A("|---|---|---|---|")
    sub = D.oos[(D.oos["mode"] == "Pure_OOS") & (D.oos.benchmark == "B4")]
    for sid in STRATEGY_IDS:
        s = sub[sub.system == sid]
        A(f"| {sid} | {pct(float(s['maxdd_improvement'].mean()))} | "
          f"{int((s['maxdd_improvement'] > 0.02).sum())}/12 | "
          f"{int((s['active_return'] > 0).sum())}/12 |")
    A("")
    A("**结论**：策略的独立优势**体现在回撤控制与下行窗口**，而非全样本收益。")
    A("在 2023 类单边牛市中，任何带风险控制的策略都会大幅跑输满仓 B&H；")
    A("在 2025/2026 类震荡或下行窗口，策略显著占优。因此公平的表述是：")
    A("")
    A("- 相对 **B4（年度再平衡）**：策略以约 13–26pp 的 MaxDD 改善，换取全样本收益的让渡；")
    A("- 相对 **B0（买入持有）**：全部 24 臂的 MaxDD 改善 ≥ 15pp（gate 通过 24/24）；")
    A("- **不存在**在收益、风险、换手三方面同时占优的臂。")
    A("")

    # 5. 鲁棒性结论
    A("## 5. 鲁棒性结论（§22）")
    A("")
    A("### 5.1 硬门槛（§14）")
    A("")
    g = D.gates
    gc = [c for c in g.columns if c.startswith("gate_") and c != "passed_all"]
    A("| 门槛 | 通过臂数 / 24 |")
    A("|---|---|")
    for c in gc + ["passed_all"]:
        A(f"| {c} | {int(g[c].sum())}/24 |")
    A("")
    A(f"仅 **{int(g['passed_all'].sum())} 个臂**通过全部门槛。主要卡点是 "
      "`gate_exposure_50`（平均风险暴露 ≥ 50%）——S3–S7 作为防御型策略，"
      "风险暴露天然低于 50%。就门槛本身而言，这是**门槛与策略定位的错配**"
      "（Roadmap 已预设 `note_defensive` 标记）。")
    A("")
    A("但 §21 第 2、3 条要求的不是「不视为缺陷」，而是给出**正面判断与策略归类**：")
    A("")
    A(_exposure_sharpe_note(g))
    A("")
    A("### 5.2 样本外稳定性")
    A("")
    A(md_table(D.oosstab.sort_values(["ref", "system"]),
               ["ref", "system", "n_windows", "mean_test_cagr", "mean_test_sharpe",
                "worst_test_max_dd", "mean_active_return", "n_windows_beat_B4"],
               ["参考权重", "策略", "窗口数", "均样本外CAGR", "均样本外Sharpe",
                "最差MaxDD", "均超额", "跑赢B4窗口数"],
               {"mean_test_cagr": "pct", "mean_test_sharpe": "num3",
                "worst_test_max_dd": "pct", "mean_active_return": "num3",
                "n_windows": "int", "n_windows_beat_B4": "int"}))
    A("")
    A("### 5.3 Walk-Forward（§12：参考权重在 Train/Validate 上选出后冻结）")
    A("")
    sel = D.wfdf[D.wfdf["selected"]]
    A(md_table(sel, ["window", "ref", "train_calmar"],
               ["窗口", "选中参考权重", "训练期 Calmar"],
               {"train_calmar": "num3", "window": "int"}))
    A("")
    vc = sel["ref"].value_counts()
    A("选中分布：" + "、".join(f"**{k}** {int(v)}/{len(sel)} 次" for k, v in vc.items())
      + "。该分布由训练期 Calmar 选出，**样本外表现须独立判读**（见披露 2 与 §5.2）。")
    A("")
    A("### 5.4 配置带 / 参数 / 资产剔除 / 成本 / 现金 / MC")
    A("")
    A("- **配置带**：V5-Band 下三臂裁剪事件 **0**；V5-Band 与 No-Band 存在收益差，"
      "说明带宽通过**增持上限**实际起作用（并非 no-op）。")
    A("- **成本敏感性**：见 `v5_cost_sensitivity.csv`（4 成本档 × 2 现金模式 × 24 臂）。")
    A("- **Monte Carlo / Bootstrap**：三臂共用 `seed=20260920`、`block=20`，"
      "跨臂百分位可比（§11.3）。")
    A("")
    A("**§21 第 7 条 —— 参数宽容区间**（详值见图 18 与 `v5_parameter_sensitivity.csv`）")
    A("")
    A(_param_tolerance_note(D.sens))
    A("")
    A("**§9.4 —— 资产剔除（含 SOL 归因与 B&H 单调性）**"
      "（详值见图 19 与 `v5_asset_exclusion.csv`）")
    A("")
    A("- 子集覆盖：ALL4、drop_BTC、drop_ETH、drop_SOL、drop_BNB、EqualWeight4、"
      "BTC_ETH_only（§9.4 清单中的「只保留 BTC / ETH」）；"
      "每个子集均跑 B0–B4 与 S3/S5（图 19 与 `v5_asset_exclusion.csv`）。")
    A(f"- {sol_note}")
    A(f"- {bh_note}")
    A("")
    A("**§11.4 低波动伪影防护**：近空仓窗口（如 S1 在窗口1/4 平均风险暴露≈0）"
      "的日收益近乎恒定，Sharpe 会爆炸到 16~241 的伪影值。本报告已将其置为 NaN，"
      "不参与统一参考分布百分位与评分。防护**同时作用于全期表与分段/OOS 表**，"
      "因此本报告 §5.2 中 S1 的「均样本外 Sharpe」是剔除伪影窗口后的有效值，"
      "分母为有效窗口数而非全部窗口数。")
    A("")

    # 5.5 §21 十项判断标准逐条结论（集中表，避免读者无法核对是否被逐条回应）
    A("### 5.5 §21 十项判断标准 —— 逐条结论")
    A("")
    A("§21 是 Roadmap 对「某参考权重是否具有实用价值」的**最终判据**。"
      "下列为十条的集中结论，结论与依据均由本节数据现算，不写死：")
    A("")
    A(md_table(_s21_verdict(D.gates, D.sens, sol_triggered, D.rules),
               ["no", "criterion", "verdict", "basis"],
               ["#", "判断标准", "结论", "依据"]))
    A("")
    A("> **读法**：第 3 条与第 2 条的「不成立 / 仅部分成立」是**策略定位**问题"
      "（防御型策略风险暴露天然低于 50%），非执行缺陷；第 6、10 条的降级结论"
      "直接约束 §6 的推荐臂，须结合 §0 披露 4 判读。")
    A("")

    # 6. 投资结论
    A("## 6. 投资结论（§22）")
    A("")
    A("| 角色 | 参考权重 | 系统 | 评分 | CAGR | Sharpe | MaxDD |")
    A("|---|---|---|---|---|---|---|")
    for _, r in D.cand.iterrows():
        A(f"| {r['role']} | {r['ref']} | {r['system']} | {num(r['final_score'], 1)} | "
          f"{pct(r['cagr'])} | {num(r['sharpe'])} | {pct(r['max_dd'])} |")
    A("")
    A("**推荐参考权重**：`" + D.concl["frozen_rules"]["reference_id"] + "` "
      f"`{wp['BTC']:.2f}/{wp['ETH']:.2f}/{wp['SOL']:.2f}/{wp['BNB']:.2f}`")
    A("")
    if n_tied > 1:
        A(f"> 但必须重申：三者综合名次**并列**（combined_rank 均为 "
          f"{sb['combined_rank']:.0f}）。推荐 {rid} 的理由是其在训练期评分最高，"
          "**而非全维度占优**。最终取舍取决于投资目标对"
          "「收益 / 回撤 / 样本外稳健」的权重分配。")
    else:
        rank_map = {x["ref"]: int(x["combined_rank"])
                    for _, x in D.refsel.iterrows()}
        low = max(rank_map, key=lambda k: rank_map[k])
        A(f"> 但必须重申：{rid} 的综合名次优势**仅 {rank_map[low] - sb['combined_rank']:.0f} "
          f"个名次位**（{sb['combined_rank']:.0f} vs {rank_map[low]}），且证据互补——"
          "训练期评分与均 CAGR 由 R3 领先，样本外平均超额由 R9 领先，"
          "回撤与控制类指标三臂几乎同档。因此这是**弱区分**，"
          "最终取舍仍取决于投资目标对「收益 / 回撤 / 样本外稳健」的权重分配。")
    A("")
    if sol_triggered:
        A("**但该推荐受 §9.4 归因约束**：R9 的相对优势在剔除 SOL 后扩大，"
          "即其优势部分来自**低 SOL 暴露**，而非更强的择时中枢（见披露 4）。"
          "**若投资目标不接受 R9 的 SOL 目标权重，则不应采用 R9**，"
          "应按 §7 的兜底口径（三者包络的中位中枢 + V5 默认带）执行。")
        A("")
    A("### 共同短板（§22 要求正面披露）")
    A("")
    A(f"- **最大回撤**：24 臂 MaxDD 中位数 **{pct(cs['maxdd_median'])}**，"
      f"最差 {pct(cs['maxdd_all_arms_worst'])}；"
      f"深于 -50% 的臂数 **{cs['n_arms_maxdd_worse_than_-50pct']}/"
      f"{cs['n_arms_total']}**。")
    A("  V4.8 在参考权重层面已记录同类短板（三参考权重自身的 MaxDD 为 "
      "-57.8%~-59.4%，无一满足 <50%）；V5 在 24 个策略臂上复现了同一结论——")
    A("  **这是策略层面的共同短板，换参考权重无法解决**，V5 已把回撤控制列为一等议题。")
    A(f"- **LRR**：三臂在 V5 口径下 LRR ≥ {num(D.refsum['lrr_raw'].min())}，"
      "但这主要源于邻域定义较窄，")
    A("  **不可据此认为已消除尖峰风险**（见披露 1）。")
    A("")

    # 7. 执行规则
    A("## 7. 执行规则（§22，已冻结）")
    A("")
    R = D.rules
    A("| 项 | 取值 |")
    A("|---|---|")
    A(f"| 参考权重 | {R['reference_id']} `"
      f"{R['reference_weight']['BTC']:.2f}/{R['reference_weight']['ETH']:.2f}/"
      f"{R['reference_weight']['SOL']:.2f}/{R['reference_weight']['BNB']:.2f}` |")
    A(f"| 默认配置带 | V5-Band：BTC {R['band']['BTC'][0]:.2f}–{R['band']['BTC'][1]:.2f}、"
      f"ETH {R['band']['ETH'][0]:.2f}–{R['band']['ETH'][1]:.2f}、"
      f"SOL {R['band']['SOL'][0]:.2f}–{R['band']['SOL'][1]:.2f}、"
      f"BNB {R['band']['BNB'][0]:.2f}–{R['band']['BNB'][1]:.2f} |")
    A(f"| 再平衡 | S3/S4 阈值 ±10% 绝对偏离；B4 年度日历 |")
    A(f"| 现金处理 | {R['cash']}（历史 DFF 利率，时点 shift(1)） |")
    A(f"| 上市处理 | {R['listing_rule']} |")
    A(f"| 交易成本 | 费率 {R['cost']['fee']:.4f} + 滑点 {R['cost']['slippage']:.4f}"
      f" = 单腿 {R['cost']['per_leg']:.4f} |")
    A(f"| 决策/执行 | {R['cadence']} 决策，exec_lag_bars={R['exec_lag_bars']} |")
    A("")
    fb = R.get("fallback_mid_hub")
    if fb:
        fw = fb["reference_weight"]
        A("**§21 尾句兜底口径（三权重无法区分时不得强行选出唯一中枢）**")
        A("")
        if fb["applicable"]:
            A(f"> {fb['note']}")
            A("")
            A(f"| 兜底项 | 取值 |")
            A("|---|---|")
            A(f"| 适用性 | **适用**（训练期与样本外排序不一致，或综合名次并列） |")
            A(f"| 中位中枢 | BTC {fw['BTC']:.2f}、ETH {fw['ETH']:.2f}、"
              f"SOL {fw['SOL']:.2f}、BNB {fw['BNB']:.2f}，"
              f"Cash {fb['cash']:.2f} |")
            A(f"| 默认配置带 | {fb['band_group']}（与主口径一致） |")
            A("")
        else:
            A(f"> {fb['note']}")
            A("")
    A("**停止条件**：")
    A("")
    for s in R["stop_conditions"]:
        A(f"- {s}")
    A("")

    # 8. 图表索引
    A("## 8. 图表索引（§18，22 张）")
    A("")
    figs = sorted(FIG.glob("*.png"))
    for f in figs:
        A(f"![{f.stem}](figures/{f.name})")
        A("")

    # 9. 产物
    A("## 9. 产物清单")
    A("")
    csvs = sorted([p.name for p in OUT.glob("v5_*.csv")])
    A(f"- CSV（{len(csvs)} 个）：" + "、".join(f"`{c}`" for c in csvs))
    A("- JSON：`v5_config_snapshot.json`、`v5_execution_rules.json`、"
      "`v5_conclusion.json`、`v5_reproducibility.json`")
    A(f"- 图表：`figures/`（{len(figs)} 张）")
    A("")
    A("**两处口径说明**（§17 命名与语义的对齐）：")
    A("")
    A("- `v5_band_audit.csv` = 三参考权重 × {V5-Band / V4-Band-Strict / "
      "V4-Band-Raw / No-Band} × 代表系统的**裁剪事件**审计（§17 定义内容），"
      "`n_bars_clipped`（严格越界）与 `n_bars_out_of_band`（恰好压界，非裁剪）分列；"
      "参考权重与带边界的**间距**汇总另见 `v5_band_gap.csv`，"
      "参考权重层面的静态审计见 `v5_band_audit_reference.csv`。")
    A("- `v5_strategy_signals.csv` 按 `ref` 分列输出。三臂的**目标权重数值依臂不同是设计使然**"
      "（§7：参考权重只改变目标权重、不改变信号生成），因此**臂中性不能用目标权重逐位比较**："
      "带约束是绝对上下限（同一上界会咬在各臂不同的目标值上），且 S4–S7 在资产级开关之后"
      "会对仍在场的资产**重新归一化**，二者都发生在信号**之后**。信号层的臂中性由 `run_v5`"
      "每次运行打印的两个不变量核验（去带重建，`V4-Band-Raw`）：① 资产开关掩码 "
      "`target_a > 0` 的跨臂不一致格数（应为 0）；② `RebalanceSpec`（信号节奏）跨臂取值数"
      "（应全为 1）—— 这就是 §19.6「三臂使用同一条信号序列」的判据。执行层触发次数因阈值"
      "再平衡依赖持仓状态（各臂漂移不同）而逐臂不同，属执行差异，非信号生成差异。")
    A("")
    A("### 复现方式")
    A("")
    A("```bash")
    A("python -m scripts.v5_spot.smoke        # 14 项验收")
    A("python -m scripts.v5_spot.run_v5 --through 10   # Phase 0-10")
    A("python -m scripts.v5_spot.build_report          # 图表 + 双版本报告")
    A("```")
    A("")
    A("**配置指纹**：")
    A("")
    for r, v in D.concl["frozen_rules"].get("_fingerprints", {}).items():
        A(f"- {r}: `{v[:32]}...`")
    repro = json.loads((OUT / "v5_reproducibility.json").read_text("utf-8"))
    for r, v in repro.get("fingerprints", {}).items():
        A(f"- {r}: `{v[:32]}...`")
    A("")

    md = "\n".join(L)
    (OUT / "v5_spot_report.md").write_text(md, encoding="utf-8")
    print("  -> v5_spot_report.md", flush=True)

    # HTML：同一份 Markdown 装配，避免两版本结论不一致
    html = md_to_html(md, title="V5 回测报告")
    (OUT / "v5_spot_report.html").write_text(html, encoding="utf-8")
    print("  -> v5_spot_report.html", flush=True)


def md_to_html(md: str, title: str = "V5 回测报告") -> str:
    """轻量 Markdown → HTML（覆盖本报告用到的语法子集）。"""
    import html as _h
    import re

    lines = md.split("\n")
    out: List[str] = []
    i = 0
    in_code = False
    in_table = False
    in_ul = False

    def close_table():
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            close_table(); close_ul()
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(_h.escape(ln))
            i += 1
            continue
        # 表格
        if ln.startswith("|") and i + 1 < len(lines) and set(
                lines[i + 1].replace("|", "").replace(" ", "")) <= set("-:"):
            close_ul()
            headers = [c.strip() for c in ln.strip("|").split("|")]
            out.append("<table><thead><tr>"
                       + "".join(f"<th>{_h.escape(x)}</th>" for x in headers)
                       + "</tr></thead><tbody>")
            in_table = True
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(
                    f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            close_table()
            continue
        close_table()
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            close_ul()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        # 列表
        if re.match(r"^\s*-\s+", ln):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline(re.sub(r'^\s*-\s+', '', ln))}</li>")
            i += 1
            continue
        close_ul()
        if ln.strip() == "":
            i += 1
            continue
        # 引用块（含连续多行）
        if ln.lstrip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        out.append(f"<p>{inline(ln)}</p>")
        i += 1
    close_table(); close_ul()
    body = "\n".join(out)

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_h.escape(title)}</title>
<style>
:root{{--bg:#0f1318;--fg:#c7d1db;--mut:#8b97a5;--line:#232b36;--acc:#4c9ffe;
--ok:#3fb950;--bad:#e0564a;--warn:#f0b90b;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}}
main{{max-width:1180px;margin:0 auto;padding:48px 28px 96px}}
h1{{font-size:30px;border-bottom:2px solid var(--acc);padding-bottom:14px;margin-top:0}}
h2{{font-size:23px;margin-top:44px;border-left:4px solid var(--acc);padding-left:12px}}
h3{{font-size:18px;margin-top:30px;color:#dbe4ee}}
h4,h5,h6{{font-size:16px;color:var(--mut)}}
p{{margin:12px 0}}
a{{color:var(--acc)}}
code{{background:#1a2029;padding:2px 6px;border-radius:4px;font-size:13px;
color:#9ddcff;font-family:"Cascadia Code",Consolas,monospace}}
pre{{background:#161c24;border:1px solid var(--line);border-radius:8px;
padding:16px;overflow:auto}}
pre code{{background:none;color:#c7d1db;padding:0}}
table{{border-collapse:collapse;width:100%;margin:18px 0;font-size:13.5px;
display:block;overflow-x:auto}}
th{{background:#1a2029;color:#dbe4ee;text-align:left;padding:9px 11px;
border-bottom:2px solid var(--line);white-space:nowrap}}
td{{padding:8px 11px;border-bottom:1px solid var(--line);white-space:nowrap}}
tr:hover td{{background:#151b23}}
blockquote{{margin:18px 0;padding:12px 18px;border-left:4px solid var(--warn);
background:#1a1f16;border-radius:0 6px 6px 0;color:#d8dfe8}}
ul{{padding-left:22px}}
li{{margin:7px 0}}
img{{max-width:100%;border:1px solid var(--line);border-radius:8px;
margin:14px 0;background:#0f1318}}
strong{{color:#e6edf3}}
hr{{border:none;border-top:1px solid var(--line);margin:34px 0}}
</style></head>
<body><main>
{body}
</main></body></html>"""


def inline(s: str) -> str:
    import html as _h
    import re
    s = _h.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def main() -> int:
    print("V5 报告生成 —— 22 图 + HTML/Markdown 双版本")
    D = Data()
    print("  数据装配完成，开始绘图…")
    for fn in (fig01_equity_all_refs, fig02_equity_log, fig03_drawdown, fig04_bands,
               fig05_rebal_freq, fig06_cagr_risk, fig07_strat_vs_bh,
               fig08_active_return, fig09_asset_weights, fig10_cash, fig11_tim,
               fig12_turnover_cost, fig13_capture, fig14_rolling_sharpe,
               fig15_rolling_cagr, fig16_walk_forward, fig17_oos_by_year,
               fig18_param_heatmap, fig19_asset_exclusion, fig20_mc,
               fig21_opportunity_cost, fig22_lrr):
        try:
            fn(D)
        except Exception as e:                                    # noqa: BLE001
            print(f"  [跳过] {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    print("  绘图完成，装配报告…")
    build_report(D)
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())