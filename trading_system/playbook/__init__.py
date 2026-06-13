"""Phase 4:作战手册(CSV + Markdown + 控制台,无网页)。任务 4.1。对应 v3.1 第十二章。

每日生成一个 CSV + 一份 Markdown,控制台打印关键行(作战手册的 print 是允许的例外)。
每票字段(v3.1 §12):代码/触发器/模型分与排名/SHAP 前三理由/限价买入价/股数与目标仓位%/
止损价/止盈三阶梯价/时间止损日/否决栏/风险标注(临近解禁、质押高、商誉高、近期监管函、
days_to_disclosure 与是否已发预告、过度拉升度、当前是否高低切 regime)。
页脚印当日 T_t 与阶段、m_t、w_total、刹车档、距下次 Tier 1/2 天数。
价格层:限价/止损/止盈价用 raw(INV-2)。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 每票必备列(v3.1 §12)
PLAYBOOK_COLUMNS: tuple[str, ...] = (
    "code", "trigger", "model_score", "rank", "shap_top3",
    "limit_buy_price", "shares", "target_weight_pct",
    "stop_price", "tp1_price", "tp2_price", "tp3_price", "time_stop_date",
    "veto_reason",
    # 风险标注
    "days_to_disclosure", "has_preann", "pledge_high", "goodwill_high",
    "recent_regulatory_letter", "overextension_score", "hilo_regime",
)


def build_playbook_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """规整候选表为作战手册 schema:补齐缺失列、按 rank 排序、固定列序。"""
    out = candidates.copy()
    for col in PLAYBOOK_COLUMNS:
        if col not in out.columns:
            out[col] = None
    if out["rank"].notna().any():
        out = out.sort_values("rank")
    return out[list(PLAYBOOK_COLUMNS)].reset_index(drop=True)


def _df_to_md(df: pd.DataFrame) -> str:
    """把 DataFrame 渲染为 GitHub Markdown 表(不依赖 tabulate)。"""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def _footer_md(regime: dict) -> str:
    return (
        "\n---\n"
        f"- 情绪温度 T_t = {regime.get('T_t')},阶段 = {regime.get('stage')}\n"
        f"- 总仓位乘子 m_t = {regime.get('m_t')},总敞口 w_total = {regime.get('w_total')}\n"
        f"- 刹车档 s = {regime.get('brake_level')}\n"
        f"- 距下次 Tier 1 = {regime.get('days_to_tier1')} 日,Tier 2 = {regime.get('days_to_tier2')} 日\n"
    )


def generate_playbook(
    candidates: pd.DataFrame,
    regime: dict,
    *,
    trade_date: str,
    out_dir: "str | Path",
    print_console: bool = True,
) -> "tuple[pd.DataFrame, str]":
    """生成当日作战手册:写 CSV + Markdown 到 out_dir,控制台打印关键行。返回 (表, markdown 文本)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_playbook_table(candidates)

    csv_path = out_dir / f"playbook_{trade_date}.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_lines = [f"# 作战手册 {trade_date}", "", _df_to_md(table), _footer_md(regime)]
    md_text = "\n".join(md_lines)
    (out_dir / f"playbook_{trade_date}.md").write_text(md_text, encoding="utf-8")

    if print_console:
        # 控制台打印关键行(允许的 print 例外)
        for _, r in table.iterrows():
            print(
                f"[{r['code']}] {r['trigger']} 分{r['model_score']} 排名{r['rank']} | "
                f"买{r['limit_buy_price']} 仓{r['target_weight_pct']}% | 止损{r['stop_price']} "
                f"| 否决:{r['veto_reason'] or '-'}"
            )
        print(_footer_md(regime).strip())
    return table, md_text
