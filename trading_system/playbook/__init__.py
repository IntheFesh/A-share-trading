"""Phase 4:作战手册(CSV + Markdown + 控制台,无网页)。任务 4.1。对应 v3.1 第十二章。

每日生成一个 CSV + 一份 Markdown,控制台打印关键行。每票字段:代码/触发器/模型分与排名/
SHAP 前三理由/限价买入价/股数与目标仓位%/止损价/止盈三阶梯价/时间止损日/否决栏/风险标注
(临近解禁、质押高、商誉高、近期监管函、days_to_disclosure 与是否已发预告、过度拉升度、
当前是否高低切 regime)。文件尾部印当日 T_t 与阶段、m_t、w_total、刹车档、距下次 Tier 1/2 天数。
说明:作战手册的控制台输出是允许的 print 例外。价格层:限价/止损/止盈价用 raw(INV-2)。
"""

from __future__ import annotations

_PHASE = "Phase 4 任务 4.1"


def generate_playbook(*args, **kwargs):  # noqa: ANN002, ANN003
    """生成当日作战手册(CSV + Markdown + 控制台关键行)。"""
    raise NotImplementedError(f"{_PHASE}:generate_playbook 待实现(v3.1 第十二章)。")
