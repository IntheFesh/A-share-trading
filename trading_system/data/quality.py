"""数据质检(每日跑)。Phase 0(任务 0.6)。对应 v3.1 §2.3。

检查项(全部):退市完整性、后复权连续性(相邻日收益超 ±10.5% 且非除权日 -> 报警)、
停牌剔除、ST 历史状态、新股 60 日窗、raw/adj 分离检查、涨跌停 preclose 核对、
除权日一致性、一字不可买、跌停顺延、披露日历完整性、预告字段时点检查。
价格层:连续性用 adj 收益;涨跌停核对用 raw 昨收(INV-2)。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.6"


def run_daily_quality_checks(*args, **kwargs):  # noqa: ANN002, ANN003
    """运行全部质检项,返回报告(通过/告警/失败)。"""
    raise NotImplementedError(f"{_PHASE}:run_daily_quality_checks 待实现(v3.1 §2.3)。")
