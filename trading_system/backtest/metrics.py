"""回测指标。Phase 2(任务 2.4)。对应 v3.1 第十三章。

RankIC、分块不重叠 RankIC(块长 H,避重叠虚高)、ICIR、扣费净值、MaxDD、Calmar、
换手、成交失败率;PBO/CSCV、DSR(Phase 3 审批要用,这里先写好接口)。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.4"


def rank_ic(*args, **kwargs):  # noqa: ANN002, ANN003
    raise NotImplementedError(f"{_PHASE}:rank_ic 待实现。")


def blocked_rank_ic(*args, **kwargs):  # noqa: ANN002, ANN003
    """分块不重叠 RankIC(块长 H)。"""
    raise NotImplementedError(f"{_PHASE}:blocked_rank_ic 待实现。")


def max_drawdown(*args, **kwargs):  # noqa: ANN002, ANN003
    raise NotImplementedError(f"{_PHASE}:max_drawdown 待实现。")


def calmar(*args, **kwargs):  # noqa: ANN002, ANN003
    raise NotImplementedError(f"{_PHASE}:calmar 待实现。")


def pbo_cscv(*args, **kwargs):  # noqa: ANN002, ANN003
    """PBO / CSCV(过拟合概率)。Phase 3 审批用。"""
    raise NotImplementedError(f"{_PHASE}:pbo_cscv 待实现。")


def dsr(*args, **kwargs):  # noqa: ANN002, ANN003
    """DSR(去膨胀夏普)。Phase 3 审批用。"""
    raise NotImplementedError(f"{_PHASE}:dsr 待实现。")
