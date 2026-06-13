"""成本六层。Phase 2(任务 2.1)。对应 v3.1 第四章。

c_round = c_tax + c_exchange + c_commission + c_mincomm + c_spread + c_impact + c_failure
已核验下限:c_official = 0.05%(印花税卖出) + 2×0.00341%(经手费双向) = 5.682 bp。
过户费、佣金率、最低佣金、滑点等从 config/costs.yaml 读(待核 / 可配置)。
最低佣金闸门 c_mincomm_rt(Q)=min_per_side/Q;最小订单金额过滤 Q_i>=q_min_amount。
价格层:成本基于 raw 成交金额(执行类,INV-2)。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.1"


def round_trip_cost(*args, **kwargs):  # noqa: ANN002, ANN003
    """计算一笔往返(买+卖)的六层总成本。"""
    raise NotImplementedError(f"{_PHASE}:round_trip_cost 待实现(成本六层)。")
