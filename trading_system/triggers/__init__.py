"""Phase 1:L1 触发器(粗桶,禁止网格寻优)。任务 1.5。对应 v3.1 第六章。

A 牛回头、B 缩量低位首板、C RPS 龙头。阈值只用粗桶(回撤桶/回调天数桶/缩量桶),
边界从 config/triggers.yaml 读,全部"待自验"。**代码里禁止出现网格 search / 最优点挑选逻辑。**
价格层:特征类判定用 adj;一字板等执行约束用 raw(INV-2)。
"""

from __future__ import annotations

_PHASE = "Phase 1 任务 1.5"


def trigger_pullback(*args, **kwargs):  # noqa: ANN002, ANN003
    """A 牛回头。"""
    raise NotImplementedError(f"{_PHASE}:trigger_pullback 待实现(禁止网格寻优)。")


def trigger_first_board(*args, **kwargs):  # noqa: ANN002, ANN003
    """B 缩量低位首板。"""
    raise NotImplementedError(f"{_PHASE}:trigger_first_board 待实现(禁止网格寻优)。")


def trigger_rps_leader(*args, **kwargs):  # noqa: ANN002, ANN003
    """C RPS 龙头。"""
    raise NotImplementedError(f"{_PHASE}:trigger_rps_leader 待实现(禁止网格寻优)。")
