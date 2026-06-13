"""三标签路线 + LightGBM。Phase 3(任务 3.2)。对应 v3.1 第七章 / 第八章。

三标签路线并行:A 截面秩回归、B winsorized 净收益回归、C lambdarank 分位标签(优先五分位)。
LightGBM ranker 的 group 必须按交易日(group_t = |G_t|,不能混成大表)。
L0/状态信息只以显式交互项进入(INV-4 守卫);过度拉升只以 HiLo 交互进入(INV-7)。
Tier 1:滚动 12 个月窗口、冻结超参重训权重。价格层:特征 adj、标签成交侧 raw。
"""

from __future__ import annotations

_PHASE = "Phase 3 任务 3.2"


def train_l2_model(*args, **kwargs):  # noqa: ANN002, ANN003
    """训练 L2 模型(三标签路线之一);group 按交易日。"""
    raise NotImplementedError(f"{_PHASE}:train_l2_model 待实现(group 按交易日)。")
