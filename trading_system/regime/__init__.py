"""Phase 1:L0 情绪温度与市场状态。任务 1.4。对应 v3.1 §5.1 与 v0.3。

由日线 OHLC + raw 昨收推导六指标(涨停家数、最高连板高度、晋级率、炸板率、
昨日涨停今日溢价、跌停+核按钮数)-> 合成情绪温度 T_t(250 日分位,权重从
config/regime.yaml 读)-> 五阶段 + 总敞口乘子 m_t。
HiLo 高低切状态量;HMM 状态概率为可选(Phase 3 后再做)。

注意:T_t / 披露季强度 / 高低切强度等是"组内常数"(当天同值),进 L2 须经交互(INV-4)。
价格层:六指标推导用 raw 昨收算涨跌停(INV-2);收益类用 adj。
"""

from __future__ import annotations

_PHASE = "Phase 1 任务 1.4"


def compute_six_indicators(*args, **kwargs):  # noqa: ANN002, ANN003
    """由日线推导 L0 六指标。"""
    raise NotImplementedError(f"{_PHASE}:compute_six_indicators 待实现(v3.1 §5.1)。")


def compute_temperature(*args, **kwargs):  # noqa: ANN002, ANN003
    """合成情绪温度 T_t(250 日分位)-> 五阶段 + m_t。"""
    raise NotImplementedError(f"{_PHASE}:compute_temperature 待实现。")


def compute_hilo(*args, **kwargs):  # noqa: ANN002, ANN003
    """HiLo 高低切状态量。"""
    raise NotImplementedError(f"{_PHASE}:compute_hilo 待实现。")
