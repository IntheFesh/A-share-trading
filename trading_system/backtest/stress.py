"""滑点压力矩阵。Phase 2(任务 2.4)。对应 v3.1 第四章 / 第十三章。

slippage ∈ {5,10,20,30}bp × {大市值/中流动性/小市值/缩量低位首板/牛回头/RPS龙头/
连续涨停高开边界}。审批门槛:SlippageStress_20bp>0;首板另需 30bp>0。
档位从 config/costs.yaml 的 slippage.stress_grid_bp 读。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.4"


def run_slippage_stress(*args, **kwargs):  # noqa: ANN002, ANN003
    """跑滑点 × 场景压力矩阵,返回各格扣费净收益。"""
    raise NotImplementedError(f"{_PHASE}:run_slippage_stress 待实现。")
