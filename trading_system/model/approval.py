"""上线审批协议。Phase 3(任务 3.4)。对应 v3.1 第十三章。

四模型基线对比(复杂模型须同时胜出)。上线门槛(同时满足):
  R_blind>0、DSR>0.95、PBO<0.30、ΔMaxDD<=0、SlippageStress_20bp>0;
  首板另需 SlippageStress_30bp>0;ManualVetoAudit 非毁灭;ExecutionGap_20trades<15bp。
盲测段一次性(INV-6):用过即归档(见 audit/experiment_registry.py 与 invariants.BlindSegmentLedger)。
任一不满足 -> 只观察不上线。诚实输出:此五重 AND 极严,长期可能无配置上线。
门槛值从 config/train.yaml 的 approval 块读。
"""

from __future__ import annotations

_PHASE = "Phase 3 任务 3.4"


def evaluate_approval(*args, **kwargs):  # noqa: ANN002, ANN003
    """评估上线门槛(五重 AND);返回是否放行 + 各项明细。"""
    raise NotImplementedError(f"{_PHASE}:evaluate_approval 待实现(五重 AND;INV-6)。")
