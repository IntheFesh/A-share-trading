"""上线审批协议。Phase 3(任务 3.4)。对应 v3.1 第十三章。

四模型基线对比(复杂模型须同时胜出)。上线门槛(同时满足):
  R_blind>0、DSR>0.95、PBO<0.30、ΔMaxDD<=0、SlippageStress_20bp>0;
  首板另需 SlippageStress_30bp>0;ManualVetoAudit 非毁灭;ExecutionGap_20trades<15bp。
盲测段一次性(INV-6):评估即消费该盲测段并封存,再用同段报错。
诚实:此多重 AND 极严,长期可能无配置上线——evaluate_approval 如实返回每项结果。
门槛默认值对齐 config/train.yaml 的 approval 块(逻辑不写死,可从 config 传入)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_system.invariants import BlindSegmentLedger


@dataclass
class ApprovalThresholds:
    dsr_min: float = 0.95
    pbo_max: float = 0.30
    slippage_bp_main: float = 20.0
    slippage_bp_first_board: float = 30.0
    execution_gap_max_bp: float = 15.0


@dataclass
class ApprovalResult:
    approved: bool
    checks: dict = field(default_factory=dict)

    def failed_checks(self) -> "list[str]":
        return [k for k, v in self.checks.items() if not v]


def evaluate_approval(
    metrics: dict,
    *,
    blind_ledger: BlindSegmentLedger,
    blind_segment_id: str,
    thresholds: "ApprovalThresholds | None" = None,
    is_first_board: bool = False,
) -> ApprovalResult:
    """评估五重 AND 上线门槛。**消费盲测段(INV-6)**:本段用过即封存,再评估同段报错。

    metrics 需含:r_blind, dsr, pbo, delta_maxdd, slippage_net_20bp,
    (首板)slippage_net_30bp, execution_gap_bp, manual_veto_destructive(bool),
    beats_all_baselines(bool)。
    """
    th = thresholds or ApprovalThresholds()
    blind_ledger.assert_available(blind_segment_id)  # INV-6:已封存则报错

    checks = {
        "beats_all_baselines": bool(metrics.get("beats_all_baselines", False)),
        "R_blind_positive": metrics["r_blind"] > 0,
        "DSR>0.95": metrics["dsr"] > th.dsr_min,
        "PBO<0.30": metrics["pbo"] < th.pbo_max,
        "delta_MaxDD<=0": metrics["delta_maxdd"] <= 0,
        "SlippageStress_20bp>0": metrics["slippage_net_20bp"] > 0,
        "ManualVetoAudit_nondestructive": not metrics.get("manual_veto_destructive", False),
        "ExecutionGap<15bp": metrics["execution_gap_bp"] < th.execution_gap_max_bp,
    }
    if is_first_board:
        checks["SlippageStress_30bp>0"] = metrics["slippage_net_30bp"] > 0

    # 消费盲测段(用于一次 champion-challenger 裁决),无论通过与否都封存(INV-6)
    blind_ledger.use_for_decision(blind_segment_id)

    return ApprovalResult(approved=all(checks.values()), checks=checks)
