"""Phase 4:否决审计(OPE 反事实)+ 盲测段一次性账本(INV-6)。任务 4.2。对应 v3.1 §12 升级。

记录行为策略动作概率(模型推荐什么、人何时因何否决、否决后买什么替代)——OPE 共同支撑前提。
实现 IPW(Horvitz–Thompson)、DR(双重稳健)估计否决策略价值;封闭 reason code
(含新增"过度拉升/彩票""高低切风格反转");算法厌恶护栏(监控模型刚失误后否决率是否飙升)。
价格层:奖励 r 为扣费实现收益(由引擎按 raw 记账,INV-2)。
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class ReasonCode(str, Enum):
    """否决理由码(封闭枚举,v3.0 扩充)。"""

    MAJOR_VIOLATION = "major_violation_delist"     # 重大违法/退市
    REGULATORY_LETTER = "regulatory_letter"         # 监管函/问询函
    NEGATIVE_PREANN = "negative_preann"             # 业绩雷/已发负面预告(疑问①)
    LOCKUP_RELEASE = "lockup_release"               # 解禁减持
    SUSPEND_BOUNDARY = "suspend_boundary"           # 停复牌边界
    AUCTION_UNREACHABLE = "auction_unreachable"     # 竞价不可达
    GAP_OVER_THRESHOLD = "gap_over_threshold"       # 高开超阈
    CLUSTER_CROWDED = "cluster_crowded"             # 同簇拥挤
    LOW_LIQUIDITY = "low_liquidity"                 # 流动性不足
    OVEREXTENSION_LOTTERY = "overextension_lottery" # 过度拉升/彩票(新增,疑问②)
    HILO_STYLE_REVERSAL = "hilo_style_reversal"     # 高低切风格反转(新增,疑问②)
    OTHER = "other"                                 # 其他(需手写)


def ipw_value(
    actions: "np.ndarray",
    target_actions: "np.ndarray",
    rewards: "np.ndarray",
    behavior_probs: "np.ndarray",
) -> float:
    """IPW(逆倾向加权)估计目标策略价值:E[1{a=π_target}/π_b(a) · r]。

    π_b 为行为策略选中所记动作的概率(>0)。目标策略为确定性(每样本给出 target_action)。
    """
    a = np.asarray(actions)
    tgt = np.asarray(target_actions)
    r = np.asarray(rewards, dtype="float64")
    pb = np.asarray(behavior_probs, dtype="float64")
    if np.any(pb <= 0):
        raise ValueError("behavior_probs 必须全 > 0(共同支撑前提)")
    indicator = (a == tgt).astype("float64")
    return float(np.mean(indicator / pb * r))


def dr_value(
    actions: "np.ndarray",
    target_actions: "np.ndarray",
    rewards: "np.ndarray",
    behavior_probs: "np.ndarray",
    q_taken: "np.ndarray",
    q_target: "np.ndarray",
) -> float:
    """DR(双重稳健)估计:E[q̂(s,π_target) + 1{a=π_target}/π_b(a)·(r - q̂(s,a))]。

    q_taken=q̂(s, 实际动作),q_target=q̂(s, 目标动作)。奖励模型或倾向模型之一正确即一致(方差更低)。
    """
    a = np.asarray(actions)
    tgt = np.asarray(target_actions)
    r = np.asarray(rewards, dtype="float64")
    pb = np.asarray(behavior_probs, dtype="float64")
    qt = np.asarray(q_taken, dtype="float64")
    qg = np.asarray(q_target, dtype="float64")
    if np.any(pb <= 0):
        raise ValueError("behavior_probs 必须全 > 0")
    indicator = (a == tgt).astype("float64")
    return float(np.mean(qg + indicator / pb * (r - qt)))


def hcope_lower_bound(per_sample_values: "np.ndarray", *, delta: float = 0.05) -> float:
    """HCOPE 高置信下界(Thomas 2015 精神;经验 Bernstein 不等式)。

    per_sample_values 为逐样本的重要性加权奖励 g_i(= 1{a=π_target}/π_b · r);返回否决策略价值的
    (1-delta) 置信下界。需"有把握地证明否决没毁价值"时用本下界(下界>0 才算证据)。样本越多越紧。
    """
    x = np.asarray(per_sample_values, dtype="float64")
    n = len(x)
    if n < 2:
        return float("nan")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    rng = float(x.max() - x.min())
    ln = np.log(2.0 / delta)
    # Maurer–Pontil 经验 Bernstein:LB = mean - s·sqrt(2ln/n) - 7·range·ln/(3(n-1))
    eb = s * np.sqrt(2.0 * ln / n) + 7.0 * rng * ln / (3.0 * (n - 1))
    return m - eb


def algorithm_aversion_check(
    model_was_wrong: "np.ndarray", human_vetoed: "np.ndarray", *, ratio_threshold: float = 1.5
) -> dict:
    """算法厌恶护栏(Dietvorst 2015):比较"模型刚失误后"的否决率 vs 总体否决率。

    model_was_wrong[i]:第 i-1 步模型是否明显失误(即第 i 步是否处于"刚失误后");
    human_vetoed[i]:第 i 步人是否否决。若 after_error 否决率 >= ratio_threshold × 基线 -> 触发护栏。
    """
    wrong = np.asarray(model_was_wrong, dtype=bool)
    veto = np.asarray(human_vetoed, dtype=bool)
    base = float(veto.mean()) if len(veto) else 0.0
    after = float(veto[wrong].mean()) if wrong.any() else 0.0
    flagged = base > 0 and after >= ratio_threshold * base
    return {"baseline_veto_rate": base, "after_error_veto_rate": after, "flagged": flagged}
