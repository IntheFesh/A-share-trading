"""Phase 2:L3 仓位合成。任务 2.5。对应 v3.1 §10。价格层:ATR/波动用 adj 派生,涨跌停约束用 raw。

凯利符号三档:r_t = r_0·1{f*>0}·s(r_0=0.5%,s∈{1,0.5,0});
单股上限(INV-5):w_max=min(w_hard, L_tail/ĝ)(见 invariants.single_name_cap);
总敞口:w_total = min(0.5, σ*/σ̂, m_t, w_liquidity, w_limitdown);
合成:ATR 定相对权重 w̃_i ∝ 1/N_i,单票目标=w̃_i·w_total,逐票被单股上限截断,
截断后总和<w_total 则保持(不补杠杆);拥挤簇限制(普通≤25%,小市值+低流动性+同题材≤15%)。
"""

from __future__ import annotations

import numpy as np

from trading_system.invariants import assert_hard_cap_allowed, single_name_cap  # noqa: F401


def kelly_risk_budget(f_star: float, *, r0: float = 0.005, s: float = 1.0) -> float:
    """凯利符号三档:f*>0 才有风险预算;r_t = r0·1{f*>0}·s。f*<=0 -> 0(空仓)。"""
    return r0 * s if f_star > 0 else 0.0


def total_exposure(
    *,
    sigma_star: float,
    sigma_hat: float,
    m_t: float,
    w_liquidity: float = 1.0,
    w_limitdown: float = 1.0,
    hard_ceiling: float = 0.5,
) -> float:
    """总敞口 w_total = min(0.5, σ*/σ̂, m_t, w_liquidity, w_limitdown)。σ̂<=0 时退化为硬上限。"""
    vol_term = (sigma_star / sigma_hat) if sigma_hat > 0 else hard_ceiling
    return float(min(hard_ceiling, vol_term, m_t, w_liquidity, w_limitdown))


def inverse_atr_weights(atr: np.ndarray) -> np.ndarray:
    """相对权重 w̃_i ∝ 1/N_i(ATR 越大权重越小),归一化到和为 1。atr 须全 > 0。"""
    atr = np.asarray(atr, dtype="float64")
    if np.any(atr <= 0):
        raise ValueError("ATR 必须全 > 0")
    inv = 1.0 / atr
    return inv / inv.sum()


def compose_positions(
    atr: np.ndarray,
    *,
    w_total: float,
    single_caps: np.ndarray,
    cluster_ids: "np.ndarray | None" = None,
    cluster_limits: "dict | None" = None,
) -> np.ndarray:
    """合成单票目标仓位。返回各票权重(和 <= w_total,绝不补杠杆)。

    步骤:w̃=1/N 归一 -> target=w̃·w_total -> 逐票被 single_caps 截断 -> 簇限制按比例压缩。
    """
    atr = np.asarray(atr, dtype="float64")
    single_caps = np.asarray(single_caps, dtype="float64")
    if len(atr) != len(single_caps):
        raise ValueError("atr 与 single_caps 长度不一致")
    if len(atr) == 0:
        return np.array([], dtype="float64")

    w = inverse_atr_weights(atr) * w_total
    w = np.minimum(w, single_caps)  # 逐票单股上限截断(不补杠杆)

    if cluster_ids is not None and cluster_limits:
        cluster_ids = np.asarray(cluster_ids)
        for cid in np.unique(cluster_ids):
            limit = cluster_limits.get(cid if not isinstance(cid, np.generic) else cid.item())
            if limit is None:
                continue
            mask = cluster_ids == cid
            csum = w[mask].sum()
            if csum > limit and csum > 0:
                w[mask] = w[mask] * (limit / csum)  # 按比例压缩到簇上限
    return w
