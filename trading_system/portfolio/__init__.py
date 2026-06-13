"""Phase 2:L3 仓位合成。任务 2.5。对应 v3.1 §10.3。

凯利符号三档:r_t = r_0 * 1{f*>0} * s(r_0=0.5%,s∈{1,0.5,0});
单股上限连续跌停压力公式(INV-5,见 trading_system.invariants.single_name_cap);
波动率目标 + 多重 min:w_total = min(0.5, sigma*/sigma_hat, m_t, w_liquidity, w_limitdown);
合成:ATR 定相对权重 w_tilde_i,单票目标=w_tilde_i×w_total,被单票上限逐票截断,
截断后总和<w_total 则保持不补杠杆;拥挤簇限制(普通≤25%,小市值+低流动性+同题材≤15%)。
参数从 config/risk.yaml 读。价格层:ATR / 波动用 adj;涨跌停约束用 raw(INV-2)。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.5"


def compose_positions(*args, **kwargs):  # noqa: ANN002, ANN003
    """合成目标仓位(相对权重 × 总敞口 × 多重 min,逐票截断)。"""
    raise NotImplementedError(f"{_PHASE}:compose_positions 待实现(v3.1 §10.3 / INV-5)。")
