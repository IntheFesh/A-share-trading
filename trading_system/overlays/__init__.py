"""Phase 2:overlay(披露季 + 高低切),均须 overlay test。任务 2.6。对应 v3.1 §5.4。

披露季 overlay(三档保守默认):①已发负面预告->可否决;②临近披露无负面预告->默认只降仓/
缩短持有,不直接禁新开;③触发器在披露窗历史表现不差->暂不启用待验证。业绩预告强制与否按
交易所规则版本表读取,不硬编码;只用 PIT 预告事实字段。
高低切/过度拉升交互:只以 HiLo_t × 过度拉升度 交互形式进入(INV-7),禁止无条件给动量股扣分。
overlay test:每个 overlay 跑"带/不带"两条扣费净值,通过 ΔMaxDD<0 且 ΔCalmar>0
(过度拉升交互用 ΔRankIC>0 且 ΔMaxDD≤0)才启用,否则弃用。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.6"


def disclosure_season_overlay(*args, **kwargs):  # noqa: ANN002, ANN003
    """披露季 overlay(三档保守默认)。"""
    raise NotImplementedError(f"{_PHASE}:disclosure_season_overlay 待实现(v3.1 §5.4)。")


def hilo_overcrowding_interaction(*args, **kwargs):  # noqa: ANN002, ANN003
    """高低切 × 过度拉升交互(INV-7:仅交互形式)。"""
    raise NotImplementedError(f"{_PHASE}:hilo_overcrowding_interaction 待实现(INV-7)。")


def overlay_test(*args, **kwargs):  # noqa: ANN002, ANN003
    """overlay test 框架:带/不带两条扣费净值对比,判定是否启用。"""
    raise NotImplementedError(f"{_PHASE}:overlay_test 待实现。")
