"""Phase 2:overlay(披露季 + 高低切),均须 overlay test。任务 2.6。对应 v3.1 §5.4 / 第十三章。

披露季 overlay(三档保守默认):①已发负面预告->veto;②临近披露无负面预告->降仓(reduce);
③否则 none(不一刀切,INV-7)。只用 PIT 预告事实字段,不硬编码"应否发预告"。
高低切/过度拉升:只以 HiLo_t × 过度拉升度 交互形式进入(INV-7),禁止无条件给动量股扣分。
overlay test:带/不带两条扣费净值,通过 ΔMaxDD<0 且 ΔCalmar>0(过度拉升交互用 ΔRankIC>0 且
ΔMaxDD≤0)才启用,否则弃用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_system.backtest.metrics import calmar, max_drawdown
from trading_system.invariants import assert_conditional_or_documented_override

VETO, REDUCE, NONE = "veto", "reduce", "none"


def disclosure_season_overlay(
    panel: pd.DataFrame, *, window: int = 10
) -> pd.Series:
    """披露季 overlay 动作(逐行)。需 days_to_disclosure / has_preann / preann_sign(PIT)。

    ① has_preann 且 preann_sign<0 -> VETO(已发负面预告);
    ② 临近披露(0<=days_to_disclosure<=window)且无负面预告 -> REDUCE(降仓/缩短持有,不禁新开);
    ③ 其余 -> NONE。返回动作码 Series。
    """
    req = {"days_to_disclosure", "has_preann", "preann_sign"}
    missing = req - set(panel.columns)
    if missing:
        raise ValueError(f"披露 overlay 缺字段(需 Tushare PIT): {sorted(missing)}")
    d2d = panel["days_to_disclosure"].to_numpy(dtype="float64")
    has = panel["has_preann"].to_numpy(dtype=bool)
    sign = panel["preann_sign"].to_numpy(dtype="float64")
    action = np.full(len(panel), NONE, dtype=object)
    near = (~np.isnan(d2d)) & (d2d >= 0) & (d2d <= window)
    action[near] = REDUCE
    action[has & (sign < 0)] = VETO  # 负面预告优先级最高
    return pd.Series(action, index=panel.index, name="disclosure_action")


def hilo_overextension_interaction(hilo_t: np.ndarray, overextension: np.ndarray) -> np.ndarray:
    """高低切 × 过度拉升的**交互项**(INV-7:只能以交互形式进入,禁止无条件扣分)。

    返回 hilo_t × overextension;高低切 regime(hilo_t 高,高位股跑输)下放大过度拉升的负作用。
    本函数显式声明为条件化(非无条件),通过 INV-7 守卫。
    """
    assert_conditional_or_documented_override(is_unconditional=False)  # 条件化,合规
    return np.asarray(hilo_t, dtype="float64") * np.asarray(overextension, dtype="float64")


@dataclass
class OverlayTestResult:
    dd_without: float
    dd_with: float
    delta_maxdd: float      # = dd_with - dd_without(回撤幅度差,<0 表示 overlay 降低回撤)
    calmar_without: float
    calmar_with: float
    delta_calmar: float
    enable: bool


def overlay_test(nav_without: "np.ndarray", nav_with: "np.ndarray") -> OverlayTestResult:
    """风险型 overlay 的启用判定:带/不带两条扣费净值,通过 ΔMaxDD<0 且 ΔCalmar>0 才启用。

    这里 MaxDD 取**正幅度**(|回撤|)便于"ΔMaxDD<0=回撤变小"的直觉与 v3.1 措辞一致。
    """
    dd0, dd1 = abs(max_drawdown(nav_without)), abs(max_drawdown(nav_with))
    cal0, cal1 = calmar(nav_without), calmar(nav_with)
    delta_dd = dd1 - dd0
    delta_cal = cal1 - cal0
    enable = (delta_dd < 0) and (delta_cal > 0)
    return OverlayTestResult(dd0, dd1, delta_dd, cal0, cal1, delta_cal, enable)


def overextension_interaction_test(
    rank_ic_without: float, rank_ic_with: float, dd_without: float, dd_with: float
) -> bool:
    """过度拉升交互的启用判定:ΔRankIC>0 且 ΔMaxDD<=0(|回撤|不增)。"""
    delta_ic = rank_ic_with - rank_ic_without
    delta_dd = abs(dd_with) - abs(dd_without)
    return (delta_ic > 0) and (delta_dd <= 0)
