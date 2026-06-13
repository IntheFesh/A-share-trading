"""Phase 1:L1 触发器(粗桶,禁止网格寻优)。任务 1.5。对应 v3.1 第六章。

A 牛回头 / B 缩量低位首板 / C RPS 龙头。**阈值只从 config/triggers.yaml 的粗桶取单一边界传入,
本模块不做任何 grid search / 最优点挑选**(这是结构性纪律,见 v3.1 第六章)。
触发器是事件候选定义,非买入命令。价格层:趋势/位置/动量用 adj;首板=raw 昨收算的涨停状态。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _by_code_rolling(p: pd.DataFrame, col: str, window: int, how: str) -> pd.Series:
    g = p.groupby("code")[col]
    r = g.transform(lambda s: getattr(s.rolling(window), how)())
    return r


def trigger_pullback(
    panel: pd.DataFrame,
    *,
    drawdown_low: float = 0.05,
    drawdown_high: float = 0.15,
    ma_short: int = 20,
    ma_long: int = 60,
) -> pd.Series:
    """A 牛回头:上升趋势(MA短>MA长)中,自近 ma_short 日高点回撤幅度落在 [low, high] 粗桶。

    drawdown_low/high 来自 config 的回撤粗桶边界(单一桶,非网格)。价格层:adj。
    """
    p = panel.sort_values(["code", "trade_date"]).copy()
    ma_s = _by_code_rolling(p, "close_adj", ma_short, "mean")
    ma_l = _by_code_rolling(p, "close_adj", ma_long, "mean")
    roll_high = _by_code_rolling(p, "close_adj", ma_short, "max")
    drawdown = 1.0 - p["close_adj"] / roll_high  # >=0
    uptrend = ma_s > ma_l
    trig = uptrend & (drawdown >= drawdown_low) & (drawdown <= drawdown_high)
    return pd.Series(trig.to_numpy(dtype=bool), index=p.index, name="trigger_pullback")


def trigger_first_board(
    panel: pd.DataFrame,
    *,
    low_position: float = 0.20,
    volume_shrink_max: float = 0.70,
    high_window: int = 60,
    vol_window: int = 5,
) -> pd.Series:
    """B 缩量低位首板:今首封涨停(昨未封)且位置低(距 high_window 高点 ≥ low_position)
    且量比 ≤ volume_shrink_max。阈值来自 config 粗桶。价格层:涨停=raw;位置/量=派生。
    """
    p = panel.sort_values(["code", "trade_date"]).copy()
    is_lu = p["is_limit_up"].to_numpy(dtype=bool)
    prev_lu = p.groupby("code")["is_limit_up"].shift(1).fillna(False).to_numpy(dtype=bool)
    first_board = is_lu & ~prev_lu
    roll_high = _by_code_rolling(p, "close_adj", high_window, "max")
    below_high = 1.0 - p["close_adj"] / roll_high  # 距高点比例
    low_pos = below_high.to_numpy() >= low_position
    prior_vol_mean = p.groupby("code")["volume"].transform(
        lambda s: s.shift(1).rolling(vol_window).mean()
    )
    vol_ratio = p["volume"] / prior_vol_mean
    shrink = vol_ratio.to_numpy() <= volume_shrink_max
    trig = first_board & low_pos & shrink
    return pd.Series(trig, index=p.index, name="trigger_first_board")


def trigger_rps_leader(
    panel: pd.DataFrame,
    *,
    rps_window: int = 120,
    rps_min: float = 0.90,
) -> pd.Series:
    """C RPS 龙头:个股近 rps_window 日动量的当日**截面分位** ≥ rps_min(粗桶阈值)。价格层:adj。"""
    p = panel.sort_values(["code", "trade_date"]).copy()
    p["__mom__"] = p.groupby("code")["close_adj"].pct_change(rps_window)
    rps = p.groupby("trade_date")["__mom__"].rank(pct=True)
    trig = rps.to_numpy() >= rps_min
    return pd.Series(np.where(np.isnan(rps.to_numpy()), False, trig), index=p.index,
                     name="trigger_rps_leader")
