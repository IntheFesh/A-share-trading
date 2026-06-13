"""Phase 1:标签构造(INV-1 + INV-3)。任务 1.3。对应 v3.1 第三章。

  y_prod :生产可交易标签(raw 价,tau_exit >= t+2,扣成本)。
  y_h    :固定窗口对照,h ∈ {1,2,3,5,8,10}。
  y_mtm0 :诊断标签(h=0,diagnostic 命名空间,禁止进训练/回测/审批)。

INV-3:T+1 一字/高开>7% 的成交判定从 backtest.engine 导入**同一个** is_tradeable_fill,
不另写一份(下方 re-export 使 labels.is_tradeable_fill 与 engine 的为同一对象)。
INV-1:生产/对照标签 horizon>=1(故 exit=t+1+h >= t+2);h=0 只许 diagnostic 命名空间。
价格层:入场=t+1 open(raw)、出场=close(raw),成交价/PnL 全用 raw(INV-2)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# INV-3:权威成交判定函数,从引擎导入(re-export);labels.is_tradeable_fill IS engine 的同一对象。
from trading_system.backtest.engine import DEFAULT_GAP_ABANDON, is_tradeable_fill
from trading_system.invariants import (
    DIAGNOSTIC_NAMESPACE,
    PRODUCTION_NAMESPACE,
    assert_label_namespace_allows_horizon,
    assert_production_label_horizon,
    assert_tradeable_exit,
)

__all__ = [
    "is_tradeable_fill",
    "build_y_h",
    "build_y_prod",
    "build_y_mtm0",
]

_RAW_COLS = ("open_raw", "close_raw", "preclose_raw", "is_one_price_limit", "is_limit_up")


def _label_code(g: pd.DataFrame, h: int, cost: float, gap_threshold: float) -> np.ndarray:
    """单只票:信号在 t、入场 t+1 开盘、出场 t+1+h 收盘的可交易收益(不可成交记 NaN)。"""
    o = g["open_raw"].to_numpy(dtype="float64")
    c = g["close_raw"].to_numpy(dtype="float64")
    pc = g["preclose_raw"].to_numpy(dtype="float64")
    one = g["is_one_price_limit"].to_numpy(dtype=bool)
    lu = g["is_limit_up"].to_numpy(dtype=bool)
    n = len(g)
    arr = np.full(n, np.nan, dtype="float64")
    for t in range(n):
        e, x = t + 1, t + 1 + h  # 入场日、出场日索引
        if x >= n:
            continue
        if pc[e] <= 0 or o[e] <= 0:  # 停牌/异常价:不可入场
            continue
        # INV-3:用共享函数判定 T+1 是否可成交(一字涨停/高开>阈值)
        if not is_tradeable_fill(
            open_price=o[e],
            preclose=pc[e],
            is_one_price_limit_up=bool(one[e] and lu[e]),
            gap_threshold=gap_threshold,
        ):
            continue
        arr[t] = c[x] / o[e] - 1.0 - cost
    return arr


def build_y_h(
    panel: pd.DataFrame,
    h: int,
    *,
    cost: float = 0.0,
    gap_threshold: float = DEFAULT_GAP_ABANDON,
    namespace: str = PRODUCTION_NAMESPACE,
) -> pd.Series:
    """固定窗口标签 y_h(h>=1)。返回与 panel(排序后)对齐的 Series;不可成交处为 NaN。"""
    assert_production_label_horizon(h)        # INV-1:h>=1
    assert_tradeable_exit(0, 1 + h)           # INV-1:exit=t+1+h >= t+2,与不变量函数挂钩
    assert_label_namespace_allows_horizon(namespace, h)
    p = panel.sort_values(["code", "trade_date"]).copy()
    y = pd.Series(np.nan, index=p.index, dtype="float64")
    for _, g in p.groupby("code", sort=False):
        y.loc[g.index] = _label_code(g, h, cost, gap_threshold)
    return y


def build_y_prod(
    panel: pd.DataFrame,
    *,
    holding_days: int,
    cost: float,
    gap_threshold: float = DEFAULT_GAP_ABANDON,
) -> pd.Series:
    """生产可交易标签 y_prod:raw 价、tau_exit=t+1+holding_days(>=t+2)、扣往返成本 cost。

    说明:Phase 1 的 y_prod 为"固定持有 + 扣成本"的可交易收益,用于因子统计;含硬止损/止盈的
    完整出场状态机版本在 Phase 2 引擎落地(那才是进正式净值/审批的 y_prod)。
    """
    assert holding_days >= 1, "holding_days 必须 >= 1(INV-1)"
    return build_y_h(
        panel, holding_days, cost=cost, gap_threshold=gap_threshold,
        namespace=PRODUCTION_NAMESPACE,
    )


def build_y_mtm0(panel: pd.DataFrame) -> pd.Series:
    """诊断标签 y_mtm0(h=0):close_raw[t+1]/open_raw[t+1]-1。仅 diagnostic 命名空间。

    禁止进训练/回测/审批(INV-1)。本函数显式声明诊断命名空间,绝不当生产标签。
    """
    assert_label_namespace_allows_horizon(DIAGNOSTIC_NAMESPACE, 0)  # h=0 仅诊断
    p = panel.sort_values(["code", "trade_date"]).copy()
    y = pd.Series(np.nan, index=p.index, dtype="float64")
    for _, g in p.groupby("code", sort=False):
        o = g["open_raw"].to_numpy(dtype="float64")
        c = g["close_raw"].to_numpy(dtype="float64")
        n = len(g)
        arr = np.full(n, np.nan, dtype="float64")
        for t in range(n):
            e = t + 1
            if e >= n or o[e] <= 0:
                continue
            arr[t] = c[e] / o[e] - 1.0
        y.loc[g.index] = arr
    return y
