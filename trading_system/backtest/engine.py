"""事件级回测引擎(唯一真值,最核心)。Phase 2(任务 2.2)。对应 v3.1 第十一章。

买入状态机:signal(t) -> entry_pending -> {entry_filled / entry_failed_limitup /
  entry_failed_gap}。entry_date = t+1,entry_price = open_raw[t+1];一字涨停或高开>7% 放弃。
卖出状态机:最早可卖 t+2(INV-1)。出场触发**收盘确认、次开执行**;买入当日(D=t+1)不判止损,
  故触发评估从 D+1(=t+2)收盘起,执行最早在 D+2(=t+3)开盘。涨停/跌停/停牌致次开卖不出则顺延
  (exit_delayed_limitdown / exit_delayed_suspension)。跳空使实际亏损可 > 2.5N,按次开实际价记账。
出场优先级(代码按序判定):硬止损 > 止盈阶梯 > 时间止损(事件止损/披露 veto 由 overlays 注入)。
止盈:+1R 减 1/3;+2R 再减 1/3 且余仓止损移成本;余 1/3 跟踪 max(最高收盘)-c·N。

INV-2:所有撮合/涨跌停/PnL 用 raw 列(下方对执行列做 assert_execution_uses_raw 守卫)。
INV-3:成交判定 is_tradeable_fill 的权威实现在本模块,labels/ 从此 import,不另写一份。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from trading_system.invariants import assert_execution_uses_raw, assert_tradeable_exit

_PHASE = "Phase 2 任务 2.2"

# 默认放弃高开阈值:T+1 开盘相对昨收高开 > 7% 则放弃买入(v3.1 第十一章)。
DEFAULT_GAP_ABANDON = 0.07

# 引擎执行路径只读这些 raw 列(INV-2)。
_EXEC_RAW_COLS = ("open_raw", "high_raw", "low_raw", "close_raw", "preclose_raw")
_STATE_COLS = ("is_limit_up", "is_limit_down", "is_suspended", "is_one_price_limit")


def is_tradeable_fill(
    *,
    open_price: float,
    preclose: float,
    is_one_price_limit_up: bool,
    gap_threshold: float = DEFAULT_GAP_ABANDON,
) -> bool:
    """INV-3 权威成交判定(买入侧、基于价格)。labels/ 与引擎共用本函数,不得各写一份。

    规则(T+1 开盘):一字涨停 -> 买不进(False);高开 (open/preclose-1) > gap_threshold ->
    放弃(False);否则价格上可成交(True)。流动性/参与量(≤竞价量 1%)是引擎用成交量的额外
    闸门(标签侧无盘中量,无法判),不在本共享函数内——共享的是"一字/高开"价格判定(INV-3)。
    价格层:全用 raw(INV-2)。
    """
    if is_one_price_limit_up:
        return False
    if preclose <= 0:
        raise ValueError("preclose 必须 > 0(raw 昨收)")
    if open_price / preclose - 1.0 > gap_threshold:
        return False
    return True


def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR(N):真实波幅的滚动均值。用 raw OHLC + raw 昨收(INV-2)。"""
    high = bars["high_raw"].to_numpy(dtype="float64")
    low = bars["low_raw"].to_numpy(dtype="float64")
    prev_close = bars["preclose_raw"].to_numpy(dtype="float64")
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return pd.Series(tr, index=bars.index).rolling(period).mean()


@dataclass
class Fill:
    exec_idx: int
    fraction: float
    price: float
    reason: str  # tp1 / tp2 / stop / time / eod


@dataclass
class TradeResult:
    status: str            # closed / entry_failed_limitup / entry_failed_gap / no_entry_data
    signal_idx: int
    entry_idx: Optional[int] = None
    entry_price: Optional[float] = None
    exit_idx: Optional[int] = None
    fills: list = field(default_factory=list)
    gross_return: float = float("nan")
    net_return: float = float("nan")

    @property
    def reasons(self) -> list:
        return [f.reason for f in self.fills]


def simulate_trade(
    bars: pd.DataFrame,
    signal_idx: int,
    *,
    atr: float,
    atr_mult: float = 2.5,
    trail_c: float = 2.5,
    max_holding: int = 10,
    gap_threshold: float = DEFAULT_GAP_ABANDON,
    cost_fraction: float = 0.0,
) -> TradeResult:
    """模拟单只票一笔交易的完整生命周期(逐日事件驱动)。bars 为该票按日排序的面板(raw + 状态位)。

    atr 为信号时点的 N(每股价格单位)。返回含成交分笔、毛/净收益、出场原因的 TradeResult。
    """
    missing = [c for c in (*_EXEC_RAW_COLS, *_STATE_COLS) if c not in bars.columns]
    if missing:
        raise ValueError(f"simulate_trade 缺少执行列: {missing}")
    assert_execution_uses_raw(list(_EXEC_RAW_COLS))  # INV-2:执行路径只用 raw

    b = bars.reset_index(drop=True)
    n = len(b)
    t = signal_idx
    e = t + 1
    o = b["open_raw"].to_numpy(dtype="float64")
    c = b["close_raw"].to_numpy(dtype="float64")
    pc = b["preclose_raw"].to_numpy(dtype="float64")
    is_lu = b["is_limit_up"].to_numpy(dtype=bool)
    is_ld = b["is_limit_down"].to_numpy(dtype=bool)
    is_susp = b["is_suspended"].to_numpy(dtype=bool)
    is_one = b["is_one_price_limit"].to_numpy(dtype=bool)

    if e >= n:
        return TradeResult("no_entry_data", t)

    # 买入状态机:T+1 开盘成交判定(INV-3)
    if not is_tradeable_fill(
        open_price=o[e], preclose=pc[e],
        is_one_price_limit_up=bool(is_one[e] and is_lu[e]), gap_threshold=gap_threshold,
    ):
        status = "entry_failed_limitup" if (is_one[e] and is_lu[e]) else "entry_failed_gap"
        return TradeResult(status, t, entry_idx=e)
    entry = float(o[e])
    if entry <= 0:
        return TradeResult("entry_failed_gap", t, entry_idx=e)

    risk = atr_mult * atr
    stop = entry - risk           # 硬止损(2.5N)
    tp1, tp2 = entry + risk, entry + 2.0 * risk
    remaining = 1.0
    tp1_done = tp2_done = False
    max_close = c[e]
    fills: list[Fill] = []
    pending: Optional[tuple[float, str]] = None

    # 触发评估从 D+1(=t+2)收盘起(买入当日不判止损);执行在次开。
    d = e + 1
    while d < n and remaining > 1e-9:
        # 1) 次开执行待出场单(涨停封死/停牌则顺延)
        if pending is not None and not (is_susp[d] or is_ld[d]):
            frac, reason = pending
            frac = min(frac, remaining)
            fills.append(Fill(d, frac, float(o[d]), reason))
            remaining -= frac
            pending = None
            if remaining <= 1e-9:
                break
        # 2) 收盘确认触发(优先级:硬止损 > 止盈阶梯 > 时间止损)
        cc = c[d]
        max_close = max(max_close, cc)
        if tp2_done:
            stop = max(stop, max_close - trail_c * atr)  # 余仓跟踪止损
        if pending is None:
            if cc <= stop:
                pending = (remaining, "stop")
            elif (not tp1_done) and cc >= tp1:
                pending = (1.0 / 3.0, "tp1")
                tp1_done = True
            elif (not tp2_done) and cc >= tp2:
                pending = (1.0 / 3.0, "tp2")
                tp2_done = True
                stop = max(stop, entry)  # 余仓止损移成本
            elif (d - e) >= max_holding:
                pending = (remaining, "time")
        d += 1

    # 数据耗尽仍有持仓:按最后收盘 mark-to-market 了结
    if remaining > 1e-9:
        last = n - 1
        fills.append(Fill(last, remaining, float(c[last]), "eod"))

    if not fills:
        return TradeResult("no_exit_data", t, entry_idx=e, entry_price=entry)

    first_exit = fills[0].exec_idx
    assert_tradeable_exit(t, first_exit)  # INV-1:出场 >= t+2
    avg_exit = sum(f.fraction * f.price for f in fills)  # 各 fraction 之和 = 1.0
    gross = avg_exit / entry - 1.0
    return TradeResult(
        "closed", t, entry_idx=e, entry_price=entry, exit_idx=fills[-1].exec_idx,
        fills=fills, gross_return=gross, net_return=gross - cost_fraction,
    )


class BacktestEngine:
    """组合级回放:对一组信号逐笔调用 simulate_trade(唯一真值)。"""

    def __init__(self, *, atr_period: int = 14, cost_fraction: float = 0.0, **trade_params) -> None:
        self.atr_period = atr_period
        self.cost_fraction = cost_fraction
        self.trade_params = trade_params

    def run(self, panel: pd.DataFrame, signals: pd.DataFrame) -> list[TradeResult]:
        """signals: 含 [code, signal_date] 的表。返回每个信号的 TradeResult。"""
        results: list[TradeResult] = []
        panel = panel.sort_values(["code", "trade_date"])
        for _, sig in signals.iterrows():
            g = panel[panel["code"] == sig["code"]].reset_index(drop=True)
            pos = g.index[g["trade_date"] == pd.Timestamp(sig["signal_date"])]
            if len(pos) == 0:
                continue
            t = int(pos[0])
            atr_series = compute_atr(g, self.atr_period)
            atr_val = atr_series.iloc[t]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            results.append(
                simulate_trade(
                    g, t, atr=float(atr_val), cost_fraction=self.cost_fraction,
                    **self.trade_params,
                )
            )
        return results
