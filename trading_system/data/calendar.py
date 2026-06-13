"""交易日历。Phase 0(任务 0.1)。对应 v3.1 第二章。

来源:BaoStock ``query_trade_dates``,落本地 Parquet。价格层:与价格无关(仅日期)。
供 T+1 / T+2 / embargo 计算使用。
"""

from __future__ import annotations

import datetime as _dt

_PHASE = "Phase 0 任务 0.1"


def get_trading_days(start: str, end: str) -> list[_dt.date]:
    """返回 [start, end] 区间内的交易日列表(升序)。"""
    raise NotImplementedError(f"{_PHASE}:get_trading_days 待实现。")


def shift_trading_day(date: _dt.date, n: int) -> _dt.date:
    """返回 ``date`` 之后第 ``n`` 个交易日(n 可为负)。用于 T+1、T+2、embargo。"""
    raise NotImplementedError(f"{_PHASE}:shift_trading_day 待实现。")


def is_trading_day(date: _dt.date) -> bool:
    """``date`` 是否为交易日。"""
    raise NotImplementedError(f"{_PHASE}:is_trading_day 待实现。")
