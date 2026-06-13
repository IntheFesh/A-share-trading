"""交易日历。Phase 0(任务 0.1)。对应 v3.1 §2。价格层:与价格无关(仅日期)。

来源:BaoStock ``query_trade_dates``,落本地 Parquet。本模块提供:
  - ``TradingCalendar`` 类(可由日期列表直接构造,便于单测,无需 I/O);
  - 模块级便捷函数 get_trading_days / shift_trading_day / is_trading_day,作用于一个
    全局默认日历(由 ``set_default_calendar`` 注入,或从 store 缓存加载)。
T+1 / T+2 / embargo 全部由这里的 shift_trading_day 计算,杜绝自然日错算。
"""

from __future__ import annotations

import bisect
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

DateLike = Union[str, dt.date, dt.datetime, "object"]  # 也接受 pd.Timestamp


def to_date(value: DateLike) -> dt.date:
    """把 str('YYYY-MM-DD') / date / datetime / pd.Timestamp 归一化为 datetime.date。"""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    # pandas.Timestamp 或 numpy datetime64:有 .date() 或可转 str
    if hasattr(value, "date") and callable(value.date):
        return value.date()  # type: ignore[no-any-return]
    return dt.date.fromisoformat(str(value)[:10])


class TradingCalendar:
    """不可变的交易日历;内部是升序去重的 ``datetime.date`` 列表。"""

    def __init__(self, dates: Iterable[DateLike]) -> None:
        self._dates: list[dt.date] = sorted({to_date(d) for d in dates})
        self._pos: dict[dt.date, int] = {d: i for i, d in enumerate(self._dates)}

    def __len__(self) -> int:
        return len(self._dates)

    @property
    def dates(self) -> list[dt.date]:
        return list(self._dates)

    def is_trading_day(self, date: DateLike) -> bool:
        return to_date(date) in self._pos

    def get_trading_days(self, start: DateLike, end: DateLike) -> list[dt.date]:
        """返回 [start, end] 闭区间内的交易日(升序)。start/end 不必是交易日。"""
        s, e = to_date(start), to_date(end)
        if s > e:
            return []
        lo = bisect.bisect_left(self._dates, s)
        hi = bisect.bisect_right(self._dates, e)
        return self._dates[lo:hi]

    def shift_trading_day(self, date: DateLike, n: int) -> dt.date:
        """返回 ``date`` 之后第 ``n`` 个交易日(n 可负)。要求 ``date`` 本身是交易日。

        用于 T+1(n=1)、T+2(n=2)、embargo 偏移。越界抛 IndexError(诚实失败,不静默回退)。
        """
        d = to_date(date)
        i = self._pos.get(d)
        if i is None:
            raise KeyError(f"{d} 不是交易日,shift 无定义;请先对齐到交易日。")
        j = i + n
        if j < 0 or j >= len(self._dates):
            raise IndexError(f"shift_trading_day 越界:{d} 偏移 {n} 超出日历范围。")
        return self._dates[j]

    def offset(self, start: DateLike, end: DateLike) -> int:
        """返回从 start 到 end 的交易日个数差(end_index - start_index)。

        二者均须为交易日。end 在 start 之后为正。用于 days_to_disclosure、标签窗计数。
        """
        i, j = self._pos.get(to_date(start)), self._pos.get(to_date(end))
        if i is None or j is None:
            raise KeyError("offset 的两端都必须是交易日。")
        return j - i

    def next_trading_day_on_or_after(self, date: DateLike) -> dt.date:
        """返回 >= date 的最早交易日(date 本身是交易日则返回它)。越界抛 IndexError。"""
        d = to_date(date)
        i = bisect.bisect_left(self._dates, d)
        if i >= len(self._dates):
            raise IndexError(f"{d} 之后(含)无交易日。")
        return self._dates[i]


# ── 全局默认日历(模块级便捷函数用)─────────────────────────────────────────
_DEFAULT: Optional[TradingCalendar] = None


def set_default_calendar(calendar: TradingCalendar) -> None:
    """注入全局默认日历(测试或生产装配时调用)。"""
    global _DEFAULT
    _DEFAULT = calendar


def get_default_calendar() -> TradingCalendar:
    if _DEFAULT is None:
        raise RuntimeError(
            "默认交易日历未初始化:请先 set_default_calendar(...),"
            "或 load_calendar_from_parquet(...) 后注入。"
        )
    return _DEFAULT


def load_calendar_from_parquet(path: "str | Path") -> TradingCalendar:
    """从本地 Parquet 缓存(列含 calendar_date / is_trading_day)加载交易日历。"""
    import pandas as pd

    df = pd.read_parquet(path)
    col_date = "calendar_date" if "calendar_date" in df.columns else df.columns[0]
    if "is_trading_day" in df.columns:
        df = df[df["is_trading_day"].astype(bool)]
    return TradingCalendar(df[col_date].tolist())


def get_trading_days(start: DateLike, end: DateLike) -> list[dt.date]:
    return get_default_calendar().get_trading_days(start, end)


def shift_trading_day(date: DateLike, n: int) -> dt.date:
    return get_default_calendar().shift_trading_day(date, n)


def is_trading_day(date: DateLike) -> bool:
    return get_default_calendar().is_trading_day(date)
