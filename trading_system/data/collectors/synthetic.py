"""合成数据源(测试 / 开发用,**非真实市场数据**)。Phase 0。

用途:在无网络 / 无 token 的环境里,用**已知性质**的确定性数据端到端验证
calendar→price_layers→universe→store→quality 的逻辑正确性(对标 v3.1 附录的构造性方法)。
**严禁**把本源产出当作市场实证;真实数据采集走 baostock.py / tushare.py,真实验收见 run/。
价格层:产出 RAW_INPUT_FIELDS(原始价 + adj_factor),adj 与状态位由 build_price_layers 派生。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from trading_system.data.calendar import TradingCalendar


def make_calendar(start: str, n_days: int) -> TradingCalendar:
    """从 start 起取 n_days 个工作日(周一~周五)作为合成交易日历(忽略节假日,够测试用)。"""
    dates: list[dt.date] = []
    d = dt.date.fromisoformat(start)
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    return TradingCalendar(dates)


def make_raw_panel(
    codes: "list[str]",
    calendar: TradingCalendar,
    *,
    seed: int = 0,
    start_price: float = 10.0,
    max_abs_daily: float = 0.08,
) -> pd.DataFrame:
    """生成"干净"的随机游走原始价面板(日收益限制在 ±max_abs_daily,故不会误触涨跌停)。

    adj_factor 恒为 1.0(无除权);preclose_raw = 前一交易日 close_raw,首日 = start_price。
    用于 store / universe / 流水线测试。需要除权/涨跌停/停牌等**确切事件**的测试请在用例内
    手工构造小表(全控、可手算),不要依赖随机生成器。
    """
    rng = np.random.default_rng(seed)
    dates = calendar.dates
    rows = []
    for ci, code in enumerate(codes):
        rets = rng.uniform(-max_abs_daily, max_abs_daily, size=len(dates))
        close = start_price * np.cumprod(1.0 + rets)
        prev_close = np.empty_like(close)
        prev_close[0] = start_price
        prev_close[1:] = close[:-1]
        # 用收盘附近的小幅扰动构造 O/H/L,保证 low<=O,C<=high
        intraday = np.abs(rng.uniform(0.0, 0.01, size=len(dates))) * close
        open_ = close - rng.uniform(-0.005, 0.005, size=len(dates)) * close
        high = np.maximum(open_, close) + intraday
        low = np.minimum(open_, close) - intraday
        vol = rng.integers(1_000_00, 5_000_00, size=len(dates)).astype("float64")
        for i, d in enumerate(dates):
            rows.append(
                {
                    "code": code,
                    "trade_date": pd.Timestamp(d),
                    "open_raw": round(float(open_[i]), 2),
                    "high_raw": round(float(high[i]), 2),
                    "low_raw": round(float(low[i]), 2),
                    "close_raw": round(float(close[i]), 2),
                    "preclose_raw": round(float(prev_close[i]), 2),
                    "volume": vol[i],
                    "amount": round(float(vol[i] * close[i]), 2),
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)
