"""BaoStock 采集器(主源)。Phase 0(任务 0.2)。对应 v3.1 §2.2。

后复权(adjustflag=1)与不复权(adjustflag=3)同源可取;后复权只用本源(INV-2)。
``fetch_raw_with_factor`` 同时取 raw 与 hfq,令 ``adj_factor = close_hfq / close_raw``,
产出 schema.RAW_INPUT_FIELDS,交给 price_layers.build_price_layers 派生 adj 与状态位。
注意:BaoStock 不支持多线程(串行 + 重试);需登录会话。重型依赖(baostock/pandas)惰性导入。
本模块的网络路径需 BaoStock 可达,无法在离线 CI 验证(对应测试 skip 并写明原因)。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# query_history_k_data_plus 的字段(不复权/后复权同字段集)
_K_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag"


def login():
    """登录 BaoStock;失败抛 RuntimeError(诚实失败,不静默)。"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_code} {lg.error_msg}")
    logger.info("BaoStock login ok")
    return lg


def logout() -> None:
    import baostock as bs

    bs.logout()


def _query_k(code: str, start: str, end: str, adjustflag: int):
    import baostock as bs
    import pandas as pd

    rs = bs.query_history_k_data_plus(
        code,
        _K_FIELDS,
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=str(adjustflag),
    )
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock 查询失败 {code}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def fetch_daily(code: str, start: str, end: str, adjustflag: int):
    """取单只票日线;adjustflag=1 后复权 / 3 不复权。返回原始字符串列已转 float 的 DataFrame。"""
    import pandas as pd

    df = _query_k(code, start, end, adjustflag)
    for col in ("open", "high", "low", "close", "preclose", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["date"])
    return df


def fetch_raw_with_factor(code: str, start: str, end: str):
    """同时取 raw(adjustflag=3)与 hfq(adjustflag=1),产出 schema.RAW_INPUT_FIELDS。

    ``adj_factor = close_hfq / close_raw``(后复权因子);特征层 adj 由 build_price_layers 重建。
    """
    import numpy as np
    import pandas as pd

    raw = fetch_daily(code, start, end, adjustflag=3)
    hfq = fetch_daily(code, start, end, adjustflag=1)
    merged = raw.merge(
        hfq[["trade_date", "close"]].rename(columns={"close": "close_hfq"}),
        on="trade_date",
        how="left",
    )
    factor = merged["close_hfq"].to_numpy("float64") / merged["close"].to_numpy("float64")
    out = pd.DataFrame(
        {
            "code": code,
            "trade_date": merged["trade_date"],
            "open_raw": merged["open"],
            "high_raw": merged["high"],
            "low_raw": merged["low"],
            "close_raw": merged["close"],
            "preclose_raw": merged["preclose"],
            "volume": merged["volume"],
            "amount": merged["amount"],
            "adj_factor": factor,
        }
    )
    return out


def query_trade_dates(start: str, end: str):
    """取交易日历(用于 calendar 缓存)。返回列 [calendar_date, is_trading_day]。"""
    import baostock as bs
    import pandas as pd

    rs = bs.query_trade_dates(start_date=start, end_date=end)
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock 日历查询失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(columns={"calendar_date": "calendar_date", "is_trading_day": "is_trading_day"})
    df["is_trading_day"] = df["is_trading_day"].astype(int).astype(bool)
    return df
