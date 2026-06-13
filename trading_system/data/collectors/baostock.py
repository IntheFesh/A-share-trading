"""BaoStock 采集器(**唯一信息来源**:行情 / 交易日历 / ST 状态 / 退市 / 上市日)。Phase 0。对应 v3.1 §2.2。

架构纪律(用户约束):本系统的"信息来源"只有 BaoStock;Tushare 仅用于财报(业绩预告/预约披露日),
不作行情/退市/日历的备源。后复权也只用本源(INV-2,不跨源拼接复权价)。

**adjustflag 复权约定(以 BaoStock 官方中文文档为准,勿被英文博客误译):**
  ``1 = 后复权(hfq)``、``2 = 前复权(qfq)``、``3 = 不复权``。
本系统执行用不复权(3),特征/复权因子用后复权(1),后复权只用本源(INV-2,不跨源拼接)。
``fetch_raw_with_factor`` 同时取 raw(3)与 hfq(1),令 ``adj_factor = close_hfq / close_raw``,
产出 schema.RAW_INPUT_FIELDS(+ PIT 的 ``is_st``),交 price_layers.build_price_layers 派生 adj 与状态位。

日线字段:date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,adjustflag。
其中 ``tradestatus``(1 正常交易/0 停牌)、``isST``(1 当日为 ST,**PIT 历史状态**)是 BaoStock 直接给的,
比"量=0 推停牌""默认非 ST"更准——尤其 isST 决定 ST 的 5% 涨跌停(否则误用 10%)。
注意:BaoStock 不支持多线程(串行 + 重试);需登录会话。重型依赖惰性导入;网络路径离线 NOT RUN。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# query_history_k_data_plus 日线字段(不复权/后复权同字段集);含 turn/tradestatus/isST。
_K_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,isST,adjustflag"


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
    """取单只票日线;adjustflag:1 后复权 / 2 前复权 / 3 不复权。数值列转 float;tradestatus/isST 保留字符串。"""
    import pandas as pd

    df = _query_k(code, start, end, adjustflag)
    for col in ("open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["date"])
    return df


def fetch_raw_with_factor(code: str, start: str, end: str):
    """同时取 raw(adjustflag=3)与 hfq(adjustflag=1),产出 schema.RAW_INPUT_FIELDS。

    ``adj_factor = close_hfq / close_raw``(后复权因子);特征层 adj 由 build_price_layers 重建。
    """
    import numpy as np
    import pandas as pd

    raw = fetch_daily(code, start, end, adjustflag=3)   # 3 = 不复权(执行层)
    hfq = fetch_daily(code, start, end, adjustflag=1)   # 1 = 后复权(派生因子)
    merged = raw.merge(
        hfq[["trade_date", "close"]].rename(columns={"close": "close_hfq"}),
        on="trade_date",
        how="left",
    )
    close_raw = merged["close"].to_numpy("float64")
    # adj_factor = 后复权收盘 / 不复权收盘;close_raw<=0(异常/停牌无价)记 NaN,避免 inf
    factor = np.where(close_raw > 0, merged["close_hfq"].to_numpy("float64") / close_raw, np.nan)
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
            "turn": merged["turn"] if "turn" in merged.columns else np.nan,  # 换手率(CGO/换手族)
            "adj_factor": factor,
            # PIT 历史 ST 状态:决定 5% 涨跌停(build_price_layers 会用到 is_st)
            "is_st": (merged["isST"].astype(str) == "1") if "isST" in merged.columns else False,
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


def query_stock_basic(code: str = "", code_name: str = ""):
    """证券基本资料(BaoStock 为唯一信息来源:上市/退市状态、上市日、退市日)。

    返回列:code, code_name, ipoDate, outDate, type(1 股票/2 指数/…), status(1 上市/0 退市)。
    用途:退市股识别(status=0 / outDate 非空,回测防幸存者偏差)、次新股 60 日窗(ipoDate)。
    **退市/ST/日历一律走 BaoStock,不用 Tushare(Tushare 仅财报)。**
    """
    import baostock as bs
    import pandas as pd

    rs = bs.query_stock_basic(code=code, code_name=code_name)
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock 基本资料查询失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)
