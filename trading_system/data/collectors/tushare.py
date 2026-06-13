"""Tushare 采集器(备源 + 披露日历)。Phase 0(任务 0.2)。对应 v3.1 第二章。

token 从 ``config/data.yaml``(或环境变量 TUSHARE_TOKEN)读。不与 BaoStock 复权价拼接(INV-2)。
披露字段遵守 PIT:预约披露日按交易所当时公告版本,改期按改期公告更新,绝不用实际披露日回填。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.2"


def fetch_daily(code: str, start: str, end: str):
    """日线(备源,不复权)。"""
    raise NotImplementedError(f"{_PHASE}:tushare.fetch_daily 待实现。")


def fetch_disclosure_date(*args, **kwargs):  # noqa: ANN002, ANN003
    """预约披露日(PIT:存当时公告版本)。"""
    raise NotImplementedError(f"{_PHASE}:tushare.fetch_disclosure_date 待实现。")


def fetch_forecast(*args, **kwargs):  # noqa: ANN002, ANN003
    """业绩预告:是否已发 / 方向 / 公告日 / 类型(PIT 事实字段)。"""
    raise NotImplementedError(f"{_PHASE}:tushare.fetch_forecast 待实现。")


def fetch_delist_list(*args, **kwargs):  # noqa: ANN002, ANN003
    """退市股列表(回测防幸存者偏差)。"""
    raise NotImplementedError(f"{_PHASE}:tushare.fetch_delist_list 待实现。")
