"""Tushare 采集器(备源 + 披露日历)。Phase 0(任务 0.2)。对应 v3.1 §2.2 / §5.4。

token 优先读环境变量 TUSHARE_TOKEN,其次读 config/data.yaml。不与 BaoStock 复权价拼接(INV-2)。
披露字段遵守 PIT:预约披露日按交易所当时公告版本,改期按改期公告更新,绝不用实际披露日回填;
业绩预告只取事实字段(是否已发、方向、公告日、类型),不在代码里判断"是否应该发"。
本模块需 token + 网络,离线 CI 无法验证(对应测试 skip 并写明原因)。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_token(config_token: "str | None" = None) -> str:
    """获取 token:环境变量 TUSHARE_TOKEN 优先,其次传入的 config 值。缺失则抛错。"""
    token = os.environ.get("TUSHARE_TOKEN") or (config_token or "")
    if not token:
        raise RuntimeError(
            "Tushare token 缺失:请设置环境变量 TUSHARE_TOKEN 或在 config/data.yaml 填入。"
        )
    return token


def _pro(config_token: "str | None" = None):
    import tushare as ts

    ts.set_token(get_token(config_token))
    return ts.pro_api()


def fetch_daily(code: str, start: str, end: str, config_token: "str | None" = None):
    """日线(备源,不复权)。code 用 tushare 的 ts_code(如 '600000.SH')。"""
    pro = _pro(config_token)
    return pro.daily(ts_code=code, start_date=start, end_date=end)


def fetch_disclosure_date(period: str, config_token: "str | None" = None):
    """预约披露日(PIT:存当时公告版本)。period 形如 '20240331'。"""
    pro = _pro(config_token)
    return pro.disclosure_date(end_date=period)


def fetch_forecast(period: str, config_token: "str | None" = None):
    """业绩预告:是否已发 / 方向 / 公告日 / 类型(PIT 事实字段)。"""
    pro = _pro(config_token)
    return pro.forecast(period=period)


def fetch_delist_list(config_token: "str | None" = None):
    """退市股列表(回测防幸存者偏差):取 list_status='D' 的标的。"""
    pro = _pro(config_token)
    return pro.stock_basic(list_status="D", fields="ts_code,name,list_date,delist_date")
