"""Tushare 采集器——**仅财报获取**(业绩预告 + 预约披露日)。Phase 0(任务 0.2)。对应 v3.1 §5.4。

**架构纪律(用户约束):Tushare 不作信息来源,只用于财报。** 行情 / 交易日历 / ST 状态 / 退市列表
一律来自 BaoStock(唯一信息来源,见 collectors/baostock.py;后复权单一来源 INV-2)。本模块因此
**只保留**:预约披露日(disclosure_date)、业绩预告(forecast)。不提供日线、不提供退市/日历。
token 优先读环境变量 TUSHARE_TOKEN,其次 config/data.yaml。需 token + 网络,离线 NOT RUN。
披露字段遵守 PIT:按公告日对齐,绝不用最终实际披露日回填;只取事实字段(是否已发、方向、公告日、类型)。
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


def fetch_disclosure_date(period: str, config_token: "str | None" = None):
    """预约披露日(财报;PIT:存当时公告版本)。period 形如 '20240331'。"""
    pro = _pro(config_token)
    return pro.disclosure_date(end_date=period)


def fetch_forecast(period: str, config_token: "str | None" = None):
    """业绩预告(财报;PIT 事实字段:是否已发 / 方向 / 公告日 / 类型)。"""
    pro = _pro(config_token)
    return pro.forecast(period=period)
