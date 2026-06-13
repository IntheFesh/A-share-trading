"""BaoStock 采集器(主源)。Phase 0(任务 0.2)。对应 v3.1 第二章。

日线后复权(adjustflag=1)与不复权(adjustflag=3)同源可取;返回统一 schema 的 DataFrame。
注意:BaoStock 不支持多线程,用多进程或串行 + 重试。后复权(adj)只用本源(INV-2)。
重型依赖(baostock/pandas)在函数内导入,保持模块 import 轻量。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.2"


def login():
    """登录 BaoStock 会话。"""
    raise NotImplementedError(f"{_PHASE}:baostock.login 待实现。")


def logout():
    """登出 BaoStock 会话。"""
    raise NotImplementedError(f"{_PHASE}:baostock.logout 待实现。")


def fetch_daily(code: str, start: str, end: str, adjustflag: int):
    """取日线:adjustflag=1 后复权 / 3 不复权。返回统一 schema 的 DataFrame。"""
    raise NotImplementedError(f"{_PHASE}:baostock.fetch_daily 待实现。")
