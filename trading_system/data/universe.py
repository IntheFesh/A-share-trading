"""交易池过滤。Phase 0(任务 0.5)。对应 v3.1 §1.1。

沪深主板(60/000 开头),剔除 ST/*ST、退市整理、停牌、上市未满 60 日次新、入场时刻一字板。
必须含退市股历史数据(回测防幸存者偏差):Tushare 退市列表 + BaoStock 历史拼全。
规则阈值从 ``config/data.yaml`` 的 ``universe`` 读取(禁止硬编码)。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.5"


def filter_universe(*args, **kwargs):  # noqa: ANN002, ANN003
    """按 v3.1 §1.1 过滤交易池,返回每个交易日的可交易代码集合。"""
    raise NotImplementedError(f"{_PHASE}:filter_universe 待实现(v3.1 §1.1)。")
