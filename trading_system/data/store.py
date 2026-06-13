"""存储与增量更新(单一数据出口)。Phase 0(任务 0.4)。对应 v3.1 第二章。

Parquet(ZSTD)按年分区落盘;DuckDB 直接对 Parquet 跑 SQL。
``read(...)`` 是所有模块取数的**唯一入口**,保证沙盒/回测/生产看到同一份数据。
增量更新永不全量重拉。价格层:返回 raw 与 adj 双层(见 price_layers)。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.4"


def read(codes, start, end, fields):  # noqa: ANN001 — Phase 0 落地时定型
    """唯一取数入口:按代码 / 区间 / 字段读取 Parquet(raw+adj 双层)。"""
    raise NotImplementedError(f"{_PHASE}:read 待实现(单一数据出口)。")


def update_incremental():
    """增量更新:每只票只拉 ``max(本地)+1 → 今日``,append 后按 (code, trade_date) 去重。"""
    raise NotImplementedError(f"{_PHASE}:update_incremental 待实现。")


def query(sql: str):
    """DuckDB 查询接口:直接对 Parquet 跑 SQL。"""
    raise NotImplementedError(f"{_PHASE}:query 待实现。")
