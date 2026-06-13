"""双价格层构造(INV-2 核心)。Phase 0(任务 0.3)。对应 v3.1 第二章 / INV-2。

输出表同时含原始价与后复权价 + 状态位 + 披露字段(PIT):
  执行类(raw):open_raw/high_raw/low_raw/close_raw/preclose_raw/volume/amount/adj_factor
  特征类(adj,后复权,只用 BaoStock 一家来源):open_adj/high_adj/low_adj/close_adj
  状态位:is_suspended/is_st/is_limit_up/is_limit_down/is_one_price_limit
  披露(PIT):sched_disclosure_date/has_preann/preann_sign/days_to_disclosure

纪律:涨跌停价用 raw 昨收(见 trading_system.invariants.limit_up_price);
后复权绝不与 Tushare 复权价拼接;PIT 绝不用最终实际披露日回填。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.3"


def build_price_layers(*args, **kwargs):  # noqa: ANN002, ANN003 — Phase 0 落地时定型签名
    """由原始 OHLCV + 复权因子构造含 raw/adj 双层 + 状态位 + 披露字段的完整表。"""
    raise NotImplementedError(f"{_PHASE}:build_price_layers 待实现(INV-2)。")
