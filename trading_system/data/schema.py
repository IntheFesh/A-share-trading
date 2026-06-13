"""统一数据表 schema(v3.1 §2.1)。Phase 0。

集中定义列名分组,服务 INV-2:执行类只取 RAW_PRICE_FIELDS,特征类只取 ADJ_PRICE_FIELDS。
store.read(fields=...) 配合本文件做"按用途取数",从数据出口处就支持 raw/adj 分离。
"""

from __future__ import annotations

KEY_FIELDS: tuple[str, ...] = ("code", "trade_date")

# 执行层(原始价;成交/涨跌停/止损止盈/股数/成本/PnL 只用这些)——INV-2
RAW_PRICE_FIELDS: tuple[str, ...] = (
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "preclose_raw",
)
VOLUME_FIELDS: tuple[str, ...] = ("volume", "amount")

# 特征层(后复权价;收益/均线/波动/CGO/RPS 等只用这些)——INV-2
ADJ_FACTOR_FIELD: str = "adj_factor"
ADJ_PRICE_FIELDS: tuple[str, ...] = (
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
)

# 特征层附加列(换手率;来自 BaoStock 日线,服务换手率族与 CGO 族)。无来源时为 NaN。
FEATURE_EXTRA_FIELDS: tuple[str, ...] = ("turn",)

# 状态位
STATE_FIELDS: tuple[str, ...] = (
    "is_suspended",
    "is_st",
    "is_limit_up",
    "is_limit_down",
    "is_one_price_limit",
)

# 披露季事件字段(PIT;来自 Tushare,token-gated)
DISCLOSURE_FIELDS: tuple[str, ...] = (
    "sched_disclosure_date",
    "has_preann",
    "preann_sign",
    "days_to_disclosure",
)

# 构造 build_price_layers 的最小输入列(不含 adj/状态/披露,这些由本系统派生)
RAW_INPUT_FIELDS: tuple[str, ...] = (
    *KEY_FIELDS,
    *RAW_PRICE_FIELDS,
    *VOLUME_FIELDS,
    ADJ_FACTOR_FIELD,
)

# 双价格层 + 状态位的完整列(不含披露;披露由 attach_disclosure_fields 追加)
PRICE_LAYER_FIELDS: tuple[str, ...] = (
    *KEY_FIELDS,
    *RAW_PRICE_FIELDS,
    *VOLUME_FIELDS,
    ADJ_FACTOR_FIELD,
    *ADJ_PRICE_FIELDS,
    *FEATURE_EXTRA_FIELDS,
    *STATE_FIELDS,
)

ALL_FIELDS: tuple[str, ...] = (*PRICE_LAYER_FIELDS, *DISCLOSURE_FIELDS)


def assert_no_adj_in_execution(fields: "list[str] | tuple[str, ...]") -> None:
    """便捷转发:执行用字段不得含后复权列(INV-2)。见 invariants.assert_execution_uses_raw。"""
    from trading_system.invariants import assert_execution_uses_raw

    assert_execution_uses_raw(fields)
