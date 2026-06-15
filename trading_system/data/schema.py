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

# 特征层附加列(来自 BaoStock 日线)。无来源时为 NaN。
#  - turn:换手率,服务换手率族与 CGO 族(已接入)。
#  - peTTM / pbMRQ / psTTM:估值字段(批 5,alpha 采集)。**仅采集落盘备用,进模型前需单独
#    RankIC/ICIR 验证**(用户纪律:alpha 因子排队验证,默认不臃肿);当前无任何因子引用它们。
FEATURE_EXTRA_FIELDS: tuple[str, ...] = ("turn", "peTTM", "pbMRQ", "psTTM")
# 估值字段(批 5):仅采集,未接入打分。单列出来便于"哪些是待验证 alpha"一目了然。
VALUATION_FIELDS: tuple[str, ...] = ("peTTM", "pbMRQ", "psTTM")

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

# 季频财务字段(来自 BaoStock 免费财务接口,独立频率/独立落盘,不混进日频 RAW_INPUT_FIELDS)。
# PIT 关键:必须保留 pubDate(实际公告日)——可见性对齐只能用 pubDate,绝不能用 statDate(报告期),
# 否则会在报告期当日就"看到"尚未公告的财报,构成未来函数泄漏。字段名以 BaoStock 实际返回为准:
#   roeAvg=净资产收益率(query_profit_data)、netProfit=净利润(query_profit_data)、
#   YOYNI=净利润同比(query_growth_data)、liabilityToAsset=资产负债率(query_balance_data)。
FINANCIAL_FIELDS: tuple[str, ...] = (
    "code",
    "statDate",
    "pubDate",
    "roeAvg",
    "netProfit",
    "YOYNI",
    "liabilityToAsset",
)
# 财务面板主键(季频,与日频行情的 (code, trade_date) 不同)。
FINANCIAL_KEY_FIELDS: tuple[str, ...] = ("code", "statDate")
# 财务数值列(落盘前转 float;日期列 statDate/pubDate 转 datetime)。
FINANCIAL_NUMERIC_FIELDS: tuple[str, ...] = ("roeAvg", "netProfit", "YOYNI", "liabilityToAsset")

# 行业分类字段(批 4;来自 BaoStock query_stock_industry,低频近静态,独立落盘,主键 code)。
# industryClassification=分类标准(申万),industry=所属行业名。供行业中性化/板块共振识别用。
INDUSTRY_FIELDS: tuple[str, ...] = ("code", "industry", "industryClassification")

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
