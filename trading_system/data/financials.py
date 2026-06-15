"""财务过滤层(批 3:避雷逻辑,不是 alpha,不进模型打分、不需 RankIC 验证)。

把批 2 采集的季频财务,接成一个**可见性正确(PIT)的过滤器**:构建交易池时额外剔除"基本面恶化"的票。
用户硬纪律"剔除业绩预警/连续亏损"由此自动化。**默认关闭**(enable_financial_filter=False),不破坏 baseline。

★ 红线:可见性对齐必须用 pubDate(实际公告日)。对任一交易日 t,该票可用的最新一期财务 = 满足
  pubDate <= t 的、最近公告的那一期(merge_asof backward on pubDate)。绝不能用 statDate(报告期),
  否则会在报告期当日就"看到"尚未公告的财报,构成未来函数泄漏,使回测虚高、实盘失效。

⚠ 单位假设(需用户用真实 BaoStock 输出确认):本模块默认把 YOYNI / liabilityToAsset 当作**小数**
  (0.85 = 85%、-0.50 = -50%)。若实际 BaoStock 返回的是百分数(85、-50),请在 config/函数参数调整阈值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.data.schema import FINANCIAL_NUMERIC_FIELDS
from trading_system.data.universe import filter_universe

# 过滤默认阈值(温和保守:宁可少杀不可错杀;均可配)。单位假设见模块 docstring。
DEFAULT_MAX_LIABILITY_TO_ASSET = 0.80     # 资产负债率上限(> 此值视为畸高 → 剔除)
DEFAULT_YOYNI_DETERIORATE = -0.50         # 净利润同比恶化阈值(< 此值且当期亏损 → 剔除)

# 对齐后附加到日频面板的财务来源日期列名(便于核对 PIT)。
FIN_STATDATE_COL = "fin_statDate"
FIN_PUBDATE_COL = "fin_pubDate"


def attach_visible_financials(
    daily_panel: pd.DataFrame,
    fin_panel: "pd.DataFrame | None",
    *,
    fields: "tuple[str, ...] | list[str]" = FINANCIAL_NUMERIC_FIELDS,
) -> pd.DataFrame:
    """对日频面板每个 (code, trade_date),join 上"在该交易日已公告的最新一期财务"(PIT)。

    规则:取满足 ``pubDate <= trade_date`` 的、最近公告的一期(``pd.merge_asof`` backward on pubDate),
    **绝不用 statDate**。从未公告/次新(无可见财报)→ 财务列为 NaN(不剔除,交由过滤层决定)。
    返回:daily_panel 原列 + fin_statDate / fin_pubDate + 各财务字段(缺失为 NaN),保持原行序。
    """
    fields = list(fields)
    out = daily_panel.reset_index(drop=True).copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])

    if fin_panel is None or len(fin_panel) == 0:
        for c in fields:
            out[c] = np.nan
        out[FIN_STATDATE_COL] = pd.NaT
        out[FIN_PUBDATE_COL] = pd.NaT
        return out

    fin = fin_panel.copy()
    fin["pubDate"] = pd.to_datetime(fin["pubDate"], errors="coerce")
    fin["statDate"] = pd.to_datetime(fin["statDate"], errors="coerce")
    fin = fin.dropna(subset=["pubDate"])              # 无公告日 → 不可 PIT 对齐,排除出右表
    # 同一 (code, pubDate) 取 statDate 最大(保险:同日多期取最新报告期)
    fin = (fin.sort_values(["code", "pubDate", "statDate"])
              .drop_duplicates(["code", "pubDate"], keep="last"))
    right = fin[["code", "pubDate", "statDate", *[c for c in fields if c in fin.columns]]].rename(
        columns={"statDate": FIN_STATDATE_COL, "pubDate": FIN_PUBDATE_COL})

    out["__pos"] = np.arange(len(out))
    left_sorted = out.sort_values("trade_date")       # merge_asof 要求按 on 键全局有序
    right_sorted = right.sort_values(FIN_PUBDATE_COL)
    merged = pd.merge_asof(
        left_sorted, right_sorted,
        left_on="trade_date", right_on=FIN_PUBDATE_COL,
        by="code", direction="backward",             # 取 pubDate <= trade_date 的最近一期
    )
    merged = merged.sort_values("__pos").reset_index(drop=True).drop(columns="__pos")
    for c in fields:                                  # 该票无任何可见财报 → 列缺失则补 NaN
        if c not in merged.columns:
            merged[c] = np.nan
    return merged


def financial_exclusion_mask(
    panel: pd.DataFrame,
    *,
    max_liability_to_asset: float = DEFAULT_MAX_LIABILITY_TO_ASSET,
    yoyni_deteriorate: float = DEFAULT_YOYNI_DETERIORATE,
) -> pd.Series:
    """在已 PIT 对齐财务的面板上,返回布尔 Series(True = 基本面恶化应剔除)。

    规则(温和保守,缺失放行):
      1) 当期净利润为负(netProfit < 0)**且**净利润同比大幅恶化(YOYNI < yoyni_deteriorate)→ 剔除;
      2) 资产负债率畸高(liabilityToAsset > max_liability_to_asset)→ 剔除;
      3) 财务缺失(NaN,从未公告/次新)→ **不剔除**(NaN 比较为 False,自动放行,避免误杀次新)。
    """
    npf = pd.to_numeric(panel.get("netProfit"), errors="coerce")
    yoy = pd.to_numeric(panel.get("YOYNI"), errors="coerce")
    lta = pd.to_numeric(panel.get("liabilityToAsset"), errors="coerce")
    loss_and_deteriorate = (npf < 0) & (yoy < yoyni_deteriorate)   # 亏损 且 同比大幅恶化
    high_leverage = lta > max_liability_to_asset
    # NaN 的比较结果为 False → 缺失自动放行;fillna(False) 双保险
    return (loss_and_deteriorate | high_leverage).fillna(False)


def filter_universe_with_financials(
    daily_panel: pd.DataFrame,
    fin_panel: "pd.DataFrame | None" = None,
    *,
    enable_financial_filter: bool = False,
    max_liability_to_asset: float = DEFAULT_MAX_LIABILITY_TO_ASSET,
    yoyni_deteriorate: float = DEFAULT_YOYNI_DETERIORATE,
    **filter_universe_kwargs,
) -> pd.DataFrame:
    """先跑既有 filter_universe(板块/ST/停牌/次新/一字板/退市),再(可选)叠加财务过滤。

    enable_financial_filter=False(默认)→ 行为与 filter_universe 完全一致(不破坏 baseline)。
    开启时:PIT 对齐财务后,把"基本面恶化"的 (code,trade_date) 从 is_in_universe 中剔除(AND ~mask)。
    """
    out = filter_universe(daily_panel, **filter_universe_kwargs)
    if not enable_financial_filter:
        return out
    aligned = attach_visible_financials(out, fin_panel)
    exclude = financial_exclusion_mask(
        aligned, max_liability_to_asset=max_liability_to_asset, yoyni_deteriorate=yoyni_deteriorate)
    aligned["is_in_universe"] = aligned["is_in_universe"].to_numpy(dtype=bool) & ~exclude.to_numpy(dtype=bool)
    return aligned
