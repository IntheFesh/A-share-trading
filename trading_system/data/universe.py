"""交易池过滤。Phase 0(任务 0.5)。对应 v3.1 §1.1。价格层:状态位用 raw 派生。

沪深主板(600/601/603/605/000/001/002),剔除 ST/*ST、退市整理、停牌、上市未满 60 日次新、入场时刻一字板。
退市股历史数据必须保留在 store(回测防幸存者偏差),但其退市后不进当日可交易池。
规则阈值默认值对齐 config/data.yaml 的 universe 块(逻辑里不写死,调用方从 config 传入)。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# 沪深主板全集(用户账户可交易范围)。三位精确前缀,startswith 匹配——唯一真相源。
# 沪市主板:600/601/603/605;深市主板:000/001/002(002 原中小板已并入深主板)。
# 刻意排除:创业板 300/301、科创板 688、北交所 8xx/4xx/920、深B 200、沪B 900。
# 注意:必须用三位精确前缀,不能用两位 "00"/"60"(会误收深B 200 等),逐项验证无误收无漏收。
MAIN_BOARD_PREFIXES: "tuple[str, ...]" = ("600", "601", "603", "605", "000", "001", "002")


def stock_code_core(code: str) -> str:
    """从 'sh.600000' / '600000.SH' / '600000' 等格式抽取 6 位股票代码核心。"""
    digits = re.sub(r"\D", "", str(code))
    if len(digits) < 6:
        return digits
    # 取前 6 位(各源里 6 位股票代码总在最前)
    return digits[:6]


def board_allowed(code: str, boards: "tuple[str, ...] | list[str]") -> bool:
    """代码是否属于允许的板块前缀(如 ('60','000') = 沪深主板)。"""
    core = stock_code_core(code)
    return any(core.startswith(b) for b in boards)


def filter_universe(
    panel: pd.DataFrame,
    *,
    boards: "tuple[str, ...] | list[str]" = MAIN_BOARD_PREFIXES,
    exclude_st: bool = True,
    exclude_suspended: bool = True,
    new_listing_min_days: int = 60,
    exclude_one_price_at_entry: bool = True,
    delisting_col: "str | None" = None,
) -> pd.DataFrame:
    """在 panel 上追加布尔列 ``is_in_universe``(逐 (code, trade_date) 判定)。

    panel 须含 code, trade_date, is_st, is_suspended, is_one_price_limit。
    上市天数按"该 code 在 panel 中累计出现的交易日数"计(要求 panel 自上市首日起;
    否则请用真实上市日另算)。delisting_col 若提供,则该列为 True 的行排除。
    """
    required = {"code", "trade_date", "is_st", "is_suspended", "is_one_price_limit"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"filter_universe 缺少列: {sorted(missing)}")

    out = panel.sort_values(["code", "trade_date"]).copy()

    board_ok = out["code"].map(lambda c: board_allowed(c, boards)).to_numpy(dtype=bool)
    # 上市满 N 个交易日:同一 code 的第 k 行(从 1 计)>= new_listing_min_days
    listing_days = out.groupby("code", sort=False).cumcount().to_numpy() + 1
    seasoned = listing_days >= int(new_listing_min_days)

    mask = board_ok & seasoned
    if exclude_st:
        mask &= ~out["is_st"].to_numpy(dtype=bool)
    if exclude_suspended:
        mask &= ~out["is_suspended"].to_numpy(dtype=bool)
    if exclude_one_price_at_entry:
        mask &= ~out["is_one_price_limit"].to_numpy(dtype=bool)
    if delisting_col is not None:
        if delisting_col not in out.columns:
            raise ValueError(f"delisting_col '{delisting_col}' 不在 panel 中")
        mask &= ~out[delisting_col].to_numpy(dtype=bool)

    out["is_in_universe"] = mask
    return out.reset_index(drop=True)
