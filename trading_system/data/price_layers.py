"""双价格层构造(INV-2 核心)。Phase 0(任务 0.3)。对应 v3.1 §2.1 / INV-2。

输出表同时含原始价(执行层)与后复权价(特征层)+ 状态位 + 披露字段(PIT):
  执行类(raw):open/high/low/close/preclose_raw、volume、amount、adj_factor
  特征类(adj,后复权):open/high/low/close_adj = 对应 raw × adj_factor
  状态位:is_suspended / is_st / is_limit_up / is_limit_down / is_one_price_limit
  披露(PIT):sched_disclosure_date / has_preann / preann_sign / days_to_disclosure

纪律(INV-2):
  - 涨跌停价**只用原始昨收** preclose_raw:limit = round(preclose_raw×(1±ratio), 2);
    用后复权昨收会算错(除权日 preclose_raw 已是除权参考价,adj 昨收不是)。
  - 后复权只用单一来源(BaoStock)的 adj_factor 派生,绝不与其它复权价拼接。
  - 披露字段遵守 PIT:预约披露日按交易所当时公告版本,绝不用最终实际披露日回填。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from trading_system.data.calendar import TradingCalendar, to_date
from trading_system.data.schema import (
    ADJ_FACTOR_FIELD,
    FEATURE_EXTRA_FIELDS,
    PRICE_LAYER_FIELDS,
    RAW_INPUT_FIELDS,
)
from trading_system.invariants import round_half_up_2

_RAW_TO_ADJ = (
    ("open_raw", "open_adj"),
    ("high_raw", "high_adj"),
    ("low_raw", "low_adj"),
    ("close_raw", "close_adj"),
)

# 默认涨跌停比例(主板 10%、ST 5%);科创/创业等其它比例由调用方传入,禁止硬编码于逻辑。
MAIN_BOARD_RATIO = 0.10
ST_RATIO = 0.05
# 收盘价与涨跌停价比较的容差(半分,抵消浮点),用于判定"封死"在涨跌停。
LIMIT_TOL = 0.005


def build_price_layers(
    df: pd.DataFrame,
    *,
    main_board_ratio: float = MAIN_BOARD_RATIO,
    st_ratio: float = ST_RATIO,
    limit_tol: float = LIMIT_TOL,
) -> pd.DataFrame:
    """由原始 OHLCV + adj_factor 构造含 raw/adj 双层 + 状态位的完整表(单只票或多票)。

    输入须含 RAW_INPUT_FIELDS(code, trade_date, *_raw, volume, amount, adj_factor),
    可选含布尔列 ``is_st``(历史 PIT 状态;缺省视为 False)。
    后复权:``*_adj = *_raw × adj_factor``。状态位用 raw + preclose_raw 计算(INV-2)。
    """
    missing = set(RAW_INPUT_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"build_price_layers 缺少输入列: {sorted(missing)}")

    out = df.copy()
    if "is_st" not in out.columns:
        out["is_st"] = False
    out["is_st"] = out["is_st"].astype(bool)
    for extra in FEATURE_EXTRA_FIELDS:   # 特征层附加列(turn / 估值 peTTM/pbMRQ/psTTM);无来源时 NaN
        if extra not in out.columns:
            out[extra] = np.nan

    # ── 特征层:后复权 = 原始价 × 复权因子 ──
    factor = out[ADJ_FACTOR_FIELD].to_numpy(dtype="float64")
    for raw_col, adj_col in _RAW_TO_ADJ:
        out[adj_col] = out[raw_col].to_numpy(dtype="float64") * factor

    # ── 状态位:涨跌停价只用原始昨收 preclose_raw(INV-2)──
    ratio = np.where(out["is_st"].to_numpy(dtype=bool), st_ratio, main_board_ratio)
    preclose = out["preclose_raw"].to_numpy(dtype="float64")
    limit_up = round_half_up_2(preclose * (1.0 + ratio))
    limit_down = round_half_up_2(preclose * (1.0 - ratio))

    o = out["open_raw"].to_numpy(dtype="float64")
    h = out["high_raw"].to_numpy(dtype="float64")
    low = out["low_raw"].to_numpy(dtype="float64")
    c = out["close_raw"].to_numpy(dtype="float64")
    vol = out["volume"].to_numpy(dtype="float64")

    is_limit_up = np.abs(c - limit_up) <= limit_tol           # 收盘封死涨停
    is_limit_down = np.abs(c - limit_down) <= limit_tol       # 收盘封死跌停
    # 一字板:全日 O=H=L=C 且 = 涨/跌停价
    ohlc_flat = (
        (np.abs(o - h) <= limit_tol)
        & (np.abs(h - low) <= limit_tol)
        & (np.abs(low - c) <= limit_tol)
    )
    is_one_price = ohlc_flat & (is_limit_up | is_limit_down)
    is_suspended = vol == 0.0

    out["is_suspended"] = is_suspended
    out["is_limit_up"] = is_limit_up
    out["is_limit_down"] = is_limit_down
    out["is_one_price_limit"] = is_one_price

    return out[list(PRICE_LAYER_FIELDS)].reset_index(drop=True)


def compute_days_to_disclosure(
    trade_dates: "pd.Series | list",
    sched_dates: "pd.Series | list",
    calendar: TradingCalendar,
) -> list[float]:
    """逐行计算"距预约披露日的交易日数"(PIT)。

    定义:从 trade_date(不含)到 sched_disclosure_date(含)之间的交易日个数;
    当日披露=0;无预约日或已披露(sched 在 trade_date 之前)记 NaN。
    sched 若非交易日,顺延到其后最近交易日再计数。价格层:与价格无关。
    """
    result: list[float] = []
    for td, sd in zip(list(trade_dates), list(sched_dates)):
        if sd is None or (isinstance(sd, float) and np.isnan(sd)) or pd.isna(sd):
            result.append(float("nan"))
            continue
        td_d, sd_d = to_date(td), to_date(sd)
        if sd_d < td_d:
            result.append(float("nan"))
            continue
        try:
            td_aligned = calendar.next_trading_day_on_or_after(td_d)
            sd_aligned = calendar.next_trading_day_on_or_after(sd_d)
            result.append(float(calendar.offset(td_aligned, sd_aligned)))
        except (KeyError, IndexError):
            result.append(float("nan"))
    return result


def attach_disclosure_fields(
    df: pd.DataFrame,
    *,
    sched_disclosure: pd.DataFrame,
    preann: pd.DataFrame,
    calendar: TradingCalendar,
) -> pd.DataFrame:
    """追加披露季 PIT 字段(token-gated:sched/preann 由 Tushare 提供)。

    参数:
      sched_disclosure: 列 [code, sched_disclosure_date](每只票当前预约披露日;PIT 版本)。
      preann: 列 [code, ann_date, preann_sign](业绩预告事实,按公告日 ann_date 对齐)。
    输出在 df 上追加 sched_disclosure_date / has_preann / preann_sign / days_to_disclosure。
    has_preann/preann_sign 为"截至该 trade_date 已公告"的 as-of 结果(绝不前视)。
    """
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])

    sched = sched_disclosure.set_index("code")["sched_disclosure_date"]
    out["sched_disclosure_date"] = pd.to_datetime(out["code"].map(sched))

    out["days_to_disclosure"] = compute_days_to_disclosure(
        out["trade_date"], out["sched_disclosure_date"], calendar
    )

    # as-of 业绩预告:对每只票,取 ann_date <= trade_date 的最近一条
    has_preann = np.zeros(len(out), dtype=bool)
    preann_sign = np.zeros(len(out), dtype="float64")
    if preann is not None and len(preann) > 0:
        pa = preann.copy()
        pa["ann_date"] = pd.to_datetime(pa["ann_date"])
        by_code = {code: g.sort_values("ann_date") for code, g in pa.groupby("code")}
        codes = out["code"].to_numpy()
        tds = out["trade_date"].to_numpy()
        for i in range(len(out)):
            g = by_code.get(codes[i])
            if g is None:
                continue
            past = g[g["ann_date"].to_numpy() <= tds[i]]
            if len(past) > 0:
                has_preann[i] = True
                preann_sign[i] = float(past["preann_sign"].iloc[-1])
    out["has_preann"] = has_preann
    out["preann_sign"] = preann_sign
    return out
