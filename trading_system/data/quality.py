"""数据质检(每日跑)。Phase 0(任务 0.6)。对应 v3.1 §2.3。价格层:连续性用 adj;涨跌停核对用 raw。

每个检查返回 (check, status, n_flagged, detail);status ∈ {PASS, WARN, FAIL, SKIP}。
SKIP 仅用于"确实缺少所需输入(如披露字段 token-gated)"且明确写出原因——绝不用来掩盖失败。
``run_daily_quality_checks`` 汇总所有检查;``assert_passed`` 在任一 FAIL 时抛错(供验收脚本)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from trading_system.data.price_layers import MAIN_BOARD_RATIO, ST_RATIO
from trading_system.data.schema import ADJ_PRICE_FIELDS, RAW_PRICE_FIELDS
from trading_system.invariants import round_half_up_2

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class CheckResult:
    check: str
    status: str
    n_flagged: int
    detail: str


def _limit_prices(panel: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
    is_st = panel.get("is_st", pd.Series(False, index=panel.index)).to_numpy(dtype=bool)
    ratio = np.where(is_st, ST_RATIO, MAIN_BOARD_RATIO)
    pc = panel["preclose_raw"].to_numpy(dtype="float64")
    return round_half_up_2(pc * (1 + ratio)), round_half_up_2(pc * (1 - ratio))


def check_raw_adj_separation(panel: pd.DataFrame) -> CheckResult:
    have_raw = all(c in panel.columns for c in RAW_PRICE_FIELDS)
    have_adj = all(c in panel.columns for c in ADJ_PRICE_FIELDS)
    if have_raw and have_adj:
        return CheckResult("raw_adj_separation", PASS, 0, "raw 与 adj 双层均在")
    return CheckResult(
        "raw_adj_separation", FAIL, 0, f"缺列 raw={have_raw} adj={have_adj}"
    )


def check_limit_price_consistency(panel: pd.DataFrame, tol: float = 0.005) -> CheckResult:
    """is_limit_up/down 的行,收盘价必须等于按 raw 昨收重算的涨/跌停价(INV-2)。"""
    lu, ld = _limit_prices(panel)
    c = panel["close_raw"].to_numpy(dtype="float64")
    up = panel["is_limit_up"].to_numpy(dtype=bool)
    dn = panel["is_limit_down"].to_numpy(dtype=bool)
    bad_up = up & (np.abs(c - lu) > tol)
    bad_dn = dn & (np.abs(c - ld) > tol)
    n = int(bad_up.sum() + bad_dn.sum())
    if n == 0:
        return CheckResult("limit_price_consistency", PASS, 0, "涨跌停收盘=raw 昨收重算值")
    return CheckResult("limit_price_consistency", FAIL, n, f"{n} 行涨跌停价与 raw 昨收不符")


def check_adj_continuity(panel: pd.DataFrame, max_daily: float = 0.105) -> CheckResult:
    """后复权连续性:相邻交易日 adj 收益 |r|>10.5% 且非除权日、非涨跌停、非停牌前后 -> 数据错误。

    除权日由 adj_factor 变化识别(后复权下除权日收益应连续);停牌缺口跳过避免误报。
    """
    flagged = 0
    for _, g in panel.sort_values(["code", "trade_date"]).groupby("code", sort=False):
        adj_c = g["close_adj"].to_numpy(dtype="float64")
        factor = g["adj_factor"].to_numpy(dtype="float64")
        susp = g["is_suspended"].to_numpy(dtype=bool)
        lu = g["is_limit_up"].to_numpy(dtype=bool)
        ld = g["is_limit_down"].to_numpy(dtype=bool)
        for i in range(1, len(g)):
            if adj_c[i - 1] <= 0:
                continue
            ret = adj_c[i] / adj_c[i - 1] - 1.0
            ex_div = abs(factor[i] - factor[i - 1]) > 1e-12
            gap = susp[i] or susp[i - 1]
            at_limit = lu[i] or ld[i]
            if abs(ret) > max_daily and not ex_div and not gap and not at_limit:
                flagged += 1
    if flagged == 0:
        return CheckResult("adj_continuity", PASS, 0, "后复权收益无异常跳变")
    return CheckResult("adj_continuity", FAIL, flagged, f"{flagged} 行 adj 收益异常(疑数据错)")


def check_one_price_consistency(panel: pd.DataFrame, tol: float = 0.005) -> CheckResult:
    """一字板一致性:is_one_price_limit ⇒ OHLC 相等 且 (涨停|跌停)。"""
    one = panel["is_one_price_limit"].to_numpy(dtype=bool)
    o = panel["open_raw"].to_numpy(dtype="float64")
    h = panel["high_raw"].to_numpy(dtype="float64")
    low = panel["low_raw"].to_numpy(dtype="float64")
    c = panel["close_raw"].to_numpy(dtype="float64")
    lu = panel["is_limit_up"].to_numpy(dtype=bool)
    ld = panel["is_limit_down"].to_numpy(dtype=bool)
    flat = (np.abs(o - h) <= tol) & (np.abs(h - low) <= tol) & (np.abs(low - c) <= tol)
    bad = one & ~(flat & (lu | ld))
    n = int(bad.sum())
    if n == 0:
        return CheckResult("one_price_consistency", PASS, 0, "一字板标记自洽")
    return CheckResult("one_price_consistency", FAIL, n, f"{n} 行一字板标记不自洽")


def check_suspension_consistency(panel: pd.DataFrame) -> CheckResult:
    """停牌一致性:is_suspended ⇒ volume==0。"""
    susp = panel["is_suspended"].to_numpy(dtype=bool)
    vol = panel["volume"].to_numpy(dtype="float64")
    bad = susp & (vol != 0.0)
    n = int(bad.sum())
    if n == 0:
        return CheckResult("suspension_consistency", PASS, 0, "停牌行成交量为 0")
    return CheckResult("suspension_consistency", FAIL, n, f"{n} 行停牌但量非 0")


def check_disclosure_fields(panel: pd.DataFrame, window: int = 10) -> CheckResult:
    """披露字段检查:临近披露(days_to_disclosure<=window)的行必须有预约披露日;
    预告方向取值须 ∈ {-1,0,1}。披露字段未附(token-gated)时 SKIP 并写明原因。
    """
    if "days_to_disclosure" not in panel.columns:
        return CheckResult("disclosure_fields", SKIP, 0, "披露字段未附(需 Tushare token)")
    d2d = panel["days_to_disclosure"].to_numpy(dtype="float64")
    sched = panel["sched_disclosure_date"]
    near = (~np.isnan(d2d)) & (d2d <= window)
    missing_sched = int((near & sched.isna().to_numpy()).sum())
    sign_ok = panel["preann_sign"].isin([-1, 0, 1]).all() if "preann_sign" in panel else True
    n = missing_sched + (0 if sign_ok else 1)
    if n == 0:
        return CheckResult("disclosure_fields", PASS, 0, "披露字段完整、预告方向合法")
    return CheckResult(
        "disclosure_fields", WARN, n, f"临近披露缺预约日={missing_sched}, 预告方向异常={not sign_ok}"
    )


def diagnose_adj_continuity(panel: pd.DataFrame, max_daily: float = 0.105) -> pd.DataFrame:
    """列出被 check_adj_continuity 标记为异常的所有行及其上下文,供人工核查(真错 vs 复权边界)。

    返回 DataFrame[code, trade_date, adj_ret, ex_div, at_limit, gap]:
      ex_div=相邻日 adj_factor 变化(除权日,后复权下收益应连续);at_limit=当日涨跌停;
      gap=当日或前一日停牌(缺口)。这三类为合理边界;均为 False 的大跳变才更可能是真数据错。
    """
    rows = []
    for _, g in panel.sort_values(["code", "trade_date"]).groupby("code", sort=False):
        g = g.reset_index(drop=True)
        adj_c = g["close_adj"].to_numpy(dtype="float64")
        factor = g["adj_factor"].to_numpy(dtype="float64")
        susp = g["is_suspended"].to_numpy(dtype=bool)
        lu = g["is_limit_up"].to_numpy(dtype=bool)
        ld = g["is_limit_down"].to_numpy(dtype=bool)
        for i in range(1, len(g)):
            if adj_c[i - 1] <= 0:
                continue
            ret = adj_c[i] / adj_c[i - 1] - 1.0
            ex_div = bool(abs(factor[i] - factor[i - 1]) > 1e-12)
            gap = bool(susp[i] or susp[i - 1])
            at_limit = bool(lu[i] or ld[i])
            if abs(ret) > max_daily and not ex_div and not gap and not at_limit:
                rows.append({"code": g.loc[i, "code"], "trade_date": g.loc[i, "trade_date"],
                             "adj_ret": round(float(ret), 4), "ex_div": ex_div,
                             "at_limit": at_limit, "gap": gap})
    return pd.DataFrame(rows, columns=["code", "trade_date", "adj_ret", "ex_div", "at_limit", "gap"])


def check_adj_factor_monotonic(panel: pd.DataFrame, tol: float = 1e-9) -> CheckResult:
    """后复权因子单调性:同一只票的 adj_factor 应随时间非递减(后复权随分红累积上调);
    出现下降疑为数据错误。无除权时不变。"""
    flagged = 0
    for _, g in panel.sort_values(["code", "trade_date"]).groupby("code", sort=False):
        f = g["adj_factor"].to_numpy(dtype="float64")
        flagged += int((np.diff(f) < -tol).sum())
    if flagged == 0:
        return CheckResult("adj_factor_monotonic", PASS, 0, "后复权因子非递减")
    return CheckResult("adj_factor_monotonic", WARN, flagged, f"{flagged} 处后复权因子下降(疑数据错)")


def run_daily_quality_checks(panel: pd.DataFrame) -> "list[CheckResult]":
    """运行全部质检项,返回结果列表。"""
    return [
        check_raw_adj_separation(panel),
        check_limit_price_consistency(panel),
        check_adj_continuity(panel),
        check_adj_factor_monotonic(panel),
        check_one_price_consistency(panel),
        check_suspension_consistency(panel),
        check_disclosure_fields(panel),
    ]


def report_to_frame(results: "list[CheckResult]") -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def assert_passed(results: "list[CheckResult]") -> None:
    """任一 FAIL 即抛错(供验收脚本调用)。WARN/SKIP 不抛,但应在报告中可见。"""
    fails = [r for r in results if r.status == FAIL]
    if fails:
        lines = "; ".join(f"{r.check}({r.n_flagged}): {r.detail}" for r in fails)
        raise AssertionError(f"数据质检 FAIL: {lines}")
