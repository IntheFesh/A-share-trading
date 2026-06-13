"""一键采集训练数据(用户唯一需运行的脚本)。补丁交付物。对应 v3.1 §2 / 第十五章。

用法(默认纯 BaoStock 行情):
    python -m trading_system.data.fetch_training_data --start 2019-01-01
加披露(需 token,真实路径 NOT RUN):
    TUSHARE_TOKEN=xxx python -m trading_system.data.fetch_training_data --enable-disclosure

数据源分级:**BaoStock=唯一信息来源 + 硬依赖**(行情/日历/ST/退市;login/整体失败 → 非零退出);
**Tushare=仅财报 + 软依赖**(只取业绩预告/预约披露日;任何异常 → 降级置空披露字段,行情照常落盘)。
落盘走既有 store.py(单一出口,增量去重)。
INV-2 双价格层 / INV-3 / 披露 PIT 全部沿用 Phase 0 逻辑;不另起存储、不上多线程、不伪造数据。
NULL(未采集/未知)与 has_preann=False(已确认未发)在写入/读取时严格区分(nullable boolean)。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.data import quality
from trading_system.data.calendar import TradingCalendar
from trading_system.data.collectors.baostock_collector import BaostockCollector
from trading_system.data.collectors.tushare_collector import TushareCollector
from trading_system.data.price_layers import attach_disclosure_fields, build_price_layers
from trading_system.data.schema import DISCLOSURE_FIELDS
from trading_system.data.store import ParquetStore

logger = logging.getLogger("fetch_training_data")

DEFAULT_STORE_PATH = Path(__file__).resolve().parents[2] / "data_store"


def _with_null_disclosure(panel: pd.DataFrame) -> pd.DataFrame:
    """披露未采集 → 置 NULL(语义=未知)。has_preann 用 nullable boolean 的 NA,
    与"已确认未发预告"(False)严格区分。"""
    out = panel.copy()
    out["sched_disclosure_date"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    out["has_preann"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["preann_sign"] = np.nan
    out["days_to_disclosure"] = np.nan
    return out


def _coerce_disclosure_for_storage(panel: pd.DataFrame) -> pd.DataFrame:
    """采集成功路径:把 has_preann 统一为 nullable boolean(True/False,无 NA),保证落盘 dtype 一致。"""
    out = panel.copy()
    out["has_preann"] = out["has_preann"].astype("boolean")
    return out


def _start_by_code(codes, store: ParquetStore, default_start: str, incremental: bool) -> dict:
    """增量:每只票起点 = max(本地)+1 日;无本地或全量 → default_start。"""
    if not incremental:
        return {c: default_start for c in codes}
    local_max = store.local_max_dates()
    out = {}
    for c in codes:
        mx = local_max.get(c)
        out[c] = ((mx + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if mx is not None
                  else default_start)
    return out


def _market_quality(panel: pd.DataFrame, *, check_disclosure: bool):
    """行情质检必做;披露质检仅在采集成功时做(NULL 不检,避免误报)。"""
    results = [
        quality.check_raw_adj_separation(panel),
        quality.check_limit_price_consistency(panel),
        quality.check_adj_continuity(panel),
        quality.check_one_price_consistency(panel),
        quality.check_suspension_consistency(panel),
    ]
    if check_disclosure:
        results.append(quality.check_disclosure_fields(panel))
    return results


def run_fetch(
    *,
    start: str = "2019-01-01",
    end: "str | None" = None,
    universe: str = "main_board",
    enable_disclosure: bool = False,
    tushare_token: "str | None" = None,
    incremental: bool = True,
    out: "str | Path | None" = None,
    store: "ParquetStore | None" = None,
    baostock_collector: "BaostockCollector | None" = None,
    tushare_collector_factory=None,
    universe_codes: "list[str] | None" = None,
    universe_day: "str | None" = None,
    fail_rate_threshold: float = 0.05,
) -> int:
    """采集主流程(依赖可注入,便于离线测试)。返回进程退出码:0 成功 / 2 行情硬失败。"""
    end = end or dt.date.today().strftime("%Y-%m-%d")
    store = store or ParquetStore(out or DEFAULT_STORE_PATH)
    bc = baostock_collector or BaostockCollector()
    boards = ("60", "000")  # main_board

    # ── 行情(硬依赖)──
    try:
        with bc:
            codes = universe_codes or bc.list_universe(universe_day or end, boards=boards)
            if not codes:
                logger.error("交易池为空,无法采集;退出。")
                return 2
            start_by_code = _start_by_code(codes, store, start, incremental)
            panel_raw, failed = bc.fetch_many(codes, start_by_code, end)
    except Exception as e:  # noqa: BLE001 — login/会话失败=硬失败
        logger.error("BaoStock 会话/登录失败(行情硬依赖,非零退出): %r", e)
        return 2

    total = len(codes)
    if total and len(failed) == total:
        logger.error("整体拉取失败:%d/%d 全部失败,非零退出。", len(failed), total)
        return 2
    if total and len(failed) / total > fail_rate_threshold:
        logger.warning("行情失败率 %.1f%% 超阈值 %.0f%%(失败 %d/%d)。",
                       100 * len(failed) / total, 100 * fail_rate_threshold, len(failed), total)
    if panel_raw.empty:
        logger.info("无新增行情(增量无更新);本次不写入,退出 0。")
        return 0

    panel = build_price_layers(panel_raw)  # INV-2 双价格层 + 状态位

    # ── 披露(软依赖,失败降级)──
    disclosure_status = "disabled"
    if enable_disclosure:
        token = tushare_token or os.environ.get("TUSHARE_TOKEN")
        if not token:
            logger.warning("启用了披露但无 Tushare token → 跳过披露采集,披露字段置 NULL,主流程继续。")
            panel = _with_null_disclosure(panel)
            disclosure_status = "skipped_no_token"
        else:
            factory = tushare_collector_factory or (lambda tk: TushareCollector(tk))
            try:
                tc = factory(token)
                sched, preann = tc.fetch_disclosure(codes, start, end)
                cal = TradingCalendar(sorted(pd.to_datetime(panel["trade_date"]).unique()))
                panel = attach_disclosure_fields(panel, sched_disclosure=sched, preann=preann,
                                                 calendar=cal)
                panel = _coerce_disclosure_for_storage(panel)
                disclosure_status = "collected"
            except Exception as e:  # noqa: BLE001 — 任何 Tushare 异常 → 降级,绝不丢行情
                logger.warning("Tushare 采集失败,降级置空披露字段,主流程继续。原因=%r", e)
                panel = _with_null_disclosure(panel)
                disclosure_status = "failed_degraded"
    else:
        panel = _with_null_disclosure(panel)

    # ── 落盘(单一出口,增量去重)──
    written = store.update_incremental(panel) if incremental else store.write(panel)

    # ── 质检 ──
    results = _market_quality(panel, check_disclosure=(disclosure_status == "collected"))
    q_fails = [r for r in results if r.status == quality.FAIL]
    for r in results:
        logger.info("质检 %s: %s (%d) %s", r.check, r.status, r.n_flagged, r.detail)
    if q_fails:
        logger.error("行情质检存在 FAIL:%s(数据已落盘,请复查)。", [r.check for r in q_fails])

    # ── 总结 + 下一步 ──
    dmin = pd.to_datetime(panel["trade_date"]).min().date()
    dmax = pd.to_datetime(panel["trade_date"]).max().date()
    logger.info("=== 采集完成 ===")
    logger.info("行情:成功 %d 只 / 失败 %d 只;写入 %d 行;日期 %s ~ %s。",
                total - len(failed), len(failed), written, dmin, dmax)
    logger.info("披露:%s(disclosure 字段 %s)。", disclosure_status,
                "已填充" if disclosure_status == "collected" else "NULL=未采集/未知")
    logger.info("落盘目录:%s", store.root)
    logger.info("下一步:python -m trading_system.run.phase1_factor_report"
                " 或 Phase 3 训练脚本(直接从 store 读,无需搬运)。")
    if disclosure_status != "collected":
        logger.info("提示:披露字段为 NULL → Phase 2 披露 overlay 自动短路(默认不启用),与 v3.1 一致。")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="一键采集训练数据(BaoStock 行情主源 + 可选 Tushare 披露)")
    p.add_argument("--start", default="2019-01-01", help="起始日期(默认 2019-01-01)")
    p.add_argument("--end", default=None, help="结束日期(默认今天)")
    p.add_argument("--universe", default="main_board", help="交易池(默认 main_board:沪深主板非ST)")
    p.add_argument("--enable-disclosure", action="store_true",
                   help="采集财报/披露(默认关闭;需 Tushare token,无则降级置空)")
    p.add_argument("--tushare-token", default=None, help="Tushare token(也可用环境变量 TUSHARE_TOKEN)")
    inc = p.add_mutually_exclusive_group()
    inc.add_argument("--incremental", dest="incremental", action="store_true", default=True,
                     help="增量模式(默认):只拉 max(本地)+1→今日")
    inc.add_argument("--full", dest="incremental", action="store_false", help="全量重拉")
    p.add_argument("--out", default=None, help="落盘目录(默认走 store.py 既定路径)")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    args = build_arg_parser().parse_args(argv)
    return run_fetch(
        start=args.start, end=args.end, universe=args.universe,
        enable_disclosure=args.enable_disclosure, tushare_token=args.tushare_token,
        incremental=args.incremental, out=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
