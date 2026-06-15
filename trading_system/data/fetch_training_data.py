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
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.data import quality
from trading_system.data.calendar import TradingCalendar
from trading_system.data.collectors.baostock_collector import (
    DEFAULT_REQUEST_TIMEOUT_SEC,
    BaostockCollector,
)
from trading_system.data.collectors.quota import QuotaExceeded, RequestQuota
from trading_system.data.collectors.tushare_collector import TushareCollector
from trading_system.data.price_layers import attach_disclosure_fields, build_price_layers
from trading_system.data.financial_store import FinancialStore
from trading_system.data.industry_store import IndustryStore
from trading_system.data.schema import DISCLOSURE_FIELDS
from trading_system.data.store import ParquetStore
from trading_system.data.universe import MAIN_BOARD_PREFIXES

logger = logging.getLogger("fetch_training_data")

DEFAULT_STORE_PATH = Path(__file__).resolve().parents[2] / "data_store"

# 每拉 N 只票落盘一次(分批落盘默认批大小);中途中断不丢已落盘批次,天然支持断点续传。
DEFAULT_BATCH_SAVE_SIZE = 200
# 待拉/失败代码清单文件名(落盘到 output_dir;崩溃可见,便于人工核对)。
PENDING_FILE = "pending_codes.json"
FAILED_FILE = "failed_codes.json"
# BaoStock 单日请求配额计数文件(落盘到 data_dir;跨进程、按 Asia/Shanghai 自然日)。
QUOTA_FILE = ".baostock_quota.json"
# 配额默认值(config.yaml 未配置时的兜底)。
DEFAULT_DAILY_REQUEST_LIMIT = 50000
DEFAULT_DAILY_REQUEST_SAFETY_MARGIN = 5000


def _log_quota_stop(quota: RequestQuota) -> None:
    """配额达单日上限时的统一优雅停止提示(清晰告知用户:已用多少、明日再续、断点续传会续上)。"""
    logger.error(
        "════ 配额保护:今日 BaoStock 请求已用 %d 次,达到阈值 %d(上限 %d − 安全余量 %d)。"
        "停止发起新请求,优雅退出;已落盘数据与待拉列表均已保存。请明日(配额按国内自然日重置)"
        "再运行增量取数,断点续传会自动从未拉的票续上。 ════",
        quota.count, quota.threshold, quota.limit, quota.margin,
    )


def _chunked(seq: "list[str]", n: int):
    """把代码列表切成每段至多 n 只(分批落盘单位)。"""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _quarters_range(start_year: int, end: "str | dt.date | None" = None) -> "list[tuple[int, int]]":
    """生成 [(start_year,1) ... (含 end 所在季)] 的 (year, quarter) 列表(季频财务采集用)。

    末尾到 end 所在季为止(未公告的季 BaoStock 返回空,fetch_financials 自动跳过,不报错)。
    """
    if end is None:
        end_date = dt.date.today()
    elif isinstance(end, str):
        end_date = dt.date.fromisoformat(end[:10])
    else:
        end_date = end
    end_q = (end_date.month - 1) // 3 + 1
    out: "list[tuple[int, int]]" = []
    for y in range(int(start_year), end_date.year + 1):
        last_q = end_q if y == end_date.year else 4
        for q in range(1, last_q + 1):
            out.append((y, q))
    return out


def _write_codes_json(path: "str | Path", codes, *, note: str = "") -> None:
    """落盘待拉/失败代码清单(崩溃可见)。断点续传靠 store 主键去重自动完成,不依赖本文件;
    本文件仅供人工核对"哪些票还没拉到"。落盘失败不中断采集主流程。"""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "count": len(set(codes)),
            "codes": sorted(set(codes)),
            "note": note,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:  # noqa: BLE001 — 清单落盘失败不应拖垮行情采集
        logger.warning("写代码清单失败 %s: %r", path, e)


def _attach_disclosure(
    panel: pd.DataFrame, *, codes, start, end, enable_disclosure, tushare_token,
    tushare_collector_factory,
) -> "tuple[pd.DataFrame, str]":
    """披露字段附加(软依赖,失败降级)。返回 (面板, disclosure_status)。与原内联逻辑等价。"""
    if not enable_disclosure:
        return _with_null_disclosure(panel), "disabled"
    token = tushare_token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.warning("启用了披露但无 Tushare token → 跳过披露采集,披露字段置 NULL,主流程继续。")
        return _with_null_disclosure(panel), "skipped_no_token"
    factory = tushare_collector_factory or (lambda tk: TushareCollector(tk))
    try:
        tc = factory(token)
        sched, preann = tc.fetch_disclosure(codes, start, end)
        cal = TradingCalendar(sorted(pd.to_datetime(panel["trade_date"]).unique()))
        panel = attach_disclosure_fields(panel, sched_disclosure=sched, preann=preann, calendar=cal)
        panel = _coerce_disclosure_for_storage(panel)
        return panel, "collected"
    except Exception as e:  # noqa: BLE001 — 任何 Tushare 异常 → 降级,绝不丢行情
        logger.warning("Tushare 采集失败,降级置空披露字段,主流程继续。原因=%r", e)
        return _with_null_disclosure(panel), "failed_degraded"


def _summarize_quality(batches: "list[list]") -> "tuple[bool, list]":
    """汇总(可能分批的)质检结果:同名检查合并 n_flagged、取最差状态。返回 (有无FAIL, 汇总列表)。"""
    order = {quality.PASS: 0, quality.SKIP: 0, quality.WARN: 1, quality.FAIL: 2}
    agg: "dict[str, quality.CheckResult]" = {}
    for batch in batches:
        for r in batch:
            cur = agg.get(r.check)
            if cur is None:
                agg[r.check] = quality.CheckResult(r.check, r.status, r.n_flagged, r.detail)
            else:
                cur.n_flagged += r.n_flagged
                if order.get(r.status, 0) > order.get(cur.status, 0):
                    cur.status, cur.detail = r.status, r.detail
    merged = list(agg.values())
    has_fail = any(r.status == quality.FAIL for r in merged)
    return has_fail, merged


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
    limit: "int | None" = None,
    request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    batch_save_size: int = DEFAULT_BATCH_SAVE_SIZE,
    output_dir: "str | Path | None" = None,
    quota: "RequestQuota | None" = None,
    daily_request_limit: int = DEFAULT_DAILY_REQUEST_LIMIT,
    daily_request_safety_margin: int = DEFAULT_DAILY_REQUEST_SAFETY_MARGIN,
    enable_financials: bool = False,
    fin_out: "str | Path | None" = None,
    financial_store: "FinancialStore | None" = None,
    financials_start_year: int = 2019,
    enable_industry: bool = False,
    industry_out: "str | Path | None" = None,
    industry_store: "IndustryStore | None" = None,
) -> int:
    """采集主流程(依赖可注入,便于离线测试)。返回进程退出码:0 成功 / 2 行情硬失败。

    健壮机制(防全量拉取卡死/丢数据/触发限流):
      1) 单票超时看门狗:由注入/默认的 BaostockCollector(request_timeout_sec) 实现,超时跳过该票。
      2) 分批落盘:每 batch_save_size 只票落盘一次(默认主路径=增量+无披露,走 update_incremental)。
      3) 待拉列表 + 失败重拉:超时/失败的票记入 output_dir/pending_codes.json,全部拉完后自动重拉一轮;
         仍失败者写 output_dir/failed_codes.json 并日志列出(不伪成功、不无限重试)。
      4) 断点续传:分批落盘 + 增量去重 → 中断后重跑(增量)自动跳过已落盘的票,从未拉的票继续。
      5) 配额保护:跨进程按国内自然日持久化 BaoStock 请求计数(data_dir/.baostock_quota.json),
         每次实际请求(登录/登出/拉取/重试)都计入;当日累计 >= limit-margin 即停止发起新请求,
         优雅退出(已落盘数据 + 待拉列表保住),防止反复重拉累积触发服务端限流/封 IP。
    最终数据与"理想一次性拉取"一致(按 (code,trade_date) 主键去重,双价格层完整;写入顺序无关)。
    """
    end = end or dt.date.today().strftime("%Y-%m-%d")
    store = store or ParquetStore(out or DEFAULT_STORE_PATH)
    # 配额计数器:跨进程、按 Asia/Shanghai 自然日(用户在美国,绝不用本地时区)。注入可测。
    quota = quota or RequestQuota(Path(store.root) / QUOTA_FILE,
                                  daily_limit=daily_request_limit,
                                  safety_margin=daily_request_safety_margin)
    bc = baostock_collector or BaostockCollector(request_timeout_sec=request_timeout_sec,
                                                 quota=quota)
    boards = MAIN_BOARD_PREFIXES  # 沪深主板 600/601/603/605/000/001/002(含 001/002,唯一真相源)
    report_dir = Path(output_dir) if output_dir is not None else Path(store.root)
    batch_save_size = max(1, int(batch_save_size))

    # 启动即按自然日加载今日已用次数(跨进程累计:关终端重拉也接着算,不从 0 起)。
    quota.load_today()
    logger.info("当日 BaoStock 请求已用 %d / %d(安全余量 %d;阈值 %d)。",
                quota.count, quota.limit, quota.margin, quota.threshold)
    if quota.exceeded():                  # 开跑前就已达阈值 → 不登录、不取数,优雅退出
        _log_quota_stop(quota)
        return 0

    # 分批"即时落盘"仅用于默认主路径(增量 + 无披露):此时分批结果与一次性落盘严格一致,且天然续传。
    # 其余模式(全量=store.write 覆盖语义;披露=需对全量面板建全局日历)→ 攒到末尾一次性处理+落盘,
    # 语义与原实现完全一致(INV/质检/披露 PIT 不变)。两种模式都享有超时+待拉+重拉+失败报告。
    per_batch_save = incremental and not enable_disclosure

    codes: "list[str]" = []
    total = 0
    final_failed: "list[str]" = []
    accumulated: "list[pd.DataFrame]" = []   # 非即时落盘模式:攒原始面板
    batch_q: "list[list]" = []               # 即时落盘模式:逐批质检结果
    total_written = 0
    success_codes: set = set()
    dmins: "list[pd.Timestamp]" = []
    dmaxs: "list[pd.Timestamp]" = []
    saved_any = False
    quota_stopped = False                    # 配额达上限提前停止(优雅退出,数据/待拉已保住)
    fin_written = 0                           # 季频财务采集写入行数(批 2;--enable-financials 才启用)
    fin_failed: "list" = []                   # 财务采集失败的 (code, year, quarter)
    ind_written = 0                           # 行业分类采集落盘总行数(批 4;--enable-industry 才启用)

    def _save_batch(panel_raw: pd.DataFrame) -> None:
        """即时落盘一批(默认主路径):双价格层 + 状态位 + 置空披露 → update_incremental。"""
        nonlocal total_written, saved_any
        panel = build_price_layers(panel_raw)          # INV-2 双价格层 + 状态位(与正常路径一致)
        panel = _with_null_disclosure(panel)
        total_written += store.update_incremental(panel)
        success_codes.update(pd.unique(panel["code"]))
        batch_q.append(_market_quality(panel, check_disclosure=False))
        dmins.append(pd.to_datetime(panel["trade_date"]).min())
        dmaxs.append(pd.to_datetime(panel["trade_date"]).max())
        saved_any = True

    def _ingest(panel_raw: pd.DataFrame) -> None:
        """一批拉取结果入库:即时落盘模式直接落盘;否则攒起来末尾统一处理。"""
        if panel_raw.empty:
            return
        if per_batch_save:
            _save_batch(panel_raw)
        else:
            accumulated.append(panel_raw)

    # ── 行情(硬依赖):分批拉取 + 分批落盘;login/会话失败=硬失败非零退出 ──
    try:
        with bc:
            codes = universe_codes or bc.list_universe(universe_day or end, boards=boards)
            if not codes:
                logger.error("交易池为空,无法采集;退出。")
                return 2
            if limit is not None and limit > 0:
                codes = sorted(codes)[:limit]   # 排序后取前 limit 只,保证可复现
                logger.info("已限制交易池为前 %d 只(--limit);全量请去掉 --limit。", len(codes))
            start_by_code = _start_by_code(codes, store, start, incremental)
            total = len(codes)

            # 第一轮:分批拉取,每批落盘一次(中断不丢已落盘批次)
            pending: "list[str]" = []
            n_batches = (total + batch_save_size - 1) // batch_save_size
            for bi, chunk in enumerate(_chunked(codes, batch_save_size), start=1):
                panel_raw, failed = bc.fetch_many(chunk, start_by_code, end)
                if failed:
                    pending.extend(failed)
                    _write_codes_json(report_dir / PENDING_FILE, pending,
                                      note="超时/失败,待重拉(增量重跑会自动续传)")
                _ingest(panel_raw)
                logger.info("批 %d/%d:拉 %d 只,得 %d 行,失败 %d 只(累计待拉 %d);"
                            "当日配额已用 %d / %d(余 %d)。",
                            bi, n_batches, len(chunk), len(panel_raw), len(failed), len(pending),
                            quota.count, quota.limit, max(0, quota.remaining()))
                if getattr(bc, "quota_stopped", False):    # 本批因配额耗尽提前停止
                    remaining = list(getattr(bc, "remaining_codes", [])) + \
                        list(codes[bi * batch_save_size:])
                    deferred = sorted(set(pending) | set(remaining))
                    _write_codes_json(report_dir / PENDING_FILE, deferred,
                                      note="配额达单日上限,暂停;明日配额重置后增量重跑断点续传")
                    quota_stopped = True
                    break

            # 第二轮:对"待拉列表"重拉一轮(同一会话、同样超时保护);仍失败者计入最终失败。
            # 配额已停则不重拉(没有配额可用),把待拉留到明日续传。
            if pending and not quota_stopped:
                retry_codes = sorted(set(pending))
                logger.info("待拉列表重拉一轮:%d 只(同样超时保护)...", len(retry_codes))
                panel_raw2, final_failed = bc.fetch_many(retry_codes, start_by_code, end)
                _ingest(panel_raw2)
                if getattr(bc, "quota_stopped", False):    # 重拉途中也可能耗尽配额
                    quota_stopped = True

            # 季频财务采集(批 2:避雷数据,仅采集落盘,默认关 → 不触碰现有流程)。同一会话内追加一轮。
            if enable_financials and codes and not quota_stopped:
                fin_store = financial_store or FinancialStore(
                    fin_out if fin_out is not None
                    else Path(str(store.root).rstrip("/\\") + "_fin"))
                yqs = _quarters_range(financials_start_year, end)
                logger.info("追加季频财务采集:%d 只 × %d 季(%dQ1~);独立落盘 %s。",
                            len(codes), len(yqs), financials_start_year, fin_store.root)
                fin_panel, fin_failed = bc.fetch_financials(codes, yqs)
                fin_written = fin_store.update_incremental(fin_panel)
                logger.info("财务采集:写入 %d 行;失败 %d 个 (code,季)%s。",
                            fin_written, len(fin_failed),
                            "(配额提前停止)" if getattr(bc, "quota_stopped", False) else "")

            # 行业分类采集(批 4:基础设施,仅采集落盘,默认关 → 不触碰)。一次请求取全市场,极省配额。
            if enable_industry and codes and not quota_stopped:
                ind_store = industry_store or IndustryStore(
                    industry_out if industry_out is not None
                    else Path(str(store.root).rstrip("/\\") + "_industry"))
                logger.info("追加行业分类采集(一次取全市场);独立落盘 %s。", ind_store.root)
                ind_panel = bc.fetch_industry(codes)
                ind_written = ind_store.update(ind_panel)
                logger.info("行业采集:本次 %d 只匹配到行业;落盘累计 %d 行。", len(ind_panel), ind_written)
    except QuotaExceeded:  # 交易池列举阶段即达配额阈值:未取任何行情,优雅退出
        quota_stopped = True
        _log_quota_stop(quota)
        return 0
    except Exception as e:  # noqa: BLE001 — login/会话失败=硬失败
        logger.error("BaoStock 会话/登录失败(行情硬依赖,非零退出): %r", e)
        return 2

    # ── 待拉/失败列表最终落盘(配额停止时已写过 deferred 待拉列表,不覆盖、不报"永久失败")──
    if not quota_stopped:
        _write_codes_json(report_dir / PENDING_FILE, final_failed, note="重拉后仍待拉(空=全部拉到)")
        if final_failed:
            _write_codes_json(report_dir / FAILED_FILE, final_failed,
                              note="重拉后仍失败:确未拉到(非伪成功);可日后增量重跑补拉")
            shown = final_failed if len(final_failed) <= 30 else final_failed[:30] + ["...(更多见文件)"]
            logger.error("最终失败 %d/%d 只(重拉后仍失败,已写 %s):%s",
                         len(final_failed), total, FAILED_FILE, shown)

    # ── 失败率红线(与原实现一致:全失败硬退;超阈值告警)。配额停止属优雅暂停,不计硬失败。──
    if not quota_stopped:
        if total and len(final_failed) == total:
            logger.error("整体拉取失败:%d/%d 全部失败,非零退出。", len(final_failed), total)
            return 2
        if total and len(final_failed) / total > fail_rate_threshold:
            logger.warning("行情失败率 %.1f%% 超阈值 %.0f%%(失败 %d/%d)。",
                           100 * len(final_failed) / total, 100 * fail_rate_threshold,
                           len(final_failed), total)

    # ── 非即时落盘模式:统一双价格层 + 披露 + 一次性落盘 + 整体质检(语义同原实现)。
    #    配额提前停止时,accumulated 里已拉到的部分照常落盘,绝不丢。──
    disclosure_status = "disabled"
    if not per_batch_save:
        if not accumulated:
            logger.info("无新增行情(增量无更新);本次不写入,退出 0。")
            if quota_stopped:
                _log_quota_stop(quota)
            return 0
        panel = build_price_layers(pd.concat(accumulated, ignore_index=True))
        panel, disclosure_status = _attach_disclosure(
            panel, codes=codes, start=start, end=end, enable_disclosure=enable_disclosure,
            tushare_token=tushare_token, tushare_collector_factory=tushare_collector_factory)
        total_written = store.update_incremental(panel) if incremental else store.write(panel)
        success_codes.update(pd.unique(panel["code"]))
        batch_q = [_market_quality(panel, check_disclosure=(disclosure_status == "collected"))]
        dmins = [pd.to_datetime(panel["trade_date"]).min()]
        dmaxs = [pd.to_datetime(panel["trade_date"]).max()]
    elif not saved_any:
        logger.info("无新增行情(增量无更新);本次不写入,退出 0。")
        if quota_stopped:
            _log_quota_stop(quota)
        return 0

    # ── 质检汇总(分批时合并同名检查)──
    has_fail, merged_q = _summarize_quality(batch_q)
    for r in merged_q:
        logger.info("质检 %s: %s (%d) %s", r.check, r.status, r.n_flagged, r.detail)
    if has_fail:
        logger.error("行情质检存在 FAIL:%s(数据已落盘,请复查)。",
                     [r.check for r in merged_q if r.status == quality.FAIL])

    # ── 总结 + 下一步 ──
    dmin = min(dmins).date() if dmins else None
    dmax = max(dmaxs).date() if dmaxs else None
    logger.info("=== 采集完成 ===")
    logger.info("行情:成功 %d 只 / 失败 %d 只;写入 %d 行;日期 %s ~ %s。",
                len(success_codes), len(final_failed), total_written, dmin, dmax)
    logger.info("披露:%s(disclosure 字段 %s)。", disclosure_status,
                "已填充" if disclosure_status == "collected" else "NULL=未采集/未知")
    if enable_financials:
        logger.info("财务(季频,独立落盘):写入 %d 行;失败 %d 个 (code,季)。", fin_written, len(fin_failed))
    if enable_industry:
        logger.info("行业(独立落盘):落盘累计 %d 行。", ind_written)
    logger.info("当日 BaoStock 请求累计 %d / %d(安全余量 %d)。", quota.count, quota.limit, quota.margin)
    logger.info("落盘目录:%s", store.root)
    logger.info("下一步:python -m trading_system.run.phase1_factor_report"
                " 或 Phase 3 训练脚本(直接从 store 读,无需搬运)。")
    if disclosure_status != "collected":
        logger.info("提示:披露字段为 NULL → Phase 2 披露 overlay 自动短路(默认不启用),与 v3.1 一致。")
    if quota_stopped:                         # 配额提前停止:已落盘部分保住,清晰提示明日续传
        _log_quota_stop(quota)
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
    p.add_argument("--output-dir", default=None,
                   help="待拉/失败清单(pending_codes.json/failed_codes.json)落盘目录;默认=落盘目录")
    p.add_argument("--request-timeout-sec", type=float, default=DEFAULT_REQUEST_TIMEOUT_SEC,
                   help="单票 BaoStock 请求超时(秒);超时跳过并记入待拉列表(防进程僵死)")
    p.add_argument("--batch-save-size", type=int, default=DEFAULT_BATCH_SAVE_SIZE,
                   help="每拉 N 只票落盘一次(分批落盘;中断不丢已落盘批次,支持断点续传)")
    p.add_argument("--daily-request-limit", type=int, default=DEFAULT_DAILY_REQUEST_LIMIT,
                   help="BaoStock 单日请求次数上限(经验值,按账户调整;跨进程按国内自然日计)")
    p.add_argument("--daily-request-safety-margin", type=int,
                   default=DEFAULT_DAILY_REQUEST_SAFETY_MARGIN,
                   help="配额安全余量;当日累计 >= limit-margin 即停止取数优雅退出")
    p.add_argument("--enable-financials", action="store_true",
                   help="额外采集季频财务(profit/growth/balance,独立落盘 data_store_fin/;默认关)")
    p.add_argument("--fin-out", default=None,
                   help="财务独立落盘目录(默认 = 行情落盘目录同名加 _fin 后缀)")
    p.add_argument("--enable-industry", action="store_true",
                   help="额外采集申万行业分类(独立落盘 data_store_industry/;默认关)")
    p.add_argument("--industry-out", default=None,
                   help="行业独立落盘目录(默认 = 行情落盘目录同名加 _industry 后缀)")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    args = build_arg_parser().parse_args(argv)
    return run_fetch(
        start=args.start, end=args.end, universe=args.universe,
        enable_disclosure=args.enable_disclosure, tushare_token=args.tushare_token,
        incremental=args.incremental, out=args.out, output_dir=args.output_dir,
        request_timeout_sec=args.request_timeout_sec, batch_save_size=args.batch_save_size,
        daily_request_limit=args.daily_request_limit,
        daily_request_safety_margin=args.daily_request_safety_margin,
        enable_financials=args.enable_financials, fin_out=args.fin_out,
        enable_industry=args.enable_industry, industry_out=args.industry_out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
