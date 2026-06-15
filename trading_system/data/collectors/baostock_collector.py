"""BaoStock 行情采集器(主源,硬依赖)。补丁:网络路径 + 会话管理 + 顺序重试 + 单票降级 + 单票超时看门狗。

在已有低层封装 ``collectors/baostock.py``(login/logout/fetch_raw_with_factor/query_*)之上做编排:
  - 会话上下文(``with BaostockCollector() as bc``);login 失败直接抛出(行情是硬依赖)。
  - 单只票 try/except + 重试(≤max_retries),失败计入 failed_list 不中断整批(整批仅 login 失败才硬退)。
  - **单进程顺序**拉取(BaoStock 不支持多线程),重试间 sleep。
  - **单票超时看门狗**:给"单个串行请求"套 threading + join(timeout)(不是并发拉取);
    超时视为该次失败,触发重试;重试仍超时则抛出 → 由 fetch_many 计入 failed,绝不阻塞整批。
依赖注入:login_fn / logout_fn / fetch_fn / all_stock_fn 均可注入,便于离线 mock 测试(网络路径 NOT RUN)。
价格层:fetch_fn 同源返回不复权 + 后复权因子(INV-2,后复权只用 BaoStock 单源)。
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pandas as pd

from trading_system.data.collectors import baostock as bs_api
from trading_system.data.collectors.quota import QuotaExceeded, RequestQuota
from trading_system.data.schema import (
    FINANCIAL_FIELDS,
    FINANCIAL_NUMERIC_FIELDS,
    INDUSTRY_FIELDS,
    RAW_INPUT_FIELDS,
)
from trading_system.data.universe import MAIN_BOARD_PREFIXES, board_allowed

logger = logging.getLogger(__name__)

# 单票请求默认超时(秒)。BaoStock 不支持多线程并发,这里只给"单个串行请求"套看门狗,
# 防止某只票请求挂起(socket 无响应)导致整个串行采集进程僵死。正常拉取约 8~15 秒/票。
DEFAULT_REQUEST_TIMEOUT_SEC = 45.0

# 单票一次行情拉取消耗的 BaoStock 请求数:fetch_raw_with_factor 同源拉"不复权 + 后复权"两次
# query_history_k_data_plus(见 baostock.fetch_raw_with_factor),故计 2 次,用于配额计数。
FETCH_QUERY_COST = 2


def _call_with_timeout(fn, args, timeout):
    """给单个(串行)请求套超时看门狗:在 daemon 线程里跑 fn(*args),主线程 join(timeout)。

    - timeout 为 None/<=0 时直接同步调用(不套看门狗,便于离线 mock)。
    - 超时(线程未在 timeout 内结束)→ 抛 TimeoutError;daemon 线程留给后台自然结束/报错,
      绝不 join 等待挂起的请求(否则看门狗失效),也不会阻塞进程退出(daemon)。
    - fn 内部抛的异常原样透传(语义不变:仍走既有重试/失败逻辑)。
    这是"单请求看门狗",不是并发拉取——同一时刻只有一个 BaoStock 请求在跑。
    """
    if not timeout or timeout <= 0:
        return fn(*args)
    box: dict = {}
    done = threading.Event()

    def _worker():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 — 原样透传给主线程,语义不变
            box["error"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_worker, name="baostock-fetch", daemon=True)
    t.start()
    if not done.wait(timeout):
        raise TimeoutError(f"BaoStock 单票请求超时(>{timeout:g}s),视为失败")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _default_all_stock(day: str) -> pd.DataFrame:
    """低层:某交易日全市场代码(code, tradeStatus, code_name)。需 baostock 会话。"""
    import baostock as bs

    rs = bs.query_all_stock(day=day)
    if rs.error_code != "0":
        raise RuntimeError(f"query_all_stock 失败: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def _pick(df: "pd.DataFrame | None", cols: "list[str]") -> "pd.DataFrame | None":
    """从单期财务表抽取 (code, statDate, pubDate, *cols);空/None 返回 None。缺列容忍(以实际返回为准)。"""
    if df is None or len(df) == 0:
        return None
    keep = [c for c in ("code", "statDate", "pubDate", *cols) if c in df.columns]
    return df[keep].copy()


def merge_financial(profit, growth, balance) -> "pd.DataFrame | None":
    """合并单个 (code,year,quarter) 的 profit/growth/balance 三表为一行,产出 FINANCIAL_FIELDS。

    - 按 (code, statDate) 外连接抽取 roeAvg/netProfit(profit)、YOYNI(growth)、liabilityToAsset(balance);
    - pubDate 跨三表取同一报告期的公告日(coalesce;三表同报告应一致),**原样保留**(PIT 对齐用);
    - statDate/pubDate 转 datetime、数值列转 float(原值不变,仅落盘 dtype 规整)。三表全空返回 None。
    字段名以 BaoStock 实际返回为准:roeAvg/netProfit/YOYNI/liabilityToAsset。
    """
    p = _pick(profit, ["roeAvg", "netProfit"])
    g = _pick(growth, ["YOYNI"])
    b = _pick(balance, ["liabilityToAsset"])
    present = [x for x in (p, g, b) if x is not None]
    if not present:
        return None

    # pubDate:汇集三表的 (code, statDate, pubDate),按报告期去重取首个非空(同报告期公告日一致)
    pubs = [x[["code", "statDate", "pubDate"]] for x in present if "pubDate" in x.columns]
    metric_only = [x.drop(columns=[c for c in ("pubDate",) if c in x.columns]) for x in present]
    out = metric_only[0]
    for nxt in metric_only[1:]:
        out = out.merge(nxt, on=["code", "statDate"], how="outer")
    if pubs:
        pub = pd.concat(pubs, ignore_index=True).dropna(subset=["pubDate"])
        pub = pub[pub["pubDate"].astype(str).str.len() > 0]
        pub = pub.drop_duplicates(subset=["code", "statDate"], keep="first")
        out = out.merge(pub, on=["code", "statDate"], how="left")
    if "pubDate" not in out.columns:
        out["pubDate"] = pd.NaT

    for c in FINANCIAL_NUMERIC_FIELDS:                 # 缺失列补 NaN,保证 schema 齐全
        if c not in out.columns:
            out[c] = np.nan
    out = out.reindex(columns=list(FINANCIAL_FIELDS))
    out["statDate"] = pd.to_datetime(out["statDate"], errors="coerce")
    out["pubDate"] = pd.to_datetime(out["pubDate"], errors="coerce")
    for c in FINANCIAL_NUMERIC_FIELDS:                 # 统一 float64,避免跨票 int/float dtype 漂移
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def normalize_industry(df: "pd.DataFrame | None") -> pd.DataFrame:
    """规整行业分类原始表为 INDUSTRY_FIELDS(code, industry, industryClassification)。

    BaoStock 对未分类票返回空 industry → 统一为空串 ""(保留行,便于"已知未分类"与"未采集"区分)。
    按 code 去重(保留首条)。空输入返回带 schema 列的空表。
    """
    cols = list(INDUSTRY_FIELDS)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    out = out[cols].copy()
    out["code"] = out["code"].astype(str)
    out["industry"] = out["industry"].fillna("").astype(str)
    out["industryClassification"] = out["industryClassification"].fillna("").astype(str)
    out = out[out["code"].str.len() > 0].drop_duplicates(subset=["code"], keep="first")
    return out.reset_index(drop=True)


class BaostockCollector:
    """BaoStock 行情采集编排。"""

    def __init__(
        self,
        *,
        login_fn=bs_api.login,
        logout_fn=bs_api.logout,
        fetch_fn=bs_api.fetch_raw_with_factor,
        all_stock_fn=_default_all_stock,
        profit_fn=bs_api.query_profit,
        growth_fn=bs_api.query_growth,
        balance_fn=bs_api.query_balance,
        industry_fn=bs_api.query_industry,
        max_retries: int = 2,
        sleep_sec: float = 0.3,
        request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
        quota: "RequestQuota | None" = None,
        fetch_query_cost: int = FETCH_QUERY_COST,
    ) -> None:
        self.login_fn = login_fn
        self.logout_fn = logout_fn
        self.fetch_fn = fetch_fn
        self.all_stock_fn = all_stock_fn
        # 季频财务接口(批 2):profit/growth/balance 三表,均可注入便于离线 mock。
        self.profit_fn = profit_fn
        self.growth_fn = growth_fn
        self.balance_fn = balance_fn
        self.industry_fn = industry_fn          # 行业分类接口(批 4)
        self.max_retries = max_retries
        self.sleep_sec = sleep_sec
        # 单票请求超时(秒);<=0 关闭看门狗(直接同步调用)。超时→视为该次失败→重试/跳过。
        self.request_timeout_sec = request_timeout_sec
        # 配额计数器(跨进程按自然日);None=不计配额(离线 mock 默认)。每次实际请求都计入。
        self.quota = quota
        self.fetch_query_cost = int(fetch_query_cost)
        # fetch_many 一旦因配额耗尽提前停止,置位并记下"本批未拉的剩余代码",供上层写待拉列表。
        self.quota_stopped = False
        self.remaining_codes: "list[str]" = []

    def _quota_add(self, n: int) -> None:
        if self.quota is not None:
            self.quota.add(n)

    def __enter__(self) -> "BaostockCollector":
        # login 失败直接抛出(行情硬依赖,由上层转为非零退出)
        self.login_fn()
        self._quota_add(1)                 # 登录也消耗配额(重连场景每次 login 都计入)
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.logout_fn()
            self._quota_add(1)             # 登出一并计入(保守);随后强制 flush 固化计数
        except Exception as e:  # noqa: BLE001 — 登出失败不掩盖主流程
            logger.warning("baostock logout 异常(忽略): %r", e)
        finally:
            if self.quota is not None:
                self.quota.flush()         # 进程退出前务必落盘,避免丢失跨进程计数

    def list_universe(self, day: str, *, boards=MAIN_BOARD_PREFIXES) -> "list[str]":
        """某交易日的主板(600/601/603/605/000/001/002)非 ST 代码列表(交易池)。
        一次 query_all_stock 请求,计入配额。"""
        if self.quota is not None:
            self.quota.check()             # 配额已满则不再发起(优雅停止)
        df = self.all_stock_fn(day)
        self._quota_add(1)
        codes = []
        for _, r in df.iterrows():
            code = r["code"]
            name = str(r.get("code_name", "")).upper()
            if board_allowed(code, boards) and "ST" not in name:
                codes.append(code)
        return codes

    def _resilient_request(self, fn, args, *, cost: int):
        """单个 BaoStock 请求的统一健壮封装:发起前查配额 → 超时看门狗 → 失败重试 → 计入配额。

        返回 fn 的结果;配额达阈值抛 QuotaExceeded(在 try 外查,未发起的请求不计数);
        重试耗尽抛最后一次异常(含 TimeoutError)。每次"实际发起"的请求(成功/失败/超时)都计 cost
        ——即便超时,请求很可能已达服务端,保守计数避免低估而触发服务端限流。
        """
        last_err = None
        for attempt in range(self.max_retries + 1):
            if self.quota is not None:
                self.quota.check()         # 发起请求前查配额:已满则抛 QuotaExceeded,绝不再发起
            try:
                return _call_with_timeout(fn, args, self.request_timeout_sec)
            except Exception as e:  # noqa: BLE001 — 含 TimeoutError:超时即视为该次失败
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.sleep_sec)
            finally:
                self._quota_add(cost)      # 配额检查抛出时不进入 try,故不会误计
        raise last_err  # type: ignore[misc]

    def fetch_code(self, code: str, start: str, end: str) -> "pd.DataFrame | None":
        """拉单只票(不复权 + 后复权因子);单请求套超时看门狗 + 重试 ≤max_retries。

        无数据返回 None;超时/反复失败抛出(由 fetch_many 计入 failed,不阻塞整批)。
        """
        t0 = time.time()
        df = self._resilient_request(self.fetch_fn, (code, start, end), cost=self.fetch_query_cost)
        logger.info("baostock ok code=%s rows=%d %.2fs", code,
                    0 if df is None else len(df), time.time() - t0)
        return df if (df is not None and len(df) > 0) else None

    def _quota_stop(self, codes: "list[str]", i: int) -> None:
        """配额耗尽:记录本批从第 i 只起的剩余代码(供上层写待拉列表),并置停止位。"""
        self.quota_stopped = True
        self.remaining_codes = list(codes[i:])
        logger.warning(
            "配额接近上限,停止本批后续 %d 只(本批已处理 %d 只);剩余记入待拉列表。",
            len(codes) - i, i,
        )

    def fetch_many(
        self, codes: "list[str]", start_by_code, end: str
    ) -> "tuple[pd.DataFrame, list[str]]":
        """顺序拉多只票。start_by_code 可为 dict(逐票增量起点)或单一字符串。

        返回 (合并的 RAW_INPUT_FIELDS 面板, 失败代码列表)。单票失败不中断;配额耗尽则提前停止
        (置 quota_stopped,把本批剩余代码记入 remaining_codes),并把已拉到的部分照常返回(不丢)。
        """
        self.quota_stopped = False
        self.remaining_codes = []
        frames, failed = [], []
        for i, code in enumerate(codes):
            if self.quota is not None and self.quota.exceeded():  # 本批起步前/逐票前先查配额
                self._quota_stop(codes, i)
                break
            start = start_by_code.get(code) if isinstance(start_by_code, dict) else start_by_code
            try:
                df = self.fetch_code(code, start, end)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except QuotaExceeded:  # 重试途中配额耗尽:停止本批,该票及之后记入剩余,已拉部分照常返回
                self._quota_stop(codes, i)
                break
            except Exception as e:  # noqa: BLE001 — 单票失败计入 failed,继续
                failed.append(code)
                logger.warning("baostock fetch 失败 code=%s err=%r(已重试 %d 次)",
                               code, e, self.max_retries)
        panel = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame(columns=list(RAW_INPUT_FIELDS))
        )
        return panel, failed

    # ── 季频财务采集(批 2:避雷数据,仅采集落盘,不接入打分)──────────────────
    def fetch_financials(
        self, codes: "list[str]", years_quarters: "list[tuple[int, int]]"
    ) -> "tuple[pd.DataFrame, list[tuple[str, int, int]]]":
        """对每个 (code, year, quarter) 拉 profit/growth/balance 三表并合并为一行(含 pubDate)。

        每次接口请求都套超时看门狗 + 重试 + 配额;单个 (code,year,quarter) 失败计入 failed 不中断整批;
        配额耗尽则提前停止(置 quota_stopped)。返回 (FINANCIAL_FIELDS 列的合并面板, failed 列表)。
        **PIT 关键:pubDate 原样保留,后续可见性对齐只能用 pubDate(见 schema.FINANCIAL_FIELDS)。**
        """
        self.quota_stopped = False
        frames: "list[pd.DataFrame]" = []
        failed: "list[tuple[str, int, int]]" = []
        for code in codes:
            if self.quota_stopped:
                break
            for year, quarter in years_quarters:
                if self.quota is not None and self.quota.exceeded():
                    self.quota_stopped = True
                    logger.warning("配额接近上限,停止财务采集后续(已采 %d 行)。", len(frames))
                    break
                try:
                    row = self._fetch_one_financial(code, int(year), int(quarter))
                    if row is not None and len(row) > 0:
                        frames.append(row)
                except QuotaExceeded:
                    self.quota_stopped = True
                    logger.warning("配额耗尽,停止财务采集(已采 %d 行)。", len(frames))
                    break
                except Exception as e:  # noqa: BLE001 — 单个季度失败不中断整批
                    failed.append((code, int(year), int(quarter)))
                    logger.warning("财务采集失败 code=%s %dQ%d err=%r(已重试 %d 次)",
                                   code, year, quarter, e, self.max_retries)
        panel = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame(columns=list(FINANCIAL_FIELDS))
        )
        return panel, failed

    def _fetch_one_financial(self, code: str, year: int, quarter: int) -> "pd.DataFrame | None":
        """拉单个 (code,year,quarter) 的三表(各套看门狗 + 重试 + 配额),合并为一行。"""
        profit = self._resilient_request(self.profit_fn, (code, year, quarter), cost=1)
        growth = self._resilient_request(self.growth_fn, (code, year, quarter), cost=1)
        balance = self._resilient_request(self.balance_fn, (code, year, quarter), cost=1)
        return merge_financial(profit, growth, balance)

    # ── 行业分类采集(批 4:基础设施,仅采集落盘,不接入打分)──────────────────
    def fetch_industry(self, codes: "list[str] | None" = None) -> pd.DataFrame:
        """采集申万行业分类(query_stock_industry 一次取全市场,最省请求),套超时看门狗 + 重试 + 配额。

        返回 INDUSTRY_FIELDS 面板(code, industry, industryClassification)。codes 非空则过滤到这些票。
        低频近静态:全市场一次请求即可,无需逐票。
        """
        df = self._resilient_request(self.industry_fn, ("",), cost=1)   # code="" → 全市场一次取
        out = normalize_industry(df)
        if codes is not None:
            out = out[out["code"].isin(list(codes))].reset_index(drop=True)
        return out
