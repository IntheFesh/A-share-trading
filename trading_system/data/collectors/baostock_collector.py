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

import pandas as pd

from trading_system.data.collectors import baostock as bs_api
from trading_system.data.schema import RAW_INPUT_FIELDS
from trading_system.data.universe import board_allowed

logger = logging.getLogger(__name__)

# 单票请求默认超时(秒)。BaoStock 不支持多线程并发,这里只给"单个串行请求"套看门狗,
# 防止某只票请求挂起(socket 无响应)导致整个串行采集进程僵死。正常拉取约 8~15 秒/票。
DEFAULT_REQUEST_TIMEOUT_SEC = 45.0


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


class BaostockCollector:
    """BaoStock 行情采集编排。"""

    def __init__(
        self,
        *,
        login_fn=bs_api.login,
        logout_fn=bs_api.logout,
        fetch_fn=bs_api.fetch_raw_with_factor,
        all_stock_fn=_default_all_stock,
        max_retries: int = 2,
        sleep_sec: float = 0.3,
        request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> None:
        self.login_fn = login_fn
        self.logout_fn = logout_fn
        self.fetch_fn = fetch_fn
        self.all_stock_fn = all_stock_fn
        self.max_retries = max_retries
        self.sleep_sec = sleep_sec
        # 单票请求超时(秒);<=0 关闭看门狗(直接同步调用)。超时→视为该次失败→重试/跳过。
        self.request_timeout_sec = request_timeout_sec

    def __enter__(self) -> "BaostockCollector":
        # login 失败直接抛出(行情硬依赖,由上层转为非零退出)
        self.login_fn()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.logout_fn()
        except Exception as e:  # noqa: BLE001 — 登出失败不掩盖主流程
            logger.warning("baostock logout 异常(忽略): %r", e)

    def list_universe(self, day: str, *, boards=("60", "000")) -> "list[str]":
        """某交易日的主板非 ST 代码列表(交易池)。"""
        df = self.all_stock_fn(day)
        codes = []
        for _, r in df.iterrows():
            code = r["code"]
            name = str(r.get("code_name", "")).upper()
            if board_allowed(code, boards) and "ST" not in name:
                codes.append(code)
        return codes

    def fetch_code(self, code: str, start: str, end: str) -> "pd.DataFrame | None":
        """拉单只票(不复权 + 后复权因子);单请求套超时看门狗 + 重试 ≤max_retries。

        无数据返回 None;超时/反复失败抛出(由 fetch_many 计入 failed,不阻塞整批)。
        """
        last_err = None
        t0 = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                df = _call_with_timeout(
                    self.fetch_fn, (code, start, end), self.request_timeout_sec
                )
                logger.info("baostock ok code=%s rows=%d %.2fs", code,
                            0 if df is None else len(df), time.time() - t0)
                return df if (df is not None and len(df) > 0) else None
            except Exception as e:  # noqa: BLE001 — 含 TimeoutError:超时即视为该次失败
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.sleep_sec)
        raise last_err  # type: ignore[misc]

    def fetch_many(
        self, codes: "list[str]", start_by_code, end: str
    ) -> "tuple[pd.DataFrame, list[str]]":
        """顺序拉多只票。start_by_code 可为 dict(逐票增量起点)或单一字符串。

        返回 (合并的 RAW_INPUT_FIELDS 面板, 失败代码列表)。单票失败不中断。
        """
        frames, failed = [], []
        for code in codes:
            start = start_by_code.get(code) if isinstance(start_by_code, dict) else start_by_code
            try:
                df = self.fetch_code(code, start, end)
                if df is not None and len(df) > 0:
                    frames.append(df)
            except Exception as e:  # noqa: BLE001 — 单票失败计入 failed,继续
                failed.append(code)
                logger.warning("baostock fetch 失败 code=%s err=%r(已重试 %d 次)",
                               code, e, self.max_retries)
        panel = (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame(columns=list(RAW_INPUT_FIELDS))
        )
        return panel, failed
