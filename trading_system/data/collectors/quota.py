"""BaoStock 单日请求配额计数器(跨进程、按自然日持久化)。补丁:feature/upgrade-v2。

为什么需要:BaoStock 免费账户有单日请求次数上限(经验约 5 万次/日)。进程卡死后用户关终端重拉,
**进程内计数会清零,但 BaoStock 服务端的当日计数不会清零**;多次重拉累积可能触发限流/封 IP。
因此配额必须:
  1) 落盘持久化(``data_store/.baostock_quota.json``,内容 {"date","count"}),跨进程累计;
  2) 按**国内自然日**(Asia/Shanghai)重置——用户人在美国,本地时区不同,**绝不能用本地时区判断**
     (否则跨午夜会误判,导致服务端已是新一天而本地仍计旧账,或反之);
  3) 达到 ``limit - safety_margin`` 即停止发起新请求,优雅退出(已落盘数据 + 待拉列表都保住,
     增量重跑断点续传)。

线程安全:采集用看门狗线程跑单个请求,计数可能从不同线程触达,故用 Lock 保护。落盘用"临时文件 +
原子 rename",崩溃不会写出半个文件。默认 flush_every=1(每次请求即落盘),配合采集器 ``__exit__``
的 flush,保证"进程退出/异常时计数已在盘上";若为减少 IO 把 flush_every 调大,则靠 ``__exit__`` flush。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# BaoStock 按国内自然日计数 → 用 Asia/Shanghai。中国自 1991 年起无夏令时,等价固定 UTC+8;
# 若运行环境缺 IANA 时区库(zoneinfo),退回固定 +8 偏移(对中国恒等,保证健壮)。
_SHANGHAI_FALLBACK = timezone(timedelta(hours=8), name="CST+8")


def _shanghai_tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Shanghai")
    except Exception:  # noqa: BLE001 — 缺 tzdata 等 → 退回固定 +8(对中国正确)
        return _SHANGHAI_FALLBACK


class QuotaExceeded(RuntimeError):
    """当日请求配额达上限(>= limit - margin):应停止发起新请求,优雅退出。"""


class RequestQuota:
    """跨进程、按自然日(Asia/Shanghai)持久化的请求计数器 + 配额保护。

    用法:
        q = RequestQuota(path, daily_limit=50000, safety_margin=5000)
        q.load_today()                 # 启动时:把今日服务端口径的已用次数加载为起点(跨进程累计)
        if q.exceeded(): ...           # 达到 limit - margin?
        q.add(2)                       # 发起请求后累加(并落盘)
    """

    def __init__(
        self,
        path: "str | Path",
        *,
        daily_limit: int = 50000,
        safety_margin: int = 5000,
        tz=None,
        now_fn=None,
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.daily_limit = int(daily_limit)
        self.safety_margin = int(safety_margin)
        self._tz = tz or _shanghai_tz()
        # now_fn 返回"当前 UTC 感知时间";注入便于测试时区/跨日。默认取真实 UTC。
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._flush_every = max(1, int(flush_every))
        self._lock = threading.Lock()
        self._count = 0
        self._date = self._today()
        self._unflushed = 0
        self._load_locked()                  # 初始化即按自然日加载/重置(跨进程累计的关键)

    # ── 时间(国内自然日)──
    def _today(self) -> str:
        """当前 Asia/Shanghai 自然日(YYYY-MM-DD)。绝不用本地时区。"""
        return self._now_fn().astimezone(self._tz).strftime("%Y-%m-%d")

    # ── 持久化 ──
    def _read_file(self) -> "dict | None":
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):        # 不存在/损坏 → 视为无记录(从 0 起)
            return None

    def _write_locked(self) -> None:
        """原子落盘(临时文件 + rename),崩溃不留半个文件。"""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = {"date": self._date, "count": self._count}
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)       # 原子替换
            self._unflushed = 0
        except OSError as e:                 # 落盘失败不应拖垮采集(但会丢失跨进程计数,告警)
            logger.warning("配额计数落盘失败 %s: %r", self.path, e)

    def _load_locked(self) -> None:
        """读盘:同一自然日 → 沿用已用次数;跨日 → 归零并更新日期(并立即落盘固化重置)。"""
        today = self._today()
        data = self._read_file()
        if data and data.get("date") == today:
            self._count = int(data.get("count", 0))
            self._date = today
        else:
            self._count = 0                  # 新的一天(或无记录/损坏)→ 计数归零
            self._date = today
            self._write_locked()             # 固化重置,避免午夜后崩溃丢失"已重置"事实

    def load_today(self) -> int:
        """启动取数前调用:按自然日加载今日已用次数作为起点(跨进程累计)。返回当前计数。"""
        with self._lock:
            self._load_locked()
            return self._count

    # ── 计数 + 配额判断 ──
    def add(self, n: int = 1) -> int:
        """发起 n 次请求后累加(跨日自动滚动到新一天再加)。按 flush_every 落盘。返回新计数。"""
        with self._lock:
            today = self._today()
            if today != self._date:          # 累加过程中跨过午夜 → 先滚动归零
                self._count = 0
                self._date = today
            self._count += int(n)
            self._unflushed += int(n)
            if self._unflushed >= self._flush_every:
                self._write_locked()
            return self._count

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def limit(self) -> int:
        return self.daily_limit

    @property
    def margin(self) -> int:
        return self.safety_margin

    @property
    def threshold(self) -> int:
        """触发停止的阈值 = limit - margin。"""
        return self.daily_limit - self.safety_margin

    def remaining(self) -> int:
        """距阈值还剩多少次(可发起的安全请求数);<=0 表示应停止。"""
        with self._lock:
            return self.threshold - self._count

    def exceeded(self) -> bool:
        """当日累计是否已 >= limit - margin(达到则应停止发起新请求)。"""
        with self._lock:
            return self._count >= self.threshold

    def check(self) -> None:
        """达阈值则抛 QuotaExceeded(供"发起请求前"调用,确保不再发起新请求)。"""
        if self.exceeded():
            raise QuotaExceeded(
                f"当日 BaoStock 请求已用 {self.count} 次,达到阈值 {self.threshold} "
                f"(上限 {self.daily_limit} − 安全余量 {self.safety_margin});停止发起新请求。"
            )

    def flush(self) -> None:
        """强制落盘(进程退出/异常时务必调用,避免丢失跨进程计数)。"""
        with self._lock:
            if self._unflushed > 0:
                self._write_locked()
