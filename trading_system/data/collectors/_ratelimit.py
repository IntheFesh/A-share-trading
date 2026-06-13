"""盘中快照限频工具。Phase 4。对应 v3.1 §2.2(批量 ≤100 只/请求、全局 ≤10 次/秒)。

纯逻辑(限频 + 分批),不依赖网络,可单测。
"""

from __future__ import annotations

import time
from typing import Iterable


class RateLimiter:
    """简单的全局速率限制器:保证相邻 ``wait()`` 间隔 ≥ 1/max_per_sec 秒。"""

    def __init__(self, max_per_sec: float = 10.0) -> None:
        self.min_interval = 1.0 / float(max_per_sec)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        gap = now - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


def chunked(seq: "list", size: int) -> "Iterable[list]":
    """把 seq 切成每块 ≤ size 的若干块。"""
    if size < 1:
        raise ValueError("size 必须 >= 1")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
