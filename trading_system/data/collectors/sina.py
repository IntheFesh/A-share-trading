"""新浪盘中快照采集器(备源)。Phase 0 先就位,Phase 4 主用。对应 v3.1 第二章。

需带 Referer 头(见 config/data.yaml)。限频同腾讯:批量 <= 100 只/请求、全局 <= 10 次/秒。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.2(Phase 4 主用)"


def fetch_snapshot(codes: list[str]):
    """批量取盘中快照(带 Referer 头)。"""
    raise NotImplementedError(f"{_PHASE}:sina.fetch_snapshot 待实现。")
