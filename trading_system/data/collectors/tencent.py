"""腾讯盘中快照采集器。Phase 0 先就位,Phase 4 主用。对应 v3.1 第二章 / 第十二章。

解析 qt.gtimg.cn 返回(GBK 编码、'~' 分隔)。限频从 config/data.yaml 读:
批量 <= 100 只/请求、全局 <= 10 次/秒。返回的是当日盘中价(执行类参考,raw)。
"""

from __future__ import annotations

_PHASE = "Phase 0 任务 0.2(Phase 4 主用)"


def fetch_snapshot(codes: list[str]):
    """批量取盘中快照,解析 GBK / '~' 分隔返回。"""
    raise NotImplementedError(f"{_PHASE}:tencent.fetch_snapshot 待实现。")
