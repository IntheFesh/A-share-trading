"""purging / embargo(INV 核心)。Phase 3(任务 3.1)。对应 v3.1 第九章。

embargo = H_max + K_limitdown + 1(H_max=10、K=2 -> gap>=13,从 config/train.yaml 读)。
purged 时序交叉验证:训练/验证间留 embargo,标签窗口重叠的样本必须 purge。
"""

from __future__ import annotations

_PHASE = "Phase 3 任务 3.1"


def purged_time_series_split(*args, **kwargs):  # noqa: ANN002, ANN003
    """生成 purged + embargo 的时序 CV 划分。"""
    raise NotImplementedError(f"{_PHASE}:purged_time_series_split 待实现。")
