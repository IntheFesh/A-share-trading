"""监控(分核心/增强两层,落盘不起服务)。Phase 4(任务 4.3)。对应 v3.1 第十三章。

核心必做:分块 RankIC、扣费净值与 MaxDD、成交失败率、执行差距、触发器分桶、L0 overlay test、
披露窗样本表现、单股/同簇暴露。
增强可选(主线稳定后再做):HMM 状态概率、ADWIN/DDM、PSI、HCOPE、拥挤代理相关性、HiLo。
输出落盘 PNG/HTML 静态图 + 日志,不起 Web 服务。含降级规则与策略级退役熔断
(滚动净 alpha 衰减 -> 整体下线)。
"""

from __future__ import annotations

_PHASE = "Phase 4 任务 4.3"


def run_monitor(*args, **kwargs):  # noqa: ANN002, ANN003
    """生成核心监控面板(落盘 PNG/HTML + 日志)。"""
    raise NotImplementedError(f"{_PHASE}:run_monitor 待实现(核心层)。")
