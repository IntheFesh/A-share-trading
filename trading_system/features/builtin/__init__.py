"""特征族实现。Phase 1(任务 1.2)。对应 v3.1 §7.3。

导入本包即把内置特征注册进 features.registry.REGISTRY。各族(每日截面秩变换 + winsorize 1%,
一律用 adj 价):量价基础 / 趋势 / 反转彩票 / 过度拉升 / 流动性(代表性子集)。
CGO / 换手率族需流通股本,数据补齐后再接入(families.py 内有说明)。
"""

from trading_system.features.builtin import families  # noqa: F401  导入即注册

__all__ = ["families"]
