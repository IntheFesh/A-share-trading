"""实验注册表(持久化版;INV-6 落地)。Phase 3 起使用。对应 v3.1 第十三章。

记录每个盲测段被使用的次数;一旦某段用于 champion-challenger 换届裁决,标记为 archived,
再次用于调参/选择则报错。全 Optuna trial 也写入(供 DSR 的 N 与 PBO 记账)。
纯逻辑内核见 trading_system.invariants.BlindSegmentLedger;本模块在其上加 SQLite 持久化。
DB 路径见 config/train.yaml 的 optuna.registry_db。
"""

from __future__ import annotations

_PHASE = "Phase 3 任务 3.3/3.4"


class ExperimentRegistry:
    """SQLite 持久化的实验 / 盲测段注册表(占位)。"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise NotImplementedError(f"{_PHASE}:ExperimentRegistry 待实现(INV-6 持久化)。")
