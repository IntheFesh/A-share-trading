"""Optuna 调参(Tier 2)。Phase 3(任务 3.3)。对应 v3.1 第八章。

搜索空间从 config/train.yaml 预注册块读取(先注册后运行,禁止边搜边扩)。
目标=训练窗内 purged 时序 CV 的扣费 top-K 净收益;MedianPruner;1-SE 规则选参。
trial 预算 30~50 起;全 trial 写入实验注册表(SQLite),供 DSR 的 N 与 PBO 记账(INV-6)。
"""

from __future__ import annotations

_PHASE = "Phase 3 任务 3.3"


def tune_hyperparams(*args, **kwargs):  # noqa: ANN002, ANN003
    """跑 Optuna 搜索(预注册空间;全 trial 入注册表)。"""
    raise NotImplementedError(f"{_PHASE}:tune_hyperparams 待实现(先注册后运行)。")
