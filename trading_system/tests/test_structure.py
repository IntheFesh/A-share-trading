"""结构冒烟:导入全部子包 / 模块,确保骨架 import 干净。

stub 不在顶层引入重型依赖(pandas/lightgbm 等),故即便尚未 pip install 也应能 import。
若某模块顶层误引重型依赖,本测试会立刻暴露。另校验 config/*.yaml 齐备。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

MODULES = [
    "trading_system",
    "trading_system.invariants",
    "trading_system.check_env",
    # data(Phase 0)
    "trading_system.data",
    "trading_system.data.calendar",
    "trading_system.data.price_layers",
    "trading_system.data.store",
    "trading_system.data.quality",
    "trading_system.data.universe",
    "trading_system.data.collectors",
    "trading_system.data.collectors.baostock",
    "trading_system.data.collectors.tushare",
    "trading_system.data.collectors.tencent",
    "trading_system.data.collectors.sina",
    # features / regime / triggers / labels(Phase 1)
    "trading_system.features",
    "trading_system.features.registry",
    "trading_system.features.builtin",
    "trading_system.regime",
    "trading_system.triggers",
    "trading_system.labels",
    # backtest / portfolio / overlays(Phase 2)
    "trading_system.backtest",
    "trading_system.backtest.engine",
    "trading_system.backtest.costs",
    "trading_system.backtest.baselines",
    "trading_system.backtest.metrics",
    "trading_system.backtest.stress",
    "trading_system.portfolio",
    "trading_system.overlays",
    # model(Phase 3)
    "trading_system.model",
    "trading_system.model.cv",
    "trading_system.model.train",
    "trading_system.model.tune",
    "trading_system.model.approval",
    # playbook / audit / reports(Phase 4)
    "trading_system.playbook",
    "trading_system.audit",
    "trading_system.audit.experiment_registry",
    "trading_system.reports",
    "trading_system.reports.monitor",
    # run entry points
    "trading_system.run",
    "trading_system.run.phase0_acceptance",
    "trading_system.run.phase1_factor_report",
    "trading_system.run.phase2_acceptance",
    "trading_system.run.phase3_acceptance",
]

EXPECTED_CONFIGS = (
    "data.yaml",
    "costs.yaml",
    "triggers.yaml",
    "risk.yaml",
    "exit.yaml",
    "regime.yaml",
    "train.yaml",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_clean(module_name: str) -> None:
    importlib.import_module(module_name)


def test_all_configs_present() -> None:
    config_dir = Path(importlib.import_module("trading_system").__file__).resolve().parent / "config"
    missing = [name for name in EXPECTED_CONFIGS if not (config_dir / name).exists()]
    assert not missing, f"缺少配置文件: {missing}"
