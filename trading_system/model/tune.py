"""Optuna 调参(Tier 2)。Phase 3(任务 3.3)。对应 v3.1 §8。

搜索空间从 config/train.yaml 预注册块读取(**先注册后运行,禁止边搜边扩**:本模块只会 suggest
space 中已声明的键)。目标=训练窗内 purged 时序 CV 的扣费 top-K 净收益(由调用方以 objective
传入);MedianPruner;1-SE 规则选参;全 trial 可写入实验注册表(供 DSR 的 N 与 PBO 记账,INV-6)。
"""

from __future__ import annotations

from typing import Callable


def _suggest(trial, name: str, spec):
    """按预注册 spec 生成一个建议值。spec 形如 {'type':'float','low':..,'high':..,'log':bool}
    或 {'type':'int',...} 或 {'type':'categorical','choices':[...]}。"""
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"未知搜索空间类型: {t}")


def tune_hyperparams(
    objective: "Callable[[dict], float]",
    search_space: dict,
    *,
    n_trials: int = 30,
    direction: str = "maximize",
    seed: int = 0,
):
    """跑 Optuna 搜索。objective(params)->float;params 的键严格来自预注册 search_space。

    返回 (study, best_params)。best_params 的键 ⊆ search_space 的键(结构性保证不越界扩张)。
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        return objective(params)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(_objective, n_trials=n_trials)
    # 结构性纪律:最优参数键必须是预注册空间的子集
    assert set(study.best_params).issubset(set(search_space)), "调参越界扩张(违反先注册后运行)"
    return study, study.best_params
