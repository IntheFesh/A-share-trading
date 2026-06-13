"""四模型基线 + 随机候选池。Phase 2(任务 2.3)。对应 v3.1 第十三章。

① 候选集等权随机买入(1000 次抽样得净值分布,算策略所处分位);
② 单最佳因子排序;③ ElasticNet 截面秩回归;④ LightGBM 回归(此处为基线,非 Phase 3 完整排序模型)。
价格层:净值由引擎按 raw 成交记账;特征用 adj。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.3"


def random_candidate_baseline(*args, **kwargs):  # noqa: ANN002, ANN003
    """① 候选集等权随机买入,返回净值分布与策略分位。"""
    raise NotImplementedError(f"{_PHASE}:random_candidate_baseline 待实现。")


def single_factor_baseline(*args, **kwargs):  # noqa: ANN002, ANN003
    """② 单最佳因子排序基线。"""
    raise NotImplementedError(f"{_PHASE}:single_factor_baseline 待实现。")


def elasticnet_baseline(*args, **kwargs):  # noqa: ANN002, ANN003
    """③ ElasticNet 截面秩回归基线。"""
    raise NotImplementedError(f"{_PHASE}:elasticnet_baseline 待实现。")


def lightgbm_regression_baseline(*args, **kwargs):  # noqa: ANN002, ANN003
    """④ LightGBM 回归基线(非 Phase 3 完整排序模型)。"""
    raise NotImplementedError(f"{_PHASE}:lightgbm_regression_baseline 待实现。")
