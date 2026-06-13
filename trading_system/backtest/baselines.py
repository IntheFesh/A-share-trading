"""四模型基线 + 随机候选池。Phase 2(任务 2.3)。对应 v3.1 §7.1 / 第十三章。

① 候选集等权随机买入(抽样得净值分布,算策略所处分位);② 单最佳因子排序;
③ ElasticNet 截面秩回归;④ LightGBM 回归(Phase 3 装好 lightgbm 后接入,见 model/)。
复杂模型须**同时**战胜这四者方可成为 challenger。价格层:特征 adj、收益由标签按 raw 算好。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.backtest.metrics import mean_rank_ic


def random_candidate_baseline(
    candidate_returns: "np.ndarray", k: int, *, n_samples: int = 1000, seed: int = 0
) -> np.ndarray:
    """① 从候选集等权随机抽 k 只,重复 n_samples 次,返回组合平均收益的分布。"""
    cr = np.asarray(candidate_returns, dtype="float64")
    cr = cr[~np.isnan(cr)]
    if len(cr) < k:
        raise ValueError("候选集规模小于 k")
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(cr, size=k, replace=False).mean() for _ in range(n_samples)])


def strategy_percentile(strategy_return: float, dist: np.ndarray) -> float:
    """策略收益在随机分布中的分位(0~1):越高越说明策略胜过随机。"""
    return float((np.asarray(dist) < strategy_return).mean())


def single_factor_baseline(
    df: pd.DataFrame, feature_cols: "list[str]", label_col: str, *, date_col: str = "trade_date"
) -> "tuple[str, float]":
    """② 选 |mean RankIC| 最大的单因子作基线。返回 (因子名, 其带符号 RankIC)。"""
    best_name, best_ic, best_abs = None, 0.0, -1.0
    for col in feature_cols:
        ic = mean_rank_ic(df.dropna(subset=[col, label_col]), col, label_col, date_col=date_col)
        if np.isfinite(ic) and abs(ic) > best_abs:
            best_name, best_ic, best_abs = col, ic, abs(ic)
    return best_name, best_ic


def elasticnet_baseline(
    df: pd.DataFrame,
    feature_cols: "list[str]",
    label_col: str,
    *,
    date_col: str = "trade_date",
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
) -> "tuple[float, np.ndarray]":
    """③ ElasticNet 截面秩回归基线:用(已秩变换的)特征拟合标签,返回 (预测分的 mean RankIC, 系数)。

    说明:此处为同窗拟合的基线参照(非样本外);Phase 3 的正式评估走 purged walk-forward。
    """
    from sklearn.linear_model import ElasticNet

    sub = df.dropna(subset=[*feature_cols, label_col]).copy()
    X = sub[feature_cols].to_numpy(dtype="float64")
    y = sub[label_col].to_numpy(dtype="float64")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000)
    model.fit(X, y)
    sub["__pred__"] = model.predict(X)
    ic = mean_rank_ic(sub, "__pred__", label_col, date_col=date_col)
    return ic, model.coef_


def lightgbm_regression_baseline(
    df: pd.DataFrame,
    feature_cols: "list[str]",
    label_col: str,
    *,
    date_col: str = "trade_date",
    params: "dict | None" = None,
) -> "tuple[float, object]":
    """④ LightGBM 回归基线:复杂模型须同时战胜本基线方可成为 challenger(v3.1 §7.1)。

    返回 (预测分的 mean RankIC, 模型)。同窗拟合的参照;正式样本外评估走 Phase 3 purged walk-forward。
    """
    import lightgbm as lgb

    from trading_system.model.train import DEFAULT_LGB_PARAMS

    sub = df.dropna(subset=[*feature_cols, label_col]).copy()
    model = lgb.LGBMRegressor(**{**DEFAULT_LGB_PARAMS, **(params or {})})
    model.fit(sub[feature_cols], sub[label_col])
    sub["__pred__"] = model.predict(sub[feature_cols])
    ic = mean_rank_ic(sub, "__pred__", label_col, date_col=date_col)
    return ic, model
