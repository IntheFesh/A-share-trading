"""三标签路线 + LightGBM。Phase 3(任务 3.2)。对应 v3.1 §7.1/§7.2/§8。

三标签路线:A 截面秩回归、B winsorized 净收益回归、C lambdarank 分位标签(优先五分位)。
LightGBM ranker 的 group **必须按交易日**(group_t=|G_t|,不能混成大表)——本模块在装配时强制。
L0/状态信息只以显式交互项进入(INV-4 守卫:assemble_l2 调用不变量检查)。
价格层:特征 adj、标签成交侧 raw(由 labels 模块保证)。lightgbm 惰性导入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.invariants import (
    FeatureSpec,
    assert_group_constant_only_via_interaction,
)

DEFAULT_LGB_PARAMS = dict(n_estimators=50, num_leaves=15, min_child_samples=5,
                          learning_rate=0.05, verbose=-1)


def assemble_l2(
    df: pd.DataFrame,
    feature_specs: "list[FeatureSpec]",
    label_col: str,
    *,
    date_col: str = "trade_date",
):
    """装配 L2 训练矩阵并强制 INV-4。返回 (X, y, group_sizes, sorted_df)。

    INV-4:group_constant 特征必须以交互项进入(否则报错)。
    group_sizes:按交易日的样本数(ranker 的 group 必须按日,数据已按日排序使各组连续)。
    """
    assert_group_constant_only_via_interaction(feature_specs)  # INV-4
    names = [s.name for s in feature_specs]
    d = (
        df.sort_values([date_col])
        .dropna(subset=[*names, label_col])
        .reset_index(drop=True)
    )
    X = d[names].to_numpy(dtype="float64")
    y = d[label_col].to_numpy(dtype="float64")
    group_sizes = d.groupby(date_col, sort=False).size().to_numpy()
    assert int(group_sizes.sum()) == len(d), "group 按日切分必须覆盖所有样本(INV)"
    return X, y, group_sizes, d


def train_l2_model(
    df: pd.DataFrame,
    feature_specs: "list[FeatureSpec]",
    label_col: str,
    *,
    route: str = "C",
    n_quantiles: int = 5,
    params: "dict | None" = None,
    date_col: str = "trade_date",
):
    """训练 L2 模型。route ∈ {A 秩回归, B 净收益回归, C lambdarank 分位}。返回 (model, feature_names)。

    A/B 用 LGBMRegressor;C 用 LGBMRanker 且 group 按交易日(INV)。
    """
    import lightgbm as lgb

    p = {**DEFAULT_LGB_PARAMS, **(params or {})}
    _, y, groups, d = assemble_l2(df, feature_specs, label_col, date_col=date_col)
    names = [s.name for s in feature_specs]
    x_df = d[names]  # 用带列名的 DataFrame 拟合,使 fit/predict 特征名一致(避免 sklearn 警告)

    if route in ("A", "B"):
        model = lgb.LGBMRegressor(**p)
        model.fit(x_df, y)
    elif route == "C":
        # 分位相关性标签:每日把 label 分 n_quantiles 档(0..q-1),高 label 高相关
        rel = d.groupby(date_col)[label_col].transform(
            lambda s: pd.qcut(s.rank(method="first"), min(n_quantiles, max(1, s.nunique())),
                              labels=False, duplicates="drop")
        ).fillna(0).astype(int)
        model = lgb.LGBMRanker(**p)
        model.fit(x_df, rel.to_numpy(), group=groups)
    else:
        raise ValueError(f"未知标签路线: {route}(应为 A/B/C)")
    return model, names


def train_and_save(
    dataset: pd.DataFrame,
    *,
    train_end: str,
    config: dict,
    model_dir: "str | Path",
    label_horizon: int = 5,
    route: str = "C",
):
    """用截至 train_end 的数据训练 L2 模型并存为出生证明包。补丁(run_train 的核心)。

    本函数是模型的**唯一生产者**:不做对比、不做上线决策。流程:断言 embargo(不触碰盲测段)→
    按 config['features'] 顺序算特征(截面秩变换)→ 构造标签 → train_l2_model → 落盘 ModelCard。
    特征清单/区间/参数从 config 读(勿在调用处硬编码)。返回模型文件路径。
    """
    from pathlib import Path

    import trading_system.features.builtin  # noqa: F401  确保内置特征已注册
    from trading_system.features.registry import compute_feature
    from trading_system.invariants import FeatureSpec
    from trading_system.labels import build_y_h
    from trading_system.feature_cache import cache_from_config, dataset_fingerprint, make_key
    from trading_system.model.cv import assert_train_end_safe
    from trading_system.model.model_io import ModelCard, save_model

    features = list(config["features"])
    splits = config["splits"]
    # 纪律:训练禁止触碰盲测段与其前的 embargo 间隔
    assert_train_end_safe(train_end, splits["blind_segment_start"], int(splits["embargo_days"]))

    panel = dataset[pd.to_datetime(dataset["trade_date"]) <= pd.Timestamp(train_end)].copy()
    if panel.empty:
        raise ValueError(f"train_end={train_end} 之前无数据可训练")
    work = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)

    # 特征/标签缓存(命中即读,内容不变;只加速,不改变结果)
    cache = cache_from_config(config)
    label_spec = {"type": "fixed_h", "h": label_horizon}
    key = make_key(dataset_fingerprint(work), features, label_spec)
    cached = cache.get(key)
    if cached is not None:
        work = cached
    else:
        for feat in features:
            work[feat] = compute_feature(feat, work).reindex(work.index)
        work["__label__"] = build_y_h(work, label_horizon)
        cache.put(key, work[["code", "trade_date", *features, "__label__"]])

    specs = [FeatureSpec(f) for f in features]
    model, names = train_l2_model(work, specs, "__label__", route=route)

    card = ModelCard(
        model=model,
        feature_names=names,
        train_start=str(pd.to_datetime(panel["trade_date"]).min().date()),
        train_end=str(pd.Timestamp(train_end).date()),
        params={"label_horizon": label_horizon, "route": route,
                "lgb": DEFAULT_LGB_PARAMS, "risk": config.get("risk"), "cost": config.get("cost")},
        config_snapshot=config,
        route=route,
    )
    return save_model(card, Path(model_dir))


def predict_scores(model, df: pd.DataFrame, feature_names: "list[str]") -> np.ndarray:
    """对 df 用已训练模型打分(缺失特征行得 NaN)。"""
    sub = df[feature_names]
    mask = sub.notna().all(axis=1).to_numpy()
    scores = np.full(len(df), np.nan, dtype="float64")
    if mask.any():
        scores[mask] = model.predict(sub[mask])  # 传带列名 DataFrame,与 fit 一致
    return scores
