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


def compute_time_decay_weights(dates, train_end, half_life: float) -> np.ndarray:
    """指数时间衰减样本权重 w = 0.5 ** (交易日差 / half_life)。

    交易日差用 numpy busday_count(date, train_end) 近似(与 assert_train_end_safe 同口径);
    越接近 train_end 的样本权重越接近 1,越远越小。half_life 单位=交易日(与候选 250/500/750 一致)。
    """
    te = np.datetime64(str(pd.Timestamp(train_end).date()))
    ds = np.array([np.datetime64(str(pd.Timestamp(d).date())) for d in dates])
    diff = np.array([np.busday_count(d, te) for d in ds], dtype="float64")  # date<=train_end -> >=0
    diff = np.clip(diff, 0.0, None)
    return np.power(0.5, diff / float(half_life))


def train_l2_model(
    df: pd.DataFrame,
    feature_specs: "list[FeatureSpec]",
    label_col: str,
    *,
    route: str = "C",
    n_quantiles: int = 5,
    params: "dict | None" = None,
    date_col: str = "trade_date",
    time_decay: "dict | None" = None,
):
    """训练 L2 模型。route ∈ {A 秩回归, B 净收益回归, C lambdarank 分位}。返回 (model, feature_names)。

    A/B 用 LGBMRegressor;C 用 LGBMRanker 且 group 按交易日(INV)。
    time_decay={"half_life":H,"train_end":te} 时按时间衰减给样本权重(默认 None=不加权,行为不变)。
    """
    import lightgbm as lgb

    p = {**DEFAULT_LGB_PARAMS, **(params or {})}
    _, y, groups, d = assemble_l2(df, feature_specs, label_col, date_col=date_col)
    names = [s.name for s in feature_specs]
    x_df = d[names]  # 用带列名的 DataFrame 拟合,使 fit/predict 特征名一致(避免 sklearn 警告)

    sw = None
    if time_decay is not None:
        sw = compute_time_decay_weights(d[date_col], time_decay["train_end"], time_decay["half_life"])

    if route in ("A", "B"):
        model = lgb.LGBMRegressor(**p)
        model.fit(x_df, y, sample_weight=sw)
    elif route == "C":
        # 分位相关性标签:每日把 label 分 n_quantiles 档(0..q-1),高 label 高相关
        rel = d.groupby(date_col)[label_col].transform(
            lambda s: pd.qcut(s.rank(method="first"), min(n_quantiles, max(1, s.nunique())),
                              labels=False, duplicates="drop")
        ).fillna(0).astype(int)
        model = lgb.LGBMRanker(**p)
        model.fit(x_df, rel.to_numpy(), group=groups, sample_weight=sw)
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
    from trading_system.labels import build_y_h, build_y_prod_engine
    from trading_system.feature_cache import cache_from_config, dataset_fingerprint, make_key
    from trading_system.model.cv import assert_train_end_safe
    from trading_system.model.model_io import ModelCard, save_model

    tcfg = config.get("training", {}) or {}
    # 路线/标签类型/时间衰减/调参均从 config 读(默认维持原行为:route=C,fixed_h,不衰减,不调参)
    route = tcfg.get("route", route)
    lcfg = tcfg.get("label", {}) or {}
    label_type = lcfg.get("type", "fixed_h")
    label_horizon = int(lcfg.get("fixed_horizon", label_horizon))
    features = list(config["features"])
    splits = config["splits"]
    max_holding = int(config.get("risk", {}).get("max_holding_days", 25))
    # 纪律:训练禁止触碰盲测段与其前的 embargo 间隔
    assert_train_end_safe(train_end, splits["blind_segment_start"], int(splits["embargo_days"]))

    panel = dataset[pd.to_datetime(dataset["trade_date"]) <= pd.Timestamp(train_end)].copy()
    if panel.empty:
        raise ValueError(f"train_end={train_end} 之前无数据可训练")
    work = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)

    # 特征/标签缓存(命中即读,内容不变;只加速,不改变结果)。label_spec 含标签类型 -> fixed_h/engine 不串。
    cache = cache_from_config(config)
    label_spec = ({"type": "fixed_h", "h": label_horizon} if label_type == "fixed_h"
                  else {"type": "engine", "max_holding": max_holding})
    key = make_key(dataset_fingerprint(work), features, label_spec)
    cached = cache.get(key)
    if cached is not None:
        work = cached
    else:
        for feat in features:
            work[feat] = compute_feature(feat, work).reindex(work.index)
        # 标签:fixed_h=固定窗(快);engine=引擎动态出场净收益(对齐实际交易,慢)。均按信号日 t 对齐。
        if label_type == "engine":
            work["__label__"] = build_y_prod_engine(work, max_holding=max_holding).reindex(work.index)
        else:
            work["__label__"] = build_y_h(work, label_horizon)
        cache.put(key, work[["code", "trade_date", *features, "__label__"]])

    # 时间衰减样本权重(默认关;开启则按信号日衰减,半衰期 = half_life_active 交易日)
    tdcfg = tcfg.get("time_decay", {}) or {}
    time_decay = ({"half_life": float(tdcfg.get("half_life_active", 500)), "train_end": train_end}
                  if tdcfg.get("enabled", False) else None)

    # Optuna 调参(默认关;开启则在 purged CV 上搜超参,绝不触碰盲测段,空间预注册)
    params = None
    tune_cfg = tcfg.get("tune", {}) or {}
    if tune_cfg.get("enabled", False):
        params = _tune_lgb_params(work, features, "__label__", config,
                                  n_trials=int(tune_cfg.get("n_trials", 30)), route=route)

    specs = [FeatureSpec(f) for f in features]
    model, names = train_l2_model(work, specs, "__label__", route=route, params=params,
                                  time_decay=time_decay)

    card = ModelCard(
        model=model,
        feature_names=names,
        train_start=str(pd.to_datetime(panel["trade_date"]).min().date()),
        train_end=str(pd.Timestamp(train_end).date()),
        params={"label_horizon": label_horizon, "route": route, "label_type": label_type,
                "time_decay": time_decay, "tuned_params": params,
                "lgb": params or DEFAULT_LGB_PARAMS,
                "risk": config.get("risk"), "cost": config.get("cost")},
        config_snapshot=config,
        route=route,
    )
    return save_model(card, Path(model_dir))


def _tune_lgb_params(work, features, label_col, config, *, n_trials=30, route="C"):
    """Optuna 在 purged CV 上粗调 LightGBM 超参(预注册空间;绝不触碰盲测段)。返回 best 参数 dict。"""
    from trading_system.backtest.metrics import mean_rank_ic
    from trading_system.invariants import FeatureSpec
    from trading_system.model.cv import purged_walk_forward_splits, row_masks
    from trading_system.model.tune import tune_hyperparams

    space = {  # 预注册的粗搜索空间(先注册后运行)
        "num_leaves": {"type": "int", "low": 7, "high": 63},
        "n_estimators": {"type": "int", "low": 30, "high": 150},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
    }
    w = work.dropna(subset=[*features, label_col]).sort_values("trade_date")
    dates = np.sort(w["trade_date"].unique())
    embargo = int(config["splits"]["embargo_days"])
    splits = purged_walk_forward_splits(dates, n_splits=2, embargo=embargo) if len(dates) > (embargo + 4) else []
    specs = [FeatureSpec(f) for f in features]

    def objective(params):
        if not splits:
            return 0.0
        ics = []
        for tr_dates, va_dates in splits:
            tr_mask, va_mask = row_masks(w["trade_date"].to_numpy(), tr_dates, va_dates)
            if tr_mask.sum() < 10 or va_mask.sum() < 5:
                continue
            model, names = train_l2_model(w[tr_mask], specs, label_col, route=route,
                                          params={"verbose": -1, **params})
            val = w[va_mask].copy()
            val["__s__"] = predict_scores(model, val, names)
            ics.append(mean_rank_ic(val.dropna(subset=["__s__"]), "__s__", label_col))
        return float(np.nanmean(ics)) if ics else 0.0

    _, best = tune_hyperparams(objective, space, n_trials=n_trials, direction="maximize")
    return best


def compare_label_routes(dataset, config, *, train_end=None, n_splits=2):
    """训练 A/B/C 三路线并在 purged CV 验证集上对比 RankIC(仅供人工择优,不自动选)。返回对比表。

    A=截面秩回归;B=winsorized 净收益回归;C=lambdarank 分位。对应 v3.1 §7.3。
    """
    import trading_system.features.builtin  # noqa: F401
    from trading_system.backtest.metrics import blocked_rank_ic, daily_rank_ic, icir, mean_rank_ic
    from trading_system.features.registry import compute_feature
    from trading_system.invariants import FeatureSpec
    from trading_system.labels import build_y_h
    from trading_system.model.cv import purged_walk_forward_splits, row_masks

    features = list(config["features"])
    horizon = int((config.get("training", {}).get("label", {}) or {}).get("fixed_horizon", 5))
    df = dataset.copy()
    if train_end is not None:
        df = df[pd.to_datetime(df["trade_date"]) <= pd.Timestamp(train_end)]
    work = df.sort_values(["code", "trade_date"]).reset_index(drop=True)
    for feat in features:
        work[feat] = compute_feature(feat, work).reindex(work.index)
    work["__ret__"] = build_y_h(work, horizon)
    # 三路线标签
    work["__A__"] = work.groupby("trade_date")["__ret__"].transform(lambda s: s.rank(pct=True))
    work["__B__"] = work.groupby("trade_date")["__ret__"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    specs = [FeatureSpec(f) for f in features]

    dates = np.sort(work.dropna(subset=["__ret__"])["trade_date"].unique())
    embargo = int(config["splits"]["embargo_days"])
    splits = purged_walk_forward_splits(dates, n_splits=n_splits, embargo=embargo)
    route_label = {"A": "__A__", "B": "__B__", "C": "__ret__"}
    rows = []
    for route, lbl in route_label.items():
        ic_series = []
        for tr_dates, va_dates in splits:
            tr_mask, va_mask = row_masks(work["trade_date"].to_numpy(), tr_dates, va_dates)
            tr = work[tr_mask].dropna(subset=[*features, lbl])
            if len(tr) < 10 or va_mask.sum() < 5:
                continue
            model, names = train_l2_model(tr, specs, lbl, route=route)
            val = work[va_mask].copy()
            val["__s__"] = predict_scores(model, val, names)
            ic_series.append(daily_rank_ic(val.dropna(subset=["__s__", "__ret__"]), "__s__", "__ret__"))
        if ic_series:
            allic = pd.concat(ic_series)
            rows.append({"route": route, "mean_rank_ic": float(allic.mean()),
                         "icir": icir(allic), "n_obs": int(allic.notna().sum())})
        else:
            rows.append({"route": route, "mean_rank_ic": float("nan"), "icir": float("nan"), "n_obs": 0})
    return pd.DataFrame(rows)


def predict_scores(model, df: pd.DataFrame, feature_names: "list[str]") -> np.ndarray:
    """对 df 用已训练模型打分(缺失特征行得 NaN)。"""
    sub = df[feature_names]
    mask = sub.notna().all(axis=1).to_numpy()
    scores = np.full(len(df), np.nan, dtype="float64")
    if mask.any():
        scores[mask] = model.predict(sub[mask])  # 传带列名 DataFrame,与 fit 一致
    return scores
