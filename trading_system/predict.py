"""每日预测编排(run_predict 的核心)。补丁。对应 v3.1 第十二章。

只加载已训练模型,**绝不在内部训练**(防用未来数据重训)。加载后第一步硬核对特征名/顺序与模型
出生证明逐一吻合,不一致即报错退出。所有特征均为 T 日收盘可得(断言不使用 > asof 的数据,呼应
INV-1/3)。产出作战手册(代码/分数/排名/限价/止损/止盈三阶梯/时间止损/风险标注)。
参数从 config 读;价格层:限价/止损/止盈用 raw(INV-2)。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def review_prev_recommendations(
    dataset: pd.DataFrame, prev_codes: "list[str]", prev_date, today_date
) -> "dict[str, float]":
    """昨日推荐票今日实际涨跌幅(观感参考,非严谨业绩)。用 raw 收盘的简单涨跌幅。"""
    out = {}
    for code in prev_codes:
        g = dataset[(dataset["code"] == code)]
        p = g[pd.to_datetime(g["trade_date"]) == pd.Timestamp(prev_date)]
        t = g[pd.to_datetime(g["trade_date"]) == pd.Timestamp(today_date)]
        if len(p) and len(t) and float(p["close_raw"].iloc[0]) > 0:
            out[code] = float(t["close_raw"].iloc[0]) / float(p["close_raw"].iloc[0]) - 1.0
    return out


def run_prediction(
    dataset: pd.DataFrame,
    *,
    asof_date: str,
    config: dict,
    model_path: "str",
    output_dir: "str",
    top_k: int = 20,
    print_console: bool = True,
    risk_flags: "dict | None" = None,
):
    """加载冠军模型 → 校验特征一致性 → PIT 算特征 → 排序 → 出作战手册。返回 (playbook_df, info)。

    risk_flags(可选):外部风险数据接口,code -> {pledge_high, goodwill_high, recent_regulatory_letter};
    提供则填入对应列,不提供则这些列为空(不臆造,不影响任何计算)。1E-4。
    """
    import trading_system.features.builtin  # noqa: F401  确保内置特征已注册
    from trading_system.backtest.engine import compute_atr
    from trading_system.features.registry import compute_feature
    from trading_system.model.model_io import assert_feature_consistency, load_model
    from trading_system.model.train import predict_scores
    from trading_system.playbook import generate_playbook
    from trading_system import portfolio as port

    card = load_model(model_path)
    features = list(config["features"])
    # 硬闸:当前特征清单(来自 config)必须与模型出生证明逐一吻合
    assert_feature_consistency(card, features)

    asof = pd.Timestamp(asof_date)
    panel = dataset[pd.to_datetime(dataset["trade_date"]) <= asof].copy()
    if panel.empty:
        raise ValueError(f"asof={asof_date} 之前无数据")
    # PIT 断言:绝不使用 > asof 的数据(无 T+1)
    assert pd.to_datetime(panel["trade_date"]).max() <= asof, "预测使用了 asof 之后的数据(违反 PIT)"

    work = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)
    # 特征缓存(命中即读;只加速,不改变结果)
    from trading_system.feature_cache import cache_from_config, compute_features_cached
    work = compute_features_cached(work, features, cache_from_config(config), spec_tag="predict")
    work["__score__"] = predict_scores(card.model, work, card.feature_names)

    asof_rows = work[pd.to_datetime(work["trade_date"]) == asof].dropna(subset=["__score__"]).copy()
    asof_rows["rank"] = asof_rows["__score__"].rank(ascending=False, method="first")
    top = asof_rows.nsmallest(top_k, "rank")

    # 数据质量:作战手册 ATR 稳健化(config 驱动;此处 panel 已按 asof 截断,故 PIT)
    dq = config.get("data_quality", {}) or {}
    robust_atr = bool(dq.get("robust_atr", False))
    winsor_q = float(dq.get("atr_winsorize_quantile", 0.99))

    # SHAP(LightGBM 原生 pred_contrib,不强依赖 shap 库):每票贡献最大的前三特征 + 方向
    fn = card.feature_names
    shap_map = {}
    try:
        contrib = np.atleast_2d(np.asarray(card.model.predict(top[fn], pred_contrib=True),
                                           dtype="float64"))
        for j, idx in enumerate(top.index):
            c = contrib[j][: len(fn)]
            order = np.argsort(-np.abs(c))[:3]
            shap_map[idx] = ", ".join(f"{fn[k]}{'↑' if c[k] >= 0 else '↓'}" for k in order)
    except Exception as exc:  # noqa: BLE001 — SHAP 失败不影响主流程
        logger.warning("SHAP 贡献计算失败,shap_top3 置空: %r", exc)

    # L0 regime(**仅展示**;绝不进入 model_score 计算、不作模型特征;接入打分需 INV-4 交互并验证=未来工作)
    T_t = stage = m_t = hilo_t = None
    try:
        from trading_system.regime import compute_hilo, compute_six_indicators, compute_temperature
        temp = compute_temperature(compute_six_indicators(panel))
        hilo_df = compute_hilo(panel)
        if asof in temp.index:
            T_t = round(float(temp.loc[asof, "T_t"]), 4)
            stage = int(temp.loc[asof, "stage"])
            m_t = round(float(temp.loc[asof, "m_t"]), 4)
        if asof in hilo_df.index and np.isfinite(hilo_df.loc[asof, "hilo_t"]):
            hilo_t = round(float(hilo_df.loc[asof, "hilo_t"]), 4)
    except Exception as exc:  # noqa: BLE001 — regime 仅展示,失败不影响预测
        logger.warning("regime 展示指标计算失败(不影响预测): %r", exc)

    # 每票止损/止盈(用 raw + ATR);限价买入价为 T+1 开盘的限价参考(以 T 收盘为基准)
    stop_mult = float(config["risk"]["stop_loss_atr_mult"])
    cap = float(config["risk"]["single_stock_cap_normal"]) * 100.0
    rf = risk_flags or {}
    rows = []
    for idx, r in top.iterrows():
        code = r["code"]
        g = panel[panel["code"] == code].sort_values("trade_date")
        atr = (float(compute_atr(g, 14, robust=robust_atr, winsor_q=winsor_q).iloc[-1])
               if len(g) >= 15 else np.nan)
        close = float(r["close_raw"])
        score = float(r["__score__"])
        risk = stop_mult * atr if np.isfinite(atr) else np.nan
        stop_price = round(close - risk, 2) if np.isfinite(risk) and close > 0 else None
        stop_distance_pct = (round((close - stop_price) / close * 100.0, 2)
                             if stop_price is not None and close > 0 else None)
        amihud = ((g["close_adj"].pct_change().abs() / g["amount"].replace(0, np.nan))
                  * 1e9).rolling(20).mean()
        amihud_illiq = float(amihud.iloc[-1]) if len(amihud) and np.isfinite(amihud.iloc[-1]) else None
        flags = rf.get(code, {})  # 1E-4 外部数据接口:无输入则为空,不影响
        rows.append({
            "code": code, "trigger": None, "model_score": round(score, 4),
            "rank": int(r["rank"]), "shap_top3": shap_map.get(idx),
            "limit_buy_price": close, "target_weight_pct": cap,
            "stop_price": stop_price,
            "tp1_price": round(close + risk, 2) if np.isfinite(risk) else None,
            "tp2_price": round(close + 2 * risk, 2) if np.isfinite(risk) else None,
            "tp3_price": None,
            "time_stop_date": str(np.busday_offset(asof.date(),
                                  int(config["risk"]["max_holding_days"]), roll="forward")),
            "veto_reason": None,
            "atr_n": round(atr, 4) if np.isfinite(atr) else None,
            "single_cap_pct": cap,
            "kelly_suggest_pct": round(port.kelly_risk_budget(score) * 100.0, 4),
            "stop_distance_pct": stop_distance_pct,
            "amihud_illiq": amihud_illiq,
            "days_to_disclosure": r.get("days_to_disclosure"),
            "has_preann": r.get("has_preann"),
            # 仅展示:个股过度拉升度(距MA20乖离的截面分位)与当日高低切;不进入打分
            "overextension_score": (round(float(r["dist_ma20"]), 4)
                                    if ("dist_ma20" in r and pd.notna(r.get("dist_ma20"))) else None),
            "hilo_regime": hilo_t,
            # 需额外数据源(质押/商誉/监管函):risk_flags 提供则填,否则空(不臆造)
            "pledge_high": flags.get("pledge_high"),
            "goodwill_high": flags.get("goodwill_high"),
            "recent_regulatory_letter": flags.get("recent_regulatory_letter"),
        })
    candidates = pd.DataFrame(rows)
    regime = {"T_t": T_t, "stage": stage, "m_t": m_t, "w_total": None,
              "brake_level": None, "days_to_tier1": None, "days_to_tier2": None, "hilo_t": hilo_t}
    table, md = generate_playbook(candidates, regime, trade_date=asof_date, out_dir=output_dir,
                                  print_console=False)

    info = {"model_path": str(model_path), "model_train_end": card.train_end,
            "n_recommended": len(table)}
    if print_console:
        # 让用户下单前知道模型新旧(允许的 print 例外:作战手册/控制台)
        print(f"[模型] 使用 {model_path};训练截止 {card.train_end}")
        print(f"[手册] {asof_date} 共 {len(table)} 只推荐,已写入 {output_dir}")
        print("[提示] 昨日推荐回顾为观感参考,非严谨业绩;实盘含 T+1 开盘成交、成本与人工否决,严谨收益见回测。")
    return table, info
