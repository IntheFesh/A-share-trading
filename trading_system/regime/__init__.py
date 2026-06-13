"""Phase 1:L0 情绪温度与市场状态。任务 1.4。对应 v3.1 §5.1/§5.3 与 v0.3 第十七章。

由日线 OHLC + raw 昨收推导**六指标**(涨停家数、最高连板高度、晋级率、炸板率、
昨日涨停今日溢价、跌停+核按钮数)→ 合成情绪温度 T_t(250 日分位、初始等权,方向修正)→
五阶段 + 总敞口乘子 m_t;另出 HiLo 高低切状态量。

INV-4:T_t / HiLo 是"组内常数"(当天同值),默认是覆盖层;进 L2 须以显式交互项并过检验。
价格层:涨跌停/炸板用 raw 昨收算(INV-2);溢价/收益用 adj。权重/阈值从 config/regime.yaml 读。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.invariants import round_half_up_2

# 六指标方向:+1 越大越热,-1 越大越冷(合成温度时对 -1 者取 1-分位)。
INDICATOR_DIRECTION: dict[str, int] = {
    "limit_up_count": +1,
    "max_consecutive_boards": +1,
    "promotion_rate": +1,
    "blowup_rate": -1,
    "prev_limitup_premium": +1,
    "limitdown_plus_nuclear": -1,
}
SIX_INDICATORS = tuple(INDICATOR_DIRECTION.keys())


def compute_six_indicators(
    panel: pd.DataFrame,
    *,
    main_ratio: float = 0.10,
    st_ratio: float = 0.05,
    nuclear_reversal: float = 0.05,
    limit_tol: float = 0.005,
) -> pd.DataFrame:
    """由日线面板推导 L0 六指标(逐交易日一行)。对应 v3.1 §5.1。

    炸板率=触及涨停但未封住 / 触及涨停;晋级率=今封板 ∧ 昨封板 / 昨封板;
    核按钮=盘中冲高(>+5%)却尾盘跳水(<-5%)的反转票数(近似,可由 config 调阈)。
    """
    p = panel.sort_values(["code", "trade_date"]).copy()
    ratio = np.where(p["is_st"].to_numpy(dtype=bool), st_ratio, main_ratio)
    pc = p["preclose_raw"].to_numpy(dtype="float64")
    limit_up_px = round_half_up_2(pc * (1.0 + ratio))
    high = p["high_raw"].to_numpy(dtype="float64")
    close = p["close_raw"].to_numpy(dtype="float64")

    p["touched_up"] = high >= (limit_up_px - limit_tol)
    p["blowup"] = p["touched_up"].to_numpy(dtype=bool) & ~p["is_limit_up"].to_numpy(dtype=bool)
    intraday_up = high / pc - 1.0
    close_ret = close / pc - 1.0
    p["nuclear"] = (intraday_up >= nuclear_reversal) & (close_ret <= -nuclear_reversal)

    # 连板高度:同一 code 连续涨停的累计;遇非涨停或换 code 归零
    s = p["is_limit_up"].astype(int)
    new_block = (s == 0) | (p["code"] != p["code"].shift(1))
    block = new_block.cumsum()
    p["streak"] = s.groupby(block).cumsum()

    p["prev_limit_up"] = (
        p.groupby("code")["is_limit_up"].shift(1).fillna(False).astype(bool)
    )
    p["ret_today"] = p.groupby("code")["close_adj"].pct_change(1)
    p["promoted"] = p["is_limit_up"].to_numpy(dtype=bool) & p["prev_limit_up"].to_numpy(dtype=bool)

    daily = p.groupby("trade_date").agg(
        limit_up_count=("is_limit_up", "sum"),
        limit_down_count=("is_limit_down", "sum"),
        touched_up_count=("touched_up", "sum"),
        blowup_count=("blowup", "sum"),
        nuclear_count=("nuclear", "sum"),
        promoted_count=("promoted", "sum"),
        prev_lu_count=("prev_limit_up", "sum"),
    )
    mcb = p[p["is_limit_up"]].groupby("trade_date")["streak"].max()
    daily["max_consecutive_boards"] = mcb.reindex(daily.index).fillna(0)
    daily["promotion_rate"] = (
        daily["promoted_count"] / daily["prev_lu_count"].replace(0, np.nan)
    ).fillna(0.0)
    daily["blowup_rate"] = (
        daily["blowup_count"] / daily["touched_up_count"].replace(0, np.nan)
    ).fillna(0.0)
    prem = p[p["prev_limit_up"]].groupby("trade_date")["ret_today"].mean()
    daily["prev_limitup_premium"] = prem.reindex(daily.index).fillna(0.0)
    daily["limitdown_plus_nuclear"] = daily["limit_down_count"] + daily["nuclear_count"]

    return daily[list(SIX_INDICATORS)].copy()


def compute_temperature(
    indicators: pd.DataFrame,
    *,
    weights: "dict[str, float] | None" = None,
    window: int = 250,
    min_periods: int = 1,
    thresholds: "list[float]" = (0.2, 0.4, 0.6, 0.8),
    m_t_by_stage: "list[float]" = (0.2, 0.5, 0.8, 1.0, 0.6),
) -> pd.DataFrame:
    """合成情绪温度 T_t ∈ [0,1] 并映射五阶段 + m_t。对应 v3.1 §5.1。

    每个指标先做滚动 ``window`` 日分位(PIT),按方向修正(冷向取 1-分位),按权重(默认等权)
    加权平均得 T_t;再按 thresholds 切五阶段,查 m_t_by_stage。返回列:T_t / stage / m_t。
    """
    if weights is None:
        weights = {k: 1.0 / len(SIX_INDICATORS) for k in SIX_INDICATORS}
    comp = pd.Series(0.0, index=indicators.index)
    wsum = 0.0
    for name in SIX_INDICATORS:
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        pct = indicators[name].rolling(window, min_periods=min_periods).rank(pct=True)
        if INDICATOR_DIRECTION[name] < 0:
            pct = 1.0 - pct
        comp = comp + w * pct.fillna(0.5)
        wsum += w
    t_t = comp / wsum if wsum > 0 else comp
    stage = np.digitize(t_t.to_numpy(), np.asarray(thresholds))  # 0..len(thresholds)
    m_arr = np.asarray(m_t_by_stage, dtype="float64")
    m_t = m_arr[np.clip(stage, 0, len(m_arr) - 1)]
    return pd.DataFrame({"T_t": t_t.to_numpy(), "stage": stage, "m_t": m_t}, index=indicators.index)


def compute_hilo(
    panel: pd.DataFrame,
    *,
    rank_window: int = 20,
    n_layers: int = 10,
    spread_window: int = 5,
    quantile_window: int = 250,
    min_periods: int = 1,
) -> pd.DataFrame:
    """HiLo 高低切状态量。对应 v3.1 §5.3(b)。

    每日按近 rank_window 日涨幅分 n_layers 层;HiLo_raw = 高层近 spread_window 日收益均值 −
    低层近 spread_window 日收益均值。正=高位股跑赢(动量);转负=高低切信号。再做滚动分位 HiLo_t。
    """
    p = panel.sort_values(["code", "trade_date"]).copy()
    p["ret_rank"] = p.groupby("code")["close_adj"].pct_change(rank_window)
    p["ret_spread"] = p.groupby("code")["close_adj"].pct_change(spread_window)

    def _daily_spread(g: pd.DataFrame) -> float:
        gg = g.dropna(subset=["ret_rank", "ret_spread"])
        if len(gg) < n_layers:
            return np.nan
        try:
            layer = pd.qcut(gg["ret_rank"], n_layers, labels=False, duplicates="drop")
        except ValueError:
            return np.nan
        top, bot = layer.max(), layer.min()
        hi = gg.loc[layer == top, "ret_spread"].mean()
        lo = gg.loc[layer == bot, "ret_spread"].mean()
        return float(hi - lo)

    hilo_raw = p.groupby("trade_date")[["ret_rank", "ret_spread"]].apply(_daily_spread)
    hilo_raw = hilo_raw.sort_index()
    hilo_t = hilo_raw.rolling(quantile_window, min_periods=min_periods).rank(pct=True)
    return pd.DataFrame({"hilo_raw": hilo_raw.to_numpy(), "hilo_t": hilo_t.to_numpy()},
                        index=hilo_raw.index)
