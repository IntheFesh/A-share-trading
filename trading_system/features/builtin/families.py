"""内置特征族(代表性子集)。Phase 1(任务 1.2)。对应 v3.1 §7.3。

全部用后复权价(adj,特征侧 INV-2);全部 point-in-time(只用截至当日的历史,过截断等变性)。
导入本模块即触发注册(见 features.builtin.__init__)。每个特征的时序原值由 registry
计算,再由 registry.cross_sectional_rank 做每日截面 winsorize + 秩变换。

覆盖:量价基础 / 趋势 / 反转彩票 / 过度拉升 / 流动性 / 换手率族 / CGO 族。
换手率族与 CGO 族用 BaoStock 提供的 ``turn``(换手率)实现——CGO 用换手率衰减加权的参考成本价
(Grinblatt–Han),不臆造、不用代理冒充流通股本(无 turn 来源时为 NaN)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.features.registry import register


# ── 量价基础族 ──────────────────────────────────────────────────────────────
@register("ret_1", "量价基础", lookback=1)
def ret_1(g: pd.DataFrame) -> pd.Series:
    """1 日后复权收益。"""
    return g["close_adj"].pct_change(1)


@register("ret_5", "量价基础", lookback=5)
def ret_5(g: pd.DataFrame) -> pd.Series:
    return g["close_adj"].pct_change(5)


@register("ret_20", "量价基础", lookback=20)
def ret_20(g: pd.DataFrame) -> pd.Series:
    return g["close_adj"].pct_change(20)


@register("vol_20", "量价基础", lookback=20)
def vol_20(g: pd.DataFrame) -> pd.Series:
    """20 日已实现波动(后复权日收益滚动标准差)。"""
    return g["close_adj"].pct_change(1).rolling(20).std()


@register("volume_ratio_5", "量价基础", lookback=6)
def volume_ratio_5(g: pd.DataFrame) -> pd.Series:
    """量比:当日量 / 过去 5 日(不含当日)均量。"""
    prior_mean = g["volume"].shift(1).rolling(5).mean()
    return g["volume"] / prior_mean


@register("amihud_20", "流动性", lookback=20)
def amihud_20(g: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性:|日收益|/成交额 的 20 日均值(放大便于截面比较)。"""
    illiq = (g["close_adj"].pct_change(1).abs() / g["amount"].replace(0, np.nan)) * 1e9
    return illiq.rolling(20).mean()


# ── 趋势族 ──────────────────────────────────────────────────────────────────
@register("ma_ratio_20", "趋势", lookback=20)
def ma_ratio_20(g: pd.DataFrame) -> pd.Series:
    """收盘 / 20 日均线 - 1。"""
    return g["close_adj"] / g["close_adj"].rolling(20).mean() - 1.0


@register("ma_ratio_60", "趋势", lookback=60)
def ma_ratio_60(g: pd.DataFrame) -> pd.Series:
    return g["close_adj"] / g["close_adj"].rolling(60).mean() - 1.0


# ── 反转彩票族 ──────────────────────────────────────────────────────────────
@register("reversal_5", "反转彩票", lookback=5)
def reversal_5(g: pd.DataFrame) -> pd.Series:
    """短期反转:近 5 日收益取负(高者预期反转向下)。"""
    return -g["close_adj"].pct_change(5)


@register("max_ret_5", "反转彩票", lookback=5)
def max_ret_5(g: pd.DataFrame) -> pd.Series:
    """MAX 彩票特征:近 5 日单日最大涨幅(过度拉升/博彩需求代理)。"""
    return g["close_adj"].pct_change(1).rolling(5).max()


# ── 过度拉升族(服务疑问②;只作截面原值,进 L2 须与 HiLo 交互——INV-7)──────
@register("dist_ma20", "过度拉升", lookback=20)
def dist_ma20(g: pd.DataFrame) -> pd.Series:
    """距 20 日均线乖离率。"""
    return g["close_adj"] / g["close_adj"].rolling(20).mean() - 1.0


@register("dist_high_20", "过度拉升", lookback=20)
def dist_high_20(g: pd.DataFrame) -> pd.Series:
    """距 20 日最高收盘的距离(<=0;越接近 0 越接近新高=越拉升)。"""
    return g["close_adj"] / g["close_adj"].rolling(20).max() - 1.0


# ── 换手率族(用 BaoStock turn;无换手数据时为 NaN)──────────────────────────
@register("turnover_mean_20", "换手率", lookback=20)
def turnover_mean_20(g: pd.DataFrame) -> pd.Series:
    """20 日平均换手率。"""
    return g["turn"].rolling(20).mean()


@register("turnover_chg_20", "换手率", lookback=20)
def turnover_chg_20(g: pd.DataFrame) -> pd.Series:
    """当日换手相对过去 20 日均值的变化(放量/缩量倾向)。"""
    return g["turn"] / g["turn"].shift(1).rolling(20).mean() - 1.0


# ── CGO 族(Grinblatt–Han 资本利得突出量;换手率衰减加权参考价)──────────────
def _cgo(g: pd.DataFrame, window: int) -> pd.Series:
    """CGO = (P_t - RP_t)/P_t,RP_t 为换手率衰减加权的参考成本价(只用过去,point-in-time)。

    RP_t = Σ_{k=1..n} w_{t,k}·P_{t-k},w_{t,k} = V_{t-k}·Π_{j=1..k-1}(1-V_{t-j}),归一化;
    V 为换手率(turn/100,截断到 [0,1])。窗口内任一缺失则该行为 NaN。服务疑问②"切高位浮盈股"。
    半向量化实现:外层逐日(保持 point-in-time 与截断等变性),内层用 numpy 一次算完窗口权重。
    """
    p = g["close_adj"].to_numpy(dtype="float64")
    v = np.clip(g["turn"].to_numpy(dtype="float64") / 100.0, 0.0, 1.0)
    n = len(g)
    rp = np.full(n, np.nan)
    for t in range(n):
        m = min(t, window)            # 窗口内可用的滞后阶数 k=1..m
        if m == 0:
            continue
        idx = np.arange(t - 1, t - 1 - m, -1)   # 滞后 P_{t-1}..P_{t-m}、V_{t-1}..V_{t-m}
        big_p, big_v = p[idx], v[idx]
        if np.isnan(big_v).any() or np.isnan(big_p).any():
            continue                  # 与旧实现一致:窗口内任一 NaN -> rp NaN
        # remaining[k-1] = Π_{j=1..k-1}(1-V_{t-j});w[k-1] = V_{t-k}·remaining[k-1]
        remaining = np.concatenate(([1.0], np.cumprod(1.0 - big_v)[:-1]))
        w = big_v * remaining
        wsum = w.sum()
        if wsum > 0:
            rp[t] = float((w * big_p).sum() / wsum)
    return pd.Series((p - rp) / p, index=g.index)


@register("cgo_60", "CGO", lookback=60)
def cgo_60(g: pd.DataFrame) -> pd.Series:
    """60 日窗口的资本利得突出量。"""
    return _cgo(g, 60)


@register("cgo_120", "CGO", lookback=120)
def cgo_120(g: pd.DataFrame) -> pd.Series:
    """120 日窗口的资本利得突出量。"""
    return _cgo(g, 120)
