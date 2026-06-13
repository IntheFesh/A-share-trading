"""内置特征族(代表性子集)。Phase 1(任务 1.2)。对应 v3.1 §7.3。

全部用后复权价(adj,特征侧 INV-2);全部 point-in-time(只用截至当日的历史,过截断等变性)。
导入本模块即触发注册(见 features.builtin.__init__)。每个特征的时序原值由 registry
计算,再由 registry.cross_sectional_rank 做每日截面 winsorize + 秩变换。

注:CGO / 换手率族需流通股本(本仓 Phase 0 schema 未含),故此处未实现,留待数据补齐后接入
(不臆造,不用代理冒充)。当前覆盖:量价基础 / 趋势 / 反转彩票 / 过度拉升 / 流动性。
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
