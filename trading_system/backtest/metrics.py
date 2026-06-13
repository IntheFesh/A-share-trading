"""回测指标。Phase 2(任务 2.4),Phase 1.6 因子体检亦用。对应 v3.1 第十三章。

RankIC、分块不重叠 RankIC(块长 H,避重叠虚高)、ICIR、扣费净值、MaxDD、Calmar。
PBO/CSCV、DSR 在 Phase 3 审批落地(本文件留接口,见 model/approval.py 调用前会实现)。
价格层:指标基于收益序列,与 raw/adj 无关(收益已由各自正确价层算好)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    date_col: str = "trade_date",
    min_names: int = 3,
) -> pd.Series:
    """逐交易日的截面 RankIC(score 与 label 的秩相关 = 秩的 Pearson 相关)。返回按日 IC 序列。"""

    def _ic(g: pd.DataFrame) -> float:
        s, y = g[score_col], g[label_col]
        m = s.notna() & y.notna()
        if int(m.sum()) < min_names:
            return np.nan
        return float(s[m].rank().corr(y[m].rank()))

    return df.groupby(date_col)[[score_col, label_col]].apply(_ic)


def icir(ic_series: pd.Series) -> float:
    """ICIR = mean(IC)/std(IC)(样本标准差)。IC 全 NaN 或无波动则返回 NaN。"""
    s = ic_series.dropna()
    if len(s) < 2 or s.std(ddof=1) == 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1))


def mean_rank_ic(df, score_col, label_col, **kw) -> float:
    return float(daily_rank_ic(df, score_col, label_col, **kw).mean())


def blocked_rank_ic(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    block_len: int,
    date_col: str = "trade_date",
) -> pd.Series:
    """分块不重叠 RankIC:把 IC 按交易日切成长度 block_len 的不重叠块,每块取首日 IC 作独立样本。

    避免持有期 block_len 的标签在相邻日重叠导致 IC 自相关、ICIR/t 统计虚高(v3.1 第十三章)。
    """
    if block_len < 1:
        raise ValueError("block_len 必须 >= 1")
    ic = daily_rank_ic(df, score_col, label_col, date_col=date_col).sort_index()
    return ic.iloc[::block_len]


def max_drawdown(nav: "pd.Series | np.ndarray") -> float:
    """最大回撤(返回 <=0 的数)。nav 为净值序列。"""
    nav = pd.Series(np.asarray(nav, dtype="float64"))
    if len(nav) == 0:
        return 0.0
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def nav_from_returns(returns: "pd.Series | np.ndarray", start: float = 1.0) -> np.ndarray:
    """由周期收益序列累乘成净值(起点 start)。"""
    r = np.asarray(returns, dtype="float64")
    return start * np.cumprod(1.0 + r)


def annualized_return(nav: "pd.Series | np.ndarray", periods_per_year: int = 252) -> float:
    """由净值首尾按几何年化。"""
    nav = np.asarray(nav, dtype="float64")
    n = len(nav)
    if n < 2 or nav[0] <= 0:
        return float("nan")
    return float((nav[-1] / nav[0]) ** (periods_per_year / (n - 1)) - 1.0)


def calmar(nav: "pd.Series | np.ndarray", periods_per_year: int = 252) -> float:
    """Calmar = 年化收益 / |最大回撤|。最大回撤为 0 时返回 inf(无回撤)。"""
    mdd = max_drawdown(nav)
    ann = annualized_return(nav, periods_per_year)
    if mdd == 0:
        return float("inf") if ann > 0 else float("nan")
    return float(ann / abs(mdd))


def turnover(weights_by_day: pd.DataFrame) -> float:
    """平均单边换手:相邻两日权重变化绝对值之和的一半,跨日取均值。weights_by_day:行=日,列=票。"""
    w = weights_by_day.fillna(0.0)
    diffs = w.diff().abs().sum(axis=1) / 2.0
    return float(diffs.iloc[1:].mean()) if len(w) > 1 else 0.0
