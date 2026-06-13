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

    if df.empty:
        return pd.Series(dtype="float64")

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
    s = daily_rank_ic(df, score_col, label_col, **kw)
    return float(s.mean()) if len(s) else float("nan")


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


# ── PBO / DSR(Phase 3 审批用;对应 v3.1 第十三章,López de Prado / Bailey)──────
_EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: "pd.Series | np.ndarray") -> float:
    """每期(非年化)夏普 = mean/std(ddof=1)。"""
    r = pd.Series(np.asarray(returns, dtype="float64")).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1))


def probabilistic_sharpe_ratio(
    sharpe: float, n_obs: int, *, sr_benchmark: float = 0.0, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """PSR:观测夏普 > 基准夏普的概率(Bailey & López de Prado 2014)。每期夏普口径。"""
    from scipy.stats import norm

    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(max(1e-12, 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2))
    z = (sharpe - sr_benchmark) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(var_sharpe_trials: float, n_trials: int) -> float:
    """N 次试验下零假设的期望最大夏普 SR0(用于 DSR 去膨胀)。"""
    from scipy.stats import norm

    if n_trials < 2 or var_sharpe_trials <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sharpe_trials) * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b))


def deflated_sharpe_ratio(
    sharpe: float, n_obs: int, *, n_trials: int, var_sharpe_trials: float,
    skew: float = 0.0, kurtosis: float = 3.0,
) -> float:
    """DSR:以 N 次试验的期望最大夏普 SR0 为基准的 PSR(去膨胀)。返回 [0,1]。"""
    sr0 = expected_max_sharpe(var_sharpe_trials, n_trials)
    return probabilistic_sharpe_ratio(
        sharpe, n_obs, sr_benchmark=sr0, skew=skew, kurtosis=kurtosis
    )


def cusum(series: "pd.Series | np.ndarray", *, threshold: "float | None" = None) -> dict:
    """RankIC 的 CUSUM 漂移监控(v3.1 §13 核心)。返回去均值累积和、最大绝对偏移与是否越限。

    s_t = Σ_{i<=t}(x_i - mean(x));|s| 越大表示均值持续偏离(如 RankIC 系统性转负)。
    """
    s = pd.Series(np.asarray(series, dtype="float64")).dropna()
    if len(s) == 0:
        return {"cumsum": np.array([]), "max_abs": float("nan"), "breach": None}
    cs = (s - s.mean()).cumsum().to_numpy()
    max_abs = float(np.abs(cs).max())
    return {"cumsum": cs, "max_abs": max_abs,
            "breach": (bool(max_abs > threshold) if threshold is not None else None)}


def pbo_cscv(block_perf: "np.ndarray") -> float:
    """PBO(组合对称交叉验证)。block_perf: 形状 (S_blocks, N_trials) 的分块绩效(S 为偶数)。

    对所有 C(S, S/2) 种 IS/OOS 划分:IS 选样本内均值最大的 trial,看其在 OOS 的相对秩;
    若 OOS 秩低于中位(logit<=0)记一次过拟合。PBO = 过拟合次数 / 总划分数(López de Prado 2015)。
    """
    from itertools import combinations

    M = np.asarray(block_perf, dtype="float64")
    s, n = M.shape
    if s < 2 or s % 2 != 0:
        raise ValueError("block_perf 需 (S_blocks, N_trials),S 为偶数")
    overfit = total = 0
    all_blocks = set(range(s))
    for is_blocks in combinations(range(s), s // 2):
        oos = sorted(all_blocks - set(is_blocks))
        is_mean = M[list(is_blocks)].mean(axis=0)
        oos_mean = M[oos].mean(axis=0)
        best = int(np.argmax(is_mean))
        # best 在 OOS 的相对秩 w∈(0,1);w<=0.5 -> OOS 表现低于中位 -> 过拟合
        rank = (oos_mean <= oos_mean[best]).sum()  # 1..n(含自身)
        w = rank / (n + 1)
        if w <= 0.5:
            overfit += 1
        total += 1
    return overfit / total if total else float("nan")
