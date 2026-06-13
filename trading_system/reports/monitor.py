"""监控(核心层 + 增强层,落盘不起服务)。Phase 4(任务 4.3)。对应 v3.1 第十三章。

核心必做:分块不重叠 RankIC + CUSUM、扣费净值与 MaxDD/Calmar、成交失败率、执行差距。
增强层(本版已实现):特征 PSI、Page-Hinkley 在线漂移、与拥挤代理相关性、HMM 状态概率、
HiLo 高低切、HCOPE 否决下界——按是否提供对应输入计算,未提供则标注"未提供"。
输出:落盘 PNG(matplotlib Agg,不起服务)+ Markdown。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.backtest import metrics


# ── 增强层指标 ───────────────────────────────────────────────────────────────
def psi(expected, actual, *, bins: int = 10) -> float:
    """Population Stability Index:特征分布漂移。<0.1 稳 / 0.1–0.2 关注 / >0.2 重训信号。"""
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) < bins or len(a) < 1:
        return float("nan")
    edges = np.quantile(e, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(e, edges)[0] / len(e), 1e-6, None)
    a_pct = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def page_hinkley(series, *, delta: float = 0.0, threshold: float = 0.05) -> dict:
    """Page-Hinkley 在线漂移检测(检测均值**下行**,如 RankIC 衰减)。返回是否报警与统计量。"""
    x = pd.Series(np.asarray(series, dtype="float64")).dropna().to_numpy()
    mean = 0.0
    m = 0.0
    m_max = 0.0
    ph = 0.0
    for i, xi in enumerate(x):
        mean += (xi - mean) / (i + 1)
        m += xi - mean + delta
        m_max = max(m_max, m)
        ph = max(ph, m_max - m)
    return {"ph": float(ph), "alarm": bool(ph > threshold)}


def crowding_correlation(strategy_series, proxy_series) -> float:
    """策略暴露与公开量化拥挤代理的相关性(高相关=拥挤共振风险)。"""
    a = pd.Series(np.asarray(strategy_series, dtype="float64"))
    b = pd.Series(np.asarray(proxy_series, dtype="float64"))
    if a.std(ddof=1) == 0 or b.std(ddof=1) == 0 or len(a) < 2:
        return float("nan")
    return float(a.corr(b))


def run_monitor(
    nav,
    daily_ic: pd.Series,
    *,
    out_dir,
    block_len: int = 10,
    fill_failure_rate: "float | None" = None,
    execution_gap_bp: "float | None" = None,
    tag: str = "monitor",
    cusum_threshold: "float | None" = None,
    # 增强层可选输入
    feature_expected=None,
    feature_actual=None,
    market_returns=None,
    hilo_series=None,
    ope_values=None,
    crowding_strategy=None,
    crowding_proxy=None,
) -> dict:
    """生成核心 + 增强监控面板:计算指标、落盘净值 PNG + Markdown 报告。返回指标 dict。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nav_arr = np.asarray(nav, dtype="float64")

    mdd = metrics.max_drawdown(nav_arr)
    cal = metrics.calmar(nav_arr)
    mean_ic = float(daily_ic.dropna().mean()) if len(daily_ic) else float("nan")
    blocked = daily_ic.dropna().iloc[::block_len]
    blocked_ic = float(blocked.mean()) if len(blocked) else float("nan")
    icir = metrics.icir(daily_ic)
    cusum = metrics.cusum(daily_ic, threshold=cusum_threshold)

    result = {
        "max_drawdown": mdd, "calmar": cal, "mean_rank_ic": mean_ic,
        "blocked_rank_ic": blocked_ic, "icir": icir,
        "cusum_max_abs": cusum["max_abs"], "cusum_breach": cusum["breach"],
        "fill_failure_rate": fill_failure_rate, "execution_gap_bp": execution_gap_bp,
    }

    # ── 增强层(按提供的输入计算)──
    enh: dict = {}
    if feature_expected is not None and feature_actual is not None:
        enh["psi"] = psi(feature_expected, feature_actual)
    if daily_ic is not None and len(daily_ic):
        enh["ic_page_hinkley"] = page_hinkley(daily_ic.dropna().to_numpy())
    if market_returns is not None:
        from trading_system.regime.hmm import compute_regime_state_probs

        probs, order = compute_regime_state_probs(np.asarray(market_returns, dtype="float64"))
        enh["hmm_bear_prob_last"] = float(probs[-1, order[0]])  # 最低均值=熊态
    if hilo_series is not None:
        enh["hilo_last"] = float(pd.Series(hilo_series).dropna().iloc[-1])
    if ope_values is not None:
        from trading_system.audit import hcope_lower_bound

        enh["hcope_lower_bound"] = hcope_lower_bound(np.asarray(ope_values, dtype="float64"))
    if crowding_strategy is not None and crowding_proxy is not None:
        enh["crowding_correlation"] = crowding_correlation(crowding_strategy, crowding_proxy)
    result["enhanced"] = enh

    # ── 落盘净值 PNG(Agg,纯文件)──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nav_arr)
    ax.set_title(f"{tag} net value (MaxDD={mdd:.2%}, Calmar={cal:.2f})")
    ax.set_xlabel("trading day index"); ax.set_ylabel("NAV")
    png_path = out_dir / f"{tag}_nav.png"
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    def _fmt(k):
        return enh.get(k, "未提供")

    md = [
        f"# 监控面板 {tag}", "",
        "## 核心指标",
        f"- 最大回撤 MaxDD: {mdd:.2%};Calmar: {cal:.2f}",
        f"- 平均 RankIC: {mean_ic:.4f};分块不重叠 RankIC(H={block_len}): {blocked_ic:.4f};ICIR: {icir:.3f}",
        f"- RankIC CUSUM 最大绝对偏移: {cusum['max_abs']:.4f}(越限={cusum['breach']})",
        f"- 成交失败率: {fill_failure_rate};执行差距(bp): {execution_gap_bp}",
        "",
        "## 增强层",
        f"- 特征 PSI: {_fmt('psi')}",
        f"- RankIC Page-Hinkley 漂移: {_fmt('ic_page_hinkley')}",
        f"- HMM 熊态概率(最新): {_fmt('hmm_bear_prob_last')}",
        f"- HiLo 高低切(最新): {_fmt('hilo_last')}",
        f"- HCOPE 否决价值下界: {_fmt('hcope_lower_bound')}",
        f"- 与拥挤代理相关性: {_fmt('crowding_correlation')}",
        "",
        f"![nav]({png_path.name})",
    ]
    md_path = out_dir / f"{tag}_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    result["png_path"] = str(png_path)
    result["md_path"] = str(md_path)
    return result
