"""组合净化交叉验证(CPCV)换届裁决。批5。对应 v3.1 第九/十三章(López de Prado)。

单次盲测只给一条样本外路径,无法区分本事与运气;CPCV 产生多条样本外路径,给出绩效"分布"而非单点,
使虚假发现概率可忽略。本模块把已有零件串成 CPCV:把时间切成 S 个块,用 pbo_cscv 做组合 IS/OOS 划分
算 PBO,用 deflated_sharpe_ratio 算 DSR(N=候选/试验数),据此做换届裁决。

**纪律(绝不动)**:CPCV 是新增的并行验证路径,不删除/弱化 INV-6 盲测段一次性;换届仍由"绩效是否
真的更好"决定(挑战者 PBO<阈值 且 DSR>阈值 且 样本外分布优于冠军才换),绝不到期强制换届,
不放松 PBO<0.30 / DSR>0.95。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_block_perf(
    panel: pd.DataFrame, score_cols: "list[str]", fwd_col: str,
    *, n_blocks: int = 8, top_frac: float = 0.2,
) -> np.ndarray:
    """把时间切成 n_blocks 个不重叠块,算每个候选(score_col)在每块的样本外绩效(top_frac 多头日均前瞻收益)。

    返回 (n_blocks, n_candidates) 矩阵,供 pbo_cscv / DSR 使用。fwd_col 为每行的前瞻收益(回测实现收益)。
    """
    p = panel.dropna(subset=[fwd_col]).sort_values("trade_date")
    dates = np.sort(p["trade_date"].unique())
    blocks = np.array_split(dates, n_blocks)
    M = np.full((n_blocks, len(score_cols)), np.nan)
    for bi, bdates in enumerate(blocks):
        sub = p[p["trade_date"].isin(set(pd.Series(bdates).tolist()))]
        for ci, sc in enumerate(score_cols):
            day_perf = []
            for _, g in sub.dropna(subset=[sc]).groupby("trade_date"):
                k = max(1, int(len(g) * top_frac))
                day_perf.append(float(g.nlargest(k, sc)[fwd_col].mean()))
            M[bi, ci] = float(np.mean(day_perf)) if day_perf else np.nan
    return M


def cpcv_evaluate(block_perf: "np.ndarray") -> dict:
    """对块绩效矩阵做 CPCV 评估:PBO(组合 IS/OOS)+ 各试验 OOS 夏普 + 最优试验 DSR。"""
    from trading_system.backtest.metrics import deflated_sharpe_ratio, pbo_cscv

    M = np.asarray(block_perf, dtype="float64")
    s, n = M.shape
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1) if s > 1 else np.ones(n)
    sharpe = np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
    var_tr = float(np.var(sharpe, ddof=1)) if n > 1 else 0.0
    best = int(np.argmax(mu))
    return {
        "pbo": pbo_cscv(M),
        "per_trial_mean": mu,
        "per_trial_sharpe": sharpe,
        "best_trial": best,
        "dsr_best": deflated_sharpe_ratio(float(sharpe[best]), n_obs=s, n_trials=n,
                                          var_sharpe_trials=var_tr),
    }


def cpcv_switch_decision(
    block_perf: "np.ndarray", *, challenger_idx: int = 0, champion_idx: int = 1,
    pbo_max: float = 0.30, dsr_min: float = 0.95,
) -> dict:
    """CPCV 换届裁决:挑战者须 PBO<pbo_max 且(挑战者)DSR>dsr_min 且 样本外均值 ≥ 冠军,才换届。

    block_perf 列含挑战者(challenger_idx)与冠军(champion_idx)及可选其它试验。绝不放松门槛。
    """
    from trading_system.backtest.metrics import deflated_sharpe_ratio

    M = np.asarray(block_perf, dtype="float64")
    s, n = M.shape
    ev = cpcv_evaluate(M)
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1) if s > 1 else np.ones(n)
    sharpe = np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)
    var_tr = float(np.var(sharpe, ddof=1)) if n > 1 else 0.0
    dsr_chal = deflated_sharpe_ratio(float(sharpe[challenger_idx]), n_obs=s, n_trials=n,
                                     var_sharpe_trials=var_tr)
    chal_mean, champ_mean = float(mu[challenger_idx]), float(mu[champion_idx])
    switch = (ev["pbo"] < pbo_max) and (dsr_chal > dsr_min) and (chal_mean >= champ_mean)
    return {
        "switch": bool(switch), "pbo": ev["pbo"], "dsr_challenger": dsr_chal,
        "challenger_mean": chal_mean, "champion_mean": champ_mean,
        "method": "cpcv",
    }
