"""walk-forward 回测运行器(完全体:用事件级引擎逐笔成交)。Phase 2/3。对应 v3.1 第十一/十三章。

区别于 run_backtest 中的轻量 naive_topk:本运行器用**唯一真值引擎** simulate_trade 逐笔模拟选中标的的
T+1 成交、硬止损/止盈阶梯/跟踪/时间止损、涨跌停顺延与扣费,保证回测口径与实盘一致。
每个交易日按 score_col 选 top-K,逐票走引擎;按信号日聚合等权净收益成净值。价格层:引擎按 raw 记账(INV-2)。
可选 trigger_col(布尔)作为 L1 候选过滤(只在触发为真处建仓)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.backtest.engine import compute_atr, simulate_trade


def walk_forward_backtest(
    panel: pd.DataFrame,
    score_col: str,
    *,
    top_k: int = 20,
    atr_period: int = 14,
    cost_fraction: float = 0.0,
    trigger_col: "str | None" = None,
    atr_mult: float = 2.5,
    trail_c: float = 2.5,
    max_holding: int = 10,
    weights: "list | None" = None,
    position_mode: str = "equal",
    total_exposure_cap: float = 0.5,
    single_cap: float = 0.08,
) -> dict:
    """逐日 top-K + 引擎逐笔回测。返回 trades / 信号日净收益 / 名义与扣费收益 / 净值。

    position_mode="equal"(默认,行为不变):前 N 名等权(或按 weights 名次加权,按已成交归一)。
    position_mode="risk":portfolio 风控仓位——逆 ATR 反波动相对权重 × 总敞口硬顶,逐票被单股上限
    截断,不补杠杆(组合总权重 ≤ total_exposure_cap;未投部分计 0 收益)。簇限制/动态 regime 待题材数据接入。
    价格层:ATR 用 raw(INV-2)。
    """
    p = panel.sort_values(["code", "trade_date"]).copy()
    by_code = {code: g.reset_index(drop=True) for code, g in p.groupby("code", sort=False)}
    atr_by_code = {code: compute_atr(g, atr_period) for code, g in by_code.items()}

    day_rows: dict = {}  # date -> [(rank_i, net, gross), ...]
    trades = []
    for date, day in p.groupby("trade_date", sort=True):
        cand = day.dropna(subset=[score_col])
        if trigger_col is not None:
            cand = cand[cand[trigger_col].astype(bool)]
        if cand.empty:
            continue
        top = cand.nlargest(min(top_k, len(cand)), score_col)
        for rank_i, code in enumerate(top["code"]):
            g = by_code[code]
            pos_idx = g.index[g["trade_date"] == date]
            if len(pos_idx) == 0:
                continue
            pos = int(pos_idx[0])
            atr = atr_by_code[code].iloc[pos]
            if not np.isfinite(atr) or atr <= 0:
                continue
            res = simulate_trade(g, pos, atr=float(atr), atr_mult=atr_mult, trail_c=trail_c,
                                 max_holding=max_holding, cost_fraction=cost_fraction)
            if res.status == "closed":
                trades.append(res)
                day_rows.setdefault(date, []).append(
                    (rank_i, res.net_return, res.gross_return, float(atr)))

    def _agg(rows, idx):
        nets = np.array([r[idx] for r in rows], dtype="float64")
        if position_mode == "risk":
            from trading_system.portfolio import inverse_atr_weights
            atrs = np.array([r[3] for r in rows], dtype="float64")
            rel = inverse_atr_weights(atrs)                       # 逆 ATR 反波动,和为 1
            w = np.minimum(rel * total_exposure_cap, single_cap)  # ×总敞口,逐票截单股上限,不补杠杆
            return float((w * nets).sum())                        # 实际敞口加权(未投部分计 0)
        if weights is not None:
            w = np.array([weights[i] if i < len(weights) else 0.0 for i, *_ in rows], dtype="float64")
        else:
            w = np.ones(len(rows), dtype="float64")
        wsum = w.sum()
        return float((w * nets).sum() / wsum) if wsum > 0 else float("nan")

    dates = [d for d in sorted(day_rows) if np.isfinite(_agg(day_rows[d], 1))]
    net_daily = np.array([_agg(day_rows[d], 1) for d in dates], dtype="float64")
    gross_daily = np.array([_agg(day_rows[d], 2) for d in dates], dtype="float64")
    net_nav = np.cumprod(1.0 + net_daily) if len(net_daily) else np.array([])
    return {
        "n_trades": len(trades),
        "signal_dates": dates,
        "net_daily": net_daily,
        "gross_daily": gross_daily,
        "net_nav": net_nav,
        "nominal_return": float(np.prod(1.0 + gross_daily) - 1.0) if len(gross_daily) else float("nan"),
        "net_return": float(np.prod(1.0 + net_daily) - 1.0) if len(net_daily) else float("nan"),
    }
