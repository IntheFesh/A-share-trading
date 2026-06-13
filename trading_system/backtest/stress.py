"""滑点压力矩阵。Phase 2(任务 2.4)。对应 v3.1 第四章 / 第十三章。

slippage ∈ {5,10,20,30}bp × 标的桶,逐档用引擎重算扣费净收益。审批门槛:20bp>0;首板另需 30bp>0。
档位从 config/costs.yaml 的 slippage.stress_grid_bp 读。价格层:引擎按 raw 记账(INV-2)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.backtest.costs import CostModel
from trading_system.backtest.engine import simulate_trade


def run_slippage_stress(
    trades: "list[dict]",
    cost_model: CostModel,
    slippage_grid_bp: "list[float]" = (5, 10, 20, 30),
) -> pd.DataFrame:
    """对一组交易在各滑点档下重算净收益。

    trades:每个元素 dict(bars, signal_idx, atr, notional[, bucket])。
    返回每档 (slippage_bp, bucket?) 的成交笔数与平均扣费净收益。
    """
    rows = []
    for s in slippage_grid_bp:
        nets, buckets = [], {}
        for tr in trades:
            cf = cost_model.round_trip_cost_fraction(tr["notional"], slippage_bp=s)
            res = simulate_trade(tr["bars"], tr["signal_idx"], atr=tr["atr"], cost_fraction=cf)
            if res.status == "closed":
                nets.append(res.net_return)
                buckets.setdefault(tr.get("bucket", "all"), []).append(res.net_return)
        rows.append(dict(slippage_bp=s, n=len(nets),
                         mean_net=float(np.mean(nets)) if nets else float("nan")))
    return pd.DataFrame(rows)


def survives_stress(stress_df: pd.DataFrame, *, slippage_bp: float) -> bool:
    """某滑点档下平均净收益是否 > 0(审批门槛 SlippageStress_xbp>0)。"""
    row = stress_df[stress_df["slippage_bp"] == slippage_bp]
    if row.empty:
        raise ValueError(f"压力矩阵无 {slippage_bp}bp 档")
    return bool(row["mean_net"].iloc[0] > 0)
