"""指标真实单元测试(RankIC / ICIR / 分块 / MaxDD / Calmar)。对已知答案核验。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.backtest import metrics


class TestRankIC:
    def test_perfect_and_anti_correlation(self):
        day = pd.Timestamp("2020-01-06")
        df = pd.DataFrame({
            "trade_date": [day] * 5,
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label_pos": [10.0, 20.0, 30.0, 40.0, 50.0],   # 完全同序
            "label_neg": [50.0, 40.0, 30.0, 20.0, 10.0],   # 完全反序
        })
        assert abs(metrics.mean_rank_ic(df, "score", "label_pos") - 1.0) < 1e-9
        assert abs(metrics.mean_rank_ic(df, "score", "label_neg") + 1.0) < 1e-9

    def test_icir(self):
        ic = pd.Series([0.1, 0.1, 0.1, 0.1])  # 无波动 -> NaN
        assert np.isnan(metrics.icir(ic))
        ic2 = pd.Series([0.0, 0.2])  # mean 0.1, std(ddof1)=0.1414 -> icir≈0.707
        assert abs(metrics.icir(ic2) - (0.1 / pd.Series([0.0, 0.2]).std(ddof=1))) < 1e-9

    def test_empty_or_all_nan_returns_nan(self):
        # 全 NaN 特征(如缺 turn 的 CGO)-> 空截面 -> RankIC 应为 NaN 而非报错
        empty = pd.DataFrame({"trade_date": [], "score": [], "label": []})
        assert len(metrics.daily_rank_ic(empty, "score", "label")) == 0
        assert np.isnan(metrics.mean_rank_ic(empty, "score", "label"))

    def test_blocked_subsamples(self):
        days = pd.date_range("2020-01-06", periods=10, freq="D")
        df = pd.concat([
            pd.DataFrame({"trade_date": [d] * 4, "score": [1, 2, 3, 4],
                          "label": [1, 2, 3, 4]}) for d in days
        ], ignore_index=True)
        blocked = metrics.blocked_rank_ic(df, "score", "label", block_len=5)
        assert len(blocked) == 2  # 10 天 / 块长 5 -> 2 个不重叠样本


class TestDrawdownCalmar:
    def test_max_drawdown(self):
        nav = [1.0, 1.2, 0.9, 1.0, 1.5]
        # 最深回撤:从 1.2 跌到 0.9 -> (0.9-1.2)/1.2 = -0.25
        assert abs(metrics.max_drawdown(nav) - (-0.25)) < 1e-12

    def test_no_drawdown(self):
        assert metrics.max_drawdown([1.0, 1.1, 1.2]) == 0.0

    def test_calmar_sign(self):
        nav = metrics.nav_from_returns([0.01] * 252)  # 稳定上涨,无回撤
        assert metrics.calmar(nav) == float("inf")
        nav2 = [1.0, 1.2, 0.9, 1.1]
        c = metrics.calmar(nav2, periods_per_year=252)
        assert np.isfinite(c)  # 有回撤 -> 有限值

    def test_nav_from_returns(self):
        nav = metrics.nav_from_returns([0.1, -0.1])
        assert abs(nav[0] - 1.1) < 1e-12 and abs(nav[1] - 1.1 * 0.9) < 1e-12
