"""Phase 2 扩展真实测试:滑点压力 / 四基线 / overlay test。对已知答案核验。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest import baselines as bl
from trading_system.backtest import stress as st
from trading_system.backtest.costs import CostModel
from trading_system.data import price_layers as pl
from trading_system.data.collectors import synthetic
import trading_system.overlays as ov


def _r(date, o, c, pc, h=None, l=None):
    h = max(o, c) if h is None else h
    l = min(o, c) if l is None else l
    return dict(code="600000", trade_date=pd.Timestamp(date), open_raw=o, high_raw=h, low_raw=l,
               close_raw=c, preclose_raw=pc, volume=10000.0, amount=10000.0 * c, adj_factor=1.0)


class TestSlippageStress:
    def test_net_decreases_with_slippage(self):
        d = synthetic.make_calendar("2020-01-06", 7).dates
        rows = [
            _r(d[0], 10.0, 10.0, 10.0), _r(d[1], 10.0, 10.5, 10.0), _r(d[2], 10.6, 11.0, 10.5),
            _r(d[3], 11.2, 12.0, 11.0), _r(d[4], 12.1, 12.5, 12.0), _r(d[5], 12.0, 11.4, 12.5),
            _r(d[6], 11.3, 11.0, 11.4),
        ]
        bars = pl.build_price_layers(pd.DataFrame(rows))
        trades = [dict(bars=bars, signal_idx=0, atr=0.4, notional=100000.0)]
        df = st.run_slippage_stress(trades, CostModel(), slippage_grid_bp=[5, 10, 20, 30])
        net = dict(zip(df["slippage_bp"], df["mean_net"]))
        assert net[5] > net[30]            # 滑点越大净收益越低
        assert st.survives_stress(df, slippage_bp=5)   # 该盈利交易 5bp 下仍为正


class TestBaselines:
    def test_random_candidate_percentile(self):
        cr = np.linspace(-0.1, 0.1, 200)
        dist = bl.random_candidate_baseline(cr, k=10, n_samples=500, seed=1)
        assert bl.strategy_percentile(0.09, dist) > 0.9   # 高收益策略分位高
        assert bl.strategy_percentile(-0.09, dist) < 0.1

    def test_single_factor_picks_best(self):
        day = pd.Timestamp("2020-01-06")
        n = 20
        df = pd.DataFrame({
            "trade_date": [day] * n,
            "good": np.arange(n, dtype="float64"),
            "bad": np.random.default_rng(0).normal(size=n),
            "label": np.arange(n, dtype="float64"),   # 与 good 完全同序
        })
        name, ic = bl.single_factor_baseline(df, ["good", "bad"], "label")
        assert name == "good" and ic > 0.9

    def test_elasticnet_recovers_positive_ic(self):
        rng = np.random.default_rng(0)
        days = pd.date_range("2020-01-06", periods=10, freq="D")
        frames = []
        for dd in days:
            x = rng.normal(size=30)
            frames.append(pd.DataFrame({"trade_date": dd, "f1": x,
                                        "f2": rng.normal(size=30),
                                        "label": x + 0.1 * rng.normal(size=30)}))
        df = pd.concat(frames, ignore_index=True)
        ic, coef = bl.elasticnet_baseline(df, ["f1", "f2"], "label")
        assert ic > 0.2   # f1 预测 label,IC 应明显为正


class TestOverlays:
    def test_disclosure_actions(self):
        panel = pd.DataFrame({
            "days_to_disclosure": [3.0, 3.0, 30.0, np.nan],
            "has_preann": [True, False, False, False],
            "preann_sign": [-1.0, 0.0, 0.0, 0.0],
        })
        act = ov.disclosure_season_overlay(panel, window=10).tolist()
        assert act == [ov.VETO, ov.REDUCE, ov.NONE, ov.NONE]

    def test_hilo_interaction_is_product(self):
        hilo = np.array([0.8, 0.2])
        ext = np.array([0.5, 0.5])
        inter = ov.hilo_overextension_interaction(hilo, ext)
        assert np.allclose(inter, [0.4, 0.1])

    def test_overlay_test_enable_and_reject(self):
        without = [1.0, 1.2, 0.9, 1.1]          # 回撤 25%
        good_with = [1.0, 1.1, 1.05, 1.15]      # 回撤更小、收益更高 -> 启用
        bad_with = [1.0, 1.08, 0.81, 0.99]      # 同回撤、收益更低 -> 弃用
        assert ov.overlay_test(without, good_with).enable is True
        assert ov.overlay_test(without, bad_with).enable is False

    def test_overextension_interaction_test(self):
        assert ov.overextension_interaction_test(0.02, 0.03, -0.2, -0.18) is True   # ΔIC>0,回撤不增
        assert ov.overextension_interaction_test(0.03, 0.02, -0.2, -0.2) is False   # ΔIC<0
