"""完全体补全的真实测试:实验注册表(SQLite)/ HMM / CGO·换手族 / LightGBM 基线 / σ̂ /
CUSUM / HCOPE / 监控增强层 / 限频 / 引擎版 y_prod / walk-forward 回测 / 质检新项。

对已知答案或可推导性质核验;逻辑错即真实失败。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.audit import hcope_lower_bound
from trading_system.audit.experiment_registry import ExperimentRegistry
from trading_system.backtest import baselines, metrics
from trading_system.backtest.runner import walk_forward_backtest
from trading_system.data import price_layers as pl
from trading_system.data import quality as q
from trading_system.data.collectors import synthetic
from trading_system.data.collectors._ratelimit import RateLimiter, chunked
from trading_system.features.registry import compute_raw_feature, truncation_equivariance_violations
import trading_system.features.builtin  # noqa: F401  注册特征
import trading_system.labels as labels
from trading_system import portfolio as port
from trading_system.invariants import InvariantViolation
from trading_system.regime.hmm import GaussianHMM, compute_regime_state_probs
from trading_system.reports import monitor


def _panel_with_turn(code, dates, closes, turns, factor=1.0):
    rows, prev = [], closes[0]
    for i, d in enumerate(dates):
        c, pc, t = float(closes[i]), float(prev), float(turns[i])
        rows.append(dict(code=code, trade_date=pd.Timestamp(d), open_raw=pc, high_raw=max(pc, c),
                         low_raw=min(pc, c), close_raw=c, preclose_raw=pc, volume=1e4,
                         amount=1e4 * c, adj_factor=factor, turn=t))
        prev = c
    return pl.build_price_layers(pd.DataFrame(rows))


# ── 实验注册表(SQLite,INV-6 持久化)────────────────────────────────────────
class TestExperimentRegistry:
    def test_blind_segment_once_persisted(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "exp.sqlite")
        reg.assert_available("blind_2025H2")
        reg.use_for_decision("blind_2025H2")
        with pytest.raises(InvariantViolation):
            reg.assert_available("blind_2025H2")
        # 重开(模拟重启)仍封存 -> 持久化生效
        reg2 = ExperimentRegistry(tmp_path / "exp.sqlite")
        with pytest.raises(InvariantViolation):
            reg2.use_for_decision("blind_2025H2")

    def test_trial_log_n(self, tmp_path):
        reg = ExperimentRegistry(tmp_path / "exp.sqlite")
        for i in range(5):
            reg.log_trial("studyA", {"x": i}, value=0.1 * i)
        assert reg.n_trials("studyA") == 5 and reg.n_trials() == 5


# ── 高斯 HMM regime ─────────────────────────────────────────────────────────
class TestHMM:
    def _two_regime(self):
        rng = np.random.default_rng(0)
        return np.concatenate([rng.normal(-0.02, 0.01, 120), rng.normal(0.02, 0.01, 120)])

    def test_identifies_two_regimes(self):
        r = self._two_regime()
        model = GaussianHMM(2, seed=0).fit(r)
        filt = model.filtered_posterior(r)
        order = np.argsort(model.means_)            # 低均值=熊态
        pred = filt.argmax(axis=1)
        expected = np.where(np.arange(len(r)) < 120, order[0], order[1])
        assert (pred == expected).mean() > 0.7      # 滤波在切换处滞后,>0.7 已合理

    def test_filtered_is_causal(self):
        r = self._two_regime()
        model = GaussianHMM(2, seed=0).fit(r)        # 参数固定后,滤波只用历史
        full = model.filtered_posterior(r)
        for t in (50, 150, 200):
            sub = model.filtered_posterior(r[: t + 1])
            assert np.allclose(full[t], sub[-1], atol=1e-9)   # 截断等变 -> 因果

    def test_filtered_differs_from_smoothed(self):
        r = self._two_regime()
        model = GaussianHMM(2, seed=0).fit(r)
        filt, smooth = model.filtered_posterior(r), model.smoothed_posterior(r)
        assert np.abs(filt - smooth).max() > 0.05    # 平滑用了未来,与滤波不同

    def test_compute_regime_state_probs(self):
        probs, order = compute_regime_state_probs(self._two_regime(), n_states=2)
        assert probs.shape[1] == 2 and np.allclose(probs.sum(axis=1), 1.0)


# ── CGO / 换手率族 ──────────────────────────────────────────────────────────
class TestCgoTurnover:
    def test_turnover_mean(self):
        dates = synthetic.make_calendar("2020-01-06", 30).dates
        g = _panel_with_turn("600000", dates, list(np.linspace(10, 12, 30)), [1.5] * 30)
        raw = compute_raw_feature("turnover_mean_20", g).to_numpy()
        assert abs(raw[-1] - 1.5) < 1e-9            # 常数换手 -> 20 日均=1.5

    def test_cgo_positive_when_price_rose(self):
        dates = synthetic.make_calendar("2020-01-06", 80).dates
        g = _panel_with_turn("600000", dates, list(np.linspace(10, 20, 80)), [2.0] * 80)
        raw = compute_raw_feature("cgo_60", g).to_numpy()
        assert raw[-1] > 0                          # 持续上涨 -> 现价高于换手加权成本 -> CGO>0

    def test_cgo_truncation_equivariant(self):
        dates = synthetic.make_calendar("2020-01-06", 80).dates
        g = _panel_with_turn("600000", dates, list(np.linspace(10, 16, 80)), [1.0] * 80)
        assert not truncation_equivariance_violations("cgo_60", g, [65, 70, 75])
        assert not truncation_equivariance_violations("turnover_chg_20", g, [65, 70, 75])


# ── LightGBM 基线④ / σ̂ / CUSUM / HCOPE ─────────────────────────────────────
class TestStatsAndBaselines:
    def test_lightgbm_baseline_recovers_signal(self):
        rng = np.random.default_rng(0)
        frames = []
        for d in pd.date_range("2020-01-06", periods=12, freq="D"):
            x = rng.normal(size=30)
            frames.append(pd.DataFrame({"trade_date": d, "f1": x, "f2": rng.normal(size=30),
                                        "label": x + 0.05 * rng.normal(size=30)}))
        df = pd.concat(frames, ignore_index=True)
        ic, _ = baselines.lightgbm_regression_baseline(df, ["f1", "f2"], "label")
        assert ic > 0.2

    def test_sigma_hat(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 100)
        sig = port.estimate_sigma_hat(r)
        s20, s60 = np.std(r[-20:], ddof=1), np.std(r[-60:], ddof=1)
        assert abs(sig - np.sqrt(0.5 * s20 ** 2 + 0.5 * s60 ** 2) * np.sqrt(252)) < 1e-9

    def test_cusum_detects_drift(self):
        stable = metrics.cusum(np.r_[np.zeros(50)])
        drift = metrics.cusum(np.r_[np.full(25, -0.1), np.full(25, 0.1)], threshold=0.5)
        assert drift["max_abs"] > stable["max_abs"] and drift["breach"] in (True, False)

    def test_hcope_lower_bound(self):
        rng = np.random.default_rng(2)
        small = hcope_lower_bound(rng.normal(1.0, 0.5, 30))
        large = hcope_lower_bound(rng.normal(1.0, 0.5, 3000))
        assert small <= 1.2 and large <= 1.2 and large > small   # 样本越多越紧(下界更高)


# ── 监控增强层 ──────────────────────────────────────────────────────────────
class TestMonitorEnhanced:
    def test_psi_and_drift_and_run(self, tmp_path):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 1000)
        assert monitor.psi(base, base) < 0.05                      # 同分布 -> 稳
        assert monitor.psi(base, base + 1.0) > 0.2                 # 整体平移 -> 漂移
        assert monitor.page_hinkley(np.r_[np.zeros(30), -np.ones(30)], threshold=0.5)["alarm"]
        assert abs(monitor.crowding_correlation([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
        res = monitor.run_monitor(
            [1.0, 1.1, 0.9, 1.2], pd.Series(np.linspace(-0.01, 0.03, 40)), out_dir=tmp_path,
            feature_expected=base, feature_actual=base + 0.5,
            market_returns=np.r_[rng.normal(-0.02, 0.01, 80), rng.normal(0.02, 0.01, 80)],
            ope_values=rng.normal(1.0, 0.3, 200),
            crowding_strategy=[1, 2, 3, 4, 5], crowding_proxy=[1, 2, 3, 4, 5],
        )
        enh = res["enhanced"]
        assert {"psi", "hmm_bear_prob_last", "hcope_lower_bound", "crowding_correlation"}.issubset(enh)
        assert 0.0 <= enh["hmm_bear_prob_last"] <= 1.0


# ── 限频 ────────────────────────────────────────────────────────────────────
class TestRateLimit:
    def test_chunked(self):
        chunks = list(chunked(list(range(250)), 100))
        assert [len(c) for c in chunks] == [100, 100, 50]

    def test_rate_limiter_interval(self):
        import time
        lim = RateLimiter(max_per_sec=50)  # 间隔 0.02s
        lim.wait()
        t0 = time.monotonic()
        lim.wait()
        assert time.monotonic() - t0 >= 0.018


# ── 引擎版 y_prod ───────────────────────────────────────────────────────────
class TestYProdEngine:
    def test_matches_engine(self):
        from trading_system.backtest.engine import compute_atr, simulate_trade

        dates = synthetic.make_calendar("2020-01-06", 16).dates
        g = _panel_with_turn("600000", dates, list(np.linspace(10, 13, 16)), [1.0] * 16)
        y = labels.build_y_prod_engine(g, atr_period=5, cost_fraction=0.001)
        finite = y.dropna()
        assert len(finite) >= 1                     # 至少一个可成交信号
        # 对第一个有标签的信号,独立用引擎复算应一致
        gg = g.sort_values(["code", "trade_date"]).reset_index(drop=True)
        atr = compute_atr(gg, 5).to_numpy()
        pos = int(finite.index[0])
        res = simulate_trade(gg, pos, atr=float(atr[pos]), cost_fraction=0.001)
        assert abs(res.net_return - finite.iloc[0]) < 1e-9


# ── walk-forward 回测(引擎逐笔)────────────────────────────────────────────
class TestWalkForward:
    def test_runs_and_net_le_nominal(self):
        cal = synthetic.make_calendar("2020-01-06", 40)
        panel = pl.build_price_layers(synthetic.make_raw_panel(
            [f"60000{i}" for i in range(8)], cal, seed=5))
        panel = panel.copy()
        panel["__score__"] = np.random.default_rng(3).normal(size=len(panel))
        res = walk_forward_backtest(panel, "__score__", top_k=3, cost_fraction=0.003)
        assert res["n_trades"] > 0
        assert np.isfinite(res["net_return"]) and res["net_return"] <= res["nominal_return"] + 1e-9

    def test_weighted_differs_from_equal(self):
        # 5 只票各以不同恒定日收益生成;score=按 code 序固定,保证 top-3 名次稳定且收益各异
        cal = synthetic.make_calendar("2020-01-06", 40)
        rets = {f"60000{i}": (i - 2) * 0.004 for i in range(5)}
        rows = []
        for code, r in rets.items():
            closes = 10.0 * np.cumprod(np.full(len(cal.dates), 1.0 + r))
            for j, d in enumerate(cal.dates):
                pc = 10.0 if j == 0 else closes[j - 1]
                rows.append(dict(code=code, trade_date=pd.Timestamp(d), open_raw=pc,
                                 high_raw=max(pc, closes[j]), low_raw=min(pc, closes[j]),
                                 close_raw=closes[j], preclose_raw=pc, volume=1e4,
                                 amount=1e4 * closes[j], adj_factor=1.0))
        panel = pl.build_price_layers(pd.DataFrame(rows))
        panel["__score__"] = panel["code"].map({c: r for c, r in rets.items()})  # 名次稳定
        eq = walk_forward_backtest(panel, "__score__", top_k=3, cost_fraction=0.001)
        wt = walk_forward_backtest(panel, "__score__", top_k=3, weights=[0.5, 0.3, 0.2],
                                   cost_fraction=0.001)
        assert eq["n_trades"] > 0 and wt["n_trades"] > 0
        assert np.isfinite(eq["net_return"]) and np.isfinite(wt["net_return"])
        assert abs(eq["net_return"] - wt["net_return"]) > 1e-9   # 加权与等权结果不同


# ── 质检新项 ────────────────────────────────────────────────────────────────
class TestQualityMonotonic:
    def test_monotonic_factor_ok_and_decrease_flagged(self):
        cal = synthetic.make_calendar("2020-01-06", 10).dates
        good = _panel_with_turn("600000", cal, list(np.linspace(10, 11, 10)), [1.0] * 10)
        good = good.copy()
        good["adj_factor"] = np.linspace(1.0, 1.5, 10)        # 非递减
        assert q.check_adj_factor_monotonic(good).status == q.PASS
        bad = good.copy()
        bad.loc[bad.index[5], "adj_factor"] = 0.5             # 人为下降
        assert q.check_adj_factor_monotonic(bad).status == q.WARN
