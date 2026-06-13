"""Phase 3 真实测试:purged/embargo CV / PBO·DSR / LightGBM 训练(INV-4 + 按日 group)/
Optuna(预注册空间)/ 审批门槛(INV-6)。对已知答案与关键性质核验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest import metrics
from trading_system.invariants import FeatureSpec, InvariantViolation
from trading_system.model import approval, cv, train, tune


# ── purged / embargo CV ─────────────────────────────────────────────────────
class TestPurgedCV:
    def test_embargo_formula(self):
        assert cv.embargo_from_config(10, 2) == 13

    def test_walk_forward_has_embargo_gap_and_no_overlap(self):
        dates = list(range(100))
        splits = cv.purged_walk_forward_splits(dates, n_splits=4, embargo=13)
        assert len(splits) == 4
        for train_d, val_d in splits:
            assert set(train_d).isdisjoint(set(val_d))          # 无重叠
            assert max(train_d) < min(val_d)                     # 训练在验证之前
            assert min(val_d) - max(train_d) - 1 == 13           # 间隔恰为 embargo

    def test_row_masks(self):
        rd = np.array([1, 2, 3, 4, 5])
        tm, vm = cv.row_masks(rd, np.array([1, 2]), np.array([4, 5]))
        assert tm.tolist() == [True, True, False, False, False]
        assert vm.tolist() == [False, False, False, True, True]


# ── PBO / DSR ───────────────────────────────────────────────────────────────
class TestPBODSR:
    def test_pbo_dominant_trial_low(self):
        rng = np.random.default_rng(0)
        M = rng.uniform(0, 0.5, size=(8, 10))
        M[:, 0] = 1.0  # trial0 在每个块都最好 -> 不过拟合
        assert metrics.pbo_cscv(M) < 0.3

    def test_pbo_noise_around_half(self):
        rng = np.random.default_rng(1)
        M = rng.normal(size=(8, 12))
        assert 0.2 < metrics.pbo_cscv(M) < 0.8

    def test_psr_increases_with_obs(self):
        assert metrics.probabilistic_sharpe_ratio(0.1, 100) < metrics.probabilistic_sharpe_ratio(0.1, 1000)

    def test_dsr_bounds_and_deflation(self):
        d = metrics.deflated_sharpe_ratio(0.1, 500, n_trials=50, var_sharpe_trials=0.01)
        assert 0.0 <= d <= 1.0
        # n_trials=1,var=0 -> SR0=0 -> DSR == PSR(基准0)
        psr0 = metrics.probabilistic_sharpe_ratio(0.1, 500, sr_benchmark=0.0)
        dsr1 = metrics.deflated_sharpe_ratio(0.1, 500, n_trials=1, var_sharpe_trials=0.0)
        assert abs(dsr1 - psr0) < 1e-12
        # 去膨胀:多试验 + 正方差 -> DSR < PSR(0)
        assert metrics.deflated_sharpe_ratio(0.1, 500, n_trials=100, var_sharpe_trials=0.02) < psr0

    def test_high_sr_many_obs_high_dsr(self):
        assert metrics.deflated_sharpe_ratio(0.2, 500, n_trials=1, var_sharpe_trials=0.0) > 0.95


# ── LightGBM 训练:INV-4 守卫 + 按日 group + 学到信号 ────────────────────────
class TestTrain:
    def _signal_df(self, n_days=15, n_names=15, seed=0):
        rng = np.random.default_rng(seed)
        days = pd.date_range("2020-01-06", periods=n_days, freq="D")
        frames = []
        for dd in days:
            f1 = rng.uniform(size=n_names)
            frames.append(pd.DataFrame({
                "trade_date": dd, "code": [f"c{i}" for i in range(n_names)],
                "f1": f1, "f2": rng.normal(size=n_names),
                "label": f1 + 0.03 * rng.normal(size=n_names),  # label 随 f1
            }))
        return pd.concat(frames, ignore_index=True)

    def test_inv4_blocks_bare_group_constant(self):
        df = self._signal_df()
        specs = [FeatureSpec("f1"), FeatureSpec("T_t", group_constant=True)]  # 裸组内常数
        with pytest.raises(InvariantViolation):
            train.assemble_l2(df, specs, "label")

    def test_group_is_by_day(self):
        df = self._signal_df(n_days=3, n_names=10)
        X, y, groups, d = train.assemble_l2(df, [FeatureSpec("f1"), FeatureSpec("f2")], "label")
        assert groups.tolist() == [10, 10, 10] and int(groups.sum()) == len(d)

    def test_ranker_learns_signal(self):
        df = self._signal_df(n_days=20, n_names=20, seed=3)
        specs = [FeatureSpec("f1"), FeatureSpec("f2")]
        model, names = train.train_l2_model(df, specs, "label", route="C", n_quantiles=5)
        df = df.copy()
        df["__score__"] = train.predict_scores(model, df, names)
        ic = metrics.mean_rank_ic(df.dropna(subset=["__score__"]), "__score__", "label")
        assert ic > 0.3   # 应学到 f1->label 的信号(样本内)

    def test_route_a_regressor_runs(self):
        df = self._signal_df(seed=5)
        model, names = train.train_l2_model(df, [FeatureSpec("f1"), FeatureSpec("f2")],
                                            "label", route="A")
        scores = train.predict_scores(model, df, names)
        assert np.isfinite(scores).all()


# ── Optuna 调参:预注册空间 ─────────────────────────────────────────────────
class TestTune:
    def test_finds_optimum_within_registered_space(self):
        space = {"x": {"type": "float", "low": 0.0, "high": 10.0}}
        study, best = tune.tune_hyperparams(lambda p: -((p["x"] - 3.0) ** 2), space,
                                            n_trials=40, direction="maximize", seed=1)
        assert abs(best["x"] - 3.0) < 1.0          # 收敛到最优附近
        assert set(best).issubset(set(space))      # 不越界扩张(先注册后运行)


# ── 审批门槛 + 盲测段一次性(INV-6)────────────────────────────────────────
class TestApproval:
    def _pass_metrics(self):
        return dict(beats_all_baselines=True, r_blind=0.05, dsr=0.97, pbo=0.20,
                    delta_maxdd=-0.01, slippage_net_20bp=0.01, execution_gap_bp=10.0,
                    manual_veto_destructive=False)

    def test_all_pass_approves(self):
        led = approval.BlindSegmentLedger()
        res = approval.evaluate_approval(self._pass_metrics(), blind_ledger=led,
                                         blind_segment_id="blind_2023H2")
        assert res.approved is True and not res.failed_checks()

    def test_one_fail_rejects(self):
        led = approval.BlindSegmentLedger()
        m = self._pass_metrics()
        m["dsr"] = 0.90  # 不达 0.95
        res = approval.evaluate_approval(m, blind_ledger=led, blind_segment_id="seg")
        assert res.approved is False and "DSR>0.95" in res.failed_checks()

    def test_blind_segment_consumed_once_inv6(self):
        led = approval.BlindSegmentLedger()
        approval.evaluate_approval(self._pass_metrics(), blind_ledger=led, blind_segment_id="seg")
        with pytest.raises(InvariantViolation):   # 同一盲测段再次裁决 -> INV-6 报错
            approval.evaluate_approval(self._pass_metrics(), blind_ledger=led, blind_segment_id="seg")

    def test_first_board_needs_30bp(self):
        led = approval.BlindSegmentLedger()
        m = self._pass_metrics()
        m["slippage_net_30bp"] = -0.001  # 首板 30bp 不存活
        res = approval.evaluate_approval(m, blind_ledger=led, blind_segment_id="seg",
                                         is_first_board=True)
        assert res.approved is False and "SlippageStress_30bp>0" in res.failed_checks()
