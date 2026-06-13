"""Phase 1 真实单元测试:特征(防未来函数三检查)/ 标签 / L0 regime / L1 触发器。

纪律(用户要求):每个断言对已知正确答案做核验;且**专门验证防泄漏守卫确实能抓出泄漏**
(静态扫描抓 shift(-)、截断等变性抓全局聚合泄漏),证明测试不是橡皮图章。逻辑错则真实失败。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest import engine as eng
from trading_system.data import price_layers as pl
from trading_system.data.collectors import synthetic
import trading_system.features.builtin  # noqa: F401  触发内置特征注册
import trading_system.labels as labels
import trading_system.regime as regime
import trading_system.triggers as triggers
from trading_system.features import registry as reg


# ── 测试用确定性构造 ────────────────────────────────────────────────────────
def _rows_from_closes(code, dates, closes, vols=None):
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    vols = np.full(n, 1000.0) if vols is None else np.asarray(vols, dtype="float64")
    preclose = np.empty(n)
    preclose[0] = closes[0]
    preclose[1:] = closes[:-1]
    rows = []
    for i in range(n):
        c, pc = float(closes[i]), float(preclose[i])
        o = pc
        rows.append(dict(
            code=code, trade_date=pd.Timestamp(dates[i]),
            open_raw=round(o, 2), high_raw=round(max(o, c), 2), low_raw=round(min(o, c), 2),
            close_raw=round(c, 2), preclose_raw=round(pc, 2),
            volume=float(vols[i]), amount=round(float(vols[i]) * c, 2), adj_factor=1.0,
        ))
    return rows


def _trend_panel(codes_rets, dates, start=10.0):
    """codes_rets: {code: daily_ret};各 code 以恒定日收益生成收盘价。返回 build_price_layers 后的面板。"""
    all_rows = []
    for code, r in codes_rets.items():
        closes = start * np.cumprod(np.full(len(dates), 1.0 + r))
        all_rows += _rows_from_closes(code, dates, closes)
    return pl.build_price_layers(pd.DataFrame(all_rows))


# =====================================================================
# 特征注册表 + 防未来函数三检查
# =====================================================================
class TestFeatureRegistryAntiLookahead:
    def test_all_builtin_features_pass_truncation_equivariance(self):
        cal = synthetic.make_calendar("2020-01-06", 90)
        raw = synthetic.make_raw_panel(["600000"], cal, seed=11)
        g = pl.build_price_layers(raw)
        positions = list(range(65, 89))  # 取 lookback(<=60)之后的位置
        names = [n for n in reg.REGISTRY if not n.startswith("_leak")]
        assert names, "应已注册内置特征"
        for name in names:
            viol = reg.truncation_equivariance_violations(name, g, positions)
            assert not viol, f"特征 {name} 截断等变性失败(未来泄漏): {viol[:3]}"

    def test_static_scan_catches_negative_shift(self):
        # 检查一:负向 shift 在注册时即被静态扫描抓出
        with pytest.raises(reg.FutureLeakError):
            @reg.register("_leak_shift", "测试")
            def _leak_shift(g):  # noqa: ANN001
                return g["close_adj"].shift(-1) / g["close_adj"] - 1.0

    def test_truncation_catches_global_aggregation_leak(self):
        # 检查二:用全序列 max 归一(静态扫描放过,但截断等变性必抓出)
        @reg.register("_leak_globalmax", "测试")
        def _leak_globalmax(g):  # noqa: ANN001
            return g["close_adj"] / g["close_adj"].max()
        try:
            # 严格递增序列 -> 全局 max 落在最后一天;任一早于末日的位置,截断 max < 全局 max,
            # 故泄漏必然暴露(不依赖随机种子)。
            dates = synthetic.make_calendar("2020-01-06", 40).dates
            closes = list(np.linspace(10.0, 30.0, 40))
            g = pl.build_price_layers(pd.DataFrame(_rows_from_closes("600000", dates, closes)))
            viol = reg.truncation_equivariance_violations("_leak_globalmax", g, [10, 20, 30])
            assert viol, "全局 max 泄漏应被截断等变性抓出"
        finally:
            reg.REGISTRY.pop("_leak_globalmax", None)

    def test_check_three_hides_raw_price(self):
        # 检查三:特征函数看不到原始价(*_raw),引用即 KeyError
        @reg.register("_leak_raw", "测试")
        def _leak_raw(g):  # noqa: ANN001
            return g["close_raw"]  # 原始价对特征不可见
        try:
            cal = synthetic.make_calendar("2020-01-06", 10)
            g = pl.build_price_layers(synthetic.make_raw_panel(["600000"], cal, seed=1))
            with pytest.raises(KeyError):
                reg.compute_raw_feature("_leak_raw", g)
        finally:
            reg.REGISTRY.pop("_leak_raw", None)

    def test_ret_5_value_hand_check(self):
        dates = synthetic.make_calendar("2020-01-06", 8).dates
        closes = [10, 10, 10, 10, 10, 10, 12, 12]  # 第 6 个(idx5)起为 10,idx6=12
        g = pl.build_price_layers(pd.DataFrame(_rows_from_closes("600000", dates, closes)))
        raw = reg.compute_raw_feature("ret_5", g).to_numpy()
        # idx6: close[6]/close[1]-1 = 12/10-1 = 0.2
        assert abs(raw[6] - 0.2) < 1e-12

    def test_cross_sectional_rank(self):
        day = pd.Timestamp("2020-01-06")
        panel = pd.DataFrame({
            "code": list("abcde"), "trade_date": [day] * 5,
            "__raw__x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        ranks = reg.cross_sectional_rank(panel, "__raw__x").to_numpy()
        assert np.allclose(ranks, [0.2, 0.4, 0.6, 0.8, 1.0])
        assert ranks.min() >= 0.0 and ranks.max() <= 1.0


# =====================================================================
# 标签(INV-1 + INV-3)
# =====================================================================
class TestLabels:
    def test_y_h_hand_check(self):
        dates = synthetic.make_calendar("2020-01-06", 6).dates
        closes = [10, 11, 12, 13, 14, 15]
        g = pl.build_price_layers(pd.DataFrame(_rows_from_closes("600000", dates, closes)))
        y = labels.build_y_h(g, h=1).to_numpy()
        # 信号 t=0:入场 open[1]=close[0]=10.0,出场 close[2]=12 -> 12/10-1=0.2
        assert abs(y[0] - 0.2) < 1e-9
        # 末两行无足够未来 -> NaN
        assert np.isnan(y[-1]) and np.isnan(y[-2])

    def test_entry_gating_one_price_limit_up_blocks(self):
        dates = synthetic.make_calendar("2020-01-06", 5).dates
        # day1 一字涨停(O=H=L=C=11=round(10*1.1)),信号 t=0 入场 day1 应买不进 -> NaN
        rows = _rows_from_closes("600000", dates, [10, 11, 12, 13, 14])
        rows[1].update(open_raw=11.0, high_raw=11.0, low_raw=11.0, close_raw=11.0, preclose_raw=10.0)
        g = pl.build_price_layers(pd.DataFrame(rows))
        assert bool(g.loc[1, "is_one_price_limit"]) and bool(g.loc[1, "is_limit_up"])
        y = labels.build_y_h(g, h=1).to_numpy()
        assert np.isnan(y[0])

    def test_entry_gating_gap_up_blocks(self):
        dates = synthetic.make_calendar("2020-01-06", 5).dates
        rows = _rows_from_closes("600000", dates, [10, 10.9, 11, 12, 13])
        rows[1].update(open_raw=10.8, preclose_raw=10.0)  # 高开 +8% > 7% -> 放弃
        g = pl.build_price_layers(pd.DataFrame(rows))
        y = labels.build_y_h(g, h=1).to_numpy()
        assert np.isnan(y[0])

    def test_inv1_h0_rejected_for_production(self):
        dates = synthetic.make_calendar("2020-01-06", 5).dates
        g = pl.build_price_layers(pd.DataFrame(_rows_from_closes("600000", dates, [10, 11, 12, 13, 14])))
        from trading_system.invariants import InvariantViolation
        with pytest.raises(InvariantViolation):
            labels.build_y_h(g, h=0)               # h=0 进生产 -> INV-1 报错
        with pytest.raises(AssertionError):
            labels.build_y_prod(g, holding_days=0, cost=0.0)

    def test_y_mtm0_is_diagnostic(self):
        dates = synthetic.make_calendar("2020-01-06", 4).dates
        rows = _rows_from_closes("600000", dates, [10, 11, 12, 13])
        rows[1].update(open_raw=10.0, close_raw=11.0)  # day1 开 10 收 11
        g = pl.build_price_layers(pd.DataFrame(rows))
        y = labels.build_y_mtm0(g).to_numpy()
        assert abs(y[0] - (11.0 / 10.0 - 1.0)) < 1e-9   # 当日 close/open-1

    def test_inv3_label_and_engine_share_fill_function(self):
        assert labels.is_tradeable_fill is eng.is_tradeable_fill


# =====================================================================
# L0 regime:六指标 / 温度 / HiLo
# =====================================================================
class TestRegime:
    @pytest.fixture
    def six_panel(self):
        d = synthetic.make_calendar("2020-01-06", 3).dates
        # Code A:连续两板(d0,d1 涨停,preclose 对齐使 close 恰为涨停价),d2 不涨停
        rowsA = [
            dict(code="600000", trade_date=pd.Timestamp(d[0]), open_raw=10.5, high_raw=11.0,
                 low_raw=10.4, close_raw=11.00, preclose_raw=10.00, volume=1000.0, amount=1.1e4,
                 adj_factor=1.0),
            dict(code="600000", trade_date=pd.Timestamp(d[1]), open_raw=11.5, high_raw=12.10,
                 low_raw=11.4, close_raw=12.10, preclose_raw=11.00, volume=1000.0, amount=1.2e4,
                 adj_factor=1.0),
            dict(code="600000", trade_date=pd.Timestamp(d[2]), open_raw=12.0, high_raw=12.2,
                 low_raw=11.9, close_raw=12.00, preclose_raw=12.10, volume=1000.0, amount=1.2e4,
                 adj_factor=1.0),
        ]
        # Code B:d1 炸板(高 22.00 触及涨停,收 21 未封)
        rowsB = [
            dict(code="600001", trade_date=pd.Timestamp(d[0]), open_raw=20.0, high_raw=20.0,
                 low_raw=20.0, close_raw=20.00, preclose_raw=20.00, volume=1000.0, amount=2.0e4,
                 adj_factor=1.0),
            dict(code="600001", trade_date=pd.Timestamp(d[1]), open_raw=20.5, high_raw=22.00,
                 low_raw=20.4, close_raw=21.00, preclose_raw=20.00, volume=1000.0, amount=2.1e4,
                 adj_factor=1.0),
            dict(code="600001", trade_date=pd.Timestamp(d[2]), open_raw=21.0, high_raw=21.1,
                 low_raw=20.9, close_raw=21.00, preclose_raw=21.00, volume=1000.0, amount=2.1e4,
                 adj_factor=1.0),
        ]
        panel = pl.build_price_layers(pd.DataFrame(rowsA + rowsB))
        return panel, d

    def test_six_indicators_hand_check(self, six_panel):
        panel, d = six_panel
        ind = regime.compute_six_indicators(panel)
        # d0:A 涨停 -> 涨停家数 1;连板高度 1
        assert ind.loc[pd.Timestamp(d[0]), "limit_up_count"] == 1
        assert ind.loc[pd.Timestamp(d[0]), "max_consecutive_boards"] == 1
        # d1:A 仍涨停(连板 2),B 炸板
        assert ind.loc[pd.Timestamp(d[1]), "limit_up_count"] == 1
        assert ind.loc[pd.Timestamp(d[1]), "max_consecutive_boards"] == 2
        assert abs(ind.loc[pd.Timestamp(d[1]), "blowup_rate"] - 0.5) < 1e-9   # A 封 + B 炸 -> 1/2
        assert abs(ind.loc[pd.Timestamp(d[1]), "promotion_rate"] - 1.0) < 1e-9  # A 昨封今封
        assert abs(ind.loc[pd.Timestamp(d[1]), "prev_limitup_premium"] - 0.10) < 1e-9  # 12.1/11-1

    def test_temperature_monotone_and_mapping(self):
        idx = pd.date_range("2020-01-06", periods=8, freq="D")
        base = {k: [1.0] * 8 for k in regime.SIX_INDICATORS}
        base["limit_up_count"] = [1, 1, 1, 1, 1, 1, 1, 8]  # 末日骤热
        ind = pd.DataFrame(base, index=idx)
        out = regime.compute_temperature(ind, window=10)
        assert (out["T_t"] >= 0).all() and (out["T_t"] <= 1).all()
        assert out["T_t"].iloc[7] > out["T_t"].iloc[3]   # 末日更热 -> T_t 更高
        # stage / m_t 映射自洽
        stage = np.digitize(out["T_t"].to_numpy(), np.asarray([0.2, 0.4, 0.6, 0.8]))
        assert (out["stage"].to_numpy() == stage).all()

    def test_hilo_positive_when_high_momentum_leads(self):
        dates = synthetic.make_calendar("2020-01-06", 25).dates
        rets = {f"60000{i}": (i - 6) * 0.003 for i in range(12)}  # 收益单调随 code
        panel = _trend_panel(rets, dates)
        hilo = regime.compute_hilo(panel, quantile_window=50)
        last = hilo.index.max()
        assert hilo.loc[last, "hilo_raw"] > 0   # 高动量层近 5 日收益 > 低动量层
        assert 0.0 <= hilo.loc[last, "hilo_t"] <= 1.0


# =====================================================================
# L1 触发器(粗桶)
# =====================================================================
class TestTriggers:
    def test_pullback_in_uptrend(self):
        dates = synthetic.make_calendar("2020-01-06", 67).dates
        closes = list(np.linspace(10, 15.9, 60)) + [15.6, 15.3, 15.0, 14.7, 14.4, 14.1, 13.8]
        g = pl.build_price_layers(pd.DataFrame(_rows_from_closes("600000", dates, closes)))
        trig = triggers.trigger_pullback(g, drawdown_low=0.05, drawdown_high=0.15)
        trig = trig.to_numpy()
        assert not trig[59]          # 趋势顶、无回撤 -> 不触发
        assert trig[63] or trig[64]  # 回撤约 7~9% 落在桶内 -> 触发

    def test_first_board_low_volume(self):
        dates = synthetic.make_calendar("2020-01-06", 62).dates
        closes = list(np.linspace(20, 10, 61)) + [11.00]  # 长期下跌 -> 低位;末日首板
        vols = [1000.0] * 61 + [500.0]                    # 末日缩量
        g = pl.build_price_layers(pd.DataFrame(_rows_from_closes("600000", dates, closes, vols)))
        assert bool(g.loc[61, "is_limit_up"])             # 10.00 -> 11.00 涨停
        trig = triggers.trigger_first_board(g, low_position=0.20, volume_shrink_max=0.70).to_numpy()
        assert trig[61]

    def test_rps_leader_cross_section(self):
        dates = synthetic.make_calendar("2020-01-06", 130).dates
        rets = {"600000": -0.004, "600001": -0.002, "600002": 0.0, "600003": 0.003, "600004": 0.006}
        panel = _trend_panel(rets, dates)
        trig = triggers.trigger_rps_leader(panel, rps_window=120, rps_min=0.90)
        panel = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)
        last_day = panel["trade_date"].max()
        top = panel[(panel["code"] == "600004") & (panel["trade_date"] == last_day)].index[0]
        bot = panel[(panel["code"] == "600000") & (panel["trade_date"] == last_day)].index[0]
        assert bool(trig.iloc[top]) and not bool(trig.iloc[bot])
