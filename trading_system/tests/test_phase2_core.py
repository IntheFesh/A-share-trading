"""Phase 2 核心真实测试:成本六层 / 事件级引擎逐笔手工对账 / 仓位合成。

引擎部分严格按 v3.1"构造已知算例,逐笔结果与手算一致"的要求:每个场景的入场价、出场价、
出场日、毛收益都可手算,引擎结果必须逐位吻合。逻辑错即真实失败,不放水。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest import costs as costmod
from trading_system.backtest import engine as eng
from trading_system.data import price_layers as pl
from trading_system.data.collectors import synthetic
from trading_system import portfolio as port


# ── 构造引擎用 bars ─────────────────────────────────────────────────────────
def _r(code, date, o, c, pc, h=None, l=None, vol=10000.0):
    h = max(o, c) if h is None else h
    l = min(o, c) if l is None else l
    return dict(code=code, trade_date=pd.Timestamp(date), open_raw=o, high_raw=h, low_raw=l,
               close_raw=c, preclose_raw=pc, volume=vol, amount=vol * c, adj_factor=1.0)


def _bars(rows):
    return pl.build_price_layers(pd.DataFrame(rows))


# =====================================================================
# 成本六层
# =====================================================================
class TestCosts:
    def test_verified_floor(self):
        # 印花税 0.05% + 经手费双向 2×0.00341% = 5.682 bp
        assert abs(costmod.VERIFIED_FLOOR_BP - 5.682) < 1e-9

    def test_round_trip_hand_calc(self):
        cm = costmod.CostModel()  # 默认费率
        # notional=20000:费率层 (0.0000441+0.0005441)*20000=11.764;佣金 max(6,5)*2=12;
        # 绝对费 23.764 -> /20000=0.0011882;变动层 滑点10bp×2=0.002 -> 合计 0.0031882
        got = cm.round_trip_cost_fraction(20000.0, slippage_bp=10)
        assert abs(got - 0.0031882) < 1e-7

    def test_min_commission_floor(self):
        cm = costmod.CostModel()
        assert cm.commission_side(10000.0) == 5.0   # 0.0003*10000=3 < 5 -> 取 5
        assert cm.commission_side(30000.0) == 9.0   # 0.0003*30000=9 > 5

    def test_min_order_amount_gate(self):
        cm = costmod.CostModel(q_min_amount_yuan=16667.0)
        assert not cm.passes_min_order(10000.0)
        assert cm.passes_min_order(20000.0)


# =====================================================================
# 事件级引擎:逐笔手工对账
# =====================================================================
class TestEngineHardStopWithGap:
    def test_stop_confirmed_close_executed_next_open_gap_through(self):
        d = synthetic.make_calendar("2020-01-06", 5).dates
        rows = [
            _r("600000", d[0], 10.0, 10.0, 10.0),                 # idx0 信号 t=0
            _r("600000", d[1], 10.0, 9.5, 10.0),                  # idx1 入场:open=10.0
            _r("600000", d[2], 9.4, 9.1, 9.5),                    # idx2 eval:9.1>9.0 不触发
            _r("600000", d[3], 9.0, 8.7, 9.1),                    # idx3 eval:8.7<=9.0 止损确认
            _r("600000", d[4], 8.5, 8.5, 8.7, h=8.6, l=8.4),      # idx4 次开 8.5 执行(跳空)
        ]
        res = eng.simulate_trade(_bars(rows), 0, atr=0.4)  # 2.5N=1.0 -> stop=9.0
        assert res.status == "closed"
        assert res.entry_price == 10.0
        assert len(res.fills) == 1
        f = res.fills[0]
        assert f.reason == "stop" and f.exec_idx == 4 and f.price == 8.5
        assert abs(res.gross_return - (-0.15)) < 1e-9   # 8.5/10-1;实际亏损 15% > 2.5N(10%)
        assert res.exit_idx >= 0 + 2                    # INV-1:出场 >= t+2


class TestEngineTakeProfitLadder:
    def test_ladder_then_trailing(self):
        d = synthetic.make_calendar("2020-01-06", 7).dates
        rows = [
            _r("600000", d[0], 10.0, 10.0, 10.0),     # idx0 信号
            _r("600000", d[1], 10.0, 10.5, 10.0),     # idx1 入场 open=10
            _r("600000", d[2], 10.6, 11.0, 10.5),     # idx2 close=11=tp1 -> 确认
            _r("600000", d[3], 11.2, 12.0, 11.0),     # idx3 次开 11.2 卖 tp1;close=12=tp2 确认
            _r("600000", d[4], 12.1, 12.5, 12.0),     # idx4 次开 12.1 卖 tp2;移止损成本;max_close=12.5
            _r("600000", d[5], 12.0, 11.4, 12.5),     # idx5 跟踪止损=12.5-1.0=11.5;11.4<=11.5 确认
            _r("600000", d[6], 11.3, 11.0, 11.4),     # idx6 次开 11.3 卖余仓 stop
        ]
        res = eng.simulate_trade(_bars(rows), 0, atr=0.4, trail_c=2.5)  # R=1.0
        assert res.status == "closed"
        assert [f.reason for f in res.fills] == ["tp1", "tp2", "stop"]
        prices = [f.price for f in res.fills]
        assert prices == [11.2, 12.1, 11.3]
        assert all(abs(f.fraction - 1 / 3) < 1e-9 for f in res.fills)
        # 加权出场 = (11.2+12.1+11.3)/3 = 11.53333 -> gross = 0.153333
        assert abs(res.gross_return - (11.53333333333333 / 10 - 1)) < 1e-6


class TestEngineEntryFailures:
    def test_entry_failed_limitup_one_price(self):
        d = synthetic.make_calendar("2020-01-06", 4).dates
        rows = [
            _r("600000", d[0], 10.0, 10.0, 10.0),
            _r("600000", d[1], 11.0, 11.0, 10.0, h=11.0, l=11.0),  # 一字涨停:O=H=L=C=11=10*1.1
            _r("600000", d[2], 11.5, 11.8, 11.0),
            _r("600000", d[3], 11.9, 12.0, 11.8),
        ]
        bars = _bars(rows)
        assert bool(bars.loc[1, "is_one_price_limit"]) and bool(bars.loc[1, "is_limit_up"])
        res = eng.simulate_trade(bars, 0, atr=0.4)
        assert res.status == "entry_failed_limitup" and not res.fills

    def test_entry_failed_gap_up(self):
        d = synthetic.make_calendar("2020-01-06", 4).dates
        rows = [
            _r("600000", d[0], 10.0, 10.0, 10.0),
            _r("600000", d[1], 10.8, 10.9, 10.0),  # 高开 +8% > 7% -> 放弃(非一字)
            _r("600000", d[2], 10.9, 11.0, 10.9),
            _r("600000", d[3], 11.0, 11.1, 11.0),
        ]
        res = eng.simulate_trade(_bars(rows), 0, atr=0.4)
        assert res.status == "entry_failed_gap" and not res.fills


class TestEngineLimitDownDelay:
    def test_exit_delayed_by_limit_down(self):
        d = synthetic.make_calendar("2020-01-06", 6).dates
        rows = [
            _r("600000", d[0], 10.0, 10.0, 10.0),
            _r("600000", d[1], 10.0, 9.5, 10.0),                       # 入场 open=10
            _r("600000", d[2], 9.4, 9.1, 9.5),                          # 不触发
            _r("600000", d[3], 9.0, 8.7, 9.1),                          # 止损确认(8.7<=9.0)
            _r("600000", d[4], 7.83, 7.83, 8.70, h=7.83, l=7.83),       # 次日一字跌停 -> 卖不出,顺延
            _r("600000", d[5], 7.90, 8.00, 7.83),                       # 再次开 7.90 执行
        ]
        bars = _bars(rows)
        assert bool(bars.loc[4, "is_limit_down"])   # 7.83 = round(8.70*0.9)
        res = eng.simulate_trade(bars, 0, atr=0.4)
        assert res.status == "closed" and len(res.fills) == 1
        f = res.fills[0]
        assert f.reason == "stop" and f.exec_idx == 5 and f.price == 7.90  # 顺延一天


# =====================================================================
# L3 仓位合成
# =====================================================================
class TestPortfolio:
    def test_kelly_three_levels(self):
        assert port.kelly_risk_budget(0.1, s=1.0) == 0.005
        assert port.kelly_risk_budget(0.1, s=0.5) == 0.0025
        assert port.kelly_risk_budget(-0.1, s=1.0) == 0.0     # f*<=0 -> 空仓
        assert port.kelly_risk_budget(0.1, s=0.0) == 0.0      # 刹车档 s=0

    def test_total_exposure_min(self):
        assert port.total_exposure(sigma_star=0.1, sigma_hat=0.2, m_t=0.8) == 0.5  # 硬顶 0.5
        assert port.total_exposure(sigma_star=0.1, sigma_hat=0.2, m_t=0.3) == 0.3  # m_t 收紧
        assert abs(port.total_exposure(sigma_star=0.05, sigma_hat=0.2, m_t=0.8) - 0.25) < 1e-12

    def test_inverse_atr_weights(self):
        w = port.inverse_atr_weights(np.array([0.4, 0.4, 0.8]))
        assert np.allclose(w, [0.4, 0.4, 0.2])

    def test_compose_single_name_cap(self):
        # w_total=0.5, w̃=[0.4,0.4,0.2] -> target [0.2,0.2,0.1];name0 上限 0.08 -> 截断
        w = port.compose_positions(np.array([0.4, 0.4, 0.8]), w_total=0.5,
                                   single_caps=np.array([0.08, 0.25, 0.25]))
        assert abs(w[0] - 0.08) < 1e-12 and abs(w[1] - 0.2) < 1e-12 and abs(w[2] - 0.1) < 1e-12
        assert w.sum() < 0.5  # 截断后不补杠杆

    def test_compose_cluster_limit(self):
        # cluster A=[0,1] 合计 0.4 > 0.25 -> 按 0.625 压缩
        w = port.compose_positions(np.array([0.4, 0.4, 0.8]), w_total=0.5,
                                   single_caps=np.array([0.25, 0.25, 0.25]),
                                   cluster_ids=np.array(["A", "A", "B"]),
                                   cluster_limits={"A": 0.25})
        assert abs(w[0] - 0.125) < 1e-9 and abs(w[1] - 0.125) < 1e-9
        assert abs(w[2] - 0.1) < 1e-12
