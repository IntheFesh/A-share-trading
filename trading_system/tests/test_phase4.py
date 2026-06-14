"""Phase 4 真实测试:OPE(IPW/DR 手算)/ reason codes / 算法厌恶护栏 / 作战手册 / 监控落盘。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system import audit
from trading_system.playbook import PLAYBOOK_COLUMNS, generate_playbook
from trading_system.reports.monitor import run_monitor


class TestOPE:
    def test_ipw_unbiased_hand_calc(self):
        # 行为 50/50 随机;目标=永远动作1;动作1奖励1、动作0奖励0 -> 目标真值=1
        actions = np.array([1, 1, 0, 0])
        target = np.array([1, 1, 1, 1])
        rewards = np.array([1.0, 1.0, 0.0, 0.0])
        pb = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(audit.ipw_value(actions, target, rewards, pb) - 1.0) < 1e-12

    def test_dr_perfect_q_and_reduces_to_ipw(self):
        actions = np.array([1, 1, 0, 0])
        target = np.array([1, 1, 1, 1])
        rewards = np.array([1.0, 1.0, 0.0, 0.0])
        pb = np.array([0.5, 0.5, 0.5, 0.5])
        # 完美 q:q(1)=1,q(0)=0 -> DR = 目标真值 = 1
        assert abs(audit.dr_value(actions, target, rewards, pb,
                                  q_taken=np.array([1.0, 1.0, 0.0, 0.0]),
                                  q_target=np.array([1.0, 1.0, 1.0, 1.0])) - 1.0) < 1e-12
        # q=0 -> DR 退化为 IPW
        zero = np.zeros(4)
        assert abs(audit.dr_value(actions, target, rewards, pb, q_taken=zero, q_target=zero)
                   - audit.ipw_value(actions, target, rewards, pb)) < 1e-12

    def test_ipw_requires_positive_support(self):
        with pytest.raises(ValueError):
            audit.ipw_value(np.array([1]), np.array([1]), np.array([1.0]), np.array([0.0]))

    def test_reason_codes_closed_with_new_ones(self):
        assert audit.ReasonCode.OVEREXTENSION_LOTTERY.value == "overextension_lottery"
        assert audit.ReasonCode.HILO_STYLE_REVERSAL.value == "hilo_style_reversal"
        assert audit.ReasonCode.NEGATIVE_PREANN in list(audit.ReasonCode)

    def test_algorithm_aversion_guard(self):
        # 模型刚失误后(idx1,2)否决率飙升 -> 触发
        flagged = audit.algorithm_aversion_check(
            model_was_wrong=[False, True, True, False, False],
            human_vetoed=[False, True, True, False, False],
        )
        assert flagged["flagged"] is True
        # 否决均匀 -> 不触发
        calm = audit.algorithm_aversion_check(
            model_was_wrong=[False, True, False, True, False],
            human_vetoed=[True, True, True, True, True],
        )
        assert calm["flagged"] is False


class TestPlaybook:
    def test_generates_files_and_columns(self, tmp_path):
        candidates = pd.DataFrame({
            "code": ["600000", "600001"], "trigger": ["pullback", "first_board"],
            "model_score": [0.9, 0.8], "rank": [1, 2],
            "limit_buy_price": [10.50, 8.20], "target_weight_pct": [5.0, 5.0],
            "stop_price": [9.80, 7.60], "veto_reason": [None, "overextension_lottery"],
            "days_to_disclosure": [3, 30], "has_preann": [False, False],
        })
        regime = dict(T_t=0.72, stage=3, m_t=1.0, w_total=0.5, brake_level=1.0,
                      days_to_tier1=4, days_to_tier2=40)
        table, md = generate_playbook(candidates, regime, trade_date="2026-06-12",
                                      out_dir=tmp_path, print_console=False)
        assert list(table.columns) == list(PLAYBOOK_COLUMNS)   # 字段齐全
        # 仓位参考指标列齐备
        for col in ("atr_n", "single_cap_pct", "kelly_suggest_pct", "stop_distance_pct", "amihud_illiq"):
            assert col in PLAYBOOK_COLUMNS and col in table.columns
        assert (tmp_path / "playbook_2026-06-12.csv").exists()
        assert (tmp_path / "playbook_2026-06-12.md").exists()
        assert "T_t = 0.72" in md and "阶段 = 3" in md          # 页脚 regime 摘要


class TestMonitor:
    def test_writes_png_and_md_with_correct_metrics(self, tmp_path):
        nav = [1.0, 1.2, 0.9, 1.1, 1.3]
        ic = pd.Series(np.linspace(-0.02, 0.05, 40))
        res = run_monitor(nav, ic, out_dir=tmp_path, block_len=10,
                          fill_failure_rate=0.05, execution_gap_bp=8.0)
        assert abs(res["max_drawdown"] - (-0.25)) < 1e-9   # 1.2->0.9
        png = tmp_path / "monitor_nav.png"
        assert png.exists() and png.stat().st_size > 0     # PNG 真落盘
        assert (tmp_path / "monitor_report.md").exists()
        assert np.isfinite(res["blocked_rank_ic"]) and np.isfinite(res["icir"])
