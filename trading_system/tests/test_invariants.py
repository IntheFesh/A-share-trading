"""七条不变量(INV-1 ~ INV-7)的 pytest 断言——系统宪法,必须常绿。

对应施工任务书"一、硬约束"。本文件是第二道防线(第一道是各业务模块的运行时 assert)。
INV-3(标签—成交同源)依赖 Phase 1/2 的共享成交函数,当前以 skip 占位,待落地后自动转强校验。
"""

from __future__ import annotations

import math

import pytest

from trading_system import invariants as inv


# ── INV-1 可交易标签优先(T+1 出信号,最早 T+2 卖) ──────────────────────────
class TestINV1TradeableLabel:
    def test_exit_must_be_at_least_t_plus_2(self) -> None:
        inv.assert_tradeable_exit(10, 12)  # 边界:恰好 T+2,OK
        inv.assert_tradeable_exit(10, 13)
        with pytest.raises(inv.InvariantViolation):
            inv.assert_tradeable_exit(10, 11)  # T+1 卖 -> 违规
        with pytest.raises(inv.InvariantViolation):
            inv.assert_tradeable_exit(10, 10)  # 当日卖 -> 违规

    def test_h0_blocked_in_production(self) -> None:
        inv.assert_production_label_horizon(1)
        inv.assert_production_label_horizon(10)
        with pytest.raises(inv.InvariantViolation):
            inv.assert_production_label_horizon(0)  # h=0 禁止进生产

    def test_h0_only_in_diagnostic_namespace(self) -> None:
        inv.assert_label_namespace_allows_horizon(inv.DIAGNOSTIC_NAMESPACE, 0)  # 诊断里 OK
        inv.assert_label_namespace_allows_horizon(inv.PRODUCTION_NAMESPACE, 1)  # 生产里 h>=1 OK
        with pytest.raises(inv.InvariantViolation):
            inv.assert_label_namespace_allows_horizon(inv.PRODUCTION_NAMESPACE, 0)


# ── INV-2 双价格层:执行用 raw,特征用 adj ──────────────────────────────────
class TestINV2PriceLayers:
    def test_limit_prices_use_raw_preclose(self) -> None:
        assert inv.limit_up_price(10.00) == 11.00
        assert inv.limit_down_price(10.00) == 9.00

    def test_adjusted_preclose_would_be_wrong(self) -> None:
        # 构造一次除权:同一日,原始昨收=10.00(实际涨停=11.00),后复权昨收=5.00。
        raw_preclose, adj_preclose, actual_limit_up = 10.00, 5.00, 11.00
        assert inv.limit_up_price(raw_preclose) == actual_limit_up   # 用 raw 昨收 -> 算对
        assert inv.limit_up_price(adj_preclose) != actual_limit_up   # 用 adj 昨收 -> 算错

    def test_exchange_half_up_rounding(self) -> None:
        # 交易所半进位:.xx5 向上;银行家舍入会出错(round(2.675,2)->2.67)。
        assert inv._round_half_up_2(2.675) == 2.68
        assert inv._round_half_up_2(2.665) == 2.67

    def test_execution_guard_rejects_adj_columns(self) -> None:
        inv.assert_execution_uses_raw(["open_raw", "close_raw", "preclose_raw"])  # OK
        with pytest.raises(inv.InvariantViolation):
            inv.assert_execution_uses_raw(["close_adj"])  # 执行路径用后复权 -> 报错
        with pytest.raises(inv.InvariantViolation):
            inv.assert_execution_uses_raw(["close_raw", "high_adj"])  # 混入一个 adj 也报错

    def test_column_classifiers(self) -> None:
        assert inv.is_raw_column("close_raw") and not inv.is_adj_column("close_raw")
        assert inv.is_adj_column("close_adj") and not inv.is_raw_column("close_adj")


# ── INV-3 标签—成交同源(Phase 1/2 落地;现以 skip 占位) ────────────────────
class TestINV3LabelFillSameSource:
    def test_label_and_engine_share_fill_function(self) -> None:
        engine = pytest.importorskip("trading_system.backtest.engine")
        labels = pytest.importorskip("trading_system.labels")
        name = inv.CANONICAL_FILL_FUNC_NAME
        if not hasattr(engine, name) or not hasattr(labels, name):
            pytest.skip(f"INV-3 待 Phase 1/2:engine 与 labels 尚未共享 {name}()。")
        assert getattr(labels, name) is getattr(engine, name), (
            "INV-3:labels 必须 import engine 的同一个成交判定函数对象,不得各写一份。"
        )


# ── INV-4 组内常数默认是覆盖层 ──────────────────────────────────────────────
class TestINV4GroupConstant:
    def test_bare_group_constant_rejected(self) -> None:
        specs = [
            inv.FeatureSpec("ret_20d"),
            inv.FeatureSpec("T_t", group_constant=True),  # 裸列 -> 违规
        ]
        with pytest.raises(inv.InvariantViolation):
            inv.assert_group_constant_only_via_interaction(specs)

    def test_interaction_allowed(self) -> None:
        specs = [
            inv.FeatureSpec("ret_20d"),
            inv.FeatureSpec("T_t_x_ret20d", group_constant=True, is_interaction=True),
        ]
        inv.assert_group_constant_only_via_interaction(specs)  # 交互形式 -> OK

    def test_non_group_constant_passthrough(self) -> None:
        specs = [inv.FeatureSpec("cgo"), inv.FeatureSpec("rps")]
        inv.assert_group_constant_only_via_interaction(specs)  # 非组内常数 -> OK


# ── INV-5 单股上限由连续跌停压力决定 ────────────────────────────────────────
class TestINV5SingleNameCap:
    def test_continuous_limit_down_loss_appendix_b(self) -> None:
        # 复核 v3.1 附录 B
        assert math.isclose(inv.continuous_limit_down_loss(0.08, 2), 0.0152, abs_tol=1e-9)
        assert math.isclose(inv.continuous_limit_down_loss(0.05, 2), 0.0095, abs_tol=1e-9)

    def test_single_name_cap_formula(self) -> None:
        assert inv.single_name_cap(0.08, 0.0152, 1.0) == min(0.08, 0.0152 / 1.0)
        assert inv.single_name_cap(0.05, 0.02, 2.0) == min(0.05, 0.02 / 2.0)
        with pytest.raises(inv.InvariantViolation):
            inv.single_name_cap(0.08, 0.0152, 0.0)  # g_hat<=0 -> 报错

    def test_forbidden_15pct_default(self) -> None:
        inv.assert_hard_cap_allowed(0.08)  # 主板档 OK
        inv.assert_hard_cap_allowed(0.05)  # 特殊档 OK
        with pytest.raises(inv.InvariantViolation):
            inv.assert_hard_cap_allowed(0.15)  # 15% 默认值禁止


# ── INV-6 盲测段一次性 ──────────────────────────────────────────────────────
class TestINV6BlindSegmentOnce:
    def test_segment_archived_after_decision(self) -> None:
        ledger = inv.BlindSegmentLedger()
        ledger.assert_available("blind_2023H2")          # 初始可用
        ledger.use_for_decision("blind_2023H2")          # 用于换届裁决 -> 封存
        with pytest.raises(inv.InvariantViolation):
            ledger.assert_available("blind_2023H2")      # 再查 -> 已封存
        with pytest.raises(inv.InvariantViolation):
            ledger.use_for_decision("blind_2023H2")      # 再用 -> 报错

    def test_independent_segments(self) -> None:
        ledger = inv.BlindSegmentLedger()
        ledger.use_for_decision("seg_a")
        ledger.assert_available("seg_b")  # 不同段互不影响


# ── INV-7 条件化优先于无条件叠加 ────────────────────────────────────────────
class TestINV7Conditional:
    def test_unconditional_blocked_by_default(self) -> None:
        inv.assert_conditional_or_documented_override(is_unconditional=False)  # 条件化 -> OK
        with pytest.raises(inv.InvariantViolation):
            inv.assert_conditional_or_documented_override(is_unconditional=True)  # 无条件无override

    def test_override_requires_justification(self) -> None:
        with pytest.raises(inv.InvariantViolation):
            inv.assert_conditional_or_documented_override(
                is_unconditional=True, unconditional_override=True, justification="  "
            )
        inv.assert_conditional_or_documented_override(
            is_unconditional=True,
            unconditional_override=True,
            justification="已通过'不劣于条件化'验证(见 §X 消融)",
        )
