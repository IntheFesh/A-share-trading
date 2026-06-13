"""事件级回测引擎(唯一真值,最核心)。Phase 2(任务 2.2)。对应 v3.1 第十一章。

买入状态机:signal_generated -> entry_pending -> {entry_filled / entry_failed_limitup /
  entry_failed_gap / entry_failed_liquidity}。entry_date = t+1;高开>7% 放弃;一字涨停或
  参与量不足(<=竞价量 1%)-> 失败。
卖出状态机:not_sellable_due_to_Tplus1 -> exit_triggered -> exit_pending ->
  {exit_filled / exit_delayed_limitdown / exit_delayed_suspension}。最早可卖 t+2(INV-1)。
止损拆两日期:stop_trigger_date(收盘确认)与 first_executable_exit_date(次开执行);
  跳空按次开实际价记账(实际亏损可 > 2.5N)。
出场优先级:硬止损 > 事件止损(含披露季 veto) > 止盈 > 模型轮动/时间止损。

INV-2 守卫:所有撮合/涨跌停/PnL 用 raw 列,检测到 *_adj 即 raise
  (见 trading_system.invariants.assert_execution_uses_raw)。
INV-3:成交判定的权威实现 ``is_tradeable_fill`` 落在本模块(Phase 1/2),labels/ 从此 import,
  函数名见 trading_system.invariants.CANONICAL_FILL_FUNC_NAME。
"""

from __future__ import annotations

_PHASE = "Phase 2 任务 2.2"

# 注意:INV-3 的权威成交判定函数 is_tradeable_fill 在 Phase 1/2 落地于本模块。
# 届时 labels/ 必须 `from trading_system.backtest.engine import is_tradeable_fill`,不得另写一份。
# 现阶段尚未定义,故 tests/test_invariants.py 的 INV-3 用例会自动 skip。


class BacktestEngine:
    """事件级引擎(占位)。Phase 2 落地状态机、出场优先级、PnL 三段归因。"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise NotImplementedError(f"{_PHASE}:BacktestEngine 待实现(唯一真值)。")
