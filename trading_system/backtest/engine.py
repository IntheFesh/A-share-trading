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

# 默认放弃高开阈值:T+1 开盘相对昨收高开 > 7% 则放弃买入(v3.1 第十一章)。
DEFAULT_GAP_ABANDON = 0.07


def is_tradeable_fill(
    *,
    open_price: float,
    preclose: float,
    is_one_price_limit_up: bool,
    gap_threshold: float = DEFAULT_GAP_ABANDON,
) -> bool:
    """INV-3 权威成交判定(买入侧、基于价格)。labels/ 与引擎共用本函数,不得各写一份。

    Phase: 落地于 Phase 2 引擎,Phase 1 标签即开始引用(INV-3)。价格层:全用 raw(INV-2)。
    规则(T+1 开盘):
      - 一字涨停 -> 买不进(entry_failed_limitup) -> False;
      - 高开 (open/preclose - 1) > gap_threshold -> 放弃(entry_failed_gap) -> False;
      - 否则价格上可成交 -> True。
    流动性/参与量(≤竞价量 1%)是引擎用成交量做的**额外**闸门(标签侧无盘中量,无法判),
    不在本共享函数内——共享的是"一字/高开"价格判定,这正是 INV-3 要求同源的部分。
    """
    if is_one_price_limit_up:
        return False
    if preclose <= 0:
        raise ValueError("preclose 必须 > 0(raw 昨收)")
    gap = open_price / preclose - 1.0
    if gap > gap_threshold:
        return False
    return True


class BacktestEngine:
    """事件级引擎(占位)。Phase 2 落地状态机、出场优先级、PnL 三段归因。"""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise NotImplementedError(f"{_PHASE}:BacktestEngine 待实现(唯一真值)。")
