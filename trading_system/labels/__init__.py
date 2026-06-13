"""Phase 1:标签构造(INV-1 + INV-3)。任务 1.3。对应 v3.1 第三章。

  y_prod :生产可交易标签(raw 价,tau_exit >= t+2,扣成本)。见 INV-1。
  y_h    :固定窗口对照,h ∈ {1,2,3,5,8,10}。
  y_mtm0 :诊断标签(h=0,单独命名空间,禁止进训练/回测/审批)。见 INV-1。

INV-3(标签—成交同源):处理"T+1 一字/高开>7% 不可成交"的判定,**必须**与回测引擎共享
同一个函数。Phase 1/2 落地时:
    from trading_system.backtest.engine import is_tradeable_fill   # 唯一权威实现
不允许在本模块另写一份。函数名见 trading_system.invariants.CANONICAL_FILL_FUNC_NAME。
价格层:成交价 / 标签收益的成交侧用 raw(INV-2)。
"""

from __future__ import annotations

_PHASE = "Phase 1 任务 1.3"


def build_y_prod(*args, **kwargs):  # noqa: ANN002, ANN003
    """生产可交易标签 y_prod(raw,tau_exit>=t+2,扣成本)。"""
    raise NotImplementedError(f"{_PHASE}:build_y_prod 待实现(INV-1/INV-3)。")


def build_y_h(*args, **kwargs):  # noqa: ANN002, ANN003
    """固定窗口对照标签 y_h(h ∈ {1,2,3,5,8,10})。"""
    raise NotImplementedError(f"{_PHASE}:build_y_h 待实现。")


def build_y_mtm0(*args, **kwargs):  # noqa: ANN002, ANN003
    """诊断标签 y_mtm0(h=0;diagnostic 命名空间,禁止进训练/回测/审批)。"""
    raise NotImplementedError(f"{_PHASE}:build_y_mtm0 待实现(仅诊断命名空间)。")
