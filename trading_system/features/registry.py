"""指标注册表 + 防未来函数三检查(INV 核心)。Phase 1(任务 1.1)。对应 v3.1 第七章。

``@register`` 装饰器登记指标元信息。注册即强制三道检查(不过不进流水线):
  1) 静态扫描:禁止 ``shift(-``、``center=True``、对全序列 ``.mean()/.std()``(只许 rolling/expanding);
  2) 截断等变性:用全历史算的第 t 行 == 只喂截至 t 的数据算的第 t 行(逐位相等);
  3) 前复权陷阱拦截:注册表只向指标函数提供后复权列(adj),前复权列不可见。

``group_constant=True`` 的特征(如 L0 情绪温度 T_t)打标,供 INV-4 的 L2 装配守卫使用
(见 trading_system.invariants.assert_group_constant_only_via_interaction)。
价格层:特征一律用 adj。
"""

from __future__ import annotations

_PHASE = "Phase 1 任务 1.1"


def register(
    name: str,
    family: str,
    params: dict | None = None,
    lookback: int | None = None,
    point_in_time: bool = True,
    group_constant: bool = False,
):
    """注册装饰器(占位)。落地时:登记元信息并触发防未来函数三检查。"""

    def _decorator(fn):
        raise NotImplementedError(f"{_PHASE}:register 三检查待实现。")

    return _decorator
