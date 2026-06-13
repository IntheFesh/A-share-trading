"""trading_system.invariants — 系统宪法(INV-1 ~ INV-7)的可复用、可测试原语。

本模块属于施工任务书"一、硬约束"(实现前置,先于所有 Phase 0~4 业务逻辑)。
它把 v3.1 的七条不变量做成**纯函数 / 守卫 / 轻量数据结构**,供两道防线使用:

  1) 各业务模块在运行时 ``import`` 并 ``assert``(第一道防线,违反即 raise);
  2) ``tests/test_invariants.py`` 做成 pytest 断言(第二道防线,违反即测试失败)。

注意:本模块**只包含"不变量本身"**,不含任何 Phase 0~4 的数据 / 特征 / 引擎 / 模型
业务逻辑。价格层纪律(INV-2)贯穿全文——**执行类只用 ``*_raw``,特征类只用 ``*_adj``**。

对应 v3.1:第十章(单股上限/凯利)、第十一章(成交状态机)、第十三章(审批/盲测)、附录 B。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Iterable

logger = logging.getLogger(__name__)


class InvariantViolation(RuntimeError):
    """任一不变量被违反时抛出;``code`` 携带 INV 编号便于定位。

    业务模块应让此异常直接冒泡(不要吞掉):宪法被违反时,宁可崩溃也不要静默产出错误结果。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


# =============================================================================
# INV-1 可交易标签优先(T+1 出信号、T+1 开盘买、最早 T+2 卖)
# =============================================================================
MIN_TRADEABLE_EXIT_OFFSET: int = 2  # T 出信号 -> T+1 开盘买 -> 最早 T+2 卖

PRODUCTION_NAMESPACE: str = "production"   # 进入回测净值 / 训练目标 / 上线审批的标签
DIAGNOSTIC_NAMESPACE: str = "diagnostic"   # h=0(当日 close/open)只能活在这里


def assert_tradeable_exit(signal_day_index: int, exit_day_index: int) -> None:
    """INV-1:卖出日必须 ``>= 信号日 + 2`` 个交易日(参数为交易日历中的整数索引位置)。

    Phase: 一、硬约束(标签侧落地于 Phase 1 ``labels/``;成交侧落地于 Phase 2
    ``backtest/engine.py``)。价格层:本约束与价格层无关(纯日期约束)。
    """
    if exit_day_index < signal_day_index + MIN_TRADEABLE_EXIT_OFFSET:
        raise InvariantViolation(
            "INV-1",
            f"tau_exit 必须 >= signal + {MIN_TRADEABLE_EXIT_OFFSET} 交易日: "
            f"signal_idx={signal_day_index}, exit_idx={exit_day_index}",
        )


def assert_production_label_horizon(horizon: int) -> None:
    """INV-1:进入生产 / 训练 / 回测 / 审批的标签 horizon 必须 ``>= 1``;h=0 仅限诊断。"""
    if horizon < 1:
        raise InvariantViolation(
            "INV-1",
            f"h={horizon} 标签禁止进入生产/训练/回测/审批路径;"
            f"h=0 只能用于 '{DIAGNOSTIC_NAMESPACE}' 命名空间。",
        )


def assert_label_namespace_allows_horizon(namespace: str, horizon: int) -> None:
    """INV-1:h=0 只能出现在诊断命名空间;在其它命名空间出现 h<1 即违规。"""
    if horizon < 1 and namespace != DIAGNOSTIC_NAMESPACE:
        raise InvariantViolation(
            "INV-1",
            f"h={horizon} 出现在命名空间 '{namespace}';h<1 只允许 "
            f"'{DIAGNOSTIC_NAMESPACE}'。",
        )


# =============================================================================
# INV-2 双价格层(最重要、最易写错):执行用 raw,特征用 adj
# =============================================================================
RAW_SUFFIX: str = "_raw"
ADJ_SUFFIX: str = "_adj"

#: 执行类计算只许用这些原始价列(成交价/涨跌停价/止损止盈触发价/PnL)。
EXECUTION_FIELDS: tuple[str, ...] = (
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "preclose_raw",
)


def is_adj_column(name: str) -> bool:
    """是否为后复权列(特征类用)。"""
    return name.endswith(ADJ_SUFFIX)


def is_raw_column(name: str) -> bool:
    """是否为原始价列(执行类用)。"""
    return name.endswith(RAW_SUFFIX)


def assert_execution_uses_raw(columns: Iterable[str]) -> None:
    """INV-2 守卫:撮合 / 涨跌停 / 止损止盈 / PnL 只能用 ``*_raw`` 列;发现 ``*_adj`` 立即报错。

    Phase: 一、硬约束;落地于 Phase 2 ``backtest/engine.py`` 的价格入口(引擎接收价格即检查)。
    """
    offending = sorted({c for c in columns if is_adj_column(c)})
    if offending:
        raise InvariantViolation(
            "INV-2",
            f"执行路径(撮合/涨跌停/PnL)禁止使用后复权列: {offending};请改用 *_raw。",
        )


def _round_half_up_2(x: float) -> float:
    """按 0.01 元四舍五入(round-half-up)。

    交易所价格按分(0.01 元)、半进位取整;Python 内置 ``round`` 用银行家舍入,会在 ``.xx5``
    处与交易所偏差,故用 ``Decimal`` 显式半进位。``Decimal(str(x))`` 避免二进制浮点表示误差。
    """
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def limit_up_price(preclose_raw: float, ratio: float = 0.10) -> float:
    """INV-2:涨停价 = ``round(原始昨收 * (1 + ratio), 2)``。**必须用 raw 昨收。**

    ``ratio`` 默认 0.10(沪深主板);ST/科创/创业等特殊比例由调用方从 config 传入,禁止硬编码。
    """
    return _round_half_up_2(preclose_raw * (1.0 + ratio))


def limit_down_price(preclose_raw: float, ratio: float = 0.10) -> float:
    """INV-2:跌停价 = ``round(原始昨收 * (1 - ratio), 2)``。**必须用 raw 昨收。**"""
    return _round_half_up_2(preclose_raw * (1.0 - ratio))


# =============================================================================
# INV-3 标签—成交同源(契约;权威实现位于 backtest/engine.py,labels/ 从中 import)
# =============================================================================
#: 标签侧与引擎侧对"T+1 一字 / 高开>7% 是否可成交"的判定,必须是**同一个函数对象**。
#: 规范:权威实现位于 ``backtest/engine.py``,名为下述符号;``labels/`` 直接 import 它,
#: 不允许各写一份。Phase 1/2 落地后,tests 会强校验两侧为同一对象。
CANONICAL_FILL_FUNC_NAME: str = "is_tradeable_fill"


# =============================================================================
# INV-4 组内常数信息默认是覆盖层(L0 情绪温度 T_t 等"当天同值"量)
# =============================================================================
@dataclass(frozen=True)
class FeatureSpec:
    """L2 特征矩阵中每一列的最小元信息(完整注册表见 Phase 1 ``features/registry.py``)。"""

    name: str
    group_constant: bool = False  # 当天所有股票同值(如 L0 情绪温度 T_t、披露季强度)
    is_interaction: bool = False  # 是否为显式交互项(如 T_t * stock_feature)


def assert_group_constant_only_via_interaction(specs: Iterable[FeatureSpec]) -> None:
    """INV-4:``group_constant=True`` 的特征只能以**显式交互项**进入 L2;裸列进入即报错。

    Phase: 一、硬约束;落地于 Phase 3 ``model/`` 的 L2 数据装配守卫。组内常数特征进 L2 的
    唯一合法形式是 ``T_t * stock_feature``,且须经 §5.2 身份二消融检验(ΔRankIC>0 且
    ΔMaxDD≤0)方可保留——本守卫只拦"未经交互"的更基础违规。
    """
    offending = [s.name for s in specs if s.group_constant and not s.is_interaction]
    if offending:
        raise InvariantViolation(
            "INV-4",
            f"组内常数特征未经交互直接进入 L2: {offending};"
            "只能以 T_t*stock_feature 形式进入。",
        )


# =============================================================================
# INV-5 单股上限由连续跌停压力决定(禁止 15% 默认值)
# =============================================================================
#: v3.1 明令禁止出现的默认上限(15%)。具体档位(主板≤8%、特殊≤5%)从 config/risk.yaml 读。
FORBIDDEN_DEFAULT_CAP: float = 0.15


def continuous_limit_down_loss(w: float, k: int, limit_ratio: float = 0.10) -> float:
    """INV-5:连续 K 天跌停的组合损失 ``L_K(w) = w * (1 - (1 - limit_ratio) ** K)``。

    复核 v3.1 附录 B:``L_2(0.08) ≈ 0.0152``、``L_2(0.05) ≈ 0.0095``(``limit_ratio=0.10``)。
    """
    return w * (1.0 - (1.0 - limit_ratio) ** k)


def single_name_cap(w_hard: float, l_tail: float, g_hat: float) -> float:
    """INV-5:单股上限 ``w_max = min(w_hard, L_tail / g_hat)``,其中 ``g_hat > 0``。"""
    if g_hat <= 0:
        raise InvariantViolation("INV-5", f"g_hat 必须 > 0,收到 {g_hat}。")
    return min(w_hard, l_tail / g_hat)


def assert_hard_cap_allowed(w_hard: float) -> None:
    """INV-5:硬上限不得达到 / 超过被禁止的 15% 默认值。具体档位从 ``config/risk.yaml`` 读。"""
    if w_hard >= FORBIDDEN_DEFAULT_CAP:
        raise InvariantViolation(
            "INV-5",
            f"硬上限 {w_hard:.3f} >= 禁止值 {FORBIDDEN_DEFAULT_CAP};主板应≤8%、特殊应≤5%。",
        )


# =============================================================================
# INV-6 盲测段一次性(用于 champion-challenger 换届裁决后即封存)
# =============================================================================
class BlindSegmentStatus(str, Enum):
    """盲测段状态。"""

    UNUSED = "unused"
    ARCHIVED = "archived"  # 已用于换届裁决,封存;再用于调参/选择即违规


@dataclass
class BlindSegmentLedger:
    """盲测段一次性账本(轻量**内存**版)。

    Phase 3 的持久化(SQLite)版本见 ``audit/experiment_registry.py``,本类是其纯逻辑内核,
    便于在不依赖落库的前提下做单元测试。规则:某盲测段一旦用于 champion-challenger 换届
    裁决 → ``ARCHIVED``;再次用于调参 / 选择 / 裁决 → 报错。
    """

    _status: dict[str, BlindSegmentStatus] = field(default_factory=dict)

    def status(self, segment_id: str) -> BlindSegmentStatus:
        return self._status.get(segment_id, BlindSegmentStatus.UNUSED)

    def assert_available(self, segment_id: str) -> None:
        """若该盲测段已封存,则报错。"""
        if self.status(segment_id) is BlindSegmentStatus.ARCHIVED:
            raise InvariantViolation(
                "INV-6",
                f"盲测段 '{segment_id}' 已封存(用过一次),禁止再次用于裁决/调参/选择。",
            )

    def use_for_decision(self, segment_id: str) -> None:
        """用于一次 champion-challenger 换届裁决,并立即封存。"""
        self.assert_available(segment_id)
        self._status[segment_id] = BlindSegmentStatus.ARCHIVED


# =============================================================================
# INV-7 条件化优先于无条件叠加(惩罚/增强信号默认以 regime 交互进入)
# =============================================================================
def assert_conditional_or_documented_override(
    *,
    is_unconditional: bool,
    unconditional_override: bool = False,
    justification: str | None = None,
) -> None:
    """INV-7:惩罚 / 增强型信号默认以 regime 交互形式进入。

    若以无条件方式(直接加减打分)叠加,必须显式 ``unconditional_override=True`` 且给出
    "已通过不劣于条件化验证"的 ``justification``,否则拦截。

    Phase: 一、硬约束;落地于 Phase 2 ``overlays/`` 与 Phase 3 ``model/``。
    """
    if is_unconditional and not unconditional_override:
        raise InvariantViolation(
            "INV-7",
            "无条件叠加惩罚/增强信号被拦截;请改为 regime 交互形式,"
            "或显式 unconditional_override=True 并附'不劣于条件化'的验证说明。",
        )
    if is_unconditional and unconditional_override and not (justification and justification.strip()):
        raise InvariantViolation(
            "INV-7",
            "unconditional_override=True 必须附带'已通过不劣于条件化验证'的说明 justification。",
        )
