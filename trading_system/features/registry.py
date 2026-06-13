"""指标注册表 + 防未来函数三检查(INV 核心)。Phase 1(任务 1.1)。对应 v3.1 §7.3。

``@register`` 登记指标元信息并即时跑**检查一(静态扫描)**。另两道检查供测试 / 流水线调用:
  检查一 静态扫描:禁 ``shift(-``(负向 shift=未来)、``center=True``(居中窗含未来);
  检查二 截断等变性:用全历史算的第 t 行 == 只喂截至 t 的数据算的第 t 行(逐位相等);
  检查三 前复权陷阱拦截:指标函数只看得到**后复权价 + 量额 + 状态位**,看不到原始价/前复权。

特征一律用后复权(adj)价(INV-2 特征侧);group_constant=True 的特征(如 T_t)打标,
供 INV-4 的 L2 装配守卫使用。每个特征:先按 code 算时序原值,再每日截面 winsorize+秩变换。
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from trading_system.data.schema import (
    ADJ_PRICE_FIELDS,
    FEATURE_EXTRA_FIELDS,
    KEY_FIELDS,
    STATE_FIELDS,
    VOLUME_FIELDS,
)

#: 特征函数只能看到的列:后复权价 + 量额 + 换手 + 状态位 + key。绝不含原始价(*_raw)与前复权 —— 检查三。
ALLOWED_FEATURE_COLUMNS: set[str] = (
    set(KEY_FIELDS) | set(ADJ_PRICE_FIELDS) | set(VOLUME_FIELDS)
    | set(FEATURE_EXTRA_FIELDS) | set(STATE_FIELDS)
)

#: 检查一:静态扫描的禁用模式。
FORBIDDEN_SOURCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\.shift\(\s*-"), "负向 shift(引用未来数据)"),
    (re.compile(r"center\s*=\s*True"), "center=True(居中窗口含未来)"),
]


class FutureLeakError(RuntimeError):
    """检测到未来函数泄漏(静态扫描或截断等变性)。"""


@dataclass
class FeatureDef:
    name: str
    family: str
    fn: Callable[[pd.DataFrame], pd.Series]
    params: dict
    lookback: Optional[int]
    point_in_time: bool
    group_constant: bool


REGISTRY: dict[str, FeatureDef] = {}


def static_scan(fn: Callable) -> None:
    """检查一:对函数源码做静态扫描;命中禁用模式即抛 FutureLeakError。

    取不到源码(如内置/C 实现)则跳过本检查——截断等变性(检查二)仍兜底。
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return
    for pat, why in FORBIDDEN_SOURCE_PATTERNS:
        if pat.search(src):
            raise FutureLeakError(f"特征 {fn.__name__} 静态扫描命中:{why}")


def register(
    name: str,
    family: str,
    *,
    params: "dict | None" = None,
    lookback: "int | None" = None,
    point_in_time: bool = True,
    group_constant: bool = False,
):
    """注册装饰器:登记元信息并即时跑静态扫描(检查一)。重复注册同名报错。"""

    def _decorator(fn: Callable[[pd.DataFrame], pd.Series]):
        static_scan(fn)
        if name in REGISTRY:
            raise ValueError(f"特征名重复注册: {name}")
        REGISTRY[name] = FeatureDef(
            name=name,
            family=family,
            fn=fn,
            params=params or {},
            lookback=lookback,
            point_in_time=point_in_time,
            group_constant=group_constant,
        )
        return fn

    return _decorator


def _restricted(g: pd.DataFrame) -> pd.DataFrame:
    """检查三:只把允许列喂给特征函数(原始价/前复权不可见)。"""
    cols = [c for c in g.columns if c in ALLOWED_FEATURE_COLUMNS]
    return g[cols]


def compute_raw_feature(name: str, panel: pd.DataFrame) -> pd.Series:
    """按 code 分组计算某特征的时序原值(未做截面变换)。返回与 panel(排序后)对齐的 Series。"""
    if name not in REGISTRY:
        raise KeyError(f"未注册的特征: {name}")
    fd = REGISTRY[name]
    p = panel.sort_values(["code", "trade_date"]).copy()
    result = pd.Series(np.nan, index=p.index, dtype="float64")
    for _, g in p.groupby("code", sort=False):
        vals = fd.fn(_restricted(g))
        result.loc[g.index] = np.asarray(vals, dtype="float64")
    return result


def cross_sectional_rank(panel: pd.DataFrame, col: str, winsor: float = 0.01) -> pd.Series:
    """每个交易日内:先 winsorize(双侧 winsor 分位)再秩变换到 [0,1]。无未来(同日截面)。"""

    def _tx(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(winsor), s.quantile(1.0 - winsor)
        return s.clip(lower=lo, upper=hi).rank(pct=True)

    return panel.groupby("trade_date")[col].transform(_tx)


def compute_feature(name: str, panel: pd.DataFrame, *, transform: bool = True) -> pd.Series:
    """计算特征:时序原值 -> (可选)每日截面 winsorize + 秩变换。"""
    p = panel.sort_values(["code", "trade_date"]).copy()
    p[f"__raw__{name}"] = compute_raw_feature(name, p)
    if not transform:
        return p[f"__raw__{name}"]
    return cross_sectional_rank(p, f"__raw__{name}")


def truncation_equivariance_violations(
    name: str, g: pd.DataFrame, sample_positions: "list[int]", tol: float = 1e-12
) -> "list[tuple[int, float, float]]":
    """检查二:对单只票时序 g,逐个位置 t 比较"全历史算的第 t 行" vs "截至 t 算的最后一行"。

    返回违规列表 [(t, full_value, truncated_value)];空列表表示通过(无未来泄漏)。
    """
    if name not in REGISTRY:
        raise KeyError(f"未注册的特征: {name}")
    fn = REGISTRY[name].fn
    g = g.sort_values("trade_date").reset_index(drop=True)
    full = np.asarray(fn(_restricted(g)), dtype="float64")
    viol: list[tuple[int, float, float]] = []
    for t in sample_positions:
        sub = np.asarray(fn(_restricted(g.iloc[: t + 1])), dtype="float64")
        a, b = full[t], sub[-1]
        same = (np.isnan(a) and np.isnan(b)) or (abs(a - b) <= tol)
        if not same:
            viol.append((t, float(a), float(b)))
    return viol
