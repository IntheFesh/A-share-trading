"""Tushare 财报采集器(**仅财报**:业绩预告 + 预约披露日;软依赖,可降级)。补丁:网络路径 + 退避重试 + 失败抛 TushareError。

架构纪律(用户约束):Tushare 不作信息来源,只用于财报获取;行情/退市/日历一律 BaoStock。

任何 Tushare 异常(网络/token 失效/接口变更/限流)都封装为 TushareError 抛给上层;
上层(fetch_training_data)捕获后**降级置空**披露字段、主流程照常产出行情数据集——
绝不让 Tushare 异常冒泡导致行情数据丢失。token-gated,真实网络路径 NOT RUN。
PIT 纪律:产出按公告日对齐的 sched_disclosure / preann 两张表,交 price_layers.attach_disclosure_fields。
依赖注入:disclosure_fn 可注入,便于离线 mock(成功/抛错两条路径都可测)。
"""

from __future__ import annotations

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


class TushareError(RuntimeError):
    """Tushare 采集失败(网络/token/限流/接口变更),触发上层降级。"""


def _default_disclosure_fn(token: str, codes, start: str, end: str):
    """默认真实实现(NOT RUN,需 token + 网络;列映射须按实时 API 核实)。

    返回 (sched_df[code, sched_disclosure_date], preann_df[code, ann_date, preann_sign])。
    """
    from trading_system.data.collectors import tushare as ts_api  # 复用低层封装

    # 说明:Tushare disclosure_date/forecast 为按 period 维度;真实落地时需按 code 聚合并核对列名。
    # 此处给出最小骨架,真实运行前必须以实时返回核验字段映射(故标 NOT RUN,不臆造列)。
    period = end.replace("-", "")[:8]
    _ = ts_api.fetch_disclosure_date(period, config_token=token)  # 触发真实调用
    _ = ts_api.fetch_forecast(period, config_token=token)
    raise TushareError(
        "默认 Tushare 列映射需对实时 API 核验后落地(NOT RUN);"
        "离线/未核验环境一律降级置空。请注入已核验的 disclosure_fn 或在 Phase 0 核对后实现。"
    )


class TushareCollector:
    """Tushare 披露采集编排(限流退避重试 → 仍失败则抛 TushareError)。"""

    def __init__(self, token: str, *, disclosure_fn=None, max_retries: int = 2,
                 sleep_sec: float = 1.0) -> None:
        if not token:
            raise TushareError("Tushare token 缺失")
        self.token = token
        self.disclosure_fn = disclosure_fn or _default_disclosure_fn
        self.max_retries = max_retries
        self.sleep_sec = sleep_sec

    def fetch_disclosure(
        self, codes: "list[str]", start: str, end: str
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """拉披露日历 + 业绩预告;退避重试 ≤max_retries;最终失败抛 TushareError。"""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                sched, preann = self.disclosure_fn(self.token, codes, start, end)
                return sched, preann
            except TushareError:
                raise
            except Exception as e:  # noqa: BLE001 — 网络/限流等 → 退避重试
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.sleep_sec)
        raise TushareError(f"Tushare 披露采集失败(重试 {self.max_retries} 次): {last_err!r}")
