"""回测三纪律(run_backtest 的护栏)。补丁。对应 v3.1 第九/十三章。

纪律一·盲测段物理隔离 + 一次性:调参只允许在盲测段之前;触碰盲测段需显式放行,且重复使用报警(INV-6)。
纪律二·三数并排:名义收益、扣费后收益、PBO 必须同时输出,禁止只报名义收益。
纪律三·粗桶:参数组合总数不得超过上限,超过报错(禁止精细网格)。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


# ── 纪律三:粗桶上限 ──────────────────────────────────────────────────────────
def count_param_combos(param_grid: dict) -> int:
    """参数网格的组合总数 = 各维取值个数之积。"""
    n = 1
    for v in param_grid.values():
        n *= max(1, len(v))
    return n


def check_param_grid(param_grid: dict, max_combos: int) -> int:
    """组合数超过 max_combos 即报错(禁止精细网格)。返回组合数。"""
    n = count_param_combos(param_grid)
    if n > max_combos:
        raise ValueError(
            f"参数组合数 {n} 超过上限 {max_combos};只允许粗桶,禁止精细网格寻优。"
        )
    return n


# ── 纪律二:三数并排 ──────────────────────────────────────────────────────────
def assemble_backtest_report(
    *, nominal_return: float, net_return: float, block_perf, pbo_warn_threshold: float = 0.30
) -> dict:
    """打包回测必须并排输出的三个数:名义收益 / 扣费后收益 / PBO,并对 PBO 过高给出警告。"""
    from trading_system.backtest.metrics import pbo_cscv

    pbo = pbo_cscv(block_perf)
    return {
        "nominal_return": float(nominal_return),
        "net_return": float(net_return),
        "pbo": float(pbo),
        "pbo_warning": bool(pbo > pbo_warn_threshold),
    }


# ── 纪律一:盲测段物理隔离 + 一次性 ───────────────────────────────────────────
def region_overlaps_blind(start, end, blind_segment_start) -> bool:
    """回测区间是否触碰盲测段(end >= 盲测段起点)。"""
    return pd.Timestamp(end) >= pd.Timestamp(blind_segment_start)


class BlindUsageLedger:
    """盲测段消耗账本(落盘 JSON):记录每次盲测段被用于调参的时间戳与参数。"""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)

    def _read(self) -> list:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def usage_count(self, segment_id: str) -> int:
        return sum(1 for r in self._read() if r.get("segment_id") == segment_id)

    def was_used(self, segment_id: str) -> bool:
        return self.usage_count(segment_id) > 0

    def record_use(self, segment_id: str, params: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        recs = self._read()
        recs.append({"segment_id": segment_id, "params": params,
                     "ts": datetime.now().isoformat(timespec="seconds")})
        self.path.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")


def precheck_backtest(
    *, start, end, param_grid: dict, config: dict, use_blind_once: bool = False,
    ledger: "BlindUsageLedger | None" = None, segment_id: str = "blind",
) -> dict:
    """回测前三纪律预检。返回 {combos, overlaps_blind, warning};违纪则报错/拒绝。"""
    combos = check_param_grid(param_grid, int(config["backtest"]["max_param_combos"]))  # 纪律三
    overlaps = region_overlaps_blind(start, end, config["splits"]["blind_segment_start"])
    warning = None
    if overlaps and not use_blind_once:  # 纪律一:默认拒绝触碰盲测段
        raise ValueError(
            "回测区间触碰盲测段;调参禁止使用盲测段。如确需一次性盲测,显式传 --use-blind-once。"
        )
    if overlaps and use_blind_once and ledger is not None:
        if ledger.was_used(segment_id):  # 纪律一:重复使用 -> INV-6 警告
            warning = f"INV-6 警告:盲测段 '{segment_id}' 已被用于调参,正在重复使用。"
        ledger.record_use(segment_id, param_grid)
    return {"combos": combos, "overlaps_blind": overlaps, "warning": warning}
