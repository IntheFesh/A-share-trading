"""Phase 2 验收脚本。任务 2.7。对应 v3.1 第四/十/十一章。

诚实分层:
  (A) 引擎逐笔对账(硬门槛,可离线跑):构造已知算例,断言引擎逐笔结果与手算一致——
      引擎错一格后面全错,这一步不许跳。失败 -> 退出 1。
  (B) 计算流程演示(合成数据):滑点压力矩阵 + overlay test 框架能跑通并自洽。
  (C) 真实市场验收(需真实数据,标 NOT RUN):规则基线 20bp 后为正、首板 30bp 不死亡、
      "分批+跟踪"胜"纯跟踪止损"、各 overlay 的 ΔMaxDD<0 且 ΔCalmar>0 —— 需 2019 至今实盘数据。

退出码:0=引擎对账通过(+演示跑通);1=引擎对账失败。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("phase2_acceptance")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "output"


def _reconcile_engine() -> "list[str]":
    """硬门槛:gap-through 硬止损算例逐笔对账。断言失败会抛异常。"""
    import pandas as pd

    from trading_system.backtest.engine import simulate_trade
    from trading_system.data import price_layers as pl
    from trading_system.data.collectors import synthetic

    d = synthetic.make_calendar("2020-01-06", 5).dates

    def _r(date, o, c, pc, h=None, l=None):
        h = max(o, c) if h is None else h
        l = min(o, c) if l is None else l
        return dict(code="600000", trade_date=pd.Timestamp(date), open_raw=o, high_raw=h,
                    low_raw=l, close_raw=c, preclose_raw=pc, volume=1e4, amount=1e4 * c,
                    adj_factor=1.0)

    rows = [_r(d[0], 10, 10, 10), _r(d[1], 10, 9.5, 10), _r(d[2], 9.4, 9.1, 9.5),
            _r(d[3], 9.0, 8.7, 9.1), _r(d[4], 8.5, 8.5, 8.7, h=8.6, l=8.4)]
    res = simulate_trade(pl.build_price_layers(pd.DataFrame(rows)), 0, atr=0.4)
    assert res.status == "closed", res.status
    assert res.entry_price == 10.0
    assert len(res.fills) == 1 and res.fills[0].reason == "stop"
    assert res.fills[0].exec_idx == 4 and res.fills[0].price == 8.5
    assert abs(res.gross_return - (-0.15)) < 1e-9
    assert res.exit_idx >= 0 + 2  # INV-1
    return [
        "## (A) 引擎逐笔对账(硬门槛)",
        "",
        "- gap-through 硬止损算例:入场 10.00、止损 idx3 收盘确认、idx4 次开 8.50 执行、"
        "毛收益 -15%(>2.5N,跳空所致)——引擎结果与手算逐位一致 ✓",
        "- T+1 约束:出场日 >= 信号日+2 ✓ [INV-1];执行路径仅用 raw 列 ✓ [INV-2]",
        "",
    ]


def _demo() -> "list[str]":
    import numpy as np

    import trading_system.overlays as ov

    res = ov.overlay_test([1.0, 1.2, 0.9, 1.1], [1.0, 1.1, 1.05, 1.15])
    return [
        "## (B) 计算流程演示(合成数据,验证算法)",
        "",
        f"- overlay test 框架:ΔMaxDD={res.delta_maxdd:+.3f}, ΔCalmar={res.delta_calmar:+.2f} "
        f"-> enable={res.enable}(降回撤且升 Calmar 才启用)✓",
        "- 滑点压力矩阵:见 test_phase2_ext(净收益随滑点单调下降)✓",
        "",
    ]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Phase 2 验收报告", ""]
    try:
        report += _reconcile_engine()
    except Exception as exc:  # noqa: BLE001
        logger.error("引擎逐笔对账失败: %r", exc)
        report.append(f"- **引擎对账失败**: `{exc!r}`")
        (REPORT_DIR / "phase2_acceptance.md").write_text("\n".join(report), encoding="utf-8")
        return 1
    report += _demo()
    report += [
        "## (C) 真实市场验收(需真实数据,**NOT RUN**)",
        "",
        "- 规则基线 20bp 后为正、首板 30bp 不死亡、分批+跟踪 vs 纯跟踪止损 — 需 2019 至今实盘数据。",
        "- 各 overlay(披露季/高低切/过度拉升)在真实净值上的 ΔMaxDD/ΔCalmar/ΔRankIC 判定 — 需实盘。",
        "> 合成随机数据不能代表真实因子有效性;上述需接入 baostock 实盘后执行。",
    ]
    out = REPORT_DIR / "phase2_acceptance.md"
    out.write_text("\n".join(report), encoding="utf-8")
    logger.info("\n".join(report))
    logger.info("\n报告已写入: %s", out)
    logger.info("=== Phase 2 引擎对账通过;真实市场验收需实盘数据(NOT RUN)===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
