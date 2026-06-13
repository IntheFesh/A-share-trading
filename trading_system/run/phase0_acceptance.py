"""Phase 0 验收脚本。任务 0.7。对应 v3.1 §2 / 第十五章。

两部分,诚实分开:
  (A) 合成流水线自检——用已知性质的合成数据端到端验证 calendar→price_layers→universe→
      store→quality 的**逻辑正确性**(不依赖网络/token)。失败 -> 退出 1。
  (B) 实盘市场验收——任务 0.7 的真实口径:20 除权事件 raw 跳变/adj 连续、20 退市股历史完整、
      20 涨跌停样本 round(preclose_raw*1.1,2) 逐笔一致、披露日历无前视。需 BaoStock 网络
      + TUSHARE_TOKEN。**数据/token 缺失时本部分标 NOT RUN,Phase 0 视为"尚未完整验收"**
      (退出 3),绝不因合成自检通过就宣称 Phase 0 通过。

退出码:0=合成自检 + 实盘验收均通过;1=合成自检失败;3=合成自检通过但实盘验收未跑。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("phase0_acceptance")

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "output"


def _synthetic_self_test() -> "list[str]":
    """跑合成流水线,返回 Markdown 报告行;断言失败会抛异常。"""
    import pandas as pd

    from trading_system.data import price_layers as pl
    from trading_system.data import quality as q
    from trading_system.data import universe as uni
    from trading_system.data.collectors import synthetic
    from trading_system.data.store import ParquetStore

    lines = ["## (A) 合成流水线自检(逻辑验证,非市场实证)", ""]

    cal = synthetic.make_calendar("2020-01-06", 80)
    raw = synthetic.make_raw_panel(["600000", "600001", "000001"], cal, seed=42)
    panel = pl.build_price_layers(raw)
    assert {"close_raw", "close_adj", "is_limit_up"}.issubset(panel.columns)
    lines.append(f"- 双价格层构造:{len(panel)} 行,raw/adj 双层 + 状态位齐备 ✓")

    universe = uni.filter_universe(panel, new_listing_min_days=60)
    n_in = int(universe["is_in_universe"].sum())
    lines.append(f"- 交易池过滤:in-universe {n_in} 行(前 60 交易日次新已剔除)✓")

    store = ParquetStore(Path(__file__).resolve().parents[2] / "data_store" / "_phase0_tmp_store")
    store.write(panel)
    back = store.read(fields=["close_raw"])
    assert "close_adj" not in back.columns and "close_raw" in back.columns
    lines.append("- 存储出口 raw/adj 分离(read(fields=raw) 无 adj 列)✓ [INV-2]")

    results = q.run_daily_quality_checks(panel)
    q.assert_passed(results)  # 任一 FAIL 抛错
    lines.append("- 数据质检:")
    for r in results:
        lines.append(f"    - {r.check}: **{r.status}** ({r.n_flagged}) — {r.detail}")
    lines.append("")
    return lines


def _live_acceptance() -> "tuple[bool, list[str]]":
    """实盘验收。返回 (是否真正跑过并通过, 报告行)。数据/token 缺失则 (False, 说明)。"""
    lines = ["## (B) 实盘市场验收(任务 0.7 真实口径)", ""]
    have_baostock = importlib.util.find_spec("baostock") is not None
    have_token = bool(os.environ.get("TUSHARE_TOKEN"))
    if not have_baostock:
        lines.append("- **NOT RUN**:未安装 baostock,无法拉实盘日线/日历。")
    if not have_token:
        lines.append("- **NOT RUN**:无 TUSHARE_TOKEN,披露日历/预告/退市 token-gated。")
    if not (have_baostock and have_token):
        lines += [
            "",
            "> 实盘验收未执行:**Phase 0 尚未完整验收**。需要:",
            "> 1) `pip install baostock` 且容器可访问 BaoStock;",
            "> 2) 设置环境变量 `TUSHARE_TOKEN`;",
            "> 然后重跑本脚本完成 20 除权 / 20 退市 / 20 涨跌停逐笔核验与披露日历无前视抽检。",
        ]
        return False, lines
    # 数据与 token 就位时,在此实现真实的 20+20+20 逐笔核验(后续接入)。
    lines.append("- 数据与 token 就位:实盘逐笔核验逻辑待接入(占位,不谎报通过)。")
    return False, lines


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Phase 0 验收报告", ""]

    try:
        report += _synthetic_self_test()
    except Exception as exc:  # noqa: BLE001 — 验收要如实暴露失败
        logger.error("合成流水线自检失败: %r", exc)
        report.append(f"- **合成自检失败**: `{exc!r}`")
        (REPORT_DIR / "phase0_acceptance.md").write_text("\n".join(report), encoding="utf-8")
        return 1

    live_ok, live_lines = _live_acceptance()
    report += live_lines
    out = REPORT_DIR / "phase0_acceptance.md"
    out.write_text("\n".join(report), encoding="utf-8")
    logger.info("\n".join(report))
    logger.info("\n报告已写入: %s", out)

    if live_ok:
        logger.info("=== Phase 0 完整验收通过 ===")
        return 0
    logger.info("=== 合成自检通过;实盘验收未跑 -> Phase 0 尚未完整验收(需数据/token)===")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
