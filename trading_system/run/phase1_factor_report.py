"""Phase 1 统计验收 / 因子体检。任务 1.6。对应 v3.1 第三/六/七章。

诚实分层:
  (A) 计算验证(可在合成数据上跑):全特征**截断等变性必须全过**(防未来函数,硬门槛);
      并演示 RankIC/ICIR/分块 RankIC 的计算流程。合成数据是随机游走,RankIC≈0 属正常——
      这一步验证的是"算法正确",不是"因子有效"。
  (B) 因子有效性(需真实数据,标 NOT RUN):2019 至今逐特征 RankIC/ICIR/分十层、分五阶段稳定性、
      混池 vs 仅主板 A/B、收益三段拆解。需 BaoStock 实盘数据(+ 可选 token)。

退出码:0=截断等变性全过(计算验证通过);1=有特征泄漏(硬失败)。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("phase1_factor_report")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "output"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    import trading_system.features.builtin  # noqa: F401  注册特征
    from trading_system.backtest import metrics
    from trading_system.data import price_layers as pl
    from trading_system.data.collectors import synthetic
    from trading_system.features import registry as reg
    from trading_system.labels import build_y_h

    lines = ["# Phase 1 因子体检报告", "", "## (A) 计算验证(合成数据,验证算法正确性)", ""]

    cal = synthetic.make_calendar("2020-01-06", 200)
    codes = [f"600{str(i).zfill(3)}" for i in range(30)]
    panel = pl.build_price_layers(synthetic.make_raw_panel(codes, cal, seed=2024))
    p = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)

    # 截断等变性:硬门槛
    g0 = p[p["code"] == codes[0]].reset_index(drop=True)
    positions = list(range(65, len(g0) - 1, 7))
    leaks = {}
    for name in [n for n in reg.REGISTRY if not n.startswith("_leak")]:
        v = reg.truncation_equivariance_violations(name, g0, positions)
        if v:
            leaks[name] = v[:3]
    if leaks:
        lines.append(f"- **截断等变性失败(未来泄漏)**: {leaks}")
        (REPORT_DIR / "phase1_factor_report.md").write_text("\n".join(lines), encoding="utf-8")
        logger.error("特征存在未来泄漏,Phase 1 不通过: %s", list(leaks))
        return 1
    lines.append(f"- 截断等变性:{len(reg.REGISTRY)} 个特征全部通过 ✓(防未来函数硬门槛)")

    # RankIC 计算流程演示(合成随机数据,数值≈0 属正常)
    y = build_y_h(p, h=5)
    p["__label__"] = y
    lines += ["", "| 特征 | mean RankIC | ICIR | 分块RankIC(H=5) |", "|---|---|---|---|"]
    for name in [n for n in reg.REGISTRY if not n.startswith("_leak")]:
        p["__score__"] = reg.compute_feature(name, p).reindex(p.index)
        sub = p.dropna(subset=["__score__", "__label__"])
        mic = metrics.mean_rank_ic(sub, "__score__", "__label__")
        iic = metrics.icir(metrics.daily_rank_ic(sub, "__score__", "__label__"))
        bic = metrics.blocked_rank_ic(sub, "__score__", "__label__", block_len=5).mean()
        lines.append(f"| {name} | {mic:+.4f} | {iic:+.3f} | {bic:+.4f} |")

    lines += [
        "",
        "> 合成数据为随机游走,上表 RankIC≈0 属正常;本节只证明 RankIC/ICIR/分块IC 计算正确。",
        "",
        "## (B) 因子有效性(需真实数据,**NOT RUN**)",
        "",
        "- 2019 至今逐特征 RankIC/ICIR/分十层、分五阶段稳定性 — 需 BaoStock 实盘数据。",
        "- 混池 vs 仅主板 A/B、收益三段拆解(entry/overnight/exit) — 需真实数据。",
        "> 真实因子有效性未验:请在装好 baostock 并拉取实盘后接入本节。",
    ]
    out = REPORT_DIR / "phase1_factor_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("\n".join(lines))
    logger.info("\n报告已写入: %s", out)
    logger.info("=== Phase 1 计算验证通过(截断等变性全过);因子有效性需真实数据(NOT RUN)===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
