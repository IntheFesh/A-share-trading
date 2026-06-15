"""Phase 1 统计验收 / 因子体检。任务 1.6。对应 v3.1 第三/六/七章。

诚实分层:
  (A) 计算验证(可在合成数据上跑):全特征**截断等变性必须全过**(防未来函数,硬门槛);
      并演示 RankIC/ICIR/分块 RankIC 的计算流程。合成数据是随机游走,RankIC≈0 属正常——
      这一步验证的是"算法正确",不是"因子有效"。
  (B) 因子有效性(真实数据单因子诊断):从 data_store 读真实数据,对 config.features 逐因子算
      mean RankIC / ICIR / 分块RankIC(块长=持有期 H,避免标签重叠致 ICIR 虚高);data_store 为空则
      优雅 NOT RUN。仅诊断单因子预测力,不代表组合可实现收益(见回测)。可设 train_end_for_report
      规避在盲测段上看因子(防数据窥探)。

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
    ]

    # ── (B) 因子有效性:真实数据单因子诊断(data_store 为空则优雅 NOT RUN)──
    from trading_system.config import load_config
    from trading_system.data.store import ParquetStore

    cfg = load_config()
    fr = cfg.get("factor_report", {}) or {}
    H = int(fr.get("label_horizon", 5))
    train_end = fr.get("train_end_for_report")
    feats = [f for f in cfg.get("features", []) if f in reg.REGISTRY]

    lines += [
        "## (B) 因子有效性(真实数据单因子诊断)", "",
        "> 本节是**因子层面的单因子有效性诊断**(每个因子各自有无预测力),不代表组合策略的可实现收益;",
        "> 真实可实现收益见回测(含 T+1 成交、成本、出场状态机)。", "",
    ]
    try:
        data = ParquetStore(cfg["paths"]["data_dir"]).read()
    except Exception as exc:  # noqa: BLE001 — 无数据/读取异常一律优雅回退
        logger.warning("读取 data_store 失败,(B) 段回退 NOT RUN: %r", exc)
        data = pd.DataFrame()

    if data.empty:
        lines += [
            "- **NOT RUN**:data_store 为空(无真实数据)。请先用 `run_fetch_data.py` 拉取实盘行情后重跑;",
            "  届时将对 config.features 逐因子输出真实 mean RankIC / ICIR / 分块 RankIC(H)。",
        ]
    else:
        rp = data.sort_values(["code", "trade_date"]).reset_index(drop=True)
        if train_end:
            rp = rp[pd.to_datetime(rp["trade_date"]) <= pd.Timestamp(train_end)].reset_index(drop=True)
        dmin = pd.to_datetime(rp["trade_date"]).min().date()
        dmax = pd.to_datetime(rp["trade_date"]).max().date()
        blind = cfg.get("splits", {}).get("blind_segment_start")
        lines += [
            f"- 本报告使用的数据区间:{dmin} ~ {dmax};{rp['code'].nunique()} 只票、{len(rp)} 行;"
            f"标签持有期 H={H}。",
            f"- train_end_for_report = {train_end!r}(null = 用全部数据)。",
        ]
        if train_end is None and blind and pd.Timestamp(dmax) >= pd.Timestamp(blind):
            lines.append(
                f"- ⚠ 数据窥探提醒:本报告数据已进入盲测段(自 {blind} 起)。反复查看盲测段内因子表现并据此"
                "选因子,等于污染盲测段(变相过拟合);如需规避,把 train_end_for_report 设为盲测段起点之前。")
        lines += ["", "| 因子 | mean RankIC | ICIR | 分块RankIC(H) |", "|---|---|---|---|"]
        rp["__label__"] = build_y_h(rp, H)
        for name in feats:
            rp["__score__"] = reg.compute_feature(name, rp).reindex(rp.index)
            sub = rp.dropna(subset=["__score__", "__label__"])
            mic = metrics.mean_rank_ic(sub, "__score__", "__label__")
            iic = metrics.icir(metrics.daily_rank_ic(sub, "__score__", "__label__"))
            bic = metrics.blocked_rank_ic(sub, "__score__", "__label__", block_len=H).mean()
            lines.append(f"| {name} | {mic:+.4f} | {iic:+.3f} | {bic:+.4f} |")

    out = REPORT_DIR / "phase1_factor_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("\n".join(lines))
    logger.info("\n报告已写入: %s", out)
    logger.info("=== Phase 1:截断等变性硬门槛通过(A);(B) 因子诊断见报告"
                "(有真实数据则已计算,data_store 为空则 NOT RUN)===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
