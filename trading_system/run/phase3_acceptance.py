"""Phase 3 验收脚本。任务 3.5。对应 v3.1 第七/八/九/十三章。

诚实分层:
  (A) 计算验证(可离线跑,硬门槛):purged/embargo 划分自洽(无重叠、间隔=embargo);
      LightGBM ranker 在合成信号上学到正 IC;PBO/DSR 计算自洽;审批五重 AND + 盲测段一次性(INV-6)。
  (B) 真实市场验收(需真实数据,标 NOT RUN):2019 至今逐月 walk-forward、复杂模型胜四基线、
      DSR>0.95 且 PBO<30%、单次盲测、10日 vs 月度节奏、分位映射 vs HMM 概率对照。

退出码:0=计算验证通过;1=失败。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("phase3_acceptance")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "output"


def _computation_checks() -> "list[str]":
    import numpy as np
    import pandas as pd

    from trading_system.backtest import metrics
    from trading_system.invariants import BlindSegmentLedger, FeatureSpec, InvariantViolation
    from trading_system.model import approval, cv, train

    lines = ["## (A) 计算验证(合成数据,硬门槛)", ""]

    # purged/embargo
    splits = cv.purged_walk_forward_splits(list(range(120)), n_splits=4,
                                           embargo=cv.embargo_from_config(10, 2))
    for tr, va in splits:
        assert set(tr).isdisjoint(set(va)) and min(va) - max(tr) - 1 == 13
    lines.append(f"- purged/embargo:{len(splits)} 折,间隔恰为 13 个交易日、训练验证无重叠 ✓ [embargo=H_max+K+1]")

    # LightGBM ranker 学信号
    rng = np.random.default_rng(0)
    frames = []
    for dd in pd.date_range("2020-01-06", periods=20, freq="D"):
        f1 = rng.uniform(size=20)
        frames.append(pd.DataFrame({"trade_date": dd, "f1": f1, "f2": rng.normal(size=20),
                                    "label": f1 + 0.03 * rng.normal(size=20)}))
    df = pd.concat(frames, ignore_index=True)
    model, names = train.train_l2_model(df, [FeatureSpec("f1"), FeatureSpec("f2")], "label",
                                        route="C")
    df["__score__"] = train.predict_scores(model, df, names)
    ic = metrics.mean_rank_ic(df.dropna(subset=["__score__"]), "__score__", "label")
    assert ic > 0.3, ic
    lines.append(f"- LightGBM ranker(group 按日,INV):合成信号样本内 RankIC={ic:.3f}>0.3 ✓")

    # PBO/DSR 自洽
    M = rng.uniform(0, 0.5, size=(8, 10)); M[:, 0] = 1.0
    pbo = metrics.pbo_cscv(M)
    dsr = metrics.deflated_sharpe_ratio(0.2, 500, n_trials=1, var_sharpe_trials=0.0)
    assert pbo < 0.3 and dsr > 0.95
    lines.append(f"- PBO(支配 trial)={pbo:.2f}<0.3、DSR(高夏普少试验)={dsr:.3f}>0.95 ✓")

    # 审批 + 盲测段一次性(INV-6)
    led = BlindSegmentLedger()
    m = dict(beats_all_baselines=True, r_blind=0.05, dsr=0.97, pbo=0.2, delta_maxdd=-0.01,
             slippage_net_20bp=0.01, execution_gap_bp=10.0, manual_veto_destructive=False)
    res = approval.evaluate_approval(m, blind_ledger=led, blind_segment_id="blind_2023H2")
    assert res.approved
    try:
        approval.evaluate_approval(m, blind_ledger=led, blind_segment_id="blind_2023H2")
        raise AssertionError("INV-6 未生效:盲测段被复用")
    except InvariantViolation:
        pass
    lines.append("- 审批五重 AND 通过示例 + 盲测段一次性(复用即报错)✓ [INV-6]")
    lines.append("")
    return lines


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = ["# Phase 3 验收报告", ""]
    try:
        report += _computation_checks()
    except Exception as exc:  # noqa: BLE001
        logger.error("Phase 3 计算验证失败: %r", exc)
        report.append(f"- **计算验证失败**: `{exc!r}`")
        (REPORT_DIR / "phase3_acceptance.md").write_text("\n".join(report), encoding="utf-8")
        return 1
    report += [
        "## (B) 真实市场验收(需真实数据,**NOT RUN**)",
        "",
        "- 2019 至今逐月 walk-forward;复杂模型须同时胜随机/单因子/ElasticNet/GBDT 回归。",
        "- DSR>0.95 且 PBO<0.30 须对**真实**入选收益流计算;单次盲测段裁决。",
        "- 10 日 vs 月度刷新节奏对照;分位映射 vs HMM 概率对照。",
        "> 真实数据缺位:上述不可在合成数据上谎报;诚实结论——'此五重 AND 极严,长期可能无配置上线'。",
    ]
    out = REPORT_DIR / "phase3_acceptance.md"
    out.write_text("\n".join(report), encoding="utf-8")
    logger.info("\n".join(report))
    logger.info("\n报告已写入: %s", out)
    logger.info("=== Phase 3 计算验证通过;真实市场审批需实盘数据(NOT RUN)===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
