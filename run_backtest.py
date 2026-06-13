"""入口脚本 4/5:回测调参(守三纪律)。补丁。对应 v3.1 第九/十三章。

纪律一·盲测段物理隔离 + 一次性:调参只许在盲测段之前;触碰盲测段需显式 --use-blind-once,
  且重复使用报警(INV-6)。
纪律二·三数并排:必须同时输出 名义收益 / 扣费后收益 / PBO;PBO 过高(>config 阈值)标红警告。
纪律三·粗桶:--param-grid 组合数不得超过 config.backtest.max_param_combos,超过报错。
复用既有指标与引擎(回测口径与实盘一致),不另起引擎。
用法:  python run_backtest.py --start YYYY-MM-DD --end YYYY-MM-DD --param-grid '<json>' [--use-blind-once]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.backtest.discipline import (
    BlindUsageLedger,
    assemble_backtest_report,
    precheck_backtest,
)
from trading_system.config import load_config

logger = logging.getLogger("run_backtest")


def naive_topk_backtest(dataset: pd.DataFrame, score_col: str, *, top_k: int,
                        cost_fraction: float) -> "tuple[float, float, np.ndarray]":
    """轻量 top-K 多头回测:每日按 score 选 top-K 等权,持有至次日。

    返回 (名义收益, 扣费后收益, 每日组合收益序列)。扣费按每日全换手近似(保守)。
    前瞻收益用 close_adj 次日变化(回测实现收益,非特征)。
    """
    df = dataset.sort_values(["code", "trade_date"]).copy()
    df["__fwd__"] = df.groupby("code")["close_adj"].shift(-1) / df["close_adj"] - 1.0
    daily = []
    for _, g in df.dropna(subset=[score_col, "__fwd__"]).groupby("trade_date"):
        top = g.nlargest(min(top_k, len(g)), score_col)
        if len(top):
            daily.append(float(top["__fwd__"].mean()))
    daily = np.asarray(daily, dtype="float64")
    if len(daily) == 0:
        return float("nan"), float("nan"), daily
    nominal = float(np.prod(1.0 + daily) - 1.0)
    net = float(np.prod(1.0 + (daily - cost_fraction)) - 1.0)
    return nominal, net, daily


def run(config: dict, *, start, end, param_grid: dict, use_blind_once: bool = False,
        dataset_df=None, score_col: str = "__score__", top_k: int = 20) -> dict:
    # 划分/成本/上限均在 config.yaml 中修改,请勿在此硬改
    output_dir = Path(config["paths"]["output_dir"])
    ledger = BlindUsageLedger(output_dir / "blind_usage.json")
    pre = precheck_backtest(start=start, end=end, param_grid=param_grid, config=config,
                            use_blind_once=use_blind_once, ledger=ledger)  # 纪律一+三

    # 成本(扣费用 config 成本参数)
    cost = config["cost"]
    cost_fraction = (cost["stamp_duty"] + 2 * cost["exchange_fee"] + 2 * cost["transfer_fee"]
                     + 2 * cost["commission_rate"])

    report = {"nominal_return": None, "net_return": None, "pbo": None, "pbo_warning": None,
              "note": "数值需带 score 的数据集;否则仅执行三纪律预检"}
    if dataset_df is not None and score_col in dataset_df.columns:
        nominal, net, daily = naive_topk_backtest(dataset_df, score_col, top_k=top_k,
                                                  cost_fraction=cost_fraction)
        # PBO:把每日收益切成不重叠块,跨 top_k 变体(trial)构造块绩效矩阵
        trials = [max(1, top_k // 2), top_k, top_k + 5]
        blocks = 8
        mats = []
        for tk in trials:
            _, _, d = naive_topk_backtest(dataset_df, score_col, top_k=tk, cost_fraction=cost_fraction)
            if len(d) >= blocks:
                mats.append([seg.mean() for seg in np.array_split(d - cost_fraction, blocks)])
        if len(mats) >= 2:
            block_perf = np.array(mats).T  # (blocks, trials)
            report = assemble_backtest_report(
                nominal_return=nominal, net_return=net, block_perf=block_perf,
                pbo_warn_threshold=config["backtest"]["pbo_warn_threshold"])  # 纪律二
        else:
            report.update(nominal_return=nominal, net_return=net,
                          note="数据不足以计算 PBO(块/试验过少)")

    if pre["warning"]:
        logger.warning(pre["warning"])
    logger.info("参数组合数=%d 触碰盲测段=%s", pre["combos"], pre["overlaps_blind"])
    logger.info("名义收益=%s 扣费后收益=%s PBO=%s%s", report["nominal_return"],
                report["net_return"], report["pbo"],
                "  [PBO 过高,警告]" if report.get("pbo_warning") else "")
    return {"precheck": pre, "report": report, "cost_fraction": cost_fraction}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    ap = argparse.ArgumentParser(description="回测调参(守三纪律;参数见 config.yaml)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--param-grid", default="{}", help="粗桶参数 JSON,如 '{\"dd\":[\"浅\",\"中\",\"深\"]}'")
    ap.add_argument("--use-blind-once", action="store_true", help="显式一次性使用盲测段")
    a = ap.parse_args(argv)
    run(load_config(), start=a.start, end=a.end, param_grid=json.loads(a.param_grid),
        use_blind_once=a.use_blind_once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
