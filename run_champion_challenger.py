"""入口脚本 5/5:冠军挑战者(影子运行式对决)。补丁。对应 v3.1 第八章换届。

挑战者 = run_train.py 用截至当下全部数据训练的完整模型(用户真会用的版本,不阉割);先不上线,
与冠军并行影子空跑(每日同样出预测、记录,但用户不照它下单);用其训练完成之后的真实前瞻成绩
与同期冠军比较。换届:仅当挑战者连续 N 个影子期累计扣费收益不低于冠军、且回撤不更差,才升为冠军。

本脚本不训练模型、不做日度调度(真实影子空跑需逐日实盘数据,标 NOT RUN);只做成绩累计、换届判定、
公平性断言与留痕。诚实:10 个交易日样本小、单期运气大,故用连续多期累计判定;无真实市场数据前不代表有效性。
用法:  python run_champion_challenger.py --periods <影子期成绩 json>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from trading_system.champion_challenger import (
    ShadowPeriodResult,
    decide_switch,
    model_in_use,
)
from trading_system.config import load_config

logger = logging.getLogger("run_champion_challenger")


def run(config: dict, *, period_results: "list[ShadowPeriodResult] | None" = None,
        champion="champion", challenger="challenger", block_perf=None) -> dict:
    # 验证方式/连胜数/CPCV 门槛在 config.yaml 中修改,请勿在此硬改
    cc = config["champion_challenger"]
    method = cc.get("validation_method", "shadow")
    if method == "cpcv":
        # CPCV 多路径裁决(新增并行路径;不弱化 INV-6、不放松 PBO/DSR 门槛)
        from trading_system.model.cpcv import cpcv_switch_decision
        if block_perf is None:
            raise ValueError("validation_method=cpcv 需提供 block_perf(块×候选 绩效矩阵)")
        decision = cpcv_switch_decision(block_perf, pbo_max=float(cc.get("cpcv_pbo_max", 0.30)),
                                        dsr_min=float(cc.get("cpcv_dsr_min", 0.95)))
        streak_info = {"pbo": decision["pbo"], "dsr_challenger": decision["dsr_challenger"]}
    else:
        required = int(cc["switch_requires_consecutive"])
        if period_results is None:
            raise ValueError("validation_method=shadow 需提供 period_results")
        decision = decide_switch(period_results, required)
        streak_info = {"required_consecutive": required, "trailing_streak": decision["trailing_streak"]}

    in_use = model_in_use(champion, challenger, decision)  # 赢了才换,没赢维持冠军(两法一致)
    out = Path(config["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rec = {"validation_method": method, "switch": decision["switch"], "in_use_after": in_use,
           **streak_info,
           "periods": [vars(r) for r in period_results] if period_results else None}
    (out / "champion_challenger_decision.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("验证方式=%s → 换届=%s;当前在用=%s", method, decision["switch"], in_use)
    logger.info("诚实提示:换届永远由绩效是否真的更好决定,绝不到期强制换届;"
                "样本小/单期运气大,故 shadow 用连续多期累计、cpcv 用多路径分布 + PBO/DSR 门槛。")
    return rec


def _parse_periods(s: str) -> "list[ShadowPeriodResult]":
    return [ShadowPeriodResult(**d) for d in json.loads(s)]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    ap = argparse.ArgumentParser(description="冠军挑战者影子对决(参数见 config.yaml)")
    ap.add_argument("--periods", required=True,
                    help="影子期成绩 JSON 列表,元素含 period_index/challenger_net/challenger_maxdd/"
                         "champion_net/champion_maxdd")
    a = ap.parse_args(argv)
    run(load_config(), period_results=_parse_periods(a.periods))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
