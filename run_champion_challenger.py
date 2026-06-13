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


def run(config: dict, *, period_results: "list[ShadowPeriodResult]",
        champion="champion", challenger="challenger") -> dict:
    # 影子期长度/换届连胜数在 config.yaml 中修改,请勿在此硬改
    required = int(config["champion_challenger"]["switch_requires_consecutive"])
    decision = decide_switch(period_results, required)
    in_use = model_in_use(champion, challenger, decision)
    # 留痕
    out = Path(config["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "required_consecutive": required,
        "trailing_streak": decision["trailing_streak"],
        "switch": decision["switch"],
        "in_use_after": in_use,
        "periods": [vars(r) for r in period_results],
    }
    (out / "champion_challenger_decision.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("连胜=%d / 需要=%d → 换届=%s;当前在用=%s",
                decision["trailing_streak"], required, decision["switch"], in_use)
    logger.info("诚实提示:10 个交易日样本小、单期运气大,故采用连续多期累计判定;"
                "影子期成绩在有真实市场数据前不代表真实有效性。")
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
