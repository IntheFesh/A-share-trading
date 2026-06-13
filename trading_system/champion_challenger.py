"""冠军挑战者(影子运行式对决)。补丁。对应 v3.1 第八章换届。

核心原则:挑战者 = run_train.py 用**截至当下全部数据(含最近 10 天)**训练的完整模型(用户真会用的
版本,**不阉割**)。公平性靠**影子运行 + 滚动前瞻**:挑战者先不上线,与冠军并行空跑(每日同样出预测、
记录,但用户不照它下单),用其**训练完成之后**才发生的真实时间段的前瞻成绩与同期冠军比较。
换届判定防单期噪声:仅当挑战者**连续 N 个影子期**累计扣费收益不低于冠军、且最大回撤不更差,才换届。

本模块只做"成绩累计 + 换届判定 + 公平性断言";不训练模型(挑战者由 run_train 产出),不做日度调度
(真实影子空跑需逐日实盘数据,标 NOT RUN)。诚实:10 个交易日样本小、单期运气大,故用连续多期累计判定。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ShadowPeriodResult:
    """一个影子期(默认 10 个交易日)内两模型的真实前瞻成绩(口径与回测引擎一致)。"""

    period_index: int
    challenger_net: float      # 挑战者扣费收益
    challenger_maxdd: float    # 挑战者最大回撤(<=0)
    champion_net: float
    champion_maxdd: float


def challenger_not_worse(r: ShadowPeriodResult) -> bool:
    """挑战者当期是否"不输":扣费收益不低于冠军 且 最大回撤不更差。"""
    return (r.challenger_net >= r.champion_net) and (abs(r.challenger_maxdd) <= abs(r.champion_maxdd))


def decide_switch(results: "list[ShadowPeriodResult]", consecutive_required: int) -> dict:
    """换届判定:仅当**最近连续** consecutive_required 个影子期挑战者均"不输",才换届。

    单期赢不换;中途任一期输则连胜清零。返回 {switch, trailing_streak, required}。
    """
    streak = 0
    for r in reversed(results):  # 从最近一期往前数连续"不输"
        if challenger_not_worse(r):
            streak += 1
        else:
            break
    return {"switch": streak >= consecutive_required, "trailing_streak": streak,
            "required": consecutive_required}


def assert_challenger_is_full_model(challenger_train_end, latest_data_date) -> None:
    """公平性断言①:挑战者用截至当下全部数据训练(train_end 不早于最新数据日),即未被阉割。"""
    if pd.Timestamp(challenger_train_end) < pd.Timestamp(latest_data_date):
        raise AssertionError(
            f"挑战者被阉割:train_end={challenger_train_end} 早于最新数据 {latest_data_date};"
            "影子对决必须验证用户真正会使用的完整模型。"
        )


def assert_forward_evaluation(challenger_train_end, shadow_period_start) -> None:
    """公平性断言②:影子期发生在挑战者训练完成之后(前瞻、未见过),而非其训练过的历史区间。"""
    if pd.Timestamp(shadow_period_start) <= pd.Timestamp(challenger_train_end):
        raise AssertionError(
            f"影子期起点 {shadow_period_start} 未晚于挑战者 train_end {challenger_train_end};"
            "对决须用训练完成之后的前瞻时间段,否则不公平。"
        )


def model_in_use(champion, challenger, decision: dict):
    """换届后才把挑战者升为在用模型;未达连胜则维持冠军(挑战者仅记录、不作下单依据)。"""
    return challenger if decision.get("switch") else champion
