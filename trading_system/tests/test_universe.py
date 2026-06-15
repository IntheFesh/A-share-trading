"""主板覆盖(沪深主板全集,含 001/002)的板块前缀测试。批 1:修 bug。

背景:旧默认前缀 ("60","000") 漏掉了 001(约118只)和 002(约921只,原中小板,已并入深主板),
合计约 1040 只主板票没进数据集——用户核心 watchlist(埃斯顿 002747 等)几乎全在 002。
本测试钉死 MAIN_BOARD_PREFIXES 的"应进池/应排除"边界,防回归(三位精确前缀,无误收无漏收)。
"""

from __future__ import annotations

import pandas as pd

from trading_system.data import universe as uni
from trading_system.data.universe import MAIN_BOARD_PREFIXES, board_allowed, filter_universe


# 应进池:沪市主板 600/601/603/605 + 深市主板 000/001/002
_INCLUDE = [
    "sh.600000",   # 浦发银行(600)
    "sh.601398",   # 工商银行(601)
    "sh.603259",   # 药明康德(603)
    "sh.605499",   # 东鹏饮料(605)
    "sz.000001",   # 平安银行(000)
    "sz.001872",   # 招商港口(001)
    "sz.002747",   # 埃斯顿(002,用户核心 watchlist)
]
# 应排除:创业板 300/301、科创板 688、北交所 8xx/4xx/920、深B 200、沪B 900
_EXCLUDE = [
    "sz.300750",   # 宁德时代(创业板 300)
    "sz.301269",   # 创业板 301
    "sh.688041",   # 科创板 688
    "bj.830799",   # 北交所 830
    "bj.430047",   # 北交所 430
    "bj.920000",   # 北交所 920
    "sz.200625",   # 深 B 200
    "sh.900957",   # 沪 B 900
]


def test_main_board_prefixes_value():
    # 唯一真相源:三位精确前缀集合,顺序无关但元素必须正好是这七个
    assert set(MAIN_BOARD_PREFIXES) == {"600", "601", "603", "605", "000", "001", "002"}


def test_board_allowed_includes_main_board_incl_001_002():
    for code in _INCLUDE:
        assert board_allowed(code, MAIN_BOARD_PREFIXES) is True, f"应进池却被排除: {code}"


def test_board_allowed_excludes_non_main_board():
    for code in _EXCLUDE:
        assert board_allowed(code, MAIN_BOARD_PREFIXES) is False, f"应排除却进池: {code}"


def test_three_digit_precision_separates_002_from_200():
    # 主板 002(应进)与深 B 200(应出)前缀相近(00x vs 20x),三位精确前缀正确区分、互不误收
    assert board_allowed("sz.002747", MAIN_BOARD_PREFIXES) is True    # 002 主板进
    assert board_allowed("sz.200625", MAIN_BOARD_PREFIXES) is False   # 200 深B 出
    assert "200625".startswith("002") is False                       # 002 前缀不会误收 200
    # 反证:用过松的两位前缀 "20" 会误收深 B 200;三位精确前缀不会
    assert board_allowed("sz.200625", ("20",)) is True               # 过松前缀的错误后果
    # 反证:用过松的一位前缀 "0" 会误收 000-009 全段;三位前缀只收 000/001/002
    assert board_allowed("sz.009999", ("0",)) is True                # 过松前缀的错误后果
    assert board_allowed("sz.009999", MAIN_BOARD_PREFIXES) is False   # 正确:009 非主板,不收


def test_filter_universe_default_admits_002():
    # filter_universe 默认 boards=MAIN_BOARD_PREFIXES → 002 主板票合规应进池
    base = dict(trade_date=pd.Timestamp("2020-01-06"), is_suspended=False, is_st=False,
                is_one_price_limit=False)
    df = pd.DataFrame([
        {"code": "002747", **base},   # 埃斯顿,002 主板,合规
        {"code": "001872", **base},   # 001 主板,合规
        {"code": "300750", **base},   # 创业板 -> 出
    ])
    out = filter_universe(df, new_listing_min_days=1)
    got = dict(zip(out["code"], out["is_in_universe"]))
    assert bool(got["002747"]) is True
    assert bool(got["001872"]) is True
    assert bool(got["300750"]) is False


def test_filter_universe_default_boards_is_constant():
    # 默认参数即引用唯一真相源,避免三处各写一份而漂移
    import inspect

    sig = inspect.signature(uni.filter_universe)
    assert sig.parameters["boards"].default == MAIN_BOARD_PREFIXES
