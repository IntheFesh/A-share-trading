"""增量补空洞测试(批 C:数据完整性)。

红线:增量能补任意位置的缺失行,不只追加比本地最新更晚的日期。覆盖:补中间年份空洞、
同主键被新版本覆盖(keep=last 纠错)、纯新日期追加、只重写受影响年份、主键唯一性不破。
"""

from __future__ import annotations

import time

import pandas as pd

from trading_system.data.store import ParquetStore


def _mini(code, dates, val=1.0):
    """最小面板(KEY + 一个标记值列),够 store 写读且能验证覆盖。"""
    return pd.DataFrame({"code": code, "trade_date": pd.to_datetime(dates), "close_raw": float(val)})


# ── 1) ★补中间年份空洞:有 2019/2021 缺 2020,增量写 2020 → 被补上 ─────────────
def test_fills_middle_year_hole(tmp_path):
    store = ParquetStore(tmp_path)
    store.write(_mini("600000", ["2019-06-01"]))
    store.update_incremental(_mini("600000", ["2021-06-01"]))
    # 旧逻辑:2020 < 本地max(2021) → 被预过滤丢弃,空洞永远补不回。新逻辑:补上。
    n = store.update_incremental(_mini("600000", ["2020-06-01"]))
    back = store.read(codes=["600000"])
    years = sorted(pd.to_datetime(back["trade_date"]).dt.year.unique().tolist())
    assert years == [2019, 2020, 2021]                   # 空洞被填补
    assert n == 1                                        # 净增 1 行
    assert not back.duplicated(subset=["code", "trade_date"]).any()


# ── 2) 同 (code,trade_date) 被新版本覆盖(keep=last,纠正历史错误数据)─────────
def test_overwrites_existing_key_keep_last(tmp_path):
    store = ParquetStore(tmp_path)
    store.write(_mini("600000", ["2024-01-08"], val=10.0))
    n = store.update_incremental(_mini("600000", ["2024-01-08"], val=99.0))   # 同 key 新值
    back = store.read(codes=["600000"])
    assert len(back) == 1                                # 不重复
    assert back["close_raw"].iloc[0] == 99.0             # 以最新抓取为准
    assert n == 0                                        # 覆盖非新增 → 净增 0


# ── 3) 纯新日期 → 正常追加 ───────────────────────────────────────────────────
def test_appends_new_dates(tmp_path):
    store = ParquetStore(tmp_path)
    store.write(_mini("600000", ["2024-01-08"]))
    n = store.update_incremental(_mini("600000", ["2024-01-09", "2024-01-10"]))
    assert n == 2 and len(store.read(codes=["600000"])) == 3


# ── 4) 只重写受影响年份(未涉及年份文件不被触碰)+ 走原子写无 .tmp 残留 ────────
def test_only_affected_year_rewritten(tmp_path):
    store = ParquetStore(tmp_path)
    store.write(_mini("600000", ["2019-06-01"]))
    store.update_incremental(_mini("600000", ["2021-06-01"]))
    mtime_2019 = store._year_file(2019).stat().st_mtime_ns
    time.sleep(0.02)
    store.update_incremental(_mini("600000", ["2020-06-01"]))   # 只动 2020
    assert store._year_file(2019).stat().st_mtime_ns == mtime_2019   # 2019 未被重写
    assert store._year_file(2020).exists()
    assert list(tmp_path.rglob("*.tmp.*")) == []                # 补空洞也走原子写(批 B 配合)


# ── 5) 多票混合:补空洞只影响该票该年,不误伤其他票 ───────────────────────────
def test_holefill_multi_code(tmp_path):
    store = ParquetStore(tmp_path)
    both = pd.concat([_mini("600000", ["2019-06-01", "2021-06-01"]),
                      _mini("600001", ["2020-06-01"])], ignore_index=True)
    store.write(both)
    store.update_incremental(_mini("600000", ["2020-06-01"], val=7.0))   # 给 600000 补 2020
    a = store.read(codes=["600000"])
    b = store.read(codes=["600001"])
    assert sorted(pd.to_datetime(a["trade_date"]).dt.year.tolist()) == [2019, 2020, 2021]
    assert a.loc[pd.to_datetime(a["trade_date"]).dt.year == 2020, "close_raw"].iloc[0] == 7.0
    assert len(b) == 1                                  # 600001 不受影响
