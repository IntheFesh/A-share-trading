"""Parquet 原子写测试(批 B:数据安全)。

红线:写一半被中断绝不损坏整年文件。覆盖:正常写无 .tmp 残留、写临时文件抛异常时旧文件完好+临时
被清理、原子替换后可正确读回、初始化清理孤儿临时文件。对 ParquetStore / FinancialStore / IndustryStore
共用的 atomic_write_parquet 一并验证。
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_system.data import store as store_mod
from trading_system.data.store import (
    ParquetStore,
    atomic_write_parquet,
    cleanup_orphan_tmp,
)


def _panel(code, dates, base=10.0):
    # 最小双价格层面板(KEY + 价列),够 store 写读
    from trading_system.data.price_layers import build_price_layers
    from trading_system.data.fetch_training_data import _with_null_disclosure
    rows = [dict(code=code, trade_date=pd.Timestamp(d), open_raw=base, high_raw=base,
                 low_raw=base, close_raw=base, preclose_raw=base, volume=1e4, amount=1e5,
                 adj_factor=1.0) for d in dates]
    return _with_null_disclosure(build_price_layers(pd.DataFrame(rows)))


# ── 1) 正常写入后:只有 part.parquet,无 .tmp 残留;read 正确读回 ───────────────
def test_normal_write_no_tmp_left(tmp_path):
    store = ParquetStore(tmp_path)
    store.write(_panel("600000", ["2024-01-08", "2024-01-09"]))
    ydir = tmp_path / "year=2024"
    files = sorted(p.name for p in ydir.iterdir())
    assert files == ["part.parquet"]                     # 无 .tmp 残留
    assert len(store.read(codes=["600000"])) == 2        # 可正确读回


# ── 2) ★写临时文件时抛异常:原 part.parquet 不被破坏,临时文件被清理 ──────────
def test_atomic_write_failure_preserves_old_file(tmp_path, monkeypatch):
    store = ParquetStore(tmp_path)
    store.write(_panel("600000", ["2024-01-08", "2024-01-09"]))   # 先有一份好数据
    good = store.read(codes=["600000"]).copy()
    target = store._year_file(2024)
    old_bytes = target.read_bytes()

    # monkeypatch DataFrame.to_parquet:写到一半抛异常(模拟中断)
    real_to_parquet = pd.DataFrame.to_parquet

    def boom(self, path, *a, **k):
        # 模拟"已创建临时文件但内容损坏/未完成"后中断
        from pathlib import Path as _P
        _P(path).write_bytes(b"HALF-WRITTEN-CORRUPT")   # 写了半个
        raise OSError("disk full mid-write (mock)")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        store._write_year(2024, _panel("600000", ["2024-01-10"]))  # 这次写会"中断"
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real_to_parquet)

    # 原文件未被破坏:字节一致、可读、内容是旧数据
    assert target.read_bytes() == old_bytes              # part.parquet 原封不动
    back = store.read(codes=["600000"])
    pd.testing.assert_frame_equal(back.reset_index(drop=True), good.reset_index(drop=True))
    # 残留临时文件被清理(异常路径里 unlink)
    assert list((tmp_path / "year=2024").glob("*.tmp.*")) == []


# ── 3) 原子替换后数据正确(成功路径):内容是新写的 ───────────────────────────
def test_atomic_replace_updates_content(tmp_path):
    target = tmp_path / "x.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), target)
    assert target.exists() and not list(tmp_path.glob("*.tmp.*"))
    atomic_write_parquet(pd.DataFrame({"a": [9]}), target)   # 覆盖
    assert pd.read_parquet(target)["a"].tolist() == [9]


# ── 4) 初始化清理孤儿临时文件 ────────────────────────────────────────────────
def test_cleanup_orphan_tmp_on_init(tmp_path):
    # 伪造上次中断遗留的孤儿临时文件(根级 + year 目录)
    (tmp_path / "year=2024").mkdir(parents=True)
    (tmp_path / "year=2024" / "part.parquet.tmp.123.abcd").write_bytes(b"junk")
    (tmp_path / "industry.parquet.tmp.9.ef01").write_bytes(b"junk")
    n = cleanup_orphan_tmp(tmp_path)
    assert n == 2
    assert list(tmp_path.rglob("*.tmp.*")) == []
    # ParquetStore 初始化也会清理
    (tmp_path / "year=2024" / "part.parquet.tmp.5.0000").write_bytes(b"junk")
    ParquetStore(tmp_path)
    assert list(tmp_path.rglob("*.tmp.*")) == []


# ── 5) Financial / Industry store 也走原子写(无 .tmp 残留)───────────────────
def test_other_stores_use_atomic_write(tmp_path):
    from trading_system.data.financial_store import FinancialStore
    from trading_system.data.industry_store import IndustryStore

    fin = FinancialStore(tmp_path / "fin")
    fin.update_incremental(pd.DataFrame({
        "code": ["sz.002747"], "statDate": [pd.Timestamp("2024-03-31")],
        "pubDate": [pd.Timestamp("2024-04-30")], "roeAvg": [0.1], "netProfit": [1.0],
        "YOYNI": [0.1], "liabilityToAsset": [0.4]}))
    assert list((tmp_path / "fin").rglob("*.tmp.*")) == []
    assert len(fin.read()) == 1

    ind = IndustryStore(tmp_path / "ind")
    ind.update(pd.DataFrame({"code": ["sz.002747"], "industry": ["机械"],
                             "industryClassification": ["申万一级"]}))
    assert list((tmp_path / "ind").rglob("*.tmp.*")) == []
    assert len(ind.read()) == 1
