"""补丁:数据采集模块的离线真实测试(全 mock,网络路径 NOT RUN)。

覆盖 6 类:开关短路 / Tushare 降级 / BaoStock 硬失败 / 增量 max+1 去重 / 双价格层完整 / PIT NULL vs False。
纪律:不伪造"看起来跑通"的真实采集;真实 baostock/tushare 网络路径继续 NOT RUN(见下方说明性 skip)。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from trading_system.data import fetch_training_data as ftd
from trading_system.data.collectors.baostock_collector import BaostockCollector
from trading_system.data.price_layers import attach_disclosure_fields, build_price_layers
from trading_system.data.schema import RAW_INPUT_FIELDS
from trading_system.data.store import ParquetStore


# ── 测试构造 ────────────────────────────────────────────────────────────────
def _raw(code, dates, *, factor=1.0, base=10.0, limit_up_idx=None):
    rows, prev = [], base
    for i, d in enumerate(dates):
        if limit_up_idx is not None and i == limit_up_idx:
            pc, c = prev, round(prev * 1.1, 2)
            o, h, l = pc, c, pc
        else:
            c = round(prev * (1 + 0.01 * ((i % 3) - 1)), 2)
            o, h, l = prev, max(prev, c), min(prev, c)
            pc = prev
        rows.append(dict(code=code, trade_date=pd.Timestamp(d), open_raw=o, high_raw=h,
                         low_raw=l, close_raw=c, preclose_raw=pc, volume=10000.0,
                         amount=10000.0 * c, adj_factor=factor))
        prev = c
    return pd.DataFrame(rows)


class FakeBC:
    """假 BaoStock 采集器:可控 login 失败 / 返回固定面板 / 记录请求起点。"""

    def __init__(self, panels_by_code, *, login_raises=False):
        self.panels = panels_by_code
        self.login_raises = login_raises
        self.received_start = None
        self.universe = list(panels_by_code)

    def __enter__(self):
        if self.login_raises:
            raise RuntimeError("baostock login failed (mock)")
        return self

    def __exit__(self, *a):
        return False

    def list_universe(self, day, *, boards=("60", "000")):
        return self.universe

    def fetch_many(self, codes, start_by_code, end):
        self.received_start = dict(start_by_code) if isinstance(start_by_code, dict) else start_by_code
        frames = [self.panels[c] for c in codes if c in self.panels]
        panel = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=list(RAW_INPUT_FIELDS)))
        return panel, []


class _RecordingFactory:
    def __init__(self, collector):
        self.collector = collector
        self.called = False

    def __call__(self, token):
        self.called = True
        return self.collector


class _TushareRaises:
    def fetch_disclosure(self, codes, start, end):
        raise RuntimeError("tushare network down (mock)")


_DATES = list(pd.bdate_range("2020-01-06", periods=5))


# ── 1) 开关:enable_disclosure=False 不触碰 Tushare,披露字段 NULL ───────────
def test_disclosure_off_skips_tushare_and_nulls(tmp_path):
    fake = FakeBC({"sh.600000": _raw("sh.600000", _DATES)})
    factory = _RecordingFactory(_TushareRaises())
    store = ParquetStore(tmp_path)
    rc = ftd.run_fetch(enable_disclosure=False, universe_codes=["sh.600000"],
                       baostock_collector=fake, store=store, tushare_collector_factory=factory)
    assert rc == 0
    assert factory.called is False                      # 未实例化/调用 Tushare
    back = store.read(codes=["sh.600000"])
    assert back["has_preann"].isna().all()              # NULL=未采集
    assert back["preann_sign"].isna().all()


# ── 2) Tushare 降级:异常 → 行情照常落盘,披露置空,exit 0,有 warning ───────
def test_tushare_failure_degrades_gracefully(tmp_path, caplog):
    fake = FakeBC({"sh.600000": _raw("sh.600000", _DATES)})
    factory = _RecordingFactory(_TushareRaises())
    store = ParquetStore(tmp_path)
    with caplog.at_level(logging.WARNING):
        rc = ftd.run_fetch(enable_disclosure=True, tushare_token="x",
                           universe_codes=["sh.600000"], baostock_collector=fake,
                           store=store, tushare_collector_factory=factory)
    assert rc == 0                                       # 不崩
    assert factory.called is True
    back = store.read(codes=["sh.600000"])
    assert len(back) == len(_DATES) and "close_raw" in back.columns   # 行情仍在
    assert back["has_preann"].isna().all()               # 披露降级置空
    assert any("降级" in m or "失败" in m for m in caplog.messages)   # 留痕


# ── 3) BaoStock 硬失败:login 失败 → exit≠0(行情缺失不静默)───────────────
def test_baostock_login_hard_fail(tmp_path):
    fake = FakeBC({"sh.600000": _raw("sh.600000", _DATES)}, login_raises=True)
    rc = ftd.run_fetch(enable_disclosure=False, universe_codes=["sh.600000"],
                       baostock_collector=fake, store=ParquetStore(tmp_path))
    assert rc == 2


# ── 4) 增量:只请求 max+1→今日,append 去重无重复 ───────────────────────────
def test_incremental_requests_max_plus_one_and_dedupes(tmp_path):
    code = "sh.600000"
    store = ParquetStore(tmp_path)
    # 预置:已有 D0..D3
    pre = build_price_layers(_raw(code, _DATES[:4]))
    store.write(ftd._with_null_disclosure(pre))
    d3 = pd.Timestamp(_DATES[3])
    expected_start = (d3 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # 假采集器返回 D3(重叠)+ D4 + D5
    new_dates = [_DATES[3], _DATES[4], pd.Timestamp("2020-01-13")]
    fake = FakeBC({code: _raw(code, new_dates)})
    rc = ftd.run_fetch(incremental=True, universe_codes=[code], baostock_collector=fake,
                       store=store, enable_disclosure=False)
    assert rc == 0
    assert fake.received_start[code] == expected_start   # 起点 = 本地最新 +1 日
    back = store.read(codes=[code])
    assert not back.duplicated(subset=["code", "trade_date"]).any()   # 无重复
    assert back["trade_date"].nunique() == len(back)                  # 每日一行
    assert pd.Timestamp("2020-01-13") in set(back["trade_date"])      # 新数据已并入


# ── 5) 双价格层完整:raw/adj/adj_factor 齐全,raw≠adj(有除权),涨停价用 raw ──
def test_double_price_layer_complete(tmp_path):
    code = "sh.600000"
    fake = FakeBC({code: _raw(code, _DATES, factor=2.0, limit_up_idx=2)})
    store = ParquetStore(tmp_path)
    ftd.run_fetch(enable_disclosure=False, universe_codes=[code], baostock_collector=fake,
                  store=store)
    back = store.read(codes=[code]).sort_values("trade_date").reset_index(drop=True)
    for col in ("open_raw", "close_raw", "open_adj", "close_adj", "adj_factor"):
        assert col in back.columns
    assert np.allclose(back["close_adj"], back["close_raw"] * 2.0)    # adj = raw × factor
    assert (back["close_adj"] != back["close_raw"]).all()            # factor=2 -> raw≠adj
    assert bool(back.loc[2, "is_limit_up"])                          # 收盘=raw昨收×1.1 -> 涨停(INV-2)


# ── 6) PIT 语义:NULL(未采集)与 has_preann=False(已确认未发)严格区分 ───────
def test_pit_null_vs_false_distinguished(tmp_path):
    code = "sh.600000"
    layer = build_price_layers(_raw(code, _DATES))

    # NULL 路径
    null_store = ParquetStore(tmp_path / "null")
    null_store.write(ftd._with_null_disclosure(layer))
    null_back = null_store.read(codes=[code])
    assert null_back["has_preann"].isna().all()          # 未采集 = NA

    # "已确认未发预告" 路径:采集成功但无预告 -> has_preann=False(非 NA)
    from trading_system.data.calendar import TradingCalendar
    cal = TradingCalendar(sorted(pd.to_datetime(layer["trade_date"]).unique()))
    sched = pd.DataFrame({"code": [code], "sched_disclosure_date": [pd.Timestamp(_DATES[-1])]})
    preann = pd.DataFrame({"code": [], "ann_date": [], "preann_sign": []})  # 无预告事实
    collected = attach_disclosure_fields(layer, sched_disclosure=sched, preann=preann, calendar=cal)
    collected = ftd._coerce_disclosure_for_storage(collected)
    ok_store = ParquetStore(tmp_path / "ok")
    ok_store.write(collected)
    ok_back = ok_store.read(codes=[code])
    assert not ok_back["has_preann"].isna().any()        # 已采集,非 NA
    assert (ok_back["has_preann"] == False).all()        # noqa: E712 — 已确认未发 = False


# ── 真实网络路径:诚实 NOT RUN(不伪造)──────────────────────────────────────
def test_real_baostock_universe_listing_not_run():
    pytest.importorskip("baostock", reason="未安装 baostock;真实交易池/行情拉取见 fetch_training_data 实跑")
    pytest.skip("BaoStock 真实会话需网络,离线不验;真实采集留给用户运行 fetch_training_data。")
