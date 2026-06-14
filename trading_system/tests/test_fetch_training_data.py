"""补丁:数据采集模块的离线真实测试(全 mock,网络路径 NOT RUN)。

覆盖 6 类:开关短路 / Tushare 降级 / BaoStock 硬失败 / 增量 max+1 去重 / 双价格层完整 / PIT NULL vs False。
纪律:不伪造"看起来跑通"的真实采集;真实 baostock/tushare 网络路径继续 NOT RUN(见下方说明性 skip)。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading_system.data import fetch_training_data as ftd
from trading_system.data.collectors import baostock_collector as bcoll
from trading_system.data.collectors.baostock_collector import BaostockCollector
from trading_system.data.collectors.quota import RequestQuota
from trading_system.data.price_layers import attach_disclosure_fields, build_price_layers
from trading_system.data.schema import RAW_INPUT_FIELDS
from trading_system.data.store import ParquetStore

_FIXED_UTC = lambda: datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)  # 固定同一国内自然日


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


# ════════════════════════════════════════════════════════════════════════════
# 健壮取数:超时跳过 + 分批落盘 + 失败重拉 + 断点续传(全 mock,网络路径 NOT RUN)
# ════════════════════════════════════════════════════════════════════════════
class ScriptedBC:
    """可编排的假采集器:按 (code -> 失败轮次) 模拟单票超时/失败,并记录每次 fetch_many 调用。

    fail_counts[code]=k:该票在前 k 次 fetch_many 调用里返回到 failed(模拟超时被看门狗跳过),
    第 k+1 次起成功。run_fetch 会先分批拉(第一轮),再对待拉列表重拉一轮——故 k=1 测"重拉成功",
    k 很大(如 99)测"重拉仍失败→最终失败报告,不无限重试"。
    """

    def __init__(self, panels_by_code, *, fail_counts=None, login_raises=False,
                 quota=None, cost=2):
        self.panels = panels_by_code
        self.login_raises = login_raises
        self._remaining = dict(fail_counts or {})
        self.calls: list[list[str]] = []          # 每次 fetch_many 的入参 codes(顺序)
        self.received_start = None                 # 最近一次 fetch_many 的 start_by_code
        self.universe = list(panels_by_code)
        self.quota = quota                          # 可选:模拟真实采集器把请求计入配额
        self.cost = cost                            # 每只票一次拉取消耗的请求数(默认 2)
        self.quota_stopped = False
        self.remaining_codes: list[str] = []

    def __enter__(self):
        if self.login_raises:
            raise RuntimeError("baostock login failed (mock)")
        if self.quota is not None:
            self.quota.add(1)                       # 登录计入配额
        return self

    def __exit__(self, *a):
        if self.quota is not None:
            self.quota.add(1)                       # 登出计入配额
            self.quota.flush()
        return False

    def list_universe(self, day, *, boards=("60", "000")):
        return list(self.universe)

    def fetch_many(self, codes, start_by_code, end):
        self.received_start = (dict(start_by_code) if isinstance(start_by_code, dict)
                               else start_by_code)
        self.calls.append(list(codes))
        self.quota_stopped = False
        self.remaining_codes = []
        frames, failed = [], []
        for i, c in enumerate(codes):
            if self.quota is not None and self.quota.exceeded():   # 配额耗尽:停止本批,余下待拉
                self.quota_stopped = True
                self.remaining_codes = list(codes[i:])
                break
            if self.quota is not None:
                self.quota.add(self.cost)           # 本次拉取计入配额
            if self._remaining.get(c, 0) > 0:      # 本轮仍判失败(模拟超时跳过)
                self._remaining[c] -= 1
                failed.append(c)
                continue
            if c in self.panels:
                frames.append(self.panels[c])
        panel = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=list(RAW_INPUT_FIELDS)))
        return panel, failed


# ── 7) 单票超时看门狗:挂起的票被跳过,正常票成功,整批不卡死 ──────────────────
def test_request_timeout_watchdog_skips_hanging_code():
    def fetch_fn(code, start, end):
        if code == "sh.600999":
            time.sleep(2.0)                        # 模拟请求挂起,远超看门狗超时
        return _raw(code, _DATES)

    bc = BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None,
                           fetch_fn=fetch_fn, all_stock_fn=lambda d: pd.DataFrame(),
                           max_retries=1, sleep_sec=0.0, request_timeout_sec=0.2)
    t0 = time.time()
    panel, failed = bc.fetch_many(["sh.600999", "sh.600000"], "2020-01-01", "2020-12-31")
    elapsed = time.time() - t0
    assert "sh.600999" in failed                   # 超时票被跳过,计入 failed
    assert "sh.600000" not in failed               # 正常票成功
    assert set(panel["code"]) == {"sh.600000"}
    assert elapsed < 2.0                            # 看门狗生效:没等满 2s,整批未卡死


# ── 8) 看门狗工具函数:超时抛 TimeoutError;快路径透传返回值/内部异常;<=0 关闭 ──
def test_call_with_timeout_semantics():
    with pytest.raises(TimeoutError):
        bcoll._call_with_timeout(lambda: time.sleep(1.0), (), 0.1)
    assert bcoll._call_with_timeout(lambda x: x + 1, (41,), 1.0) == 42   # 透传返回

    def boom():
        raise ValueError("inner-error")
    with pytest.raises(ValueError, match="inner-error"):                 # 透传内部异常
        bcoll._call_with_timeout(boom, (), 1.0)
    assert bcoll._call_with_timeout(lambda: 7, (), 0) == 7               # timeout<=0 同步直调


# ── 9) 分批落盘 + 待拉列表 + 重拉成功:全部票最终落盘,无重复,pending 清空 ─────
def test_batch_save_pending_then_retry_success(tmp_path):
    codes = ["sh.600000", "sh.600001", "sh.600002"]
    panels = {c: _raw(c, _DATES) for c in codes}
    bc = ScriptedBC(panels, fail_counts={"sh.600001": 1})   # 第一轮失败,重拉成功
    store = ParquetStore(tmp_path / "store")
    out = tmp_path / "out"
    rc = ftd.run_fetch(universe_codes=codes, baostock_collector=bc, store=store,
                       enable_disclosure=False, incremental=True,
                       batch_save_size=2, output_dir=out)
    assert rc == 0
    back = store.read(codes=codes)
    assert set(back["code"]) == set(codes)                  # 三只票都落盘(含重拉成功的)
    assert not back.duplicated(subset=["code", "trade_date"]).any()   # 无重复
    # 分批:chunk1=[600000,600001]、chunk2=[600002],再重拉 [600001]
    assert bc.calls == [["sh.600000", "sh.600001"], ["sh.600002"], ["sh.600001"]]
    pend = json.loads((out / "pending_codes.json").read_text(encoding="utf-8"))
    assert pend["codes"] == []                              # 全部拉到 → 待拉清空
    assert not (out / "failed_codes.json").exists()         # 无最终失败


# ── 10) 重拉仍失败:记入 failed_codes.json,日志列出,只重拉一次(不无限重试)──
def test_permanent_failure_recorded_and_retried_once(tmp_path, caplog):
    codes = ["sh.600000", "sh.600001"]
    panels = {c: _raw(c, _DATES) for c in codes}
    bc = ScriptedBC(panels, fail_counts={"sh.600001": 99})  # 永远失败
    store = ParquetStore(tmp_path / "store")
    out = tmp_path / "out"
    with caplog.at_level(logging.ERROR):
        rc = ftd.run_fetch(universe_codes=codes, baostock_collector=bc, store=store,
                           enable_disclosure=False, incremental=True,
                           batch_save_size=10, output_dir=out)
    assert rc == 0                                          # 非全失败 → 不硬退(仅告警)
    assert set(store.read(codes=codes)["code"]) == {"sh.600000"}      # 失败票未落盘
    failed = json.loads((out / "failed_codes.json").read_text(encoding="utf-8"))
    assert failed["codes"] == ["sh.600001"]                # 确未拉到,写入最终失败报告
    assert bc.calls == [["sh.600000", "sh.600001"], ["sh.600001"]]   # 只重拉一次,非无限
    assert any("sh.600001" in m for m in caplog.messages)  # 日志明确列出失败票


# ── 11) 断点续传:中断后重跑(增量),已落盘票跳过、未拉票从头补,最终完整无重复 ──
def test_resume_after_interruption_incremental(tmp_path):
    codes = ["sh.600000", "sh.600001"]
    panels = {c: _raw(c, _DATES) for c in codes}
    store = ParquetStore(tmp_path / "store")
    out = tmp_path / "out"

    # 第一次:模拟 600001 拉取失败(中断/未落盘),600000 已落盘
    bc1 = ScriptedBC(panels, fail_counts={"sh.600001": 99})
    rc1 = ftd.run_fetch(universe_codes=codes, baostock_collector=bc1, store=store,
                        enable_disclosure=False, incremental=True,
                        batch_save_size=10, output_dir=out)
    assert rc1 == 0
    assert set(store.read(codes=codes)["code"]) == {"sh.600000"}      # 仅 600000 落盘

    # 重跑(增量):600001 这次成功;600000 已是最新 → 起点=本地最新+1(被跳过补拉)
    bc2 = ScriptedBC(panels)
    rc2 = ftd.run_fetch(universe_codes=codes, baostock_collector=bc2, store=store,
                        enable_disclosure=False, incremental=True,
                        batch_save_size=10, output_dir=out)
    assert rc2 == 0
    a_next = (pd.Timestamp(_DATES[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert bc2.received_start["sh.600000"] == a_next       # 已落盘票:从最新+1 起(不重复拉历史)
    assert bc2.received_start["sh.600001"] == "2019-01-01"  # 未拉票:从 config.start 起
    back = store.read(codes=codes)
    assert set(back["code"]) == set(codes)                 # 两只票最终都在
    assert not back.duplicated(subset=["code", "trade_date"]).any()   # 不重复
    assert back[back["code"] == "sh.600001"]["trade_date"].nunique() == len(_DATES)  # 不丢失


# ── 12) 分批落盘 == 一次性落盘:多批结果与单批严格一致(不漏、不重、双价格层完整)──
def test_batched_equals_single_save(tmp_path):
    codes = [f"sh.60000{i}" for i in range(6)]
    panels = {c: _raw(c, _DATES, factor=1.5) for c in codes}

    single = ParquetStore(tmp_path / "single")
    ftd.run_fetch(universe_codes=codes, baostock_collector=ScriptedBC(panels), store=single,
                  enable_disclosure=False, incremental=True,
                  batch_save_size=999, output_dir=tmp_path / "o1")        # 单批
    batched = ParquetStore(tmp_path / "batched")
    ftd.run_fetch(universe_codes=codes, baostock_collector=ScriptedBC(panels), store=batched,
                  enable_disclosure=False, incremental=True,
                  batch_save_size=2, output_dir=tmp_path / "o2")          # 分 3 批

    a = single.read(codes=codes).sort_values(["code", "trade_date"]).reset_index(drop=True)
    b = batched.read(codes=codes).sort_values(["code", "trade_date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)                    # 分批与一次性逐格相等


# ── 13) 配额保护:当日累计达 limit-margin → 优雅停止,已拉部分落盘,余下记待拉,退出 0 ──
def test_run_fetch_quota_stop_graceful(tmp_path):
    codes = [f"sh.60000{i}" for i in range(5)]
    panels = {c: _raw(c, _DATES) for c in codes}
    store = ParquetStore(tmp_path / "store")
    out = tmp_path / "out"
    # limit=8, margin=2 → 阈值 6。login(+1) 后:code0→3、code1→5、code2→7;查 code3 时已 >=6 → 停。
    q = RequestQuota(store.root / ".baostock_quota.json", daily_limit=8, safety_margin=2,
                     now_fn=_FIXED_UTC)
    bc = ScriptedBC(panels, quota=q, cost=2)
    rc = ftd.run_fetch(universe_codes=codes, baostock_collector=bc, store=store, quota=q,
                       enable_disclosure=False, incremental=True,
                       batch_save_size=10, output_dir=out)
    assert rc == 0                                          # 优雅退出(非硬失败)
    saved = set(store.read(codes=codes)["code"])
    assert 0 < len(saved) < len(codes)                     # 已拉部分落盘(不丢),未拉部分未落盘
    pend = json.loads((out / "pending_codes.json").read_text(encoding="utf-8"))
    assert set(pend["codes"]) == set(codes) - saved        # 余下的票记入待拉(明日续传)
    assert not (out / "failed_codes.json").exists()        # 配额停止 ≠ 永久失败,不写失败报告
    assert q.exceeded()                                    # 配额确已达阈值


# ── 14) 配额已满启动:开跑前即超阈值 → 不登录、不取数,优雅退出 0 ─────────────────
def test_run_fetch_quota_already_exhausted_at_start(tmp_path):
    codes = ["sh.600000"]
    store = ParquetStore(tmp_path / "store")
    q = RequestQuota(store.root / ".baostock_quota.json", daily_limit=10, safety_margin=2,
                     now_fn=_FIXED_UTC)
    q.add(9)                                               # 9 >= 阈值 8 → 已超
    bc = ScriptedBC({c: _raw(c, _DATES) for c in codes}, quota=q)
    rc = ftd.run_fetch(universe_codes=codes, baostock_collector=bc, store=store, quota=q,
                       enable_disclosure=False, output_dir=tmp_path / "out")
    assert rc == 0
    assert bc.calls == []                                  # 没有发起任何拉取(也没登录消耗)
    assert store.read(codes=codes).empty                   # 未写入任何数据
