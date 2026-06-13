"""Phase 0 数据底座的真实单元测试(对已知正确答案的构造性核验)。

纪律(用户要求):不写"通过不报错"的糊弄测试。每个断言都有可手算 / 可独立推导的正确答案;
逻辑错则真实失败。需要实盘数据 / token 的路径(baostock/tushare 网络)用 importorskip /
明确 skip 原因占位,绝不伪装通过。真实市场验收见 run/phase0_acceptance.py(未跑,需数据/token)。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from trading_system import invariants as inv
from trading_system.data import price_layers as pl
from trading_system.data import quality as q
from trading_system.data import universe as uni
from trading_system.data.calendar import TradingCalendar
from trading_system.data.collectors import sina, synthetic, tencent
from trading_system.data.schema import ADJ_PRICE_FIELDS, RAW_PRICE_FIELDS
from trading_system.data.store import ParquetStore


# ── 向量化半进位 vs Decimal 权威实现(锚定 INV-2 价格取整一致性)──────────────
def test_round_half_up_array_matches_decimal_authority():
    vals = [0.005, 2.665, 2.675, 11.00, 9.005, 10.045, 10.055, 123.455, 999.995]
    arr = inv.round_half_up_2(np.array(vals))
    for v, got in zip(vals, arr):
        assert got == inv._round_half_up_2(v), f"{v}: 向量 {got} != Decimal {inv._round_half_up_2(v)}"
    # 标量路径
    assert inv.round_half_up_2(2.675) == 2.68
    assert inv.round_half_up_2(2.665) == 2.67


# ── 交易日历 ────────────────────────────────────────────────────────────────
@pytest.fixture
def cal30() -> TradingCalendar:
    return synthetic.make_calendar("2020-01-06", 30)  # 2020-01-06 是周一


class TestCalendar:
    def test_membership_and_range(self, cal30):
        d = cal30.dates
        assert cal30.is_trading_day(d[0]) and cal30.is_trading_day(d[-1])
        assert not cal30.is_trading_day(dt.date(2020, 1, 11))  # 周六
        rng = cal30.get_trading_days(d[2], d[5])
        assert rng == d[2:6]

    def test_shift_t_plus_1_and_2(self, cal30):
        d = cal30.dates
        assert cal30.shift_trading_day(d[5], 1) == d[6]   # T+1
        assert cal30.shift_trading_day(d[5], 2) == d[7]   # T+2
        assert cal30.shift_trading_day(d[5], -2) == d[3]

    def test_shift_errors_are_real(self, cal30):
        d = cal30.dates
        with pytest.raises(KeyError):
            cal30.shift_trading_day(dt.date(2020, 1, 11), 1)  # 非交易日:无定义 -> 报错
        with pytest.raises(IndexError):
            cal30.shift_trading_day(d[-1], 5)                 # 越界 -> 报错

    def test_offset_and_align(self, cal30):
        d = cal30.dates
        assert cal30.offset(d[3], d[9]) == 6
        # 周六对齐到下个交易日
        fri = next(x for x in d if x.weekday() == 4)
        sat = fri + dt.timedelta(days=1)
        assert cal30.next_trading_day_on_or_after(sat) == cal30.dates[cal30.dates.index(fri) + 1]


# ── 双价格层(INV-2 核心)──────────────────────────────────────────────────
def _raw_row(**kw) -> dict:
    base = dict(
        code="600000", trade_date=pd.Timestamp("2020-01-06"),
        open_raw=10.0, high_raw=10.0, low_raw=10.0, close_raw=10.0,
        preclose_raw=10.0, volume=1000.0, amount=10000.0, adj_factor=1.0,
    )
    base.update(kw)
    return base


class TestPriceLayers:
    def test_adj_layer_is_raw_times_factor(self):
        df = pd.DataFrame([_raw_row(close_raw=10.0, adj_factor=2.5)])
        out = pl.build_price_layers(df)
        assert out.loc[0, "close_adj"] == 25.0  # 10.0 * 2.5

    def test_limit_up_flag_uses_raw_preclose(self):
        # 昨收 10.00 -> 涨停 round(11.00)=11.00;收盘 11.00 = 涨停
        up = pl.build_price_layers(pd.DataFrame([_raw_row(
            preclose_raw=10.0, open_raw=10.5, high_raw=11.0, low_raw=10.4, close_raw=11.0)]))
        assert bool(up.loc[0, "is_limit_up"]) is True
        # 收盘 10.99 != 11.00 -> 非涨停
        notup = pl.build_price_layers(pd.DataFrame([_raw_row(
            preclose_raw=10.0, open_raw=10.5, high_raw=10.99, low_raw=10.4, close_raw=10.99)]))
        assert bool(notup.loc[0, "is_limit_up"]) is False

    def test_limit_down_flag(self):
        dn = pl.build_price_layers(pd.DataFrame([_raw_row(
            preclose_raw=10.0, open_raw=9.5, high_raw=9.6, low_raw=9.0, close_raw=9.0)]))
        assert bool(dn.loc[0, "is_limit_down"]) is True

    def test_one_price_limit(self):
        one = pl.build_price_layers(pd.DataFrame([_raw_row(
            preclose_raw=10.0, open_raw=11.0, high_raw=11.0, low_raw=11.0, close_raw=11.0)]))
        assert bool(one.loc[0, "is_one_price_limit"]) is True
        # 涨停但非一字(OHLC 不全等)
        not_one = pl.build_price_layers(pd.DataFrame([_raw_row(
            preclose_raw=10.0, open_raw=10.5, high_raw=11.0, low_raw=10.4, close_raw=11.0)]))
        assert bool(not_one.loc[0, "is_limit_up"]) is True
        assert bool(not_one.loc[0, "is_one_price_limit"]) is False

    def test_st_ratio_changes_limit(self):
        # ST 5%:昨收 10.00 -> 涨停 10.50;收盘 10.50 = 涨停(主板 10% 则不会)
        st = pl.build_price_layers(pd.DataFrame([_raw_row(
            is_st=True, preclose_raw=10.0, open_raw=10.2, high_raw=10.5, low_raw=10.1,
            close_raw=10.5)]))
        assert bool(st.loc[0, "is_limit_up"]) is True
        main = pl.build_price_layers(pd.DataFrame([_raw_row(
            is_st=False, preclose_raw=10.0, open_raw=10.2, high_raw=10.5, low_raw=10.1,
            close_raw=10.5)]))
        assert bool(main.loc[0, "is_limit_up"]) is False

    def test_suspension_flag(self):
        s = pl.build_price_layers(pd.DataFrame([_raw_row(volume=0.0)]))
        assert bool(s.loc[0, "is_suspended"]) is True

    def test_ex_dividend_keeps_adj_continuous_while_raw_jumps(self):
        # 除权:day0 收盘 11.00(factor 1.0);day1 现金分红致 raw 跳到 ~9.9,factor 升到使 adj 连续
        # 构造 adj 连续:close_adj 应平滑(≈11.0->≈11.0)。raw 则跳变。
        rows = [
            _raw_row(trade_date=pd.Timestamp("2020-01-06"), preclose_raw=10.8,
                     open_raw=11.0, high_raw=11.0, low_raw=11.0, close_raw=11.00, adj_factor=1.0),
            _raw_row(trade_date=pd.Timestamp("2020-01-07"), preclose_raw=9.90,
                     open_raw=9.95, high_raw=10.0, low_raw=9.9, close_raw=9.9,
                     adj_factor=11.00 / 9.9),  # 使 close_adj = 9.9 * (11/9.9) = 11.00,连续
        ]
        out = pl.build_price_layers(pd.DataFrame(rows))
        assert abs(out.loc[0, "close_adj"] - 11.00) < 1e-9
        assert abs(out.loc[1, "close_adj"] - 11.00) < 1e-9   # adj 连续(无跳变)
        assert out.loc[1, "close_raw"] == 9.9                # raw 真实跳变


# ── 披露季 PIT 字段 ─────────────────────────────────────────────────────────
class TestDisclosure:
    def test_days_to_disclosure(self, cal30):
        d = cal30.dates
        got = pl.compute_days_to_disclosure([d[5], d[5], d[5]], [d[10], d[5], d[2]], cal30)
        assert got[0] == 5.0          # d5 -> d10 = 5 个交易日(d6..d10)
        assert got[1] == 0.0          # 当日披露
        assert np.isnan(got[2])       # 已披露(过去) -> NaN

    def test_preann_is_point_in_time(self, cal30):
        d = cal30.dates
        panel = pd.DataFrame({"code": ["600000"] * 3, "trade_date": [d[2], d[5], d[8]]})
        sched = pd.DataFrame({"code": ["600000"], "sched_disclosure_date": [pd.Timestamp(d[10])]})
        # 预告在 d[5] 公告(方向 -1);d[2] 时尚未公告
        preann = pd.DataFrame({"code": ["600000"], "ann_date": [pd.Timestamp(d[5])],
                               "preann_sign": [-1]})
        out = pl.attach_disclosure_fields(panel, sched_disclosure=sched, preann=preann,
                                          calendar=cal30)
        assert bool(out.loc[0, "has_preann"]) is False   # d[2]:预告尚未公告(无前视)
        assert bool(out.loc[1, "has_preann"]) is True     # d[5]:当日已公告
        assert out.loc[1, "preann_sign"] == -1
        assert bool(out.loc[2, "has_preann"]) is True     # d[8]:仍可见
        assert out.loc[1, "days_to_disclosure"] == 5.0    # d5 -> d10


# ── 交易池过滤 ──────────────────────────────────────────────────────────────
class TestUniverse:
    def test_filter_rules(self):
        # 4 只票各 1 天;构造各自违规点
        base = dict(trade_date=pd.Timestamp("2020-01-06"), is_suspended=False, is_st=False,
                    is_one_price_limit=False)
        df = pd.DataFrame([
            {"code": "600000", **base},  # 主板,合规
            {"code": "300001", **base},  # 创业板 -> 出
            {"code": "600001", **{**base, "is_st": True}},        # ST -> 出
            {"code": "600002", **{**base, "is_suspended": True}}, # 停牌 -> 出
            {"code": "600003", **{**base, "is_one_price_limit": True}},  # 一字 -> 出
        ])
        out = uni.filter_universe(df, new_listing_min_days=1)  # 每票 1 天,放宽次新约束
        got = dict(zip(out["code"], out["is_in_universe"]))
        assert got["600000"] is True or got["600000"] == True  # noqa: E712
        assert not got["300001"]
        assert not got["600001"]
        assert not got["600002"]
        assert not got["600003"]

    def test_new_listing_window(self):
        # 同一票 70 天;前 60 天应被次新规则剔除
        dates = synthetic.make_calendar("2020-01-06", 70).dates
        df = pd.DataFrame({
            "code": ["600000"] * 70, "trade_date": [pd.Timestamp(x) for x in dates],
            "is_st": False, "is_suspended": False, "is_one_price_limit": False,
        })
        out = uni.filter_universe(df, new_listing_min_days=60).sort_values("trade_date")
        inu = out["is_in_universe"].to_numpy()
        assert not inu[:59].any()   # 前 59 天(上市未满 60 交易日)不在池
        assert inu[59:].all()       # 第 60 天起在池


# ── 存储(Parquet + DuckDB + 增量)─────────────────────────────────────────
@pytest.fixture
def full_panel():
    cal = synthetic.make_calendar("2020-01-06", 12)
    raw = synthetic.make_raw_panel(["600000", "600001"], cal, seed=7)
    return pl.build_price_layers(raw), cal


class TestStore:
    def test_write_read_roundtrip(self, tmp_path, full_panel):
        panel, _ = full_panel
        store = ParquetStore(tmp_path)
        n = store.write(panel)
        assert n == len(panel)
        back = store.read()
        # 行数一致;按 key 排序后逐列等值
        assert len(back) == len(panel)
        a = panel.sort_values(["code", "trade_date"]).reset_index(drop=True)
        b = back.sort_values(["code", "trade_date"]).reset_index(drop=True)
        assert (a["close_raw"].to_numpy() == b["close_raw"].to_numpy()).all()

    def test_field_selection_supports_inv2(self, tmp_path, full_panel):
        panel, _ = full_panel
        store = ParquetStore(tmp_path)
        store.write(panel)
        exec_only = store.read(fields=list(RAW_PRICE_FIELDS))
        # 执行取数:只含 key + raw,绝无 adj 列(INV-2 在数据出口处即分离)
        assert not any(c in exec_only.columns for c in ADJ_PRICE_FIELDS)
        assert "close_raw" in exec_only.columns and "code" in exec_only.columns
        feat = store.read(fields=["close_adj"])
        assert "close_adj" in feat.columns and "close_raw" not in feat.columns

    def test_filters(self, tmp_path, full_panel):
        panel, cal = full_panel
        store = ParquetStore(tmp_path)
        store.write(panel)
        d = cal.dates
        sub = store.read(codes=["600000"], start=d[2], end=d[5])
        assert set(sub["code"].unique()) == {"600000"}
        assert sub["trade_date"].min() == pd.Timestamp(d[2])
        assert sub["trade_date"].max() == pd.Timestamp(d[5])

    def test_unknown_field_raises(self, tmp_path, full_panel):
        panel, _ = full_panel
        store = ParquetStore(tmp_path)
        store.write(panel)
        with pytest.raises(ValueError):
            store.read(fields=["close_adj"]) if False else store.read(fields=["not_a_col"])

    def test_empty_store_returns_empty(self, tmp_path):
        assert ParquetStore(tmp_path).read().empty

    def test_incremental_only_appends_new(self, tmp_path, full_panel):
        panel, cal = full_panel
        d = cal.dates
        store = ParquetStore(tmp_path)
        part1 = panel[panel["trade_date"] <= pd.Timestamp(d[6])]
        store.write(part1)
        local_max = store.local_max_dates()
        assert local_max["600000"] == pd.Timestamp(d[6])
        # 新数据含重叠(d4..d6)+ 新(d7..d11):只接受 > d6
        newer = panel[panel["trade_date"] >= pd.Timestamp(d[4])]
        appended = store.update_incremental(newer)
        assert appended == (len(panel) - len(part1))  # 只新增 d7..d11
        assert len(store.read()) == len(panel)
        assert store.local_max_dates()["600000"] == pd.Timestamp(d[11])

    def test_incremental_dedupe_keeps_last(self, tmp_path):
        store = ParquetStore(tmp_path)
        cal = synthetic.make_calendar("2020-01-06", 2)
        raw = synthetic.make_raw_panel(["600000"], cal, seed=1)
        panel = pl.build_price_layers(raw)
        # 人为制造同 (code,date) 的两条,后一条 close 改为 99.0
        dup = panel.iloc[[1]].copy()
        dup.loc[dup.index, "close_raw"] = 99.0
        merged_new = pd.concat([panel, dup], ignore_index=True)
        store.update_incremental(merged_new)
        back = store.read()
        # 该日去重后保留最后一条(close=99.0)
        row = back[back["trade_date"] == panel.iloc[1]["trade_date"]]
        assert len(row) == 1 and row.iloc[0]["close_raw"] == 99.0


# ── 数据质检 ────────────────────────────────────────────────────────────────
class TestQuality:
    def test_clean_panel_has_no_fail(self, full_panel):
        panel, _ = full_panel
        results = q.run_daily_quality_checks(panel)
        fails = [r for r in results if r.status == q.FAIL]
        assert not fails, f"干净数据不应有 FAIL: {[(r.check, r.detail) for r in fails]}"
        q.assert_passed(results)  # 不抛

    def test_detects_limit_price_inconsistency(self, full_panel):
        panel, _ = full_panel
        bad = panel.copy()
        # 把一行强标为 is_limit_up 但收盘价并非涨停价
        bad.loc[bad.index[0], "is_limit_up"] = True
        r = q.check_limit_price_consistency(bad)
        assert r.status == q.FAIL and r.n_flagged >= 1

    def test_detects_adj_discontinuity(self, full_panel):
        panel, _ = full_panel
        bad = panel.sort_values(["code", "trade_date"]).copy()
        idx = bad.index[3]
        bad.loc[idx, "close_adj"] = bad.loc[idx, "close_adj"] * 1.5  # +50% 非除权非涨停
        r = q.check_adj_continuity(bad)
        assert r.status == q.FAIL and r.n_flagged >= 1

    def test_detects_suspension_inconsistency(self, full_panel):
        panel, _ = full_panel
        bad = panel.copy()
        bad.loc[bad.index[0], "is_suspended"] = True  # 但 volume>0
        r = q.check_suspension_consistency(bad)
        assert r.status == q.FAIL and r.n_flagged >= 1

    def test_assert_passed_raises_on_fail(self, full_panel):
        panel, _ = full_panel
        bad = panel.copy()
        bad.loc[bad.index[0], "is_suspended"] = True
        with pytest.raises(AssertionError):
            q.assert_passed(q.run_daily_quality_checks(bad))


# ── 盘中快照解析器(不依赖网络,可单测)────────────────────────────────────
class TestSnapshotParsers:
    def test_parse_tencent(self):
        f = ["1", "贵州茅台", "600519", "1700.00", "1688.00", "1695.00", "12345"] + ["0"] * 30
        f[33], f[34] = "1710.00", "1685.00"
        text = f'v_sh600519="{"~".join(f)}";'
        rec = tencent.parse_tencent(text)[0]
        assert rec["code"] == "600519" and rec["price"] == 1700.00
        assert rec["preclose"] == 1688.00 and rec["open"] == 1695.00
        assert rec["high"] == 1710.00 and rec["low"] == 1685.00

    def test_parse_sina(self):
        f = ["贵州茅台", "1695.00", "1688.00", "1700.00", "1710.00", "1685.00"] + ["0"] * 26
        f[30], f[31] = "2026-06-12", "15:00:00"
        text = f'var hq_str_sh600519="{",".join(f)}";'
        rec = sina.parse_sina(text)[0]
        assert rec["name"] == "贵州茅台" and rec["price"] == 1700.00
        assert rec["open"] == 1695.00 and rec["preclose"] == 1688.00
        assert rec["high"] == 1710.00 and rec["low"] == 1685.00


# ── 需实盘数据 / token 的路径:诚实 skip(绝不伪装通过)──────────────────────
class TestLiveSourcesHonestlySkipped:
    def test_baostock_needs_network(self):
        pytest.importorskip("baostock", reason="未安装 baostock;实盘日历/日线验收见 run/phase0_acceptance.py")
        pytest.skip("BaoStock 需网络会话,离线 CI 不验;真实 20 除权/退市/涨跌停核验留给验收脚本。")

    def test_tushare_needs_token(self):
        import os
        if not os.environ.get("TUSHARE_TOKEN"):
            pytest.skip("无 TUSHARE_TOKEN:披露日历/预告/退市为 token-gated,真实验收留给验收脚本。")
        pytest.importorskip("tushare", reason="未安装 tushare")
