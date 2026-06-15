"""季频财务数据采集 + 落盘的离线测试(批 2,全 mock,网络路径 NOT RUN)。

覆盖:三表合并(保留 pubDate)、单季失败计入 failed 不中断整批、PIT 关键(pubDate 原样落盘)、
独立 store 增量去重、--enable-financials 默认关时行情流程不变(不触碰任何财务接口)。
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_system.data import fetch_training_data as ftd
from trading_system.data.collectors.baostock_collector import BaostockCollector, merge_financial
from trading_system.data.financial_store import FinancialStore
from trading_system.data.schema import FINANCIAL_FIELDS, FINANCIAL_KEY_FIELDS
from trading_system.data.store import ParquetStore


# ── 构造小样本三表(以 BaoStock 实际字段名为准)────────────────────────────────
def _profit(code, stat, pub, roe, npf):
    return pd.DataFrame({"code": [code], "statDate": [stat], "pubDate": [pub],
                         "roeAvg": [roe], "netProfit": [npf]})


def _growth(code, stat, pub, yoyni):
    return pd.DataFrame({"code": [code], "statDate": [stat], "pubDate": [pub], "YOYNI": [yoyni]})


def _balance(code, stat, pub, lta):
    return pd.DataFrame({"code": [code], "statDate": [stat], "pubDate": [pub],
                         "liabilityToAsset": [lta]})


class FinFakeAPI:
    """假 BaoStock 财务接口:按 (code,year,quarter) 返回固定样本;可指定某些键抛错(模拟失败)。"""

    def __init__(self, *, raise_on=None):
        self.raise_on = set(raise_on or [])
        self.calls = []

    def _maybe_raise(self, code, year, quarter):
        self.calls.append((code, year, quarter))
        if (code, year, quarter) in self.raise_on:
            raise RuntimeError(f"mock financial fail {code} {year}Q{quarter}")

    def profit(self, code, year, quarter):
        self._maybe_raise(code, year, quarter)
        return _profit(code, f"{year}-03-31", f"{year}-04-30", "0.10", "1000")

    def growth(self, code, year, quarter):
        self._maybe_raise(code, year, quarter)
        return _growth(code, f"{year}-03-31", f"{year}-04-30", "-0.6")

    def balance(self, code, year, quarter):
        self._maybe_raise(code, year, quarter)
        return _balance(code, f"{year}-03-31", f"{year}-04-30", "85.0")


def _collector(api, **kw):
    return BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None,
                             fetch_fn=lambda *a: None, all_stock_fn=lambda d: pd.DataFrame(),
                             profit_fn=api.profit, growth_fn=api.growth, balance_fn=api.balance,
                             max_retries=0, sleep_sec=0.0, request_timeout_sec=0, **kw)


# ── 1) merge_financial:三表正确合并为一行,字段齐全,pubDate 保留 ────────────
def test_merge_financial_combines_three_tables_and_keeps_pubdate():
    m = merge_financial(
        _profit("sz.002747", "2024-03-31", "2024-04-30", "0.12", "1000"),
        _growth("sz.002747", "2024-03-31", "2024-04-30", "-0.6"),
        _balance("sz.002747", "2024-03-31", "2024-04-30", "85.0"),
    )
    assert list(m.columns) == list(FINANCIAL_FIELDS)
    row = m.iloc[0]
    assert row["code"] == "sz.002747"
    assert row["pubDate"] == pd.Timestamp("2024-04-30")     # PIT:公告日原样保留
    assert row["statDate"] == pd.Timestamp("2024-03-31")
    assert abs(row["roeAvg"] - 0.12) < 1e-9 and abs(row["YOYNI"] + 0.6) < 1e-9
    assert abs(row["liabilityToAsset"] - 85.0) < 1e-9
    assert merge_financial(pd.DataFrame(), pd.DataFrame(), None) is None   # 三表全空 → None


# ── 2) fetch_financials:多票多季合并;字段齐全 ───────────────────────────────
def test_fetch_financials_merges_codes_and_quarters():
    api = FinFakeAPI()
    bc = _collector(api)
    panel, failed = bc.fetch_financials(["sz.002747", "sz.002156"], [(2023, 1), (2024, 1)])
    assert failed == []
    assert len(panel) == 4                                  # 2 票 × 2 季
    assert set(panel.columns) == set(FINANCIAL_FIELDS)
    assert set(panel["code"]) == {"sz.002747", "sz.002156"}
    assert not panel["pubDate"].isna().any()                # 每行都保留了 pubDate


# ── 3) 单个 (code,year,quarter) 失败计入 failed,不中断整批 ────────────────────
def test_fetch_financials_failure_recorded_not_aborting():
    api = FinFakeAPI(raise_on={("sz.002156", 2024, 1)})     # 该季 profit 抛错 → 整条失败
    bc = _collector(api)
    panel, failed = bc.fetch_financials(["sz.002747", "sz.002156"], [(2024, 1)])
    assert failed == [("sz.002156", 2024, 1)]
    assert set(panel["code"]) == {"sz.002747"}              # 好的那只照常拿到,不被拖累


# ── 4) PIT 关键:落盘后 pubDate 原样保留(statDate=2024-03-31, pubDate=2024-04-30)──
def test_financial_store_preserves_pubdate_pit(tmp_path):
    api = FinFakeAPI()
    bc = _collector(api)
    panel, _ = bc.fetch_financials(["sz.002747"], [(2024, 1)])
    store = FinancialStore(tmp_path / "data_store_fin")
    n = store.update_incremental(panel)
    assert n == 1
    back = store.read(codes=["sz.002747"])
    assert len(back) == 1
    assert back["statDate"].iloc[0] == pd.Timestamp("2024-03-31")
    assert back["pubDate"].iloc[0] == pd.Timestamp("2024-04-30")   # 落盘后 pubDate 不丢、不改写


# ── 5) 财务 store 增量去重:同主键重抓以最新版本为准,不重复 ───────────────────
def test_financial_store_incremental_dedup(tmp_path):
    store = FinancialStore(tmp_path / "fin")
    base = merge_financial(
        _profit("sz.002747", "2024-03-31", "2024-04-30", "0.10", "1000"),
        _growth("sz.002747", "2024-03-31", "2024-04-30", "-0.6"),
        _balance("sz.002747", "2024-03-31", "2024-04-30", "85.0"))
    store.update_incremental(base)
    # 同 (code, statDate) 重抓,roeAvg 改了 → keep last,不新增行
    upd = merge_financial(
        _profit("sz.002747", "2024-03-31", "2024-04-30", "0.20", "2000"),
        _growth("sz.002747", "2024-03-31", "2024-04-30", "-0.6"),
        _balance("sz.002747", "2024-03-31", "2024-04-30", "85.0"))
    store.update_incremental(upd)
    back = store.read()
    assert len(back) == 1                                   # 去重:同报告期仅一行
    assert abs(back["roeAvg"].iloc[0] - 0.20) < 1e-9        # 以最新抓取为准
    assert not back.duplicated(subset=list(FINANCIAL_KEY_FIELDS)).any()


# ── 6) 季度区间生成 ───────────────────────────────────────────────────────────
def test_quarters_range():
    qs = ftd._quarters_range(2019, "2020-05-15")            # 2020Q2 所在
    assert qs[0] == (2019, 1) and qs[-1] == (2020, 2)
    assert (2019, 4) in qs and (2020, 3) not in qs


# ── 7) --enable-financials 默认关 → 行情流程不触碰任何财务接口 ─────────────────
class _SpyFinAPI(FinFakeAPI):
    pass


def test_financials_disabled_by_default_no_calls(tmp_path):
    # 用真实 BaostockCollector 注入行情 + 财务接口;run_fetch 默认 enable_financials=False
    dates = list(pd.bdate_range("2020-01-06", periods=3))

    def fetch_fn(code, start, end):
        return pd.DataFrame({  # 最小合法 RAW_INPUT_FIELDS 面板
            "code": code, "trade_date": dates,
            "open_raw": 10.0, "high_raw": 10.0, "low_raw": 10.0, "close_raw": 10.0,
            "preclose_raw": 10.0, "volume": 100.0, "amount": 1000.0, "adj_factor": 1.0})

    api = _SpyFinAPI()
    bc = BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None, fetch_fn=fetch_fn,
                           all_stock_fn=lambda d: pd.DataFrame(),
                           profit_fn=api.profit, growth_fn=api.growth, balance_fn=api.balance,
                           max_retries=0, sleep_sec=0.0, request_timeout_sec=0)
    store = ParquetStore(tmp_path / "store")
    rc = ftd.run_fetch(universe_codes=["sz.002747"], baostock_collector=bc, store=store,
                       enable_disclosure=False, output_dir=tmp_path / "out")   # 默认不开财务
    assert rc == 0
    assert api.calls == []                                  # 默认关 → 财务接口零调用
    assert not (tmp_path / "store_fin").exists()            # 不产生财务落盘目录


# ── 8) --enable-financials 开启 → 行情后追加财务采集并独立落盘 ─────────────────
def test_financials_enabled_collects_and_stores(tmp_path):
    dates = list(pd.bdate_range("2020-01-06", periods=3))

    def fetch_fn(code, start, end):
        return pd.DataFrame({
            "code": code, "trade_date": dates,
            "open_raw": 10.0, "high_raw": 10.0, "low_raw": 10.0, "close_raw": 10.0,
            "preclose_raw": 10.0, "volume": 100.0, "amount": 1000.0, "adj_factor": 1.0})

    api = FinFakeAPI()
    bc = BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None, fetch_fn=fetch_fn,
                           all_stock_fn=lambda d: pd.DataFrame(),
                           profit_fn=api.profit, growth_fn=api.growth, balance_fn=api.balance,
                           max_retries=0, sleep_sec=0.0, request_timeout_sec=0)
    store = ParquetStore(tmp_path / "store")
    fin_store = FinancialStore(tmp_path / "fin")
    rc = ftd.run_fetch(universe_codes=["sz.002747"], baostock_collector=bc, store=store,
                       enable_disclosure=False, output_dir=tmp_path / "out",
                       enable_financials=True, financial_store=fin_store,
                       financials_start_year=2024, end="2024-03-31")
    assert rc == 0
    assert len(api.calls) > 0                               # 开启 → 确实调用了财务接口
    back = fin_store.read(codes=["sz.002747"])
    assert len(back) >= 1
    assert back["pubDate"].notna().all()                   # 财务独立落盘且 pubDate 保留
