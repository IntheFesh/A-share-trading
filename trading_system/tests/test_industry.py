"""行业分类采集 + 落盘测试(批 4,全 mock,网络路径 NOT RUN)。

覆盖:fetch_industry 正确解析(一次取全市场)、IndustryStore 落盘/读取/按 code upsert、
--enable-industry 默认关时不触碰行业接口、开启时采集落盘。
"""

from __future__ import annotations

import pandas as pd

from trading_system.data import fetch_training_data as ftd
from trading_system.data.collectors.baostock_collector import BaostockCollector, normalize_industry
from trading_system.data.industry_store import IndustryStore
from trading_system.data.schema import INDUSTRY_FIELDS
from trading_system.data.store import ParquetStore


# BaoStock query_stock_industry 原始返回(含 updateDate / code_name 等多余列)
def _raw_industry():
    return pd.DataFrame({
        "updateDate": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "code": ["sz.002747", "sz.002156", "sh.600000"],
        "code_name": ["埃斯顿", "通富微电", "浦发银行"],
        "industry": ["机械设备", "电子", "银行"],
        "industryClassification": ["申万一级", "申万一级", "申万一级"],
    })


class IndFakeAPI:
    def __init__(self):
        self.calls = []

    def query(self, code=""):
        self.calls.append(code)
        return _raw_industry()


def _collector(api):
    return BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None,
                             fetch_fn=lambda *a: None, all_stock_fn=lambda d: pd.DataFrame(),
                             industry_fn=api.query,
                             max_retries=0, sleep_sec=0.0, request_timeout_sec=0)


# ── 1) normalize_industry:只留 schema 列,空 industry 归空串,按 code 去重 ────
def test_normalize_industry():
    raw = pd.concat([_raw_industry(),
                     pd.DataFrame({"code": ["sz.002747"], "industry": [None],
                                   "industryClassification": ["申万一级"]})], ignore_index=True)
    out = normalize_industry(raw)
    assert list(out.columns) == list(INDUSTRY_FIELDS)
    assert out["code"].is_unique                         # 按 code 去重
    assert (out["industry"].fillna("") == out["industry"]).all()   # 无 NaN(已归空串)
    assert normalize_industry(pd.DataFrame()).empty       # 空输入 → 空表(带 schema 列)


# ── 2) fetch_industry:一次取全市场,可按 codes 过滤 ──────────────────────────
def test_fetch_industry_one_request_and_filter():
    api = IndFakeAPI()
    bc = _collector(api)
    out = bc.fetch_industry(["sz.002747", "sz.002156"])
    assert api.calls == [""]                              # 一次请求(code="" 全市场),不逐票
    assert set(out["code"]) == {"sz.002747", "sz.002156"}  # 过滤到指定票
    assert out.loc[out["code"] == "sz.002747", "industry"].iloc[0] == "机械设备"


# ── 3) IndustryStore:落盘/读取 + 按 code upsert(更新不丢旧 code)─────────────
def test_industry_store_upsert(tmp_path):
    store = IndustryStore(tmp_path / "data_store_industry")
    store.update(normalize_industry(_raw_industry()))
    assert len(store.read()) == 3
    # 更新 002747 的行业 + 新增一只;旧的 600000 不应丢失
    upd = pd.DataFrame({"code": ["sz.002747", "sz.000001"], "industry": ["自动化", "银行"],
                        "industryClassification": ["申万一级", "申万一级"]})
    total = store.update(upd)
    back = store.read()
    assert total == 4                                     # 3 旧 + 1 新(002747 被更新非新增)
    assert back.loc[back["code"] == "sz.002747", "industry"].iloc[0] == "自动化"   # keep last
    assert "sh.600000" in set(back["code"])               # 旧 code 未丢
    assert set(store.read(codes=["sz.000001"])["code"]) == {"sz.000001"}


# ── 4) --enable-industry 默认关 → 不触碰行业接口、不落盘 ──────────────────────
def test_industry_disabled_by_default(tmp_path):
    dates = list(pd.bdate_range("2020-01-06", periods=3))

    def fetch_fn(code, start, end):
        return pd.DataFrame({"code": code, "trade_date": dates,
                             "open_raw": 10.0, "high_raw": 10.0, "low_raw": 10.0, "close_raw": 10.0,
                             "preclose_raw": 10.0, "volume": 100.0, "amount": 1000.0,
                             "adj_factor": 1.0})

    api = IndFakeAPI()
    bc = BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None, fetch_fn=fetch_fn,
                           all_stock_fn=lambda d: pd.DataFrame(), industry_fn=api.query,
                           max_retries=0, sleep_sec=0.0, request_timeout_sec=0)
    store = ParquetStore(tmp_path / "store")
    rc = ftd.run_fetch(universe_codes=["sz.002747"], baostock_collector=bc, store=store,
                       enable_disclosure=False, output_dir=tmp_path / "out")
    assert rc == 0
    assert api.calls == []                                # 默认关 → 行业接口零调用
    assert not (tmp_path / "store_industry").exists()


# ── 5) --enable-industry 开启 → 行情后采集行业并独立落盘 ──────────────────────
def test_industry_enabled_collects_and_stores(tmp_path):
    dates = list(pd.bdate_range("2020-01-06", periods=3))

    def fetch_fn(code, start, end):
        return pd.DataFrame({"code": code, "trade_date": dates,
                             "open_raw": 10.0, "high_raw": 10.0, "low_raw": 10.0, "close_raw": 10.0,
                             "preclose_raw": 10.0, "volume": 100.0, "amount": 1000.0,
                             "adj_factor": 1.0})

    api = IndFakeAPI()
    bc = BaostockCollector(login_fn=lambda: None, logout_fn=lambda: None, fetch_fn=fetch_fn,
                           all_stock_fn=lambda d: pd.DataFrame(), industry_fn=api.query,
                           max_retries=0, sleep_sec=0.0, request_timeout_sec=0)
    store = ParquetStore(tmp_path / "store")
    ind_store = IndustryStore(tmp_path / "ind")
    rc = ftd.run_fetch(universe_codes=["sz.002747"], baostock_collector=bc, store=store,
                       enable_disclosure=False, output_dir=tmp_path / "out",
                       enable_industry=True, industry_store=ind_store)
    assert rc == 0
    assert api.calls == [""]                              # 开启 → 一次全市场请求
    back = ind_store.read(codes=["sz.002747"])
    assert len(back) == 1 and back["industry"].iloc[0] == "机械设备"
