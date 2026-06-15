"""日线估值字段 peTTM/pbMRQ/psTTM 测试(批 5,alpha 采集)。

红线:这三个字段**仅采集落盘,不得进任何因子计算/模型打分**(进模型前需单独 RankIC/ICIR 验证)。
覆盖:① 正确采集 + 数值化 + 透传到 fetch_raw_with_factor 输出;② build_price_layers 保留;
③ 落盘 schema 含这三列;④ 没有任何现有因子/打分逻辑引用它们(grep 断言)。
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

import trading_system.data.collectors.baostock as bs_api
from trading_system.data.price_layers import build_price_layers
from trading_system.data.schema import (
    FEATURE_EXTRA_FIELDS,
    PRICE_LAYER_FIELDS,
    VALUATION_FIELDS,
)


def _fake_k_rows(code):
    # 模拟 _query_k 返回(字符串列,含估值字段),供 fetch_daily 数值化
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "code": [code, code],
        "open": ["10.0", "10.5"], "high": ["10.8", "10.9"], "low": ["9.9", "10.2"],
        "close": ["10.5", "10.7"], "preclose": ["10.0", "10.5"],
        "volume": ["1000", "1100"], "amount": ["10500", "11770"], "turn": ["1.2", "1.3"],
        "tradestatus": ["1", "1"], "pctChg": ["5.0", "1.9"], "isST": ["0", "0"],
        "peTTM": ["20.5", "21.0"], "pbMRQ": ["3.1", "3.2"], "psTTM": ["5.0", "5.1"],
    })


# ── 1) 采集 + 数值化 + 透传:fetch_raw_with_factor 输出含三估值字段(float)──────
def test_valuation_collected_and_passthrough(monkeypatch):
    monkeypatch.setattr(bs_api, "_query_k", lambda code, s, e, adj: _fake_k_rows(code))
    out = bs_api.fetch_raw_with_factor("sz.002747", "2024-01-01", "2024-01-31")
    for f in VALUATION_FIELDS:
        assert f in out.columns, f"估值字段未透传: {f}"
        assert out[f].dtype.kind == "f"                  # 已数值化为浮点
    assert out["peTTM"].iloc[0] == 20.5
    assert out["pbMRQ"].iloc[1] == 3.2
    assert out["psTTM"].iloc[0] == 5.0


# ── 2) build_price_layers 保留估值字段;缺失时 NaN 填充(不报错)─────────────────
def test_build_price_layers_keeps_valuation(monkeypatch):
    monkeypatch.setattr(bs_api, "_query_k", lambda code, s, e, adj: _fake_k_rows(code))
    raw = bs_api.fetch_raw_with_factor("sz.002747", "2024-01-01", "2024-01-31")
    layered = build_price_layers(raw)
    for f in VALUATION_FIELDS:
        assert f in layered.columns
    assert layered["peTTM"].iloc[0] == 20.5              # 原值保留

    # 缺失估值字段的输入(如合成测试面板)→ NaN 填充,不抛错
    bare = raw.drop(columns=list(VALUATION_FIELDS))
    layered2 = build_price_layers(bare)
    assert layered2["peTTM"].isna().all()


# ── 3) 落盘 schema:估值字段在 FEATURE_EXTRA_FIELDS / PRICE_LAYER_FIELDS 中 ─────
def test_valuation_in_store_schema():
    for f in VALUATION_FIELDS:
        assert f in FEATURE_EXTRA_FIELDS                 # 特征附加列(与 turn 同类)
        assert f in PRICE_LAYER_FIELDS                   # 随行情一起落盘


# ── 4) ★严格约束:估值字段未被任何现有因子/打分逻辑引用(仅采集,待 RankIC 验证)──
def test_valuation_not_referenced_by_any_factor():
    import trading_system.features as feats_pkg

    root = pathlib.Path(feats_pkg.__file__).parent
    offenders = []
    for py in root.rglob("*.py"):
        src = py.read_text(encoding="utf-8")
        for f in VALUATION_FIELDS:
            if f in src:                                 # 因子代码里出现估值字段名 = 违规接入
                offenders.append((str(py.relative_to(root)), f))
    assert offenders == [], (
        f"估值字段不得进因子/打分(仅采集,进模型前需单独 RankIC/ICIR 验证),发现引用: {offenders}")


# ── 5) ★严格约束:估值字段不在 config 特征清单(不进模型打分)─────────────────
def test_valuation_not_in_config_features():
    from trading_system.config import load_config

    features = load_config().get("features", [])
    for f in VALUATION_FIELDS:
        assert f not in features                         # 不在打分用特征清单里
    # 也不应作为已注册特征名出现
    from trading_system.features import registry as reg
    for f in VALUATION_FIELDS:
        assert f not in reg.REGISTRY
