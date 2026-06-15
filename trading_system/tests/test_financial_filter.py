"""财务过滤层测试(批 3:避雷逻辑)。重中之重:PIT 可见性(pubDate 对齐,防未来函数)。

挂了 PIT 测试 = 有未来函数,必须修。其余:过滤规则(亏损+恶化/高负债剔除、缺失放行)、开关默认关。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.data.financials import (
    attach_visible_financials,
    filter_universe_with_financials,
    financial_exclusion_mask,
)


def _fin_row(code, stat, pub, *, roe=0.1, npf=1000.0, yoyni=0.1, lta=0.4):
    return pd.DataFrame({"code": [code], "statDate": [pd.Timestamp(stat)],
                         "pubDate": [pd.Timestamp(pub)], "roeAvg": [roe], "netProfit": [npf],
                         "YOYNI": [yoyni], "liabilityToAsset": [lta]})


# ════════════════════════════════════════════════════════════════════════════
# ★★★ PIT 可见性(最重要):pubDate=2024-04-30 的财报,2024-04-29 看不到、当日起看得到 ★★★
# ════════════════════════════════════════════════════════════════════════════
def test_pit_visibility_pubdate_alignment():
    code = "sz.002747"
    # 交易日:公告日前一天 / 公告当日 / 公告后
    days = pd.to_datetime(["2024-04-29", "2024-04-30", "2024-05-06"])
    daily = pd.DataFrame({"code": code, "trade_date": days})
    fin = _fin_row(code, "2024-03-31", "2024-04-30", roe=0.15, npf=500.0, yoyni=0.2, lta=0.5)

    aligned = attach_visible_financials(daily, fin).sort_values("trade_date").reset_index(drop=True)

    # 2024-04-29:财报尚未公告(pubDate=04-30 > 04-29)→ 看不到 → 财务列全 NaN
    assert pd.isna(aligned.loc[0, "roeAvg"]), "未来函数!04-29 不应看到 04-30 才公告的财报"
    assert pd.isna(aligned.loc[0, "fin_pubDate"])
    # 2024-04-30 当日及之后:看得到,且来源就是该期(statDate 2024-03-31 / pubDate 2024-04-30)
    assert aligned.loc[1, "roeAvg"] == 0.15
    assert aligned.loc[1, "fin_pubDate"] == pd.Timestamp("2024-04-30")
    assert aligned.loc[1, "fin_statDate"] == pd.Timestamp("2024-03-31")
    assert aligned.loc[2, "roeAvg"] == 0.15           # 之后仍可见(直到下一期公告)


def test_pit_uses_latest_announced_period():
    code = "sz.002747"
    # 两期:Q1(公告 04-30)、Q2(公告 08-30)。8 月某日应看到 Q2,5 月某日只看到 Q1
    fin = pd.concat([
        _fin_row(code, "2024-03-31", "2024-04-30", roe=0.10),
        _fin_row(code, "2024-06-30", "2024-08-30", roe=0.20),
    ], ignore_index=True)
    daily = pd.DataFrame({"code": code, "trade_date": pd.to_datetime(["2024-05-10", "2024-09-02"])})
    aligned = attach_visible_financials(daily, fin).sort_values("trade_date").reset_index(drop=True)
    assert aligned.loc[0, "fin_statDate"] == pd.Timestamp("2024-03-31")   # 5月:只见 Q1
    assert aligned.loc[1, "fin_statDate"] == pd.Timestamp("2024-06-30")   # 9月:已见 Q2(最新)


# ── 过滤规则:亏损 + 同比大幅恶化 → 剔除 ───────────────────────────────────────
def test_exclude_loss_and_deterioration():
    panel = pd.DataFrame({
        "code": ["a", "b", "c"],
        "netProfit": [-100.0, -100.0, 200.0],   # a/b 亏损, c 盈利
        "YOYNI": [-0.6, -0.2, -0.9],            # a 大幅恶化(<-0.5), b 轻微, c 大幅但盈利
        "liabilityToAsset": [0.3, 0.3, 0.3],
    })
    mask = financial_exclusion_mask(panel)       # 默认阈值 yoyni<-0.5
    assert bool(mask.iloc[0]) is True            # a:亏损 且 同比<-50% → 剔除
    assert bool(mask.iloc[1]) is False           # b:亏损但同比仅 -20% → 保留(温和)
    assert bool(mask.iloc[2]) is False           # c:盈利 → 保留(同比恶化但不亏损)


# ── 过滤规则:资产负债率畸高 → 剔除 ───────────────────────────────────────────
def test_exclude_high_leverage():
    panel = pd.DataFrame({
        "code": ["a", "b"], "netProfit": [100.0, 100.0], "YOYNI": [0.1, 0.1],
        "liabilityToAsset": [0.85, 0.75],        # a 超 0.80 阈值
    })
    mask = financial_exclusion_mask(panel, max_liability_to_asset=0.80)
    assert bool(mask.iloc[0]) is True            # 0.85 > 0.80 → 剔除
    assert bool(mask.iloc[1]) is False           # 0.75 → 保留


# ── 过滤规则:财务缺失(NaN)→ 放行,不误杀次新 ───────────────────────────────
def test_missing_financials_not_excluded():
    panel = pd.DataFrame({
        "code": ["newstock"], "netProfit": [np.nan], "YOYNI": [np.nan],
        "liabilityToAsset": [np.nan],
    })
    mask = financial_exclusion_mask(panel)
    assert bool(mask.iloc[0]) is False           # 缺失 = 未知,不等于恶化 → 放行


# ── 开关:enable_financial_filter=False(默认)→ 池子与改动前一致 ──────────────
def test_filter_disabled_matches_baseline():
    base = dict(trade_date=pd.Timestamp("2024-05-06"), is_suspended=False, is_st=False,
                is_one_price_limit=False)
    daily = pd.DataFrame([{"code": "002747", **base}])    # 002 主板合规
    # 即便给了"会被剔除"的恶化财务,默认关时也不应生效
    fin = _fin_row("002747", "2024-03-31", "2024-04-30", npf=-100.0, yoyni=-0.9, lta=0.95)

    off = filter_universe_with_financials(daily, fin, enable_financial_filter=False,
                                          new_listing_min_days=1)
    on = filter_universe_with_financials(daily, fin, enable_financial_filter=True,
                                         new_listing_min_days=1)
    assert bool(off.loc[0, "is_in_universe"]) is True     # 默认关 → 合规票仍进池
    assert bool(on.loc[0, "is_in_universe"]) is False     # 开启 → 基本面恶化被剔除


def test_filter_on_keeps_healthy_admits_pit():
    # 开启过滤:健康票(盈利/低负债)且财报已公告 → 仍进池;PIT 对齐由 attach 保证
    base = dict(is_suspended=False, is_st=False, is_one_price_limit=False)
    daily = pd.DataFrame([{"code": "002747", "trade_date": pd.Timestamp("2024-05-06"), **base}])
    fin = _fin_row("002747", "2024-03-31", "2024-04-30", npf=500.0, yoyni=0.2, lta=0.4)
    out = filter_universe_with_financials(daily, fin, enable_financial_filter=True,
                                          new_listing_min_days=1)
    assert bool(out.loc[0, "is_in_universe"]) is True
