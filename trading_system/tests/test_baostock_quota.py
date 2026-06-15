"""BaoStock 单日请求配额计数器的离线测试(全 mock,网络路径 NOT RUN)。

覆盖四个关键场景:
  1) 跨进程读取累计(关终端重拉接着算,不从 0 起);
  2) 自然日切换归零(跨午夜按国内自然日重置);
  3) 达到配额上限优雅退出(check 抛 QuotaExceeded、exceeded 为真);
  4) 时区用 Asia/Shanghai 而非本地时区(用户在美国,绝不能用本地时区判断);
另加:采集器把每次实际请求(登录/拉取/登出)如实计入配额。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

from trading_system.data.collectors.baostock_collector import BaostockCollector
from trading_system.data.collectors.quota import QuotaExceeded, RequestQuota

# 固定 UTC 时刻(同一国内自然日 2026-06-15:UTC 06:00 → 上海 14:00)。
_SAME_DAY = lambda: datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)


# ── 1) 跨进程累计:两个独立实例指向同一文件,第二个从已用次数续起 ───────────────
def test_cross_process_accumulation(tmp_path):
    path = tmp_path / ".baostock_quota.json"
    q1 = RequestQuota(path, daily_limit=100, safety_margin=10, now_fn=_SAME_DAY)
    q1.add(30)                                  # "进程 1" 用掉 30 次并落盘
    q1.flush()
    # "进程 2":全新实例打开同一文件,应从 30 续起(而非 0)
    q2 = RequestQuota(path, daily_limit=100, safety_margin=10, now_fn=_SAME_DAY)
    assert q2.load_today() == 30                 # 跨进程累计起点 = 已用 30
    assert q2.add(5) == 35
    assert json.loads(path.read_text(encoding="utf-8"))["count"] == 35


# ── 2) 自然日切换:文件是昨天的 → 今天新实例归零并更新日期 ─────────────────────
def test_natural_day_rollover_resets(tmp_path):
    path = tmp_path / ".baostock_quota.json"
    path.write_text(json.dumps({"date": "2026-06-14", "count": 42000}), encoding="utf-8")
    q = RequestQuota(path, daily_limit=50000, safety_margin=5000, now_fn=_SAME_DAY)  # 今天=06-15
    assert q.count == 0                          # 跨日 → 计数归零
    assert q.load_today() == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["date"] == "2026-06-15" and data["count"] == 0   # 文件已更新为今日 + 0
    assert q.add(3) == 3                         # 当日内继续累加正常


# ── 3) 达配额上限:exceeded 为真,check 抛 QuotaExceeded,停止发起新请求 ────────
def test_quota_limit_blocks_new_requests(tmp_path):
    q = RequestQuota(tmp_path / "q.json", daily_limit=100, safety_margin=10, now_fn=_SAME_DAY)
    q.add(89)                                    # 阈值 = 100 − 10 = 90
    assert not q.exceeded()
    q.check()                                    # 89 < 90,不抛
    q.add(1)                                     # → 90
    assert q.exceeded()                          # 90 >= 90
    assert q.remaining() <= 0
    with pytest.raises(QuotaExceeded):
        q.check()                                # 达阈值 → 抛出,供上层优雅停止


# ── 4) 时区:按 Asia/Shanghai 判日,而非本地时区(用户在美国)──────────────────
def test_uses_shanghai_timezone_not_local(tmp_path):
    import os

    # 选取 UTC 18:30:上海=次日 06-16,而 UTC / 美国东西部均为当日 06-15。
    instant = datetime(2026, 6, 15, 18, 30, tzinfo=timezone.utc)
    old_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"     # 把进程本地时区强制设为洛杉矶(此刻本地=06-15)
    try:
        try:
            time.tzset()
            assert instant.astimezone().strftime("%Y-%m-%d") == "2026-06-15"   # 本地(洛杉矶)=15
        except AttributeError:
            pass                                 # 无 tzset 的平台(如 Windows)跳过本地反证
        q = RequestQuota(tmp_path / "q.json", now_fn=lambda: instant)   # 默认 tz = Asia/Shanghai
        q.add(1)
        data = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
        assert data["date"] == "2026-06-16"      # 用上海日期(次日),非 UTC、非本地洛杉矶(15)
    finally:                                      # 还原进程时区,避免污染其他测试
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        try:
            time.tzset()
        except AttributeError:
            pass


# ── 4b) 缺 IANA 时区库时退回固定 UTC+8(对中国恒等),仍按国内自然日 ───────────
def test_shanghai_fallback_is_utc_plus_8(tmp_path):
    from datetime import timedelta
    from trading_system.data.collectors import quota as qmod

    instant = datetime(2026, 6, 15, 18, 30, tzinfo=timezone.utc)
    q = RequestQuota(tmp_path / "q.json", tz=qmod._SHANGHAI_FALLBACK, now_fn=lambda: instant)
    q.add(1)
    assert qmod._SHANGHAI_FALLBACK.utcoffset(None) == timedelta(hours=8)
    assert json.loads((tmp_path / "q.json").read_text())["date"] == "2026-06-16"


# ── 5) 采集器把每次实际请求(登录 + 每票拉取×cost + 登出)如实计入配额 ──────────
def test_collector_counts_every_request_into_quota(tmp_path):
    q = RequestQuota(tmp_path / "q.json", daily_limit=1000, safety_margin=10, now_fn=_SAME_DAY)
    bc = BaostockCollector(
        login_fn=lambda: None, logout_fn=lambda: None,
        fetch_fn=lambda c, s, e: pd.DataFrame({"code": [c]}),   # 非空即视为成功
        all_stock_fn=lambda d: pd.DataFrame(),
        max_retries=0, sleep_sec=0.0, request_timeout_sec=0, quota=q,
    )
    with bc:                                      # 登录 +1
        panel, failed = bc.fetch_many(["sh.600000", "sh.600001"], "2020-01-01", "2020-12-31")
    # 2 只票 × cost 2 = 4,+ 登录 1 + 登出 1 = 6
    assert failed == []
    assert q.count == 6
    assert json.loads((tmp_path / "q.json").read_text())["count"] == 6


# ── 5b) 重试也计入:单票失败重试 N 次,每次实际请求都计数 ─────────────────────
def test_collector_counts_retries(tmp_path):
    q = RequestQuota(tmp_path / "q.json", daily_limit=1000, safety_margin=10, now_fn=_SAME_DAY)
    calls = {"n": 0}

    def flaky(code, start, end):
        calls["n"] += 1
        raise RuntimeError("boom")                # 每次都失败 → 触发重试

    bc = BaostockCollector(
        login_fn=lambda: None, logout_fn=lambda: None, fetch_fn=flaky,
        all_stock_fn=lambda d: pd.DataFrame(),
        max_retries=2, sleep_sec=0.0, request_timeout_sec=0, quota=q,
    )
    with bc:                                      # 登录 +1
        panel, failed = bc.fetch_many(["sh.600000"], "2020-01-01", "2020-12-31")
    assert failed == ["sh.600000"]
    assert calls["n"] == 3                         # 1 + 2 重试 = 3 次实际请求
    # 登录 1 + 3 次拉取 × cost 2 = 6 + 登出 1 = 8
    assert q.count == 1 + 3 * 2 + 1
