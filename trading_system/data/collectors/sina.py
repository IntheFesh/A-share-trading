"""新浪盘中快照采集器(备源)。Phase 0 先就位,Phase 4 主用。对应 v3.1 §2.2。

解析 hq.sinajs.cn 返回(GBK、逗号分隔)。**解析函数 parse_sina 不依赖网络,可单测**;
fetch_snapshot 需带 Referer 头(见 config/data.yaml)。限频同腾讯:批量 ≤100、全局 ≤10/s。
"""

from __future__ import annotations

import re
from typing import Optional

_LINE_RE = re.compile(r'var hq_str_(\w+)="([^"]*)"')


def _f(values: list, idx: int) -> Optional[float]:
    if idx >= len(values) or values[idx] == "":
        return None
    try:
        return float(values[idx])
    except ValueError:
        return None


def parse_sina(text: str) -> "list[dict]":
    """解析新浪返回文本(可含多行 ``var hq_str_sh600519="...";``)。

    字段下标(新浪约定):0=名称, 1=今开, 2=昨收, 3=现价, 4=最高, 5=最低, 30=日期, 31=时间。
    """
    out: list[dict] = []
    for sym, payload in _LINE_RE.findall(text):
        f = payload.split(",")
        out.append(
            {
                "symbol": sym,
                "name": f[0] if len(f) > 0 else None,
                "open": _f(f, 1),
                "preclose": _f(f, 2),
                "price": _f(f, 3),
                "high": _f(f, 4),
                "low": _f(f, 5),
                "date": f[30] if len(f) > 30 else None,
                "time": f[31] if len(f) > 31 else None,
            }
        )
    return out


def fetch_snapshot(
    codes: "list[str]",
    base_url: str = "https://hq.sinajs.cn/list=",
    referer: str = "https://finance.sina.com.cn",
    max_per_sec: float = 10.0,
) -> "list[dict]":
    """批量取盘中快照:自动按 ≤100 只/请求分批、全局 ≤max_per_sec 次/秒限频(带 Referer 头)。"""
    import requests

    from trading_system.data.collectors._ratelimit import RateLimiter, chunked

    limiter = RateLimiter(max_per_sec)
    out: list[dict] = []
    for batch in chunked(list(codes), 100):
        limiter.wait()
        resp = requests.get(base_url + ",".join(batch), headers={"Referer": referer}, timeout=5)
        resp.encoding = "gbk"
        out.extend(parse_sina(resp.text))
    return out
