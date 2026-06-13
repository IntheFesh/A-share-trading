"""腾讯盘中快照采集器。Phase 0 先就位,Phase 4 主用。对应 v3.1 §2.2 / 第十二章。

解析 qt.gtimg.cn 返回(GBK 编码、'~' 分隔)。**解析函数 parse_tencent 不依赖网络,可单测**;
fetch_snapshot 走网络(测试 skip)。限频从 config/data.yaml 读:批量 ≤100 只/请求、全局 ≤10 次/秒。
返回盘中价为执行类参考(raw)。
"""

from __future__ import annotations

import re
from typing import Optional

_LINE_RE = re.compile(r'v_(\w+)="([^"]*)"')


def _f(values: list, idx: int) -> Optional[float]:
    """安全取第 idx 个字段并转 float;越界或空值返回 None。"""
    if idx >= len(values) or values[idx] == "":
        return None
    try:
        return float(values[idx])
    except ValueError:
        return None


def parse_tencent(text: str) -> "list[dict]":
    """解析腾讯返回文本(可含多行 ``v_sh600519="...";``)。

    字段下标(gtimg 约定):1=名称, 2=代码, 3=现价, 4=昨收, 5=今开, 6=成交量(手),
    33=最高, 34=最低。未知/越界字段返回 None,不臆造。
    """
    out: list[dict] = []
    for sym, payload in _LINE_RE.findall(text):
        f = payload.split("~")
        out.append(
            {
                "symbol": sym,
                "name": f[1] if len(f) > 1 else None,
                "code": f[2] if len(f) > 2 else None,
                "price": _f(f, 3),
                "preclose": _f(f, 4),
                "open": _f(f, 5),
                "volume_hand": _f(f, 6),
                "high": _f(f, 33),
                "low": _f(f, 34),
            }
        )
    return out


def fetch_snapshot(codes: "list[str]", base_url: str = "https://qt.gtimg.cn/q=") -> "list[dict]":
    """批量取盘中快照(GBK 解码后交 parse_tencent)。codes 形如 ['sh600519','sz000001']。"""
    import requests

    if len(codes) > 100:
        raise ValueError("腾讯快照单请求 ≤100 只(见 config/data.yaml realtime_rate_limit)")
    resp = requests.get(base_url + ",".join(codes), timeout=5)
    resp.encoding = "gbk"
    return parse_tencent(resp.text)
