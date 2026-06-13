"""配置中心加载器(单一参数源)。补丁。

五个入口脚本与编排函数的常用参数一律经 ``load_config`` 读取根目录 ``config.yaml``;脚本内不得
保存该参数的硬编码副本(改 config.yaml 即影响全部脚本)。为便于测试,编排函数都接受注入的
``config`` dict——传入修改过的 config 即可改变行为,从而验证"确实读 config、无硬编码副本"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: "str | Path | None" = None) -> dict:
    """加载 config.yaml(默认根目录)。不缓存:重载即取最新值。"""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config 解析结果非映射: {p}")
    return cfg


def get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """按点路径取值,如 get(cfg, 'risk.stop_loss_atr_mult')。缺失返回 default。"""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
