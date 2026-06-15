"""特征/标签缓存(轻量、内容寻址)。批2。

为引擎版标签(逐 (票,日) 模拟,极慢)等重计算提速:以 (数据集指纹 + 特征清单 + 标签配置) 为 key,
把算好的特征/标签矩阵落盘 parquet;数据或配置变化 -> key 变 -> 缓存失效、重算。
**缓存只加速,不改变结果**:命中缓存与重算得到完全相同的矩阵(parquet 对 float64 精确往返)。
缓存目录应加入 .gitignore。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

# 影响特征/标签计算的列(指纹只对这些列取哈希;其它列变化不影响结果)
_FINGERPRINT_COLS = (
    "code", "trade_date", "open_raw", "high_raw", "low_raw", "close_raw", "preclose_raw",
    "volume", "amount", "turn", "adj_factor", "open_adj", "high_adj", "low_adj", "close_adj",
)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """对影响计算的列取内容指纹(顺序无关:按 code,trade_date 排序后哈希)。"""
    cols = [c for c in _FINGERPRINT_COLS if c in df.columns]
    sub = df[cols]
    key_cols = [c for c in ("code", "trade_date") if c in sub.columns]
    if key_cols:
        sub = sub.sort_values(key_cols)
    h = pd.util.hash_pandas_object(sub, index=False)
    return hashlib.md5(h.to_numpy().tobytes()).hexdigest()


def make_key(fingerprint: str, feature_names, label_spec: dict) -> str:
    """由 (数据指纹 + 特征清单含顺序 + 标签配置) 生成缓存 key。"""
    payload = json.dumps(
        {"fp": fingerprint, "feats": list(feature_names), "label": label_spec},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


class FeatureCache:
    """以 key 命名的 parquet 缓存。enabled=False 时所有操作为空(等同无缓存)。"""

    def __init__(self, cache_dir: "str | Path", enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.dir = Path(cache_dir)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.parquet"

    def get(self, key: str) -> "pd.DataFrame | None":
        if not self.enabled:
            return None
        p = self._path(key)
        return pd.read_parquet(p) if p.exists() else None

    def put(self, key: str, df: pd.DataFrame) -> None:
        if not self.enabled:
            return
        df.to_parquet(self._path(key), index=False)


def cache_from_config(config: dict) -> FeatureCache:
    """按 config.training.cache 构造缓存(enabled 默认 False;目录默认 ./train_cache)。"""
    cfg = (config.get("training", {}) or {}).get("cache", {}) or {}
    return FeatureCache(cfg.get("cache_dir", "./train_cache"), enabled=bool(cfg.get("enabled", False)))


def compute_features_cached(work: pd.DataFrame, feature_names, cache: FeatureCache,
                            *, spec_tag: str = "features") -> pd.DataFrame:
    """在已按 (code,trade_date) 排序的 work 上计算特征,命中缓存则读、否则算后写。

    缓存命中与重算结果完全一致(指纹相同 -> 同一份排序数据 -> 逐行位置对齐)。返回填好特征列的 work。
    """
    from trading_system.features.registry import compute_feature

    key = make_key(dataset_fingerprint(work), feature_names, {"tag": spec_tag})
    cached = cache.get(key)
    if cached is not None and len(cached) == len(work):
        for f in feature_names:
            work[f] = cached[f].to_numpy() if f in cached.columns \
                else compute_feature(f, work).reindex(work.index)
        return work
    for f in feature_names:
        work[f] = compute_feature(f, work).reindex(work.index)
    cache.put(key, work[["code", "trade_date", *feature_names]])
    return work
