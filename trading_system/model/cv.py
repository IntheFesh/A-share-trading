"""purging / embargo(INV 核心)。Phase 3(任务 3.1)。对应 v3.1 第九章。

embargo = H_max + K_limitdown + 1(H_max=10、K=2 -> gap>=13,从 config/train.yaml 读)。
purged 时序 walk-forward:训练块在验证块之前,二者间留 embargo 个交易日的间隔(该间隔 >= H_max
已涵盖标签窗重叠的 purge)。返回每折的训练/验证**日期集合**;行级 mask 由 row_masks 生成。
价格层:与价格无关(仅时间划分)。
"""

from __future__ import annotations

import numpy as np


def embargo_from_config(h_max: int, k_limitdown: int) -> int:
    """embargo = H_max + K_limitdown + 1(v3.1 第九章)。(10,2) -> 13。"""
    return h_max + k_limitdown + 1


def purged_walk_forward_splits(
    sorted_dates: "list | np.ndarray",
    *,
    n_splits: int,
    embargo: int,
    min_train: int = 1,
) -> "list[tuple[np.ndarray, np.ndarray]]":
    """purged 时序 walk-forward 划分。

    把时间轴切成 n_splits+1 块:第 0 块作初始训练,其后 n_splits 块依次作验证;每折训练集为
    "验证开始位置 - embargo" 之前的全部日期(扩张窗 + embargo 间隔)。返回 [(train_dates, val_dates)]。
    """
    dates = np.asarray(sorted_dates)
    T = len(dates)
    if n_splits < 1 or T < (n_splits + 1):
        raise ValueError("数据长度不足以做 n_splits 折划分")
    block = T // (n_splits + 1)
    splits = []
    for k in range(1, n_splits + 1):
        val_start = k * block
        val_end = (k + 1) * block if k < n_splits else T
        train_end_excl = max(0, val_start - embargo)  # purge + embargo 间隔
        train = dates[:train_end_excl]
        val = dates[val_start:val_end]
        if len(train) >= min_train and len(val) > 0:
            splits.append((train, val))
    return splits


def row_masks(
    row_dates: "np.ndarray", train_dates: "np.ndarray", val_dates: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """把日期集合映射为样本行的布尔 mask(train_mask, val_mask)。"""
    rd = np.asarray(row_dates)
    train_set, val_set = set(train_dates.tolist()), set(val_dates.tolist())
    train_mask = np.array([d in train_set for d in rd], dtype=bool)
    val_mask = np.array([d in val_set for d in rd], dtype=bool)
    return train_mask, val_mask
