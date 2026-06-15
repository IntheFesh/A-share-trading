"""行业分类独立落盘(批 4)。低频近静态:主键 ``code``,单文件,每次采集覆盖/更新(不逐日)。

与日频行情、季频财务都物理隔离。供后续行业中性化、板块共振识别使用(本批仅采集落盘,不进打分)。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_system.data.schema import INDUSTRY_FIELDS
from trading_system.data.store import atomic_write_parquet, cleanup_orphan_tmp


class IndustryStore:
    """行业分类的独立 Parquet 出口(主键 code;单文件 upsert)。"""

    def __init__(self, root: "str | Path", compression: str = "zstd") -> None:
        self.root = Path(root)
        self.compression = compression
        self.root.mkdir(parents=True, exist_ok=True)
        cleanup_orphan_tmp(self.root)        # 清理上次中断遗留的孤儿临时文件

    @property
    def _file(self) -> Path:
        return self.root / "industry.parquet"

    def read(self, codes: "list[str] | None" = None) -> pd.DataFrame:
        """读取行业分类(可按 code 过滤),按 code 排序。无数据返回带 schema 列的空表。"""
        if not self._file.exists():
            return pd.DataFrame(columns=list(INDUSTRY_FIELDS))
        out = pd.read_parquet(self._file)
        if codes is not None:
            out = out[out["code"].isin(list(codes))]
        return out.sort_values("code").reset_index(drop=True)

    def update(self, new_df: pd.DataFrame) -> int:
        """按 code upsert(keep last:同一 code 以最新采集为准,保留未在本次采集中的旧 code)。
        返回落盘后的总行数。"""
        if new_df is None or new_df.empty:
            return len(self.read())
        new_df = new_df[[c for c in INDUSTRY_FIELDS if c in new_df.columns]].copy()
        existing = self.read()
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined = combined.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)
        combined = combined.sort_values("code").reset_index(drop=True)
        atomic_write_parquet(combined, self._file, self.compression)        # 原子写,中断不损坏
        return len(combined)
