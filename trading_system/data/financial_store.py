"""季频财务数据独立落盘(批 2)。与日频行情 store 物理隔离(频率不同,混表会乱)。

主键 ``(code, statDate)``,按 statDate 年份分区(``root/year=YYYY/part.parquet``),增量去重(keep last)。
**PIT 关键:pubDate 列原样保留**——批 3 做可见性对齐时只能用 pubDate(实际公告日),绝不能用 statDate
(报告期),否则会在报告期当日就"看到"尚未公告的财报,构成未来函数泄漏。
风格对齐 store.ParquetStore(同样的年分区 + 增量去重),但键为 statDate 而非 trade_date。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trading_system.data.schema import FINANCIAL_FIELDS, FINANCIAL_KEY_FIELDS
from trading_system.data.store import atomic_write_parquet, cleanup_orphan_tmp


class FinancialStore:
    """季频财务面板的独立 Parquet 出口(主键 code+statDate;保留 pubDate)。"""

    def __init__(self, root: "str | Path", compression: str = "zstd") -> None:
        self.root = Path(root)
        self.compression = compression
        self.root.mkdir(parents=True, exist_ok=True)
        cleanup_orphan_tmp(self.root)        # 清理上次中断遗留的孤儿临时文件

    # ── 年分区(按 statDate 年份)──
    def _year_dir(self, year: int) -> Path:
        return self.root / f"year={year}"

    def _year_file(self, year: int) -> Path:
        return self._year_dir(year) / "part.parquet"

    def _existing_years(self) -> "list[int]":
        years = []
        for d in self.root.glob("year=*"):
            try:
                years.append(int(d.name.split("=", 1)[1]))
            except (ValueError, IndexError):
                continue
        return sorted(years)

    def _read_year(self, year: int) -> pd.DataFrame:
        path = self._year_file(year)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _write_year(self, year: int, df: pd.DataFrame) -> None:
        self._year_dir(year).mkdir(parents=True, exist_ok=True)
        df = df.sort_values(list(FINANCIAL_KEY_FIELDS)).reset_index(drop=True)
        atomic_write_parquet(df, self._year_file(year), self.compression)   # 原子写,中断不损坏

    @staticmethod
    def _tag_year(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["statDate"] = pd.to_datetime(out["statDate"], errors="coerce")
        out = out[out["statDate"].notna()]           # statDate 不可解析则无法按主键归位,丢弃
        out["pubDate"] = pd.to_datetime(out["pubDate"], errors="coerce")
        out["_year"] = out["statDate"].dt.year
        return out

    def update_incremental(self, new_df: pd.DataFrame) -> int:
        """按 (code, statDate) 增量合并(keep last:同一报告期以最新抓取版本为准,含 pubDate 更新)。
        只重写受影响的年分区。返回实际新增行数(净增,不含被覆盖更新的)。"""
        if new_df is None or new_df.empty:
            return 0
        nd = self._tag_year(new_df)
        if nd.empty:
            return 0
        appended = 0
        for year, g in nd.groupby("_year"):
            year = int(year)
            existing = self._read_year(year)
            g2 = g.drop(columns="_year")
            if not existing.empty:
                existing["statDate"] = pd.to_datetime(existing["statDate"], errors="coerce")
                combined = pd.concat([existing, g2], ignore_index=True)
            else:
                combined = g2
            before = len(existing)
            combined = combined.drop_duplicates(
                subset=list(FINANCIAL_KEY_FIELDS), keep="last"
            ).reset_index(drop=True)
            self._write_year(year, combined)
            appended += len(combined) - before
        return appended

    def read(self, codes: "list[str] | None" = None) -> pd.DataFrame:
        """读取财务面板(可按 code 过滤),按 (code, statDate) 排序。无数据返回空表(带 schema 列)。"""
        years = self._existing_years()
        if not years:
            return pd.DataFrame(columns=list(FINANCIAL_FIELDS))
        frames = [self._read_year(y) for y in years]
        out = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame(columns=list(FINANCIAL_FIELDS))
        if out.empty:
            return out
        out["statDate"] = pd.to_datetime(out["statDate"], errors="coerce")
        if "pubDate" in out.columns:
            out["pubDate"] = pd.to_datetime(out["pubDate"], errors="coerce")
        if codes is not None:
            out = out[out["code"].isin(list(codes))]
        return out.sort_values(list(FINANCIAL_KEY_FIELDS)).reset_index(drop=True)

    def local_keys(self) -> set:
        """已落盘的 (code, statDate) 主键集合(供断点续传判断,可选)。"""
        df = self.read()
        if df.empty:
            return set()
        return set(zip(df["code"], pd.to_datetime(df["statDate"])))
