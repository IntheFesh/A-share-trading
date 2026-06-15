"""存储与增量更新(单一数据出口)。Phase 0(任务 0.4)。对应 v3.1 §2.2。

Parquet(ZSTD)按年分区落盘(layout: ``root/year=YYYY/part.parquet``);
DuckDB 直接对 Parquet 跑 SQL。``read(...)`` 是所有模块取数的**唯一入口**(沙盒/回测/生产
同一份数据)。``update_incremental`` 只在年分区级别重写受影响分区,不全量重拉、不全量重写。
价格层:read(fields=...) 配合 schema 实现"执行取 raw、特征取 adj"的按用途取数(INV-2)。
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from trading_system.data.schema import ALL_FIELDS, KEY_FIELDS

_KNOWN_FIELDS = set(ALL_FIELDS)
# 原子写临时文件名后缀模式(供清理孤儿临时文件;形如 part.parquet.tmp.<pid>.<uuid>)。
_TMP_GLOB = "*.tmp.*"


def _to_ts(value) -> pd.Timestamp:
    return pd.Timestamp(value)


def atomic_write_parquet(df: pd.DataFrame, target: "str | Path", compression: str = "zstd") -> None:
    """原子落盘:写同目录临时文件 + ``os.replace`` 替换为正式文件。

    ``os.replace`` 在同一文件系统上是原子的——要么旧文件、要么新文件,**永不出现半个文件**;
    即使在 rename 瞬间断电,文件系统层面也只会留下旧/新之一。写临时文件过程中抛异常 → 删除残留
    临时文件,**原 target 保持不变**(旧数据不被破坏)。供 ParquetStore / Financial / Industry 共用。
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        df.to_parquet(tmp, compression=compression, index=False)
        os.replace(tmp, target)          # 原子替换(同一文件系统)
    except BaseException:                 # noqa: BLE001 — 写一半失败:清理临时文件,保住旧数据后重抛
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def cleanup_orphan_tmp(root: "str | Path") -> int:
    """清理 root 下(根级 + 各 year=* 目录)上次中断遗留的孤儿临时文件(*.tmp.*)。返回清理个数。"""
    root = Path(root)
    if not root.exists():
        return 0
    n = 0
    for t in list(root.glob(_TMP_GLOB)) + list(root.glob(f"year=*/{_TMP_GLOB}")):
        try:
            t.unlink()
            n += 1
        except OSError:
            pass
    return n


class ParquetStore:
    """按年分区的 Parquet 数据出口 + DuckDB 查询。"""

    def __init__(self, root: "str | Path", compression: str = "zstd") -> None:
        self.root = Path(root)
        self.compression = compression
        self.root.mkdir(parents=True, exist_ok=True)
        cleanup_orphan_tmp(self.root)        # 清理上次中断遗留的孤儿临时文件,避免堆积

    # ── 内部:年分区路径与现有数据 ──
    def _year_dir(self, year: int) -> Path:
        return self.root / f"year={year}"

    def _year_file(self, year: int) -> Path:
        return self._year_dir(year) / "part.parquet"

    def _existing_years(self) -> list[int]:
        years = []
        for d in self.root.glob("year=*"):
            try:
                years.append(int(d.name.split("=", 1)[1]))
            except (ValueError, IndexError):
                continue
        return sorted(years)

    def _read_year(self, year: int) -> pd.DataFrame:
        path = self._year_file(year)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _write_year(self, year: int, df: pd.DataFrame) -> None:
        self._year_dir(year).mkdir(parents=True, exist_ok=True)
        df = df.sort_values(list(KEY_FIELDS)).reset_index(drop=True)
        # 原子写:中断也不会损坏整年文件(写临时 + os.replace);失败则保住旧 part.parquet。
        atomic_write_parquet(df, self._year_file(year), self.compression)

    @staticmethod
    def _with_year(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out["_year"] = out["trade_date"].dt.year
        return out

    # ── 写入(初始全量或覆盖)──
    def write(self, df: pd.DataFrame) -> int:
        """覆盖式写入:按年分区,每个出现的年份分区被该年数据覆盖。返回写入行数。"""
        if df.empty:
            return 0
        tagged = self._with_year(df)
        for year, g in tagged.groupby("_year"):
            self._write_year(int(year), g.drop(columns="_year"))
        return len(df)

    # ── 增量更新(只拉/只改新数据)──
    def update_incremental(self, new_df: pd.DataFrame) -> int:
        """按 (code, trade_date) 增量合并:每只票只接受 > 本地最新日的新行,
        去重保留最新抓取版本,只重写受影响的年分区。返回实际新增行数。
        """
        if new_df.empty:
            return 0
        local_max = self.local_max_dates()  # code -> Timestamp
        nd = self._with_year(new_df)
        if local_max:
            keep = nd.apply(
                lambda r: r["trade_date"] > local_max.get(r["code"], pd.Timestamp.min),
                axis=1,
            )
            nd = nd[keep]
        if nd.empty:
            return 0
        appended = 0
        for year, g in nd.groupby("_year"):
            year = int(year)
            existing = self._read_year(year)
            g2 = g.drop(columns="_year")
            if not existing.empty:
                existing["trade_date"] = pd.to_datetime(existing["trade_date"])
                combined = pd.concat([existing, g2], ignore_index=True)
            else:
                combined = g2
            before = len(existing)
            combined = combined.drop_duplicates(
                subset=list(KEY_FIELDS), keep="last"
            ).reset_index(drop=True)
            self._write_year(year, combined)
            appended += len(combined) - before
        return appended

    def local_max_dates(self) -> dict:
        """返回 {code: 本地最新 trade_date(Timestamp)};无数据则空 dict。"""
        years = self._existing_years()
        if not years:
            return {}
        frames = [self._read_year(y)[["code", "trade_date"]] for y in years]
        alld = pd.concat(frames, ignore_index=True)
        alld["trade_date"] = pd.to_datetime(alld["trade_date"])
        s = alld.groupby("code")["trade_date"].max()
        return {k: v for k, v in s.items()}

    # ── 唯一取数入口 ──
    def read(
        self,
        codes: "Optional[Sequence[str]]" = None,
        start=None,
        end=None,
        fields: "Optional[Sequence[str]]" = None,
    ) -> pd.DataFrame:
        """唯一取数入口。按 code / [start,end] / 字段读取(DuckDB 谓词下推)。

        fields 不含 KEY_FIELDS 时自动补上 (code, trade_date)。请求不存在的列会显式报错
        (诚实失败,不静默丢列)。返回按 (code, trade_date) 排序的 DataFrame。
        """
        import duckdb

        years = self._existing_years()
        if not years:
            return pd.DataFrame()

        # 选择列
        if fields is None:
            select_cols = "*"
            requested = None
        else:
            requested = list(dict.fromkeys([*KEY_FIELDS, *fields]))
            unknown = [f for f in requested if f not in _KNOWN_FIELDS]
            if unknown:
                raise ValueError(f"read 请求了未知字段: {unknown}")
            # 校验列确实存在于数据集中
            available = set(self._read_year(years[0]).columns)
            missing = [f for f in requested if f not in available]
            if missing:
                raise ValueError(f"read 请求的字段不在数据集中: {missing}")
            select_cols = ", ".join(f'"{c}"' for c in requested)

        glob = str(self.root / "year=*" / "*.parquet").replace("'", "''")
        where = []
        if codes is not None:
            code_list = ", ".join("'" + str(c).replace("'", "''") + "'" for c in codes)
            where.append(f"code IN ({code_list})")
        if start is not None:
            where.append(f"trade_date >= TIMESTAMP '{_to_ts(start)}'")
        if end is not None:
            where.append(f"trade_date <= TIMESTAMP '{_to_ts(end)}'")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        sql = (
            f"SELECT {select_cols} FROM read_parquet('{glob}', hive_partitioning=0)"
            f"{where_sql} ORDER BY code, trade_date"
        )
        con = duckdb.connect()
        try:
            out = con.execute(sql).df()
        finally:
            con.close()
        if "trade_date" in out.columns:
            out["trade_date"] = pd.to_datetime(out["trade_date"])
        return out

    def query(self, sql: str) -> pd.DataFrame:
        """对底层 Parquet 直接跑 DuckDB SQL。表名用 ``store`` 占位会被替换为 read_parquet。"""
        import duckdb

        glob = str(self.root / "year=*" / "*.parquet").replace("'", "''")
        sql2 = sql.replace("store", f"read_parquet('{glob}', hive_partitioning=0)")
        con = duckdb.connect()
        try:
            return con.execute(sql2).df()
        finally:
            con.close()
