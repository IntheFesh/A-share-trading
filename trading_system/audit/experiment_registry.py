"""实验注册表(SQLite 持久化)。Phase 3 起使用。对应 v3.1 第八/十三章 / INV-6。

持久化两件事:
  1) 盲测段一次性(INV-6):某段用于 champion-challenger 换届裁决后即 archived,再用报错;
  2) 全部 Optuna trial 入库,供 DSR 的 N(完整研究账本)与 PBO 记账。
纯逻辑内核见 invariants.BlindSegmentLedger;本类在其语义上加 SQLite 落盘,跨进程/重启仍生效。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from trading_system.invariants import BlindSegmentStatus, InvariantViolation


class ExperimentRegistry:
    """SQLite 持久化的实验 / 盲测段注册表。"""

    def __init__(self, db_path: "str | Path") -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS blind_segments ("
                "segment_id TEXT PRIMARY KEY, status TEXT NOT NULL, used_at TEXT)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS trials ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, study TEXT NOT NULL, "
                "params TEXT NOT NULL, value REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )

    # ── 盲测段一次性(INV-6)──
    def status(self, segment_id: str) -> BlindSegmentStatus:
        with self._conn() as con:
            row = con.execute(
                "SELECT status FROM blind_segments WHERE segment_id=?", (segment_id,)
            ).fetchone()
        if row is None:
            return BlindSegmentStatus.UNUSED
        return BlindSegmentStatus(row["status"])

    def assert_available(self, segment_id: str) -> None:
        if self.status(segment_id) is BlindSegmentStatus.ARCHIVED:
            raise InvariantViolation(
                "INV-6", f"盲测段 '{segment_id}' 已封存(用过一次),禁止再次用于裁决/调参/选择。"
            )

    def use_for_decision(self, segment_id: str) -> None:
        """用于一次换届裁决并封存(持久化)。再用同段抛 INV-6。"""
        self.assert_available(segment_id)
        with self._conn() as con:
            con.execute(
                "INSERT INTO blind_segments(segment_id, status, used_at) "
                "VALUES(?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(segment_id) DO UPDATE SET status=excluded.status, used_at=excluded.used_at",
                (segment_id, BlindSegmentStatus.ARCHIVED.value),
            )

    # ── Optuna trial 账本(供 DSR 的 N 与 PBO 记账)──
    def log_trial(self, study: str, params: dict, value: float) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO trials(study, params, value) VALUES(?, ?, ?)",
                (study, json.dumps(params, ensure_ascii=False), float(value)),
            )

    def n_trials(self, study: "str | None" = None) -> int:
        """研究账本中的 trial 总数(DSR 去膨胀的 N)。study=None 统计全部。"""
        with self._conn() as con:
            if study is None:
                row = con.execute("SELECT COUNT(*) AS n FROM trials").fetchone()
            else:
                row = con.execute("SELECT COUNT(*) AS n FROM trials WHERE study=?", (study,)).fetchone()
        return int(row["n"])

    def list_trials(self, study: "str | None" = None) -> list:
        with self._conn() as con:
            if study is None:
                rows = con.execute("SELECT study, params, value FROM trials ORDER BY id").fetchall()
            else:
                rows = con.execute(
                    "SELECT study, params, value FROM trials WHERE study=? ORDER BY id", (study,)
                ).fetchall()
        return [{"study": r["study"], "params": json.loads(r["params"]), "value": r["value"]}
                for r in rows]
