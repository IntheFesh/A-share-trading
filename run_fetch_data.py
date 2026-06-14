"""入口脚本 1/5:取数。补丁。

职责:从 config.yaml 读路径/区间/披露开关,调用既有采集逻辑(BaoStock 硬依赖 + Tushare 软依赖降级),
落盘至既有 store(单一存储,不另起)。落盘后写一张**带时间戳、不覆盖历史**的"数据体检卡"
(dataset_card_<时间戳>.json)记录票数/日期范围/字段齐全性,供复现追溯。
用法:  python run_fetch_data.py [--end YYYY-MM-DD] [--full]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from trading_system.config import load_config
from trading_system.data.fetch_training_data import run_fetch
from trading_system.data.store import ParquetStore

logger = logging.getLogger("run_fetch_data")


def build_dataset_card(store, *, start, end, universe, disclosure, exit_code) -> dict:
    from trading_system.data.schema import PRICE_LAYER_FIELDS

    local_max = store.local_max_dates()
    n_codes = len(local_max)
    date_max = max(local_max.values()).strftime("%Y-%m-%d") if local_max else None
    fields_complete = None
    if n_codes:
        sample = store.read(codes=[next(iter(local_max))])
        fields_complete = all(c in sample.columns for c in PRICE_LAYER_FIELDS)
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "start": start, "end": end, "universe": universe, "enable_disclosure": disclosure,
        "n_codes": n_codes, "date_max": date_max,
        "price_layer_fields_complete": fields_complete, "fetch_exit_code": exit_code,
    }


def write_dataset_card(card: dict, output_dir) -> Path:
    """带时间戳落盘,已存在则追加序号,绝不覆盖历史。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = output_dir / f"dataset_card_{stamp}.json"
    i = 2
    while p.exists():
        p = output_dir / f"dataset_card_{stamp}_{i}.json"
        i += 1
    p.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def run(config: dict, *, end=None, full: bool = False, limit=None,
        baostock_collector=None, tushare_collector_factory=None, universe_codes=None) -> int:
    # 本参数在 config.yaml 中修改,请勿在此硬改(改这里会导致脚本间参数不一致)
    paths, data = config["paths"], config["data"]
    store = ParquetStore(paths["data_dir"])
    rc = run_fetch(
        start=data["start"], end=end, universe=config["universe"],
        enable_disclosure=data["enable_disclosure"], incremental=not full, store=store,
        baostock_collector=baostock_collector, tushare_collector_factory=tushare_collector_factory,
        universe_codes=universe_codes, limit=limit,
        # 健壮取数:单票超时看门狗 + 分批落盘 + 待拉/失败清单 + 单日配额保护
        # (均从 config.yaml 读,默认有兜底)
        request_timeout_sec=data.get("request_timeout_sec", 45),
        batch_save_size=data.get("batch_save_size", 200),
        output_dir=paths["output_dir"],
        daily_request_limit=data.get("daily_request_limit", 50000),
        daily_request_safety_margin=data.get("daily_request_safety_margin", 5000),
    )
    if rc == 0:
        card = build_dataset_card(store, start=data["start"], end=end, universe=config["universe"],
                                  disclosure=data["enable_disclosure"], exit_code=rc)
        path = write_dataset_card(card, paths["output_dir"])
        logger.info("数据体检卡: %s", json.dumps(card, ensure_ascii=False))
        logger.info("体检卡已写入: %s", path)
        logger.info("下一步:python run_train.py --dataset %s --train-end <日期>", paths["data_dir"])
    return rc


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    ap = argparse.ArgumentParser(description="取数(参数见 config.yaml)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--full", action="store_true", help="全量重拉(默认增量)")
    ap.add_argument("--limit", type=int, default=None,
                    help="只采集前 N 只票(按代码排序);默认 None=全部")
    a = ap.parse_args(argv)
    return run(load_config(), end=a.end, full=a.full, limit=a.limit)


if __name__ == "__main__":
    raise SystemExit(main())
