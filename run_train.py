"""入口脚本 2/5:训练(模型的唯一生产者,也是冠军挑战者中"挑战者"的来源)。补丁。

职责:用截至 --train-end 的数据训练 L2 模型,存为带时间戳、不覆盖的"出生证明"包(含特征名/顺序、
训练区间、超参、config 快照、时间戳、git hash)。运行时断言 train_end 距盲测段 ≥ embargo(13)。
本脚本只训练,不做对比、不做上线决策。
用法:  python run_train.py --dataset <数据集目录或parquet> --train-end YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trading_system.config import load_config
from trading_system.model.train import train_and_save

logger = logging.getLogger("run_train")


def _load_dataset(dataset_path: str):
    import pandas as pd

    from trading_system.data.store import ParquetStore

    p = Path(dataset_path)
    if p.is_dir():
        return ParquetStore(p).read()
    return pd.read_parquet(p)


def run(config: dict, *, train_end: str, dataset_path: str = None, dataset_df=None) -> Path:
    # 特征清单/区间划分/参数均在 config.yaml 中修改,请勿在此硬改
    if dataset_df is None:
        dataset_df = _load_dataset(dataset_path)
    model_path = train_and_save(
        dataset_df, train_end=train_end, config=config, model_dir=config["paths"]["model_dir"]
    )
    logger.info("模型已保存(出生证明包): %s;训练截止 %s", model_path, train_end)
    logger.info("下一步:python run_predict.py --model %s --asof <日期>", model_path)
    return model_path


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    ap = argparse.ArgumentParser(description="训练(参数见 config.yaml)")
    ap.add_argument("--dataset", required=True, help="数据集目录(store)或 parquet 路径")
    ap.add_argument("--train-end", required=True, help="训练截止日 YYYY-MM-DD")
    a = ap.parse_args(argv)
    run(load_config(), train_end=a.train_end, dataset_path=a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
