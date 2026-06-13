"""入口脚本 3/5:每日预测(日常使用)。补丁。

职责:只加载已训练模型(绝不在内部训练);加载后第一步硬核对特征一致性;PIT 算特征(无 T+1);
套模型排序出作战手册。显式打印所用模型文件与训练截止日,便于下单前判断模型新旧。
用法:  python run_predict.py --asof YYYY-MM-DD [--model <模型路径>] [--dataset <数据集目录>]
默认 --model 取 model_dir 下最新的模型(实际部署中应指向当前冠军)。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trading_system.config import load_config

logger = logging.getLogger("run_predict")


def latest_model(model_dir) -> Path:
    models = sorted(Path(model_dir).glob("model_*.pkl"))
    if not models:
        raise FileNotFoundError(f"{model_dir} 下无模型;请先运行 run_train.py")
    return models[-1]


def run(config: dict, *, asof: str, model_path: str = None, dataset_path: str = None, dataset_df=None):
    from trading_system.data.store import ParquetStore
    from trading_system.predict import run_prediction

    # 路径/特征清单/风控参数均在 config.yaml 中修改,请勿在此硬改
    if dataset_df is None:
        dataset_df = ParquetStore(dataset_path or config["paths"]["data_dir"]).read()
    model_path = model_path or latest_model(config["paths"]["model_dir"])
    return run_prediction(dataset_df, asof_date=asof, config=config, model_path=str(model_path),
                          output_dir=config["paths"]["output_dir"])


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    ap = argparse.ArgumentParser(description="每日预测(参数见 config.yaml)")
    ap.add_argument("--asof", required=True, help="预测基准日(T 日收盘)YYYY-MM-DD")
    ap.add_argument("--model", default=None, help="模型路径(默认取 model_dir 下最新)")
    ap.add_argument("--dataset", default=None, help="数据集目录(默认 config.paths.data_dir)")
    a = ap.parse_args(argv)
    run(load_config(), asof=a.asof, model_path=a.model, dataset_path=a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
