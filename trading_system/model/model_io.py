"""模型"出生证明"包:训练产物的持久化与一致性校验。补丁。

防特征错位的关键:模型不裸存权重,而是连同**特征名(含顺序)/训练区间/超参/config 快照/时间戳/
git hash** 一起打包。predict 加载时第一步逐一核对特征名与顺序,不一致即报错退出(硬闸)。
文件名带时间戳且不覆盖历史(如 model_20260613_153000.pkl)。
"""

from __future__ import annotations

import datetime as dt
import pickle
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class FeatureMismatchError(RuntimeError):
    """当前特征名 / 顺序与模型出生证明不一致(防止静默错位)。"""


def _git_hash() -> "str | None":
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 — 无 git 环境时静默返回 None
        return None


@dataclass
class ModelCard:
    """模型出生证明。"""

    model: Any
    feature_names: list                # 含顺序,预测时逐一核对
    train_start: str
    train_end: str
    params: dict
    config_snapshot: dict
    route: str = "C"
    trained_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    git_hash: "str | None" = field(default_factory=_git_hash)


def save_model(card: ModelCard, model_dir: "str | Path") -> Path:
    """落盘模型出生证明包;文件名带时间戳,已存在则追加序号,**绝不覆盖历史**。"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = model_dir / f"model_{stamp}.pkl"
    i = 2
    while path.exists():  # 不覆盖:同秒内多次保存时追加序号
        path = model_dir / f"model_{stamp}_{i}.pkl"
        i += 1
    with open(path, "wb") as fh:
        pickle.dump(card, fh)
    return path


def load_model(path: "str | Path") -> ModelCard:
    with open(path, "rb") as fh:
        card = pickle.load(fh)
    if not isinstance(card, ModelCard):
        raise TypeError(f"{path} 不是 ModelCard 出生证明包")
    return card


def assert_feature_consistency(card: ModelCard, current_features: "list[str]") -> None:
    """硬闸:当前特征名与顺序必须与出生证明逐一吻合,否则报错退出(防特征错位的静默错误)。"""
    if list(current_features) != list(card.feature_names):
        raise FeatureMismatchError(
            "特征名/顺序与模型出生证明不一致——拒绝预测以防错位。\n"
            f"  模型: {card.feature_names}\n  当前: {list(current_features)}"
        )
