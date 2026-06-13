"""环境自检脚本:``python -m trading_system.check_env``。

Phase: 零、技术栈与环境。检查内容:
  1) Python 版本 >= 3.11;
  2) 依赖是否就位(区分"全程必需 / 数据源 / Phase 3 才用"),并打印版本;
  3) ``config/*.yaml`` 是否齐备且可加载;
  4) 七条不变量原语(``trading_system.invariants``)是否通过冒烟断言。

退出码:全部必需项就位且不变量冒烟通过 -> 0;否则 -> 1(阻断,需先修复)。
说明:本脚本是环境诊断 CLI,使用 ``logging`` 输出到 stdout(非业务逻辑、非作战手册)。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger("check_env")

MIN_PYTHON = (3, 11)

CONFIG_DIR = Path(__file__).resolve().parent / "config"
EXPECTED_CONFIGS = (
    "data.yaml",
    "costs.yaml",
    "triggers.yaml",
    "risk.yaml",
    "exit.yaml",
    "regime.yaml",
    "train.yaml",
)

# (import 名, 是否阻断, 用途说明)。阻断=True 表示缺失则自检失败。
REQUIRED_DEPS = (
    ("pandas", True, "数据处理(全程)"),
    ("numpy", True, "数值计算(全程)"),
    ("pyarrow", True, "Parquet 落盘(Phase 0)"),
    ("duckdb", True, "本地 SQL 查询(Phase 0)"),
    ("yaml", True, "读 config/*.yaml(全程;包名 pyyaml)"),
    ("scipy", True, "统计检验(Phase 1+)"),
    ("sklearn", True, "ElasticNet 基线 + 工具(Phase 2;包名 scikit-learn)"),
    ("matplotlib", True, "落盘静态图(Phase 1+)"),
)
DATASOURCE_DEPS = (
    ("baostock", True, "日线后复权主源(Phase 0;无 token)"),
    ("tushare", False, "仅财报获取:业绩预告/预约披露日(Phase 0;需 token;非信息来源)"),
    ("requests", True, "腾讯/新浪盘中快照(Phase 4;Phase 0 先就位)"),
)
PHASE3_DEPS = (
    ("lightgbm", False, "L2 模型(Phase 3 才用)"),
    ("optuna", False, "超参搜索(Phase 3 才用)"),
)


def _check_python() -> bool:
    ok = sys.version_info[:2] >= MIN_PYTHON
    logger.info(
        "Python 版本: %s.%s.%s  (要求 >= %s.%s)  -> %s",
        *sys.version_info[:3],
        *MIN_PYTHON,
        "OK" if ok else "不满足",
    )
    return ok


def _probe(module_name: str) -> tuple[bool, str]:
    """探测某依赖是否可导入,返回 (是否就位, 版本或原因)。不导入则不引入重型依赖。"""
    if importlib.util.find_spec(module_name) is None:
        return False, "未安装"
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — 自检要如实报告任何导入异常
        return False, f"导入失败: {exc!r}"
    return True, getattr(mod, "__version__", "(无版本号)")


def _check_dep_group(title: str, deps: tuple[tuple[str, bool, str], ...]) -> bool:
    """检查一组依赖;返回该组的"阻断项是否全部就位"。"""
    logger.info("--- %s ---", title)
    group_ok = True
    for name, blocking, purpose in deps:
        present, info = _probe(name)
        tag = "OK " if present else ("缺失[阻断]" if blocking else "缺失[可选]")
        logger.info("  [%s] %-12s %-22s %s", tag, name, info if present else "", purpose)
        if blocking and not present:
            group_ok = False
    return group_ok


def _check_configs() -> bool:
    logger.info("--- 配置文件 (config/*.yaml) ---")
    yaml_present = importlib.util.find_spec("yaml") is not None
    if not yaml_present:
        logger.info("  pyyaml 未安装,跳过 YAML 解析,仅检查文件是否存在。")
    ok = True
    for fname in EXPECTED_CONFIGS:
        path = CONFIG_DIR / fname
        if not path.exists():
            logger.info("  [缺失[阻断]] %s 不存在", fname)
            ok = False
            continue
        if yaml_present:
            import yaml  # 局部导入,避免顶层依赖

            try:
                with path.open("r", encoding="utf-8") as fh:
                    yaml.safe_load(fh)
                logger.info("  [OK ] %s 可解析", fname)
            except Exception as exc:  # noqa: BLE001
                logger.info("  [解析失败[阻断]] %s -> %r", fname, exc)
                ok = False
        else:
            logger.info("  [存在] %s", fname)
    return ok


def _check_invariants() -> bool:
    """冒烟跑七条不变量原语,确保宪法模块本身可用。"""
    logger.info("--- 不变量冒烟 (trading_system.invariants) ---")
    try:
        from trading_system import invariants as inv

        # INV-1
        inv.assert_tradeable_exit(10, 12)
        _expect_raise(lambda: inv.assert_tradeable_exit(10, 11), "INV-1")
        # INV-2
        assert inv.limit_up_price(10.00) == 11.00
        assert inv.limit_down_price(10.00) == 9.00
        _expect_raise(lambda: inv.assert_execution_uses_raw(["close_adj"]), "INV-2")
        # INV-4
        _expect_raise(
            lambda: inv.assert_group_constant_only_via_interaction(
                [inv.FeatureSpec("T_t", group_constant=True)]
            ),
            "INV-4",
        )
        # INV-5(附录 B 数值)
        assert abs(inv.continuous_limit_down_loss(0.08, 2) - 0.0152) < 1e-9
        assert abs(inv.continuous_limit_down_loss(0.05, 2) - 0.0095) < 1e-9
        # INV-6
        ledger = inv.BlindSegmentLedger()
        ledger.use_for_decision("smoke")
        _expect_raise(lambda: ledger.use_for_decision("smoke"), "INV-6")
        # INV-7
        _expect_raise(
            lambda: inv.assert_conditional_or_documented_override(is_unconditional=True),
            "INV-7",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("  不变量冒烟失败: %r", exc)
        return False
    logger.info("  [OK ] 七条不变量原语冒烟通过(INV-3 待 Phase 1/2 落地)")
    return True


def _expect_raise(fn, code: str) -> None:
    """断言 ``fn()`` 抛出 InvariantViolation;否则报错。"""
    from trading_system.invariants import InvariantViolation

    try:
        fn()
    except InvariantViolation:
        return
    raise AssertionError(f"{code} 守卫未按预期抛出 InvariantViolation")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info("=== A股交易系统环境自检 (Phase 0 前置) ===")

    results = {
        "python": _check_python(),
        "required_deps": _check_dep_group("全程必需依赖", REQUIRED_DEPS),
        "datasource_deps": _check_dep_group("数据源依赖", DATASOURCE_DEPS),
        "phase3_deps": _check_dep_group("Phase 3 依赖(现在可缺)", PHASE3_DEPS),
        "configs": _check_configs(),
        "invariants": _check_invariants(),
    }

    # Phase 3 依赖现在缺失不阻断;其余阻断组都要 OK。
    blocking_ok = all(
        results[k] for k in ("python", "required_deps", "datasource_deps", "configs", "invariants")
    )
    logger.info("=== 结论: %s ===", "通过(可进入 Phase 0)" if blocking_ok else "存在阻断项,请先修复")
    if not blocking_ok:
        logger.info("提示:缺失依赖请运行  pip install -r requirements.txt")
    return 0 if blocking_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
