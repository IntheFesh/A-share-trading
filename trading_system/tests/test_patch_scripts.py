"""补丁:配置中心 + 五脚本职责 + 模型出生证明 + 回测三纪律 + 冠军挑战者(影子式)的离线测试。

全部离线可验证(不需 token/网络);真实采集/影子日度调度路径继续 NOT RUN。
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest import discipline as disc
from trading_system.champion_challenger import (
    ShadowPeriodResult,
    assert_challenger_is_full_model,
    assert_forward_evaluation,
    decide_switch,
    model_in_use,
)
from trading_system.config import load_config
from trading_system.data import price_layers as pl
from trading_system.data.collectors import synthetic
from trading_system.model import model_io
from trading_system.model.cv import assert_train_end_safe
from trading_system.model.train import (
    compare_label_routes,
    compute_time_decay_weights,
    train_and_save,
)


# ── 公共构造 ────────────────────────────────────────────────────────────────
def _dataset(n_days=40, n_codes=15, seed=0):
    cal = synthetic.make_calendar("2020-01-06", n_days)
    codes = [f"sh.60{i:04d}" for i in range(n_codes)]
    return pl.build_price_layers(synthetic.make_raw_panel(codes, cal, seed=seed)), cal


def _config(tmp_path, **overrides):
    cfg = copy.deepcopy(load_config())  # 以真实 config.yaml 为基准
    cfg["paths"] = {"data_dir": str(tmp_path / "data"), "model_dir": str(tmp_path / "models"),
                    "output_dir": str(tmp_path / "out")}
    cfg["features"] = ["ret_1", "ret_5", "ret_20"]            # 加速:小特征集
    cfg["splits"]["blind_segment_start"] = "2030-01-01"       # 远期,默认不触碰
    cfg.setdefault("training", {}).setdefault("cache", {})["cache_dir"] = str(tmp_path / "tcache")
    cfg.update(overrides)
    return cfg


# ── 1) 配置单一源 ───────────────────────────────────────────────────────────
class TestConfigSingleSource:
    def test_load_and_value_drives_behavior(self, tmp_path):
        # 写一份临时 config,embargo 改成 20,断言读到的是新值
        p = tmp_path / "config.yaml"
        p.write_text("splits:\n  embargo_days: 20\n  blind_segment_start: '2025-07-01'\n",
                     encoding="utf-8")
        cfg = load_config(p)
        assert cfg["splits"]["embargo_days"] == 20
        # 行为随 config 改变(证明读 config 而非硬编码):同一 train_end,embargo=20 时报错、=5 时通过
        with pytest.raises(ValueError):
            assert_train_end_safe("2025-06-10", "2025-07-01", cfg["splits"]["embargo_days"])
        assert_train_end_safe("2025-06-10", "2025-07-01", 5)  # 放宽即通过


# ── 2) 模型出生证明 ─────────────────────────────────────────────────────────
class TestModelCard:
    def test_train_saves_card_with_metadata(self, tmp_path):
        ds, _ = _dataset()
        cfg = _config(tmp_path)
        path = train_and_save(ds, train_end="2020-02-20", config=cfg,
                              model_dir=cfg["paths"]["model_dir"])
        card = model_io.load_model(path)
        assert card.feature_names == cfg["features"]          # 特征名 + 顺序
        assert card.train_end == "2020-02-20"
        assert card.trained_at and isinstance(card.config_snapshot, dict)
        assert card.params["label_horizon"] == 5

    def test_predict_rejects_feature_mismatch(self, tmp_path):
        card = model_io.ModelCard(model=None, feature_names=["ret_1", "ret_5"],
                                  train_start="2020-01-06", train_end="2020-02-01",
                                  params={}, config_snapshot={})
        model_io.assert_feature_consistency(card, ["ret_1", "ret_5"])  # 一致 OK
        with pytest.raises(model_io.FeatureMismatchError):
            model_io.assert_feature_consistency(card, ["ret_5", "ret_1"])  # 顺序不同 -> 报错
        with pytest.raises(model_io.FeatureMismatchError):
            model_io.assert_feature_consistency(card, ["ret_1"])          # 缺特征 -> 报错


# ── 3) embargo 断言 ─────────────────────────────────────────────────────────
class TestEmbargo:
    def test_train_end_too_close_raises(self):
        with pytest.raises(ValueError):
            assert_train_end_safe("2025-06-25", "2025-07-01", 28)   # 间隔不足 28 个交易日

    def test_train_end_safe_passes(self):
        assert assert_train_end_safe("2025-03-01", "2025-07-01", 28) >= 28

    def test_train_and_save_blocks_peeking(self, tmp_path):
        ds, _ = _dataset()
        cfg = _config(tmp_path)
        cfg["splits"]["blind_segment_start"] = "2020-02-25"  # 紧贴 train_end
        with pytest.raises(ValueError):
            train_and_save(ds, train_end="2020-02-24", config=cfg,
                           model_dir=cfg["paths"]["model_dir"])


# ── 4) 盲测段一次性 ─────────────────────────────────────────────────────────
class TestBlindSegmentOnce:
    def test_overlap_refused_without_flag(self, tmp_path):
        cfg = _config(tmp_path)
        with pytest.raises(ValueError):
            disc.precheck_backtest(start="2025-06-01", end="2030-12-31", param_grid={},
                                   config=cfg, use_blind_once=False)

    def test_reuse_warns_inv6(self, tmp_path):
        cfg = _config(tmp_path)
        cfg["splits"]["blind_segment_start"] = "2025-07-01"
        led = disc.BlindUsageLedger(tmp_path / "blind.json")
        first = disc.precheck_backtest(start="2025-07-02", end="2025-12-31", param_grid={"a": [1, 2]},
                                       config=cfg, use_blind_once=True, ledger=led)
        assert first["warning"] is None and led.was_used("blind")
        second = disc.precheck_backtest(start="2025-07-02", end="2025-12-31", param_grid={"a": [1, 2]},
                                        config=cfg, use_blind_once=True, ledger=led)
        assert second["warning"] and "INV-6" in second["warning"]


# ── 5) 回测三数并排 ─────────────────────────────────────────────────────────
class TestThreeNumbers:
    def test_report_has_all_three(self):
        rng = np.random.default_rng(0)
        block_perf = rng.normal(size=(8, 4))
        rep = disc.assemble_backtest_report(nominal_return=0.10, net_return=0.07,
                                            block_perf=block_perf)
        assert set(["nominal_return", "net_return", "pbo"]).issubset(rep)   # 三数齐全
        assert rep["net_return"] <= rep["nominal_return"]
        assert 0.0 <= rep["pbo"] <= 1.0

    def test_naive_backtest_net_le_nominal(self, tmp_path):
        ds, _ = _dataset(seed=1)
        ds = ds.copy()
        ds["__score__"] = np.random.default_rng(2).normal(size=len(ds))  # 随机打分
        import run_backtest
        nominal, net, daily = run_backtest.naive_topk_backtest(ds, "__score__", top_k=5,
                                                               cost_fraction=0.003)
        assert np.isfinite(nominal) and net <= nominal   # 扣费后不高于名义


# ── 6) 粗桶上限 ─────────────────────────────────────────────────────────────
class TestParamGridCap:
    def test_within_cap(self):
        assert disc.check_param_grid({"dd": [1, 2, 3, 4], "vol": [1, 2, 3, 4]}, 50) == 16

    def test_over_cap_raises(self):
        grid = {f"d{i}": [1, 2, 3, 4] for i in range(5)}  # 4^5 = 1024 > 50
        with pytest.raises(ValueError):
            disc.check_param_grid(grid, 50)


# ── 7) 冠军挑战者公平性(修正版)──────────────────────────────────────────────
class TestChampionChallenger:
    def _r(self, i, cn, cm, pn, pm):
        return ShadowPeriodResult(i, cn, cm, pn, pm)

    def test_challenger_not_crippled(self):
        assert_challenger_is_full_model("2026-06-12", "2026-06-12")        # 用全部数据 -> OK
        with pytest.raises(AssertionError):
            assert_challenger_is_full_model("2026-05-01", "2026-06-12")    # 早于最新 -> 阉割

    def test_forward_evaluation(self):
        assert_forward_evaluation("2026-06-12", "2026-06-15")              # 影子期在训练之后 -> OK
        with pytest.raises(AssertionError):
            assert_forward_evaluation("2026-06-12", "2026-06-01")          # 重叠训练区间 -> 报错

    def test_single_win_no_switch_consecutive_required(self):
        # 挑战者净收益更高、回撤不更差 = "不输"
        win = self._r(0, 0.05, -0.03, 0.04, -0.03)
        assert decide_switch([win], 3)["switch"] is False        # 单期赢不换
        assert decide_switch([win, win], 3)["switch"] is False   # 两期也不换
        assert decide_switch([win, win, win], 3)["switch"] is True  # 连续 3 期 -> 换

    def test_loss_resets_streak(self):
        win = self._r(0, 0.05, -0.02, 0.04, -0.02)
        loss = self._r(1, 0.01, -0.05, 0.04, -0.02)  # 收益更低 -> 输
        res = decide_switch([win, win, loss, win], 3)
        assert res["trailing_streak"] == 1 and res["switch"] is False

    def test_in_use_keeps_champion_until_switch(self):
        win = self._r(0, 0.05, -0.02, 0.04, -0.02)
        no = decide_switch([win, win], 3)
        assert model_in_use("champ", "chal", no) == "champ"      # 未达连胜 -> 仍用冠军
        yes = decide_switch([win, win, win], 3)
        assert model_in_use("champ", "chal", yes) == "chal"


# ── 8) 时间戳命名不覆盖 ─────────────────────────────────────────────────────
class TestTimestampNaming:
    def test_model_files_not_overwritten(self, tmp_path):
        card = model_io.ModelCard(model=None, feature_names=["ret_1"], train_start="a",
                                  train_end="b", params={}, config_snapshot={})
        p1 = model_io.save_model(card, tmp_path)
        p2 = model_io.save_model(card, tmp_path)
        assert p1 != p2 and p1.exists() and p2.exists()
        assert p1.name.startswith("model_") and p1.suffix == ".pkl"

    def test_dataset_card_not_overwritten(self, tmp_path):
        import run_fetch_data
        c1 = run_fetch_data.write_dataset_card({"n_codes": 1}, tmp_path)
        c2 = run_fetch_data.write_dataset_card({"n_codes": 1}, tmp_path)
        assert c1 != c2 and c1.exists() and c2.exists()
        assert c1.name.startswith("dataset_card_")


# ── 端到端:训练 → 预测(每日使用路径)──────────────────────────────────────
class TestPredictEndToEnd:
    def test_train_then_predict(self, tmp_path):
        from trading_system.predict import run_prediction

        ds, cal = _dataset(n_days=40, n_codes=15, seed=7)
        cfg = _config(tmp_path)
        model_path = train_and_save(ds, train_end="2020-02-20", config=cfg,
                                    model_dir=cfg["paths"]["model_dir"])
        asof = str(cal.dates[35])  # 训练区间之后的某交易日(仍在数据内)
        table, info = run_prediction(ds, asof_date=asof, config=cfg, model_path=str(model_path),
                                     output_dir=cfg["paths"]["output_dir"], top_k=5,
                                     print_console=False)
        assert info["model_train_end"] == "2020-02-20"
        assert len(table) > 0 and "limit_buy_price" in table.columns and "stop_price" in table.columns
        # 仓位参考指标已计算并填充(批1B)
        for col in ("atr_n", "single_cap_pct", "kelly_suggest_pct", "stop_distance_pct", "amihud_illiq"):
            assert col in table.columns
        assert table["atr_n"].notna().any() and table["stop_distance_pct"].notna().any()


# ── 批3:时间衰减 / 引擎标签 / A·B·C 对比 / Optuna ──────────────────────────
class TestTrainingUpgrades:
    def test_time_decay_weights_monotonic(self):
        import pandas as pd
        dates = list(pd.bdate_range("2020-01-06", periods=40))
        w = compute_time_decay_weights(dates, dates[-1], half_life=10)
        assert abs(w[-1] - 1.0) < 1e-12          # train_end 当日权重=1
        assert w[0] < w[-1]                        # 老样本权重 < 新样本
        assert np.all(np.diff(w) >= -1e-12)        # 随时间非递减

    def test_train_with_time_decay_enabled(self, tmp_path):
        from trading_system.model.model_io import load_model
        ds, _ = _dataset(n_days=60, n_codes=10, seed=1)
        cfg = _config(tmp_path)
        cfg["training"]["time_decay"] = {"enabled": True, "half_life_active": 30}
        card = load_model(train_and_save(ds, train_end="2020-03-10", config=cfg,
                                         model_dir=cfg["paths"]["model_dir"]))
        assert card.params["time_decay"] is not None and card.params["time_decay"]["half_life"] == 30

    def test_engine_label_records_type(self, tmp_path):
        from trading_system.model.model_io import load_model
        ds, _ = _dataset(n_days=50, n_codes=6, seed=2)
        cfg = _config(tmp_path)
        cfg["training"]["label"] = {"type": "engine", "fixed_horizon": 5}
        card = load_model(train_and_save(ds, train_end="2020-03-01", config=cfg,
                                         model_dir=cfg["paths"]["model_dir"]))
        assert card.params["label_type"] == "engine"

    def test_compare_routes_three_rows(self, tmp_path):
        ds, _ = _dataset(n_days=90, n_codes=12, seed=3)
        cfg = _config(tmp_path)
        table = compare_label_routes(ds, cfg, train_end="2020-04-30", n_splits=2)
        assert set(table["route"]) == {"A", "B", "C"} and len(table) == 3

    def test_tune_enabled_records_params(self, tmp_path):
        from trading_system.model.model_io import load_model
        ds, _ = _dataset(n_days=90, n_codes=10, seed=4)
        cfg = _config(tmp_path)
        cfg["training"]["tune"] = {"enabled": True, "n_trials": 3}
        card = load_model(train_and_save(ds, train_end="2020-04-30", config=cfg,
                                         model_dir=cfg["paths"]["model_dir"]))
        assert card.params["tuned_params"] is not None   # 调参参数已记录(粗调,purged CV)


# ── 脚本可导入 + 读 config(单一源贯通)──────────────────────────────────────
class TestScriptsReadConfig:
    def test_root_scripts_import(self):
        import run_backtest, run_champion_challenger, run_fetch_data, run_predict, run_train  # noqa: F401

    def test_backtest_cost_fraction_from_config(self, tmp_path):
        import run_backtest
        cfg = _config(tmp_path)
        out = run_backtest.run(cfg, start="2019-01-01", end="2019-06-30", param_grid={})
        base = out["cost_fraction"]
        cfg2 = _config(tmp_path)
        cfg2["cost"]["stamp_duty"] = cfg["cost"]["stamp_duty"] + 0.001  # 改 config 成本
        out2 = run_backtest.run(cfg2, start="2019-01-01", end="2019-06-30", param_grid={})
        assert out2["cost_fraction"] > base   # 成本随 config 改变 -> 证明读 config
