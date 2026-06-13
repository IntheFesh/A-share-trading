"""监控(核心层,落盘不起服务)。Phase 4(任务 4.3)。对应 v3.1 第十三章。

核心必做:分块不重叠 RankIC、扣费净值与 MaxDD/Calmar、成交失败率、执行差距、单股/同簇暴露。
增强可选(主线稳定后再做):HMM/ADWIN/DDM/PSI/HCOPE/拥挤代理/HiLo —— 见各自 TODO,不阻塞主线。
输出:落盘 PNG(matplotlib Agg,不起服务)+ Markdown,绝不起 Web 服务。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.backtest import metrics


def run_monitor(
    nav: "pd.Series | np.ndarray",
    daily_ic: "pd.Series",
    *,
    out_dir: "str | Path",
    block_len: int = 10,
    fill_failure_rate: "float | None" = None,
    execution_gap_bp: "float | None" = None,
    tag: str = "monitor",
) -> dict:
    """生成核心监控面板:计算指标、落盘净值 PNG + Markdown 报告。返回指标 dict。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nav_arr = np.asarray(nav, dtype="float64")

    mdd = metrics.max_drawdown(nav_arr)
    cal = metrics.calmar(nav_arr)
    mean_ic = float(daily_ic.dropna().mean()) if len(daily_ic) else float("nan")
    # 分块不重叠 IC:对 daily_ic 直接按块长抽样(避重叠虚高)
    blocked = daily_ic.dropna().iloc[::block_len]
    blocked_ic = float(blocked.mean()) if len(blocked) else float("nan")
    icir = metrics.icir(daily_ic)

    result = {
        "max_drawdown": mdd,
        "calmar": cal,
        "mean_rank_ic": mean_ic,
        "blocked_rank_ic": blocked_ic,
        "icir": icir,
        "fill_failure_rate": fill_failure_rate,
        "execution_gap_bp": execution_gap_bp,
    }

    # 落盘净值 PNG(Agg 后端,纯文件,不起服务)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nav_arr)
    # 用 ASCII 轴标题(默认字体无中文字形);中文叙述在 Markdown 报告中。
    ax.set_title(f"{tag} net value (MaxDD={mdd:.2%}, Calmar={cal:.2f})")
    ax.set_xlabel("trading day index"); ax.set_ylabel("NAV")
    png_path = out_dir / f"{tag}_nav.png"
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    md = [
        f"# 监控面板 {tag}", "",
        "## 核心指标",
        f"- 最大回撤 MaxDD: {mdd:.2%}",
        f"- Calmar: {cal:.2f}",
        f"- 平均 RankIC: {mean_ic:.4f};分块不重叠 RankIC(H={block_len}): {blocked_ic:.4f};ICIR: {icir:.3f}",
        f"- 成交失败率: {fill_failure_rate}",
        f"- 执行差距(bp): {execution_gap_bp}",
        "",
        "## 增强可选(主线稳定后再做)",
        "- HMM 状态概率 / ADWIN·DDM 漂移 / 特征 PSI / HCOPE 否决下界 / 拥挤代理相关性 / HiLo —— 待接入。",
        "",
        f"![nav]({png_path.name})",
    ]
    md_path = out_dir / f"{tag}_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    result["png_path"] = str(png_path)
    result["md_path"] = str(md_path)
    return result
