"""数据质量诊断:列出后复权连续性异常行,供人工核查(真数据错 vs 复权边界)。补丁 1E-5。

从 store 读真实数据,对 check_adj_continuity 标记的异常行输出明细(adj 跳变幅度 / 是否除权日 /
是否涨跌停 / 是否停牌缺口)到 reports/output;data_store 为空则优雅提示,不报错。
用法:python -m trading_system.run.diagnose_data_quality
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("diagnose_data_quality")
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "output"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    from trading_system.config import load_config
    from trading_system.data import quality
    from trading_system.data.store import ParquetStore

    cfg = load_config()
    data = ParquetStore(cfg["paths"]["data_dir"]).read()
    md = ["# 数据质量诊断:后复权连续性异常", ""]
    if data.empty:
        md.append("- **NOT RUN**:data_store 为空。请先用 run_fetch_data.py 拉取实盘数据后重跑。")
        (REPORT_DIR / "adj_continuity_anomalies.md").write_text("\n".join(md), encoding="utf-8")
        logger.info("data_store 为空,诊断 NOT RUN。")
        return 0

    flagged = quality.diagnose_adj_continuity(data)
    csv_path = REPORT_DIR / "adj_continuity_anomalies.csv"
    flagged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md += [
        f"- 共 {len(flagged)} 行后复权收益异常跳变(|日收益|>10.5% 且 非除权日/非涨跌停/非停牌缺口)。",
        "- 这些更可能是真数据错;除权/涨跌停/停牌边界已被排除。明细见 CSV。",
        f"- 涉及 {flagged['code'].nunique() if len(flagged) else 0} 只票。明细 CSV:{csv_path.name}",
        "",
        "> 处理建议:逐条核查后,若确为数据错可在数据侧修正;ATR 已可在使用环节稳健化(见 config.data_quality)。",
    ]
    if len(flagged):
        md += ["", "| code | trade_date | adj_ret |", "|---|---|---|"]
        for _, r in flagged.head(50).iterrows():
            md.append(f"| {r['code']} | {str(r['trade_date'])[:10]} | {r['adj_ret']:+.4f} |")
    (REPORT_DIR / "adj_continuity_anomalies.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("诊断完成:%d 行异常,明细写入 %s", len(flagged), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
