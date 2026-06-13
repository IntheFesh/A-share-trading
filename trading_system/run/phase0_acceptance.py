"""Phase 0 验收脚本(占位)。任务 0.7。对应 v3.1 第二章。

实现后将:抽 20 除权事件(raw 跳变、adj 收益不异常)、20 退市股(历史拉到退市)、
20 涨跌停样本(round(preclose_raw*1.1,2) 逐笔一致)、披露日历无前视;输出 Markdown 核验报告。
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("phase0_acceptance")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger.info("Phase 0 验收尚未实现:请先完成任务 0.1~0.6 后再填充本脚本(任务 0.7)。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
