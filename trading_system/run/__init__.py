"""各 Phase 的命令行入口 / 验收脚本。

用法(在仓库根目录):
    python -m trading_system.run.phase0_acceptance   # 亦有 phase1/phase2/phase3 及 diagnose_data_quality
各脚本均已实现:离线可跑的硬门槛(引擎对账/计算自洽)直接执行;需真实市场数据的部分诚实标注
**NOT RUN**(不伪造数据、不谎报通过)。
"""
