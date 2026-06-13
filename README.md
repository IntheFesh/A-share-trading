# A 股日频截面选股交易系统(v3.1 实现版)

A 股日频截面选股与交易研究系统。系统为纯后端实现,不包含任何 Web / GUI / 仪表盘服务;
全部产出为落盘工件:命令行脚本、CSV / Parquet / Markdown 文件,以及 matplotlib 落盘静态图
(`.png` / `.html`)。

系统由四层流水线构成(市场状态 → 触发候选 → 截面排序 → 组合风控 → 事件级回测 / 作战手册),
并以七条不变量(见 [§12](#12-设计不变量))作为全局约束。实现遵循 v3.1 冻结设计版,按
Phase 0 至 Phase 4 的依赖顺序分阶段交付。

设计文档优先级:v0.3 宪法 > v3.1 设计文档 > v1.0。

---

## 目录

- [1. 实现状态](#1-实现状态)
- [2. 系统架构](#2-系统架构)
- [3. 环境要求与安装](#3-环境要求与安装)
- [4. 快速开始](#4-快速开始)
- [5. 数据采集](#5-数据采集)
- [6. 运行各 Phase](#6-运行各-phase)
- [7. 运行测试](#7-运行测试)
- [8. 目录结构](#8-目录结构)
- [9. 数据表结构](#9-数据表结构)
- [10. 配置说明](#10-配置说明)
- [11. 故障排查](#11-故障排查)
- [12. 设计不变量](#12-设计不变量)
- [13. 术语表](#13-术语表)
- [14. 已知限制](#14-已知限制)
- [15. 免责声明](#15-免责声明)

---

## 1. 实现状态

Phase 0 至 Phase 4 的全部逻辑已实现,并配有单元测试。测试以可独立推导或可手工核算的期望值
为基准进行校验。

```
176 passed, 3 skipped
```

测试覆盖区分两类验收口径:

- **逻辑验收(离线可执行)**:数据层、特征防泄漏检查、事件级引擎逐笔核算、交叉验证、统计量
  (RankIC / PBO / DSR)、审批门槛等,均在合成或构造数据上完成校验。
- **市场验收(需实盘环境)**:依赖实盘数据、Tushare 凭证或训练算力的验收项——例如 Phase 0 的
  除权 / 退市 / 涨跌停逐笔核验、Phase 1 的历史 RankIC、Phase 3 的 walk-forward 与盲测——
  标记为「未执行(NOT RUN)」,在对应验收脚本中显式列示。

3 项跳过的测试均为需要外部资源(BaoStock 网络会话、Tushare 凭证)的实盘路径,在测试输出中
附明确原因。合成数据仅用于验证算法正确性,不代表因子在真实市场的有效性。

---

## 2. 系统架构

系统为四层流水线,自上而下逐层收敛至每日可执行的交易清单:

| 层 | 职责 | 模块 | Phase |
|---|---|---|---|
| L0 市场状态 | 由日线推导市场情绪温度与 regime,输出总仓位乘子 `m_t` | `regime/` | 1 |
| L1 触发候选 | 从全市场筛选事件候选(牛回头 / 缩量首板 / RPS 龙头) | `triggers/` | 1 |
| L2 截面排序 | 对候选用机器学习模型打分并排名,选取 Top-K | `model/`、`features/` | 3 |
| L3 组合风控 | 单笔仓位、单股上限、总敞口、拥挤簇限制 | `portfolio/` | 2 |
| 回测 / 手册 | 事件级引擎逐笔模拟成交(T+1、涨跌停、止损止盈);生成每日作战手册 | `backtest/`、`playbook/` | 2 / 4 |

两项专门处置:

- **披露季风险**:在已知的财报披露窗内降低暴露(不预测方向),由 `overlays/` 的披露季 overlay 实现。
- **过度拉升 / 高低切**:将"过度拉升度"以与"高低切 regime"的交互项形式进入模型,而非无条件惩罚,
  由 `regime/` 与 `features/` 协同实现。

---

## 3. 环境要求与安装

- Python 3.11 及以上。
- 安装依赖:

  ```bash
  pip install -r requirements.txt
  ```

依赖分组:

| 分组 | 组件 | 用途 |
|---|---|---|
| 全程必需 | pandas, numpy, pyarrow, duckdb, pyyaml, scipy, scikit-learn, matplotlib, pytest | 数据处理、存储、统计、绘图、测试 |
| 数据源 | baostock, tushare, requests | 行情(BaoStock)、财报(Tushare)、盘中快照 |
| Phase 3 | lightgbm, optuna | L2 模型与超参搜索 |

安装完成后执行环境自检:

```bash
python -m trading_system.check_env
```

该命令检查 Python 版本、依赖就位情况(按"全程必需 / 数据源 / Phase 3"分组)、七个配置文件可加载性,
以及七条不变量原语的冒烟断言。存在阻断项时返回非零退出码,并提示安装依赖。

---

## 4. 快速开始

以下为从零到产出报告的端到端流程。每一步标注其目的与预期结果。

**步骤 1 — 安装依赖。**

```bash
pip install -r requirements.txt
```

**步骤 2 — 环境自检。** 确认依赖、配置与不变量就位。预期末行输出"通过(可进入 Phase 0)";
若有阻断项,按提示补装依赖。

```bash
python -m trading_system.check_env
```

**步骤 3 — 运行测试。** 确认代码完整性。预期结果为 `176 passed, 3 skipped`。

```bash
pytest
```

**步骤 4 — 采集数据。** 拉取行情并落盘至 `data_store/`(需网络)。详见 [§5](#5-数据采集)。

```bash
python -m trading_system.data.fetch_training_data --start 2019-01-01
```

**步骤 5 — 运行 Phase 验收 / 报告。** 产出 Markdown 报告至 `trading_system/reports/output/`。
详见 [§6](#6-运行各-phase)。

```bash
python -m trading_system.run.phase1_factor_report
python -m trading_system.run.phase2_acceptance
python -m trading_system.run.phase3_acceptance
```

在 IDE(如 PyCharm)中,选择 3.11+ 解释器后,可直接右键运行 `run/` 下任意脚本或 `tests/` 目录。

---

## 5. 数据采集

### 5.1 数据源职责

| 数据源 | 角色 | 提供内容 | 依赖级别 |
|---|---|---|---|
| BaoStock | 唯一信息来源 | 行情(不复权 + 后复权)、交易日历、ST 状态、退市、上市日 | 硬依赖 |
| Tushare | 仅财报获取 | 业绩预告、预约披露日 | 软依赖 |

后复权序列仅取自 BaoStock 单一来源,不与其它来源拼接复权价(不变量 INV-2)。Tushare 不作为
行情 / 退市 / 日历来源。

### 5.2 命令

```bash
# 行情采集(沪深主板非 ST,2019 至今,增量模式)
python -m trading_system.data.fetch_training_data --start 2019-01-01

# 全量重拉
python -m trading_system.data.fetch_training_data --start 2019-01-01 --full

# 同时采集财报(需 Tushare 凭证;凭证缺失或采集失败时降级置空,行情仍正常落盘)
TUSHARE_TOKEN=<token> \
python -m trading_system.data.fetch_training_data --start 2019-01-01 --enable-disclosure
```

### 5.3 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--start` | `2019-01-01` | 起始日期 |
| `--end` | 当日 | 结束日期 |
| `--universe` | `main_board` | 交易池范围(沪深主板非 ST) |
| `--enable-disclosure` | 关闭 | 是否采集财报 / 披露(需凭证) |
| `--tushare-token` | 无 | Tushare 凭证;亦可通过环境变量 `TUSHARE_TOKEN` 提供 |
| `--incremental` / `--full` | `--incremental` | 增量(仅拉取「本地最新 + 1 → 当日」)/ 全量重拉 |
| `--out` | store 既定路径 | 落盘目录 |

### 5.4 执行流程

1. 列出交易池(BaoStock,主板非 ST)。
2. 采集行情(硬依赖):逐只票同源取不复权与后复权数据,计算复权因子,经双价格层构造后写入 store。
3. 采集财报(软依赖,仅在 `--enable-disclosure` 时):凭证缺失则跳过并置空;采集异常则退避重试后
   降级置空。
4. 数据质检:行情项必检;披露项仅在采集成功时检。
5. 输出总结:行情成功 / 失败数量、披露状态、落盘路径、数据日期范围,及后续步骤提示。

### 5.5 容错策略

| 情形 | 处理 |
|---|---|
| BaoStock 登录失败 / 全部票拉取失败 | 进程以非零码(2)退出,不产出空或占位数据 |
| BaoStock 单只票失败 | 重试至多 2 次后计入失败名单并继续;失败率超过 5% 时告警 |
| Tushare 凭证缺失 | 跳过财报采集,披露字段置 NULL,行情正常落盘 |
| Tushare 采集异常 | 退避重试至多 2 次,仍失败则降级置空,行情正常落盘 |

行情采集为单进程顺序执行(BaoStock 不支持多线程)。

### 5.6 产物

数据落盘至 `data_store/`,按年分区:

```
data_store/
├── year=2019/part.parquet
├── year=2020/part.parquet
└── ...
```

落盘后,Phase 1 特征流水线与 Phase 3 训练直接经 `store.read(...)` 读取,无需额外搬运。

### 5.7 关键语义

- **双价格层**:`adj_factor = 后复权收盘 / 不复权收盘`;涨跌停价仅依据不复权昨收计算;
  PIT `isST` 决定 ST 标的的 5% 涨跌停幅度。
- **PIT 语义**:`has_preann = NULL` 表示"未采集 / 未知",与 `has_preann = False`("已确认未发预告")
  通过可空布尔类型严格区分。
- **披露 overlay 默认不启用**:未启用财报采集时披露字段为 NULL,Phase 2 披露 overlay 自动短路。

### 5.8 示例输出(说明用途)

```text
INFO === 采集完成 ===
INFO 行情:成功 1180 只 / 失败 12 只;写入 1432560 行;日期 2019-01-02 ~ 2026-06-12。
INFO 披露:disabled(disclosure 字段 NULL=未采集/未知)。
INFO 落盘目录:/path/to/data_store
INFO 下一步:python -m trading_system.run.phase1_factor_report 或 Phase 3 训练脚本。
```

BaoStock / Tushare 真实网络拉取需实盘环境,标记为「未执行(NOT RUN)」;其逻辑(开关短路、
Tushare 降级、BaoStock 硬失败退出、增量去重、双价格层完整性、PIT 语义)由注入式 mock 在
`tests/test_fetch_training_data.py` 中覆盖。

---

## 6. 运行各 Phase

系统按依赖顺序分阶段实现:数据底座先行,模型最后。每个验收脚本将报告写入
`trading_system/reports/output/`,并以退出码表明结果。

| Phase | 内容 | 运行命令 | 产物 |
|---|---|---|---|
| 0 | 数据底座与双价格层 | `python -m trading_system.run.phase0_acceptance` | `phase0_acceptance.md` |
| 1 | 特征、触发器、标签 | `python -m trading_system.run.phase1_factor_report` | `phase1_factor_report.md` |
| 2 | 事件级引擎、成本、压力、overlay | `python -m trading_system.run.phase2_acceptance` | `phase2_acceptance.md` |
| 3 | L2 模型与审批 | `python -m trading_system.run.phase3_acceptance` | `phase3_acceptance.md` |
| 4 | 作战手册、否决审计、监控 | 经 `playbook/`、`audit/`、`reports/monitor.py` 调用 | CSV / Markdown / PNG |

退出码语义:

| 退出码 | 含义 |
|---|---|
| 0 | 离线逻辑验收通过(市场验收项在报告中标记为 NOT RUN) |
| 3 | (仅 Phase 0)合成流水线自检通过,但实盘逐笔核验需数据 / 凭证,尚未执行 |
| 1 | 验收失败,详见报告 |

各 Phase 的市场验收口径:

- **Phase 0**:20 个除权事件、20 只退市股、20 个涨跌停样本的逐笔核验,以及披露日历无前视抽检。
- **Phase 1**:截断等变性全部通过(离线硬门槛);历史 RankIC / ICIR、混池对照、收益三段拆解(需实盘数据)。
- **Phase 2**:引擎逐笔核算(离线硬门槛);规则基线在 20bp 滑点后为正、首板在 30bp 下存活、
  各 overlay 满足 ΔMaxDD < 0 且 ΔCalmar > 0(需实盘数据)。
- **Phase 3**:purged CV / PBO / DSR / 审批门槛(离线硬门槛);胜过四基线、walk-forward 与单次盲测(需实盘数据)。

---

## 7. 运行测试

```bash
pytest                                              # 全部测试
pytest trading_system/tests/test_invariants.py -v   # 仅七条不变量
pytest -W error::UserWarning                        # 将警告视为错误
```

测试覆盖:

| 测试文件 | 覆盖内容 |
|---|---|
| `test_invariants.py` | 七条不变量(INV-1 至 INV-7)断言 |
| `test_structure.py` | 全模块可导入性与配置完整性 |
| `test_phase0_data.py` | 交易日历、双价格层(含除权)、披露 PIT、交易池、存储增量、质检、快照解析 |
| `test_phase1.py` | 防泄漏三检查(含对守卫有效性的反向验证)、标签、regime 指标、触发器 |
| `test_metrics.py` | RankIC、ICIR、分块 RankIC、MaxDD、Calmar |
| `test_phase2_core.py` | 引擎逐笔核算(五场景)、成本六层、仓位合成 |
| `test_phase2_ext.py` | 滑点压力矩阵、四基线、overlay test |
| `test_phase3.py` | purged CV、PBO / DSR、LightGBM(INV-4 与按日 group)、Optuna、审批与 INV-6 |
| `test_phase4.py` | OPE(IPW / DR)、否决理由码、算法厌恶护栏、作战手册、监控落盘 |
| `test_fetch_training_data.py` | 采集开关、Tushare 降级、BaoStock 硬失败、增量去重、双价格层、PIT 语义 |

3 项跳过的测试分别需要 BaoStock 网络会话与 Tushare 凭证,在测试输出中附原因。

---

## 8. 目录结构

```
trading_system/
├── invariants.py              # 七条不变量的可复用原语与守卫
├── check_env.py               # 环境自检命令
├── config/                    # 配置(全部"待自验"阈值集中于此,逻辑中禁止硬编码)
├── data/                      # Phase 0:数据底座
│   ├── schema.py              #   统一表列名分组(raw / adj / 状态位 / 披露)
│   ├── calendar.py            #   交易日历(T+1 / T+2 / embargo 偏移)
│   ├── price_layers.py        #   双价格层构造与状态位、披露 PIT 字段
│   ├── universe.py            #   交易池过滤
│   ├── store.py               #   Parquet + DuckDB 存储,增量去重(单一数据出口)
│   ├── quality.py             #   每日数据质检
│   ├── fetch_training_data.py #   一键采集入口
│   └── collectors/            #   采集器(baostock / tushare / tencent / sina / synthetic)
├── features/                  # Phase 1:指标注册表(防泄漏三检查)与特征族
├── regime/                    # Phase 1:L0 六指标、情绪温度、五阶段、HiLo
├── triggers/                  # Phase 1:L1 触发器(粗桶,禁网格寻优)
├── labels/                    # Phase 1:三类标签(INV-1)与成交同源(INV-3)
├── backtest/                  # Phase 2:事件级引擎、成本、基线、指标、压力
├── portfolio/                 # Phase 2:L3 仓位合成
├── overlays/                  # Phase 2:披露季 overlay、高低切交互、overlay test
├── model/                     # Phase 3:purged CV、LightGBM 三路线、Optuna、审批
├── playbook/                  # Phase 4:作战手册(CSV + Markdown + 控制台)
├── audit/                     # Phase 4:否决审计(OPE)与盲测段账本
├── reports/                   # 落盘报告与监控(静态文件)
├── tests/                     # 单元测试
└── run/                       # 各 Phase 命令行验收脚本
```

运行期数据落盘至仓库根目录的 `data_store/`(已在版本控制中忽略)。

---

## 9. 数据表结构

存储中每条记录(一只标的一个交易日)包含以下字段:

```text
code, trade_date                                                   # 主键
open_raw, high_raw, low_raw, close_raw, preclose_raw, volume, amount    # 执行层(不复权)
adj_factor, open_adj, high_adj, low_adj, close_adj                      # 特征层(后复权)
is_suspended, is_st, is_limit_up, is_limit_down, is_one_price_limit     # 状态位
sched_disclosure_date, has_preann, preann_sign, days_to_disclosure      # 披露季事件(PIT)
```

字段使用约束:

- 执行类计算(成交价、涨跌停价、止损止盈、股数、成本、盈亏)仅使用 `*_raw` 字段。
- 特征类计算(收益、均线、波动、CGO、RPS)仅使用 `*_adj` 字段。
- 涨跌停价:`涨停 = round(preclose_raw × 1.1, 2)`,`跌停 = round(preclose_raw × 0.9, 2)`;
  ST 标的为 ±5%。

数据经 `store.read(codes=, start=, end=, fields=)` 读取;按 `fields` 取数使执行代码与特征代码
在数据出口处即分离原始价与复权价。

---

## 10. 配置说明

所有 v3.1 未确定、标注"待自验"的阈值集中于 `config/*.yaml`(带 `# TODO: Phase X 自验` 注释),
逻辑代码中不得硬编码。

| 文件 | 内容 |
|---|---|
| `data.yaml` | 数据源角色、存储路径、交易池规则、盘中快照限频 |
| `costs.yaml` | 成本六层;已核验下限(印花税 0.05% + 经手费双向 2×0.00341% = 5.682 bp)为定值,过户费 / 佣金待核 |
| `triggers.yaml` | L1 触发器粗桶边界 |
| `risk.yaml` | 单股上限档、连续跌停 K、波动率目标、凯利档、拥挤簇限制 |
| `exit.yaml` | 硬止损 2.5N、止盈三阶梯、跟踪系数、最大持有期 |
| `regime.yaml` | L0 六指标权重、五阶段阈值、HiLo 参数 |
| `train.yaml` | 标签窗口、embargo、purged CV、Optuna 搜索空间、审批门槛 |

---

## 11. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `fetch_training_data` 退出码 2,提示缺少 `baostock` | BaoStock 为行情硬依赖。执行 `pip install baostock` 并确保网络连通后重试 |
| 启用 `--enable-disclosure` 但无凭证 | 系统跳过财报采集并置空披露字段,行情正常落盘;如需财报请提供 `TUSHARE_TOKEN` |
| `phase0_acceptance` 退出码 3 | 合成自检通过,实盘逐笔核验需数据 / 凭证,属预期的分层结果,非失败 |
| `check_env` 退出码非零 | 存在未安装的阻断依赖,按提示执行 `pip install -r requirements.txt` |
| 监控 PNG 中文显示为方框 | 默认字体无中文字形,图内轴标题采用 ASCII;中文说明见配套 Markdown 报告 |

---

## 12. 设计不变量

七条不变量定义于 `trading_system/invariants.py`,并在 `tests/test_invariants.py` 中逐条断言;
任何违反将触发运行时错误或测试失败。

| 编号 | 名称 | 约束 |
|---|---|---|
| INV-1 | 可交易标签优先 | 卖出日 ≥ 信号日 + 2 个交易日;`h=0` 标签仅用于诊断命名空间 |
| INV-2 | 双价格层 | 执行类计算仅用不复权价,特征类计算仅用后复权价 |
| INV-3 | 标签—成交同源 | 标签与回测引擎使用同一成交判定函数 |
| INV-4 | 组内常数为覆盖层 | 当日同值的量(如情绪温度)仅以显式交互项进入排序模型 |
| INV-5 | 单股上限由连续跌停压力决定 | `w_max = min(w_hard, L_tail/ĝ)`;主板 ≤ 8%、高风险 ≤ 5%;禁用 15% 默认值 |
| INV-6 | 盲测段一次性 | 盲测段用于换届裁决后即封存,再次使用将报错 |
| INV-7 | 条件化优先于无条件叠加 | 惩罚 / 增强型信号默认以 regime 交互形式进入 |

---

## 13. 术语表

| 术语 | 含义 |
|---|---|
| 不复权 / 后复权 | 不复权为实际成交价(除权日跳变);后复权为消除除权影响后的连续价(用于收益计算) |
| PIT(point-in-time) | 仅使用当时可得的信息,不以未来公告回填历史,避免前视偏差 |
| T+1 | 当日买入次日方可卖出;系统信号当日生成、次日开盘买入、最早第三个交易日卖出 |
| RankIC | 当日打分排名与未来收益排名的秩相关,衡量选股有效性 |
| ICIR | RankIC 的均值除以标准差,衡量选股能力稳定性 |
| MaxDD / Calmar | 最大回撤 / 年化收益除以最大回撤 |
| embargo / purged CV | 训练集与验证集间设隔离带并剔除标签窗重叠样本,防时序泄漏 |
| PBO | 回测过拟合概率(组合对称交叉验证);上线要求低于 30% |
| DSR | 去膨胀夏普率,扣除多次尝试带来的选择偏差;上线要求高于 0.95 |
| regime | 市场状态 / 风格(如动量期与高低切期),仅可事后确认 |
| HiLo | 高位股相对低位股开始跑输的风格反转信号 |
| CGO / MAX / RPS / 量比 | 资本利得突出量 / 近期单日最大涨幅 / 相对强弱 / 当日量比过去均量 |
| overlay | 在排序之上的降仓或否决覆盖层,须通过 overlay test 方可启用 |
| ATR / 2.5N | 真实波动幅度;硬止损设于入场价减 2.5 倍 ATR |
| 凯利三档 | 按证据强度将单笔风险预算设为 100% / 50% / 0% |
| lambdarank | LightGBM 的 learning-to-rank 目标;本系统按交易日分组 |

---

## 14. 已知限制

- 当前内置特征为代表性子集(12 项)。CGO 与换手率族需流通股本数据,现有数据表暂未包含,
  待数据补齐后接入。
- L0 的 HMM 状态概率为 v3.1 标注的增强可选项,尚未实现。
- Phase 1 的生产标签 `y_prod` 为"固定持有期 + 扣费"的可交易收益;含完整止损 / 止盈状态机的版本
  由 Phase 2 事件级引擎承担。
- 次新股 60 日窗口当前以面板内计数近似;精确判定需 BaoStock `query_stock_basic` 的上市日字段
  (接口已就位,待接入)。
- 采集器的网络路径需实盘环境验证;离线测试以注入式 mock 覆盖其编排逻辑。
- regime 拐点、踩踏时点与无预告突发事件本质上不可提前精确预测;系统以事前纪律(暴露约束、
  单股上限、条件化、不加杠杆)应对,而非事中预判。

---

## 15. 免责声明

本系统仅用于个人交易系统的工程设计与研究流程规范,不构成投资建议。历史数据、回测结果、学术研究
与券商研究均不代表未来表现。
