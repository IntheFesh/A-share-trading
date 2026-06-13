# A 股日频截面选股交易系统(v3.1 实现版)

A 股日频截面选股与交易研究系统。系统为纯后端实现,不包含任何 Web / GUI / 仪表盘服务;
全部产出为落盘工件:命令行脚本、CSV / Parquet / Markdown 文件,以及 matplotlib 落盘静态图
(`.png` / `.html`)。

系统由四层流水线构成(市场状态 → 触发候选 → 截面排序 → 组合风控 → 事件级回测 / 作战手册),
以七条不变量(见 [§13](#13-设计不变量))作为全局约束。日常操作由五个入口脚本承担,常用参数集中
于根目录 `config.yaml`。实现遵循 v3.1 冻结设计版,按 Phase 0 至 Phase 4 的依赖顺序分阶段交付。

设计文档优先级:v0.3 宪法 > v3.1 设计文档 > v1.0。

---

## 目录

- [1. 实现状态](#1-实现状态)
- [2. 系统架构](#2-系统架构)
- [3. 环境要求与安装](#3-环境要求与安装)
- [4. 快速开始](#4-快速开始)
- [5. 日常运行流程(五个入口脚本与配置中心)](#5-日常运行流程五个入口脚本与配置中心)
- [6. 数据采集详解](#6-数据采集详解)
- [7. 运行各 Phase(验收脚本)](#7-运行各-phase验收脚本)
- [8. 运行测试](#8-运行测试)
- [9. 目录结构](#9-目录结构)
- [10. 数据表结构](#10-数据表结构)
- [11. 配置说明](#11-配置说明)
- [12. 故障排查](#12-故障排查)
- [13. 设计不变量](#13-设计不变量)
- [14. 术语表](#14-术语表)
- [15. 已知限制](#15-已知限制)
- [16. 免责声明](#16-免责声明)

---

## 1. 实现状态

Phase 0 至 Phase 4 的全部逻辑、五个入口脚本与配置中心均已实现,并配有单元测试。测试以可独立推导
或可手工核算的期望值为基准进行校验。

```
198 passed, 3 skipped
```

测试覆盖区分两类验收口径:

- **逻辑验收(离线可执行)**:数据层、特征防泄漏检查、事件级引擎逐笔核算、交叉验证、统计量
  (RankIC / PBO / DSR)、审批门槛、模型出生证明、回测三纪律、冠军挑战者判定等。
- **市场验收(需实盘环境)**:依赖实盘数据、Tushare 凭证或训练算力的验收项标记为「未执行(NOT RUN)」,
  在对应验收脚本中显式列示。

3 项跳过的测试均为需要外部资源(BaoStock 网络会话、Tushare 凭证)的实盘路径,在测试输出中附原因。
合成数据仅用于验证算法正确性,不代表因子在真实市场的有效性。

---

## 2. 系统架构

系统为四层流水线,自上而下逐层收敛至每日可执行的交易清单:

| 层 | 职责 | 模块 | Phase |
|---|---|---|---|
| L0 市场状态 | 由日线推导情绪温度与 regime,输出总仓位乘子 `m_t` | `regime/` | 1 |
| L1 触发候选 | 从全市场筛选事件候选(牛回头 / 缩量首板 / RPS 龙头) | `triggers/` | 1 |
| L2 截面排序 | 对候选用机器学习模型打分排名,选取 Top-K | `model/`、`features/` | 3 |
| L3 组合风控 | 单笔仓位、单股上限、总敞口、拥挤簇限制 | `portfolio/` | 2 |
| 回测 / 手册 | 事件级引擎逐笔模拟成交(T+1、涨跌停、止损止盈);生成每日作战手册 | `backtest/`、`playbook/` | 2 / 4 |

---

## 3. 环境要求与安装

- Python 3.11 及以上。
- 安装依赖:`pip install -r requirements.txt`。

| 分组 | 组件 | 用途 |
|---|---|---|
| 全程必需 | pandas, numpy, pyarrow, duckdb, pyyaml, scipy, scikit-learn, matplotlib, pytest | 数据、存储、统计、绘图、测试 |
| 数据源 | baostock, tushare, requests | 行情(BaoStock)、财报(Tushare)、盘中快照 |
| Phase 3 | lightgbm, optuna | L2 模型与超参搜索 |

安装后执行环境自检(检查版本、依赖、配置可加载性、七条不变量冒烟;存在阻断项时返回非零码):

```bash
python -m trading_system.check_env
```

---

## 4. 快速开始

```bash
pip install -r requirements.txt           # 1. 安装依赖(Python 3.11+)
python -m trading_system.check_env        # 2. 环境自检
pytest                                    # 3. 运行测试(预期 197 passed, 3 skipped)

# 4. 编辑根目录 config.yaml,设置数据/模型/输出目录等参数(见 §5.1)

python run_fetch_data.py --start 2019-01-01            # 5. 取数(需网络)
python run_train.py --dataset ./data_store --train-end 2025-06-01   # 6. 训练
python run_predict.py --asof 2026-06-12                # 7. 每日预测,出作战手册
```

在 IDE(如 PyCharm)中选择 3.11+ 解释器后,可直接右键运行根目录的 `run_*.py` 脚本或 `tests/` 目录。

---

## 5. 日常运行流程(五个入口脚本与配置中心)

日常操作由五个职责严格隔离的入口脚本承担。推荐顺序:**取数 → 训练 → 每日预测 → 定期回测 →
冠军挑战者影子对决**。所有常用参数集中于根目录 `config.yaml`,脚本内不得硬编码。

| 脚本 | 职责 | 命令示例 |
|---|---|---|
| `run_fetch_data.py` | 取数:BaoStock 行情(硬依赖)+ Tushare 财报(软依赖,失败降级);落 store + 时间戳体检卡 | `python run_fetch_data.py` |
| `run_train.py` | 训练:模型的唯一生产者,存"出生证明"包;断言不触碰盲测段 | `python run_train.py --dataset ./data_store --train-end 2025-06-01` |
| `run_predict.py` | 每日预测:只加载不训练;特征一致性硬闸 + PIT;出作战手册 | `python run_predict.py --asof 2026-06-12` |
| `run_backtest.py` | 回测调参:守三纪律(盲测隔离 / 三数并排 / 粗桶上限) | `python run_backtest.py --start 2019-01-01 --end 2024-12-31 --param-grid '{}'` |
| `run_champion_challenger.py` | 冠军挑战者影子对决:连胜判定换届 | `python run_champion_challenger.py --periods '[...]'` |

### 5.1 配置中心 config.yaml

单一参数源。修改参数请改 `config.yaml`,勿在脚本内修改(否则脚本间参数不一致)。主要分组:

| 分组 | 内容 |
|---|---|
| `paths` | `data_dir` / `model_dir` / `output_dir`(用户按本机修改) |
| `universe` | 交易池范围(`main_board`:沪深主板非 ST) |
| `data` | 起始日期、是否采集财报(`enable_disclosure`,默认关) |
| `features` | 特征清单(**顺序敏感**:训练 / 预测 / 回测必须一致) |
| `cost` | 成本参数(印花税、经手费、过户费、佣金、最低佣金) |
| `risk` | 单股上限、止损 ATR 倍数、止盈阶梯、跟踪系数、最大持有期 |
| `splits` | `embargo_days`(=13)、`blind_segment_start`(盲测段起点) |
| `champion_challenger` | 影子期长度、换届所需连续不输期数 |
| `backtest` | 参数组合上限、PBO 警告阈值 |

### 5.2 取数(run_fetch_data.py)

从 config 读路径 / 区间 / 披露开关,调用采集逻辑落盘至 store(单一存储,增量去重),并写一张
**带时间戳、不覆盖历史**的数据体检卡(`dataset_card_<时间戳>.json`,记录票数 / 日期范围 / 字段齐全性)。
数据源职责、容错策略与采集语义见 [§6](#6-数据采集详解)。

```bash
python run_fetch_data.py                      # 增量(config.data.start 至今)
python run_fetch_data.py --full               # 全量重拉
python run_fetch_data.py --end 2026-06-12     # 指定结束日
```

### 5.3 训练(run_train.py)

模型的**唯一生产者**:用截至 `--train-end` 的数据训练 L2 模型,存为带时间戳、不覆盖的
**"出生证明"包**(含特征名与顺序、训练区间、超参、config 快照、时间戳、git hash)。运行时断言
`train_end` 距盲测段起点至少 `embargo_days`(13)个交易日,违反则报错退出。本脚本不做对比、不做上线决策。

```bash
python run_train.py --dataset ./data_store --train-end 2025-06-01
```

### 5.4 每日预测(run_predict.py)

只加载已训练模型,**绝不在内部训练**。加载后第一步硬核对当前特征清单与模型出生证明逐一吻合
(不一致即报错退出,防特征错位的静默错误);随后断言仅使用 `asof` 当日及之前的数据(无 T+1,PIT),
套模型排序并输出作战手册。脚本显式打印所用模型文件及其训练截止日,便于下单前判断模型新旧;
昨日推荐回顾为观感参考,非严谨业绩。

```bash
python run_predict.py --asof 2026-06-12                     # 默认取 model_dir 下最新模型
python run_predict.py --asof 2026-06-12 --model <模型路径>   # 指定冠军模型
```

### 5.5 回测调参(run_backtest.py)

回测复用既有事件级引擎与指标,守三条纪律:

- **纪律一(盲测段隔离 + 一次性)**:调参只许在盲测段起点之前;回测区间触碰盲测段时默认拒绝,
  需显式 `--use-blind-once` 放行;重复使用同一盲测段调参将报警(对应 INV-6)。
- **纪律二(三数并排)**:同时输出名义收益、扣费后收益、PBO;PBO 超过 config 阈值时标红警告。
  禁止只输出名义收益。
- **纪律三(粗桶)**:`--param-grid` 的组合总数不得超过 `config.backtest.max_param_combos`,
  超过则报错(禁止精细网格寻优)。

```bash
python run_backtest.py --start 2019-01-01 --end 2024-12-31 \
  --param-grid '{"drawdown_bucket": ["浅","中","深","极深"], "volume_bucket": ["a","b","c","d"]}'
```

### 5.6 冠军挑战者(run_champion_challenger.py)

判断"用更新数据训练的挑战者"是否真比"在用的冠军"好,采用**影子运行式**对决:

- **挑战者 = `run_train.py` 用截至当下全部数据(含最近交易日)训练出的完整模型**,即用户真正会上线
  使用的版本,**不做任何阉割**(绝不为"公平"而排除最近数据)。
- 公平性靠**影子运行 + 滚动前瞻**实现:挑战者先不上线,与冠军并行空跑——每个交易日两者同样出预测并
  记录,但用户只照冠军下单;用挑战者**训练完成之后**才发生的真实时间段的前瞻成绩,与同期冠军比较。
  这段时间对挑战者是未见过的未来,故公平,且被验证的是完整模型而非阉割版。
- **换届判定(防单期噪声)**:仅当挑战者**连续 `switch_requires_consecutive`(默认 3)个影子期**
  累计扣费收益不低于冠军、且最大回撤不更差,才正式换届(挑战者升为冠军、上线实用);单期赢不换。
- 每个影子期成绩仅用于当期判定,不重复使用同一段历史反复评判(滚动 INV-6 精神)。

诚实说明:单个影子期(默认 10 个交易日)样本小、运气成分大,故采用连续多期累计判定;影子期成绩在
有真实市场数据前不代表真实有效性。真实的逐日影子空跑需实盘数据,标记为「未执行(NOT RUN)」;
本脚本仅做成绩累计、换届判定与留痕。

```bash
python run_champion_challenger.py --periods '<影子期成绩 JSON 列表>'
```

---

## 6. 数据采集详解

### 6.1 数据源职责

| 数据源 | 角色 | 提供内容 | 依赖级别 |
|---|---|---|---|
| BaoStock | 唯一信息来源 | 行情(不复权 + 后复权)、交易日历、ST 状态、退市、上市日 | 硬依赖 |
| Tushare | 仅财报获取 | 业绩预告、预约披露日 | 软依赖 |

后复权序列仅取自 BaoStock 单一来源,不与其它来源拼接复权价(INV-2)。Tushare 不作为行情 / 退市 /
日历来源。

### 6.2 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--start` | `config.data.start` | 起始日期 |
| `--end` | 当日 | 结束日期 |
| `--full` | 关闭(默认增量) | 全量重拉 |

其余参数(交易池、是否采集财报、路径)取自 config。采集财报需通过环境变量 `TUSHARE_TOKEN` 或
config 提供凭证。

### 6.3 容错策略

| 情形 | 处理 |
|---|---|
| BaoStock 登录失败 / 全部票拉取失败 | 进程以非零码(2)退出,不产出空或占位数据 |
| BaoStock 单只票失败 | 重试至多 2 次后计入失败名单并继续;失败率超过 5% 时告警 |
| Tushare 凭证缺失 | 跳过财报采集,披露字段置 NULL,行情正常落盘 |
| Tushare 采集异常 | 退避重试至多 2 次,仍失败则降级置空,行情正常落盘 |

行情采集为单进程顺序执行(BaoStock 不支持多线程)。

### 6.4 关键语义

- **双价格层**:`adj_factor = 后复权收盘 / 不复权收盘`;涨跌停价仅依据不复权昨收计算;
  PIT `isST` 决定 ST 标的的 5% 涨跌停幅度。
- **PIT 语义**:`has_preann = NULL`(未采集 / 未知)与 `has_preann = False`(已确认未发预告)
  通过可空布尔类型严格区分。
- **披露 overlay 默认不启用**:未启用财报采集时披露字段为 NULL,Phase 2 披露 overlay 自动短路。

### 6.5 产物

数据落盘至 `config.paths.data_dir`,按年分区:

```
<data_dir>/
├── year=2019/part.parquet
├── year=2020/part.parquet
└── ...
```

落盘后,训练与预测脚本直接经 `store.read(...)` 读取,无需额外搬运。BaoStock / Tushare 真实网络拉取
需实盘环境,标记为「未执行(NOT RUN)」;其编排逻辑由注入式 mock 在
`tests/test_fetch_training_data.py` 中覆盖。

---

## 7. 运行各 Phase(验收脚本)

系统按依赖顺序分阶段实现。每个验收脚本将报告写入 `trading_system/reports/output/`,并以退出码表明结果。

| Phase | 内容 | 命令 |
|---|---|---|
| 0 | 数据底座与双价格层 | `python -m trading_system.run.phase0_acceptance` |
| 1 | 特征、触发器、标签 | `python -m trading_system.run.phase1_factor_report` |
| 2 | 事件级引擎、成本、压力、overlay | `python -m trading_system.run.phase2_acceptance` |
| 3 | L2 模型与审批 | `python -m trading_system.run.phase3_acceptance` |
| 4 | 作战手册、否决审计、监控 | 经 `playbook/`、`audit/`、`reports/monitor.py` 调用 |

退出码语义:

| 退出码 | 含义 |
|---|---|
| 0 | 离线逻辑验收通过(市场验收项在报告中标记为 NOT RUN) |
| 3 | (仅 Phase 0)合成流水线自检通过,实盘逐笔核验需数据 / 凭证,尚未执行 |
| 1 | 验收失败,详见报告 |

各 Phase 的市场验收口径(均需实盘数据):Phase 0 的除权 / 退市 / 涨跌停逐笔核验与披露日历无前视;
Phase 1 的历史 RankIC、混池对照、收益三段拆解;Phase 2 的规则基线 20bp 后为正、首板 30bp 存活、
overlay 满足 ΔMaxDD < 0 且 ΔCalmar > 0;Phase 3 的胜过四基线、walk-forward 与单次盲测。

---

## 8. 运行测试

```bash
pytest                                              # 全部测试
pytest trading_system/tests/test_invariants.py -v   # 仅七条不变量
pytest -W error::UserWarning                        # 将警告视为错误
```

| 测试文件 | 覆盖内容 |
|---|---|
| `test_invariants.py` | 七条不变量断言 |
| `test_structure.py` | 全模块可导入性与配置完整性 |
| `test_phase0_data.py` | 日历、双价格层(含除权)、披露 PIT、交易池、存储增量、质检、快照解析 |
| `test_phase1.py` | 防泄漏三检查(含对守卫有效性的反向验证)、标签、regime 指标、触发器 |
| `test_metrics.py` | RankIC、ICIR、分块 RankIC、MaxDD、Calmar |
| `test_phase2_core.py` | 引擎逐笔核算(五场景)、成本六层、仓位合成 |
| `test_phase2_ext.py` | 滑点压力矩阵、四基线、overlay test |
| `test_phase3.py` | purged CV、PBO / DSR、LightGBM、Optuna、审批与 INV-6 |
| `test_phase4.py` | OPE(IPW / DR)、否决理由码、算法厌恶护栏、作战手册、监控落盘 |
| `test_fetch_training_data.py` | 采集开关、Tushare 降级、BaoStock 硬失败、增量去重、双价格层、PIT 语义 |
| `test_patch_scripts.py` | 配置单一源、模型出生证明、embargo 断言、盲测一次性、回测三数、粗桶上限、冠军挑战者、时间戳命名 |

---

## 9. 目录结构

```
config.yaml                    # 配置中心(单一参数源)
run_fetch_data.py              # 入口 1:取数
run_train.py                   # 入口 2:训练(模型唯一生产者)
run_predict.py                 # 入口 3:每日预测
run_backtest.py                # 入口 4:回测调参(守三纪律)
run_champion_challenger.py     # 入口 5:冠军挑战者影子对决
trading_system/
├── config.py                  # 配置加载器
├── invariants.py              # 七条不变量的可复用原语与守卫
├── check_env.py               # 环境自检
├── champion_challenger.py     # 冠军挑战者影子判定逻辑
├── predict.py                 # 每日预测编排
├── config/                    # 各类"待自验"阈值(逻辑中禁止硬编码)
├── data/                      # Phase 0:数据底座(schema/日历/双价格层/交易池/存储/质检/采集器)
├── features/                  # Phase 1:指标注册表(防泄漏三检查)与特征族
├── regime/                    # Phase 1:L0 六指标、情绪温度、五阶段、HiLo
├── triggers/                  # Phase 1:L1 触发器(粗桶)
├── labels/                    # Phase 1:三类标签
├── backtest/                  # Phase 2:引擎、成本、基线、指标、压力、回测三纪律
├── portfolio/                 # Phase 2:L3 仓位合成
├── overlays/                  # Phase 2:披露季 overlay、高低切交互
├── model/                     # Phase 3:purged CV、LightGBM、Optuna、审批、出生证明(model_io)
├── playbook/                  # Phase 4:作战手册
├── audit/                     # Phase 4:否决审计(OPE)
├── reports/                   # 落盘报告与监控
├── tests/                     # 单元测试
└── run/                       # 各 Phase 验收脚本
```

---

## 10. 数据表结构

存储中每条记录(一只标的一个交易日)包含:

```text
code, trade_date                                                   # 主键
open_raw, high_raw, low_raw, close_raw, preclose_raw, volume, amount    # 执行层(不复权)
adj_factor, open_adj, high_adj, low_adj, close_adj                      # 特征层(后复权)
is_suspended, is_st, is_limit_up, is_limit_down, is_one_price_limit     # 状态位
sched_disclosure_date, has_preann, preann_sign, days_to_disclosure      # 披露季事件(PIT)
```

- 执行类计算(成交价、涨跌停价、止损止盈、股数、成本、盈亏)仅使用 `*_raw` 字段。
- 特征类计算(收益、均线、波动、CGO、RPS)仅使用 `*_adj` 字段。
- 涨跌停价:`涨停 = round(preclose_raw × 1.1, 2)`,`跌停 = round(preclose_raw × 0.9, 2)`;ST 为 ±5%。

---

## 11. 配置说明

`config.yaml`(根目录)为五个入口脚本的单一参数源。`trading_system/config/*.yaml` 存放各 Phase 内部
"待自验"阈值(带 `# TODO: Phase X 自验` 注释),逻辑代码中不得硬编码。

| 文件 | 内容 |
|---|---|
| `config.yaml` | 入口脚本的常用参数(路径、区间、特征清单、成本、风控、划分、冠军挑战者、回测纪律) |
| `trading_system/config/data.yaml` | 数据源角色、存储路径、交易池规则、盘中快照限频 |
| `trading_system/config/costs.yaml` | 成本六层;已核验下限为定值,过户费 / 佣金待核 |
| `trading_system/config/risk.yaml` | 单股上限档、连续跌停 K、波动率目标、凯利档、簇限制 |
| `trading_system/config/exit.yaml` | 硬止损、止盈三阶梯、跟踪系数、最大持有期 |
| `trading_system/config/triggers.yaml` `regime.yaml` `train.yaml` | 触发器粗桶 / L0 参数 / 训练与审批门槛 |

---

## 12. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `run_fetch_data.py` 退出码 2,提示缺少 `baostock` | BaoStock 为行情硬依赖;执行 `pip install baostock` 并确保网络连通后重试 |
| 启用财报采集但无凭证 | 系统跳过财报并置空披露字段,行情正常落盘;如需财报请提供 `TUSHARE_TOKEN` |
| `run_train.py` 报 train_end 距盲测段不足 | 训练截止日距盲测段须 ≥ embargo(13)个交易日,调整 `--train-end` 或 config |
| `run_predict.py` 报特征不一致 | 当前 `config.features` 与模型出生证明不符;统一特征清单或改用匹配的模型 |
| `run_backtest.py` 拒绝回测区间 | 回测区间触碰盲测段;调参应避开盲测段,或显式 `--use-blind-once` 一次性使用 |
| `phase0_acceptance` 退出码 3 | 合成自检通过,实盘逐笔核验需数据 / 凭证,属预期分层结果 |
| 监控 PNG 中文显示为方框 | 默认字体无中文字形,图内轴标题采用 ASCII;中文说明见配套 Markdown 报告 |

---

## 13. 设计不变量

定义于 `trading_system/invariants.py`,并在 `tests/test_invariants.py` 中逐条断言;违反将触发运行时
错误或测试失败。

| 编号 | 名称 | 约束 |
|---|---|---|
| INV-1 | 可交易标签优先 | 卖出日 ≥ 信号日 + 2 个交易日;`h=0` 标签仅用于诊断命名空间 |
| INV-2 | 双价格层 | 执行类计算仅用不复权价,特征类计算仅用后复权价 |
| INV-3 | 标签—成交同源 | 标签与回测引擎使用同一成交判定函数 |
| INV-4 | 组内常数为覆盖层 | 当日同值的量仅以显式交互项进入排序模型 |
| INV-5 | 单股上限由连续跌停压力决定 | `w_max = min(w_hard, L_tail/ĝ)`;主板 ≤ 8%、高风险 ≤ 5%;禁用 15% 默认值 |
| INV-6 | 盲测段一次性 | 盲测段用于换届裁决后即封存,再次使用将报错 |
| INV-7 | 条件化优先于无条件叠加 | 惩罚 / 增强型信号默认以 regime 交互形式进入 |

---

## 14. 术语表

| 术语 | 含义 |
|---|---|
| 不复权 / 后复权 | 不复权为实际成交价(除权日跳变);后复权为消除除权影响后的连续价 |
| PIT(point-in-time) | 仅使用当时可得的信息,不以未来公告回填历史 |
| T+1 | 当日买入次日方可卖出;信号当日生成、次日开盘买入、最早第三个交易日卖出 |
| RankIC / ICIR | 打分排名与未来收益排名的秩相关 / 其均值除以标准差 |
| MaxDD / Calmar | 最大回撤 / 年化收益除以最大回撤 |
| embargo / purged CV | 训练集与验证集间设隔离带并剔除标签窗重叠样本,防时序泄漏 |
| PBO / DSR | 回测过拟合概率(上线 < 30%)/ 去膨胀夏普率(上线 > 0.95) |
| regime / HiLo | 市场状态(动量期与高低切期)/ 高位股相对低位股跑输的风格反转信号 |
| 出生证明 | 模型持久化包,含特征名与顺序、训练区间、超参、config 快照、时间戳 |
| 冠军 / 挑战者 / 影子运行 | 在用模型 / 待评估的最新完整模型 / 挑战者并行空跑不下单仅记录 |
| overlay | 排序之上的降仓或否决覆盖层,须通过 overlay test 方可启用 |

---

## 15. 已知限制

- 当前内置特征为代表性子集(12 项)。CGO 与换手率族需流通股本数据,现有数据表暂未包含,待补齐后接入。
- L0 的 HMM 状态概率为 v3.1 标注的增强可选项,尚未实现。
- Phase 1 的生产标签 `y_prod` 为"固定持有期 + 扣费"的可交易收益;含完整止损 / 止盈状态机的版本由
  Phase 2 事件级引擎承担。
- 次新股 60 日窗口当前以面板内计数近似;精确判定需 BaoStock `query_stock_basic` 的上市日字段(已就位,待接入)。
- 采集器与冠军挑战者影子调度的网络 / 逐日实盘路径需实盘环境验证;离线测试以注入式 mock 与构造数据覆盖其逻辑。
- regime 拐点、踩踏时点与无预告突发事件本质上不可提前精确预测;系统以事前纪律应对,而非事中预判。

---

## 16. 免责声明

本系统仅用于个人交易系统的工程设计与研究流程规范,不构成投资建议。历史数据、回测结果、学术研究
与券商研究均不代表未来表现。
