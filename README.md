# A 股日频截面选股交易系统(v3.1 实现版)

> 一套**纯后端、无前端**的 A 股日频截面选股 / 交易研究系统。所有产出都是**落盘工件**——
> Python 脚本(PyCharm 直接 Run)、命令行入口、CSV / Parquet / Markdown 文件、落盘的
> matplotlib 静态图(`.png` / `.html`)。**不起任何 Web / GUI 服务。**

它把一份冻结的设计文档(v3.1)逐字翻译成可运行、可测试的工程代码,核心理念只有一句:

> **确保你看到的每一个数字都是真的。**

为此整套系统建立在**七条不变量(宪法)**之上,并用**真实单元测试**(对已知正确答案做核验,
错就真失败,绝不写"通过不报错"的糊弄测试)守住它们。

---

## 目录

1. [当前状态与诚实边界](#1-当前状态与诚实边界)
2. [30 秒快速开始](#2-30-秒快速开始)
3. [这套系统在做什么(架构总览)](#3-这套系统在做什么架构总览)
4. [七条不变量(系统宪法)](#4-七条不变量系统宪法)
5. [仓库内容详解(每个模块干什么)](#5-仓库内容详解每个模块干什么)
6. [安装与环境自检](#6-安装与环境自检)
7. [数据采集教程(一条命令落盘)](#7-数据采集教程一条命令落盘)
8. [各 Phase 怎么跑](#8-各-phase-怎么跑)
9. [运行测试(宪法必须常绿)](#9-运行测试宪法必须常绿)
10. [数据字典(双价格层 schema)](#10-数据字典双价格层-schema)
11. [配置文件说明](#11-配置文件说明)
12. [常见问题与故障排查](#12-常见问题与故障排查)
13. [名词表(新手必看)](#13-名词表新手必看)
14. [待办与已登记边界](#14-待办与已登记边界)

---

## 1. 当前状态与诚实边界

**进度:Phase 0~4 的逻辑全部实现,并配有真实单元测试。**

```
176 passed, 3 skipped
```

- **测试纪律:** 每条断言都对"可手算 / 可独立推导的正确答案"做构造性核验(对标 v3.1 附录方法),
  逻辑错就真实失败。**没有任何"为了能跑而通过不报错"的测试。** 我们甚至专门写测试去验证防泄漏
  守卫**确实能抓出**泄漏(否则那守卫就是橡皮图章)。
- **3 个 skip 都是诚实的"需外部资源":** 2 个需 BaoStock 网络会话、1 个需 Tushare token,
  都带明确原因,绝不伪装通过。
- **重要——区分"逻辑正确"与"市场有效":** 各 Phase 的**真实市场验收口径**(Phase 0 的
  20 除权/退市/涨跌停逐笔核验、Phase 1 的 2019 至今真实 RankIC、Phase 2 的规则基线 20bp 为正、
  Phase 3 的真实 walk-forward + DSR/PBO + 单次盲测)**需要实盘数据 / Tushare token / 训练算力,
  目前一律标 `NOT RUN`,未伪造**。各 `run/phaseX_acceptance.py` 只跑"可离线验证的计算/逻辑硬门槛",
  并明确打印实盘验收尚未执行。
- **合成数据只用于验证算法正确性,不代表因子有效性**(对应 v3.1 附录 D:构造性仿真不作市场实证)。

> 设计优先级:**v0.3 宪法 > v3.1 设计文档 > v1.0**。任何冲突以宪法红线为准。

---

## 2. 30 秒快速开始

```bash
# 0) 进入仓库根目录,装依赖(Python 3.11+)
pip install -r requirements.txt

# 1) 自检环境:依赖/配置/七不变量是否就位
python -m trading_system.check_env

# 2) 跑全部测试(应为 176 passed, 3 skipped)
pytest

# 3) 采集训练数据(纯 BaoStock 行情,自动落盘;需要网络)
python -m trading_system.data.fetch_training_data --start 2019-01-01

# 4) 跑各 Phase 的离线验收 / 体检报告(产出 Markdown 到 reports/output/)
python -m trading_system.run.phase1_factor_report
python -m trading_system.run.phase2_acceptance
python -m trading_system.run.phase3_acceptance
```

> 在 PyCharm 里:选 3.11+ 解释器 → 右键任意 `run/*.py` 或 `tests/` 目录 → Run 即可。

---

## 3. 这套系统在做什么(架构总览)

系统是一条**四层流水线**,从"今天市场什么状态"一路推到"今天买什么、买多少、何时卖":

```
L0 市场状态层      →  L1 触发与候选池   →  L2 截面排序      →  L3 组合与风控     →  事件级回测 / 作战手册
(情绪温度/regime)    (牛回头/首板/RPS)     (LightGBM 打分排名)   (仓位/止损/上限)      (唯一真值 / 每日清单)
```

| 层 | 通俗解释 | 代码位置 | Phase |
|---|---|---|---|
| **L0 市场状态** | 今天大盘"情绪温度"几度?该满仓还是空仓?(养家五阶段 → 仓位乘子 `m_t`) | `regime/` | 1 |
| **L1 触发器** | 从全市场筛出"今天值得看"的候选(回调启动 / 缩量首板 / 强势龙头) | `triggers/` | 1 |
| **L2 截面排序** | 对候选用机器学习打分排名,选 Top-K | `model/` + `features/` | 3 |
| **L3 组合风控** | 每只买多少(凯利)、单股上限(连续跌停压力)、总敞口、簇限制 | `portfolio/` | 2 |
| **回测/手册** | 事件级引擎逐笔模拟成交(T+1、涨跌停、止损止盈);每日生成作战手册 | `backtest/` `playbook/` | 2/4 |

**两个贯穿全程的实战疑问**(v3.1 专门回答):
- **披露季财报暴雷** → 用"披露季 overlay"在已知高风险窗**降暴露**(不预测方向),见 `overlays/`。
- **情绪过热、维稳高低切** → 把"过度拉升"做成与"高低切 regime"的**交互项**(而非无条件扣分),见 `regime/` + `features/`。

---

## 4. 七条不变量(系统宪法)

它们写在 `trading_system/invariants.py`(可复用原语 + 运行时守卫),并在
`tests/test_invariants.py` 里被逐条断言。**任何代码违反即报错 / 测试失败。**

| 编号 | 名称 | 一句话解释 | 为什么 |
|---|---|---|---|
| **INV-1** | 可交易标签优先 | 卖出日 ≥ 信号日 + 2 个交易日;`h=0` 只能进诊断 | T+1 制度下,今天的信号最早后天才能卖,标签必须如实反映 |
| **INV-2** | 双价格层 | 执行(成交/涨跌停/止损/PnL)只用**不复权价**;特征(收益/均线/波动)只用**后复权价** | 除权日不复权价会跳变、后复权价连续;用错层会算错涨停价或泄漏 |
| **INV-3** | 标签—成交同源 | 标签里"T+1 一字/高开买不进"的判断,和回测引擎填单**用同一个函数** | 避免训练时假设能买、回测时其实买不进 |
| **INV-4** | 组内常数是覆盖层 | 情绪温度等"当天所有股票同值"的量,只能以**显式交互项**进 L2 | 裸的同值列改不了组内排序,还会隐性过拟合(附录 A 证明) |
| **INV-5** | 单股上限看连续跌停 | `w_max = min(w_hard, L_tail/ĝ)`;主板 ≤8%、高风险 ≤5%;**禁止 15%** | 极端尾部靠事前降敞口兜,不靠预测(附录 B 推导) |
| **INV-6** | 盲测段一次性 | 某段数据用于换届裁决后即封存,再用报错 | 防止把盲测段反复用成"调参集"导致过拟合 |
| **INV-7** | 条件化优先于无条件 | 惩罚/增强型信号默认以 regime 交互进入,无条件叠加须证明不更差 | 无条件惩罚易"1+1<2"打坏趋势期(附录 D 仿真) |

---

## 5. 仓库内容详解(每个模块干什么)

> 每个业务函数的 docstring 都注明:**属于哪个 Phase、对应 v3.1 哪一节、用不复权还是后复权价**。

### `trading_system/`(顶层)
| 文件 | 作用 |
|---|---|
| `invariants.py` | **系统宪法**:七条不变量的纯函数 / 守卫 / 数据结构(如涨跌停半进位取整、连续跌停损失、盲测段账本) |
| `check_env.py` | 环境自检 CLI:依赖 / 配置 / 不变量冒烟 |

### `data/`(Phase 0:数据底座)
| 文件 | 作用 |
|---|---|
| `schema.py` | 统一表的列名分组(raw / adj / 状态位 / 披露),从数据出口处支持 INV-2 按用途取数 |
| `calendar.py` | 交易日历:T+1 / T+2 / embargo 偏移、日期对齐(越界/非交易日真实报错) |
| `price_layers.py` | **双价格层构造(INV-2 核心)**:`adj = raw × adj_factor`,涨跌停价用不复权昨收,状态位(涨停/跌停/一字/停牌),披露 PIT 字段 |
| `universe.py` | 交易池过滤:沪深主板、非 ST、非停牌、上市满 60 日、非一字板 |
| `store.py` | **单一数据出口**:Parquet(ZSTD)按年分区 + DuckDB 查询 + 年分区级增量去重 |
| `quality.py` | 每日质检:raw/adj 分离、涨跌停核对、后复权连续性、一字/停牌自洽、披露字段 |
| `fetch_training_data.py` | **一键采集入口**(见 [第 7 节](#7-数据采集教程一条命令落盘)) |
| `collectors/baostock.py` | **唯一信息来源**:行情(不复权+后复权)/ 日历 / ST / 退市 / 上市日的底层封装 |
| `collectors/baostock_collector.py` | BaoStock 编排:会话管理、顺序拉取、单票重试降级、交易池列举 |
| `collectors/tushare.py` | **仅财报**:业绩预告 + 预约披露日(非信息来源) |
| `collectors/tushare_collector.py` | Tushare 编排:退避重试 → 失败抛 `TushareError`(软依赖,交上层降级) |
| `collectors/tencent.py` `sina.py` | 盘中快照解析器(Phase 4 主用;解析函数可单测) |
| `collectors/synthetic.py` | **合成数据源(仅测试/开发,非市场实证)** |

### `features/` `regime/` `triggers/` `labels/`(Phase 1)
| 文件 | 作用 |
|---|---|
| `features/registry.py` | 指标注册表 + **防未来函数三检查**(静态扫描 / 截断等变性 / 前复权陷阱拦截)+ 截面 winsorize+秩变换 |
| `features/builtin/families.py` | 12 个内置特征(全用后复权价,全过截断等变性):<br>· 量价基础:`ret_1/5/20`、`vol_20`、`volume_ratio_5`<br>· 流动性:`amihud_20`<br>· 趋势:`ma_ratio_20/60`<br>· 反转彩票:`reversal_5`、`max_ret_5`<br>· 过度拉升:`dist_ma20`、`dist_high_20` |
| `regime/__init__.py` | L0 六指标(涨停家数/连板高度/晋级率/炸板率/昨涨停今溢价/跌停+核按钮)→ 情绪温度 `T_t`(250 日分位)→ 五阶段 → 仓位乘子 `m_t`;HiLo 高低切状态量 |
| `triggers/__init__.py` | L1 触发器:A 牛回头 / B 缩量低位首板 / C RPS 龙头(**只用 config 粗桶,结构性禁止网格寻优**) |
| `labels/__init__.py` | 标签:`y_prod`(生产可交易)/ `y_h`(固定窗口)/ `y_mtm0`(诊断 h=0);落地 INV-1,且 import 引擎的 `is_tradeable_fill`(INV-3) |

### `backtest/` `portfolio/` `overlays/`(Phase 2)
| 文件 | 作用 |
|---|---|
| `backtest/engine.py` | **事件级引擎(唯一真值)**:买入/卖出状态机、出场优先级(硬止损>止盈>时间)、收盘确认次开执行、跳空实际亏损可 >2.5N、跌停/停牌顺延、T+1 最早 t+2 |
| `backtest/costs.py` | 成本六层(已核验下限 5.682bp;过户费/佣金/滑点可配;最低佣金闸门) |
| `backtest/metrics.py` | RankIC / ICIR / 分块不重叠 RankIC / MaxDD / Calmar / 换手 / **PBO / DSR** |
| `backtest/baselines.py` | 四基线:随机候选池 / 单因子 / ElasticNet 截面秩回归(LightGBM 基线见 model) |
| `backtest/stress.py` | 滑点压力矩阵(逐档重算扣费净收益)+ 存活门槛 |
| `portfolio/__init__.py` | L3 仓位合成:凯利符号三档 / 单股上限(INV-5)/ 总敞口多重 min / 逆 ATR 相对权重 + 逐票截断 + 拥挤簇限制 |
| `overlays/__init__.py` | 披露季 overlay(VETO/REDUCE/NONE)/ HiLo×过度拉升交互(INV-7)/ overlay test 框架(ΔMaxDD<0 且 ΔCalmar>0 才启用) |

### `model/`(Phase 3:L2 模型 + 审批)
| 文件 | 作用 |
|---|---|
| `cv.py` | purged + embargo 时序交叉验证(`embargo = H_max + K + 1 = 13`) |
| `train.py` | 三标签路线(秩回归 / 净收益回归 / lambdarank 分位)+ LightGBM ranker(**group 按交易日**)+ 装配时强制 INV-4 |
| `tune.py` | Optuna 调参(搜索空间**预注册**,结构性禁止越界扩张) |
| `approval.py` | 五重 AND 上线门槛(`R_blind>0`、`DSR>0.95`、`PBO<0.30`、`ΔMaxDD≤0`、`SlippageStress>0`…)+ 盲测段一次性(INV-6) |

### `playbook/` `audit/` `reports/`(Phase 4)
| 文件 | 作用 |
|---|---|
| `playbook/__init__.py` | **作战手册**:每日 CSV + Markdown + 控制台;每票含触发器/分数/SHAP 理由/限价/股数/止损止盈/否决栏/风险标注;页脚印 `T_t`/阶段/`m_t`/`w_total` |
| `audit/__init__.py` | 否决审计(OPE 反事实:IPW / DR)、封闭 reason code(12 个,含"过度拉升/彩票""高低切风格反转")、算法厌恶护栏 |
| `audit/experiment_registry.py` | 盲测段一次性账本的持久化占位(Phase 3+) |
| `reports/monitor.py` | 核心监控面板:分块 RankIC / 净值 MaxDD / Calmar / 成交失败率 / 执行差距,落盘 PNG + Markdown |

### `config/` `tests/` `run/`
- `config/*.yaml`:见 [第 11 节](#11-配置文件说明)。
- `tests/*.py`:10 个测试文件,见 [第 9 节](#9-运行测试宪法必须常绿)。
- `run/*.py`:各 Phase 的命令行验收脚本,见 [第 8 节](#8-各-phase-怎么跑)。

---

## 6. 安装与环境自检

```bash
# Python 3.11+。PyCharm: File → Settings → Project → Python Interpreter 选 3.11+。
pip install -r requirements.txt

python -m trading_system.check_env
```

`check_env` 会检查:Python 版本、依赖是否就位(区分**全程必需 / 数据源 / Phase 3 才用**)、
七个 `config/*.yaml` 是否可加载、七条不变量原语是否通过冒烟断言。**退出码非 0 = 有阻断项**
(末尾会提示 `pip install -r requirements.txt`)。

依赖分组(`requirements.txt`):
- 全程:`pandas / numpy / pyarrow / duckdb / pyyaml / scipy / scikit-learn / matplotlib / pytest`
- 数据源:`baostock`(行情主源,无需 token)、`tushare`(仅财报,需 token)、`requests`
- Phase 3:`lightgbm / optuna`

---

## 7. 数据采集教程(一条命令落盘)

**目标:用户一条命令,把训练所需数据自动落盘,无需手动操作 BaoStock。**

### 数据源职责(重要)
- **BaoStock = 唯一信息来源**:行情(不复权 + 后复权)、交易日历、ST 状态、退市、上市日。**硬依赖**。
- **Tushare = 仅财报获取**:业绩预告 + 预约披露日。**软依赖**(失败即降级,不影响行情)。

### 用法
```bash
# 默认:纯 BaoStock 行情(沪深主板非 ST,2019 至今,增量),不采集财报
python -m trading_system.data.fetch_training_data --start 2019-01-01

# 全量重拉(忽略本地已有,从头拉)
python -m trading_system.data.fetch_training_data --start 2019-01-01 --full

# 额外采集财报(需 Tushare token;无 token 或失败 → 自动降级置空,行情照常落盘)
TUSHARE_TOKEN=你的token \
python -m trading_system.data.fetch_training_data --start 2019-01-01 --enable-disclosure
```

### 命令行参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `--start` | `2019-01-01` | 起始日期 |
| `--end` | 今天 | 结束日期 |
| `--universe` | `main_board` | 交易池(沪深主板非 ST) |
| `--enable-disclosure` | 关 | 是否采集财报/披露(需 token) |
| `--tushare-token` | — | token(也可用环境变量 `TUSHARE_TOKEN`) |
| `--incremental` / `--full` | `--incremental` | 增量(只拉 `本地最新+1 → 今天`)/ 全量 |
| `--out` | store 既定路径 | 落盘目录 |

### 执行流程与防御(发生了什么)
1. **列出交易池**(BaoStock,主板非 ST)。
2. **拉行情**(硬依赖):每只票同源取不复权 + 后复权 → 算 `adj_factor` → `build_price_layers` → 写 store。
   - **整批 login 失败 / 全部票失败 → 进程非零退出(exit 2)**,绝不静默产出空/假数据。
   - 单只票失败:重试 ≤2 次后计入失败名单、继续下一只;失败率 >5% 告警。
   - **单进程顺序拉**(BaoStock 不支持多线程)。
3. **拉财报**(软依赖,仅 `--enable-disclosure`):无 token → 警告并跳过(置 NULL);有 token 但异常 →
   退避重试 ≤2 次 → 仍失败则**降级置空、主流程继续**。
4. **质检**:行情必检;披露仅在采集成功时检。
5. **总结**:行情成功/失败数、披露状态、落盘路径、数据日期范围 + 一行"下一步"提示。

### 关键语义(别踩坑)
- **双价格层(INV-2)**:`adj_factor = 后复权收盘 / 不复权收盘`;涨跌停价只用**不复权昨收**;
  PIT `isST` 决定 ST 的 5% 涨跌停(否则误用 10%)。后复权只用 BaoStock 单源,**不跨源拼接复权价**。
- **PIT 语义**:`has_preann = NULL`(未采集/未知)≠ `has_preann = False`(已确认未发预告),用
  nullable boolean 严格区分。
- **披露默认不启用**:不加 `--enable-disclosure` → 披露字段全 NULL → Phase 2 披露 overlay 自动短路
  (与 v3.1"默认不启用,待验证"一致)。

> 落盘后,Phase 1 / Phase 3 直接从 `store` 读,无需搬运。
> 真实 BaoStock / Tushare 网络拉取为 **NOT RUN**(需网络 / token);离线已用 mock 验证
> 开关短路、Tushare 降级、BaoStock 硬失败退出、增量去重、双价格层完整、PIT 语义共 6 类
> (`tests/test_fetch_training_data.py`)。

---

## 8. 各 Phase 怎么跑

> 设计纪律:**地基先行、模型最后,逐 Phase 验收后再放行下一个,不跨 Phase 提前实现。**

| Phase | 内容 | 运行命令 | 验收口径 |
|---|---|---|---|
| 0 | 数据底座 + 双价格层 | `python -m trading_system.run.phase0_acceptance` | 20 除权 + 20 退市 + 20 涨跌停逐笔核验;披露日历无前视 |
| 1 | 特征 + 触发器 + 标签 | `python -m trading_system.run.phase1_factor_report` | 截断等变性全过;真实 RankIC / 混池 A/B / 收益三段拆解 |
| 2 | 引擎 + 成本 + 压力 + overlay | `python -m trading_system.run.phase2_acceptance` | 引擎逐笔对账;规则基线 20bp 为正;overlay ΔMaxDD<0 且 ΔCalmar>0 |
| 3 | L2 模型 + 审批 | `python -m trading_system.run.phase3_acceptance` | 胜四基线;PBO<30%、DSR>0.95;walk-forward + 单次盲测 |
| 4 | 手册 + 否决审计 + 监控 | (用 `playbook/` `audit/` `reports/monitor.py`) | 执行差距/否决绩效/成交失败率可解释;回撤<8% |

**验收脚本的退出码语义(诚实分层):**
- **`phase0_acceptance` → exit 3**:合成流水线逻辑通过,但真实 20+20+20 逐笔核验 = `NOT RUN`(需数据/token),
  明示"Phase 0 尚未完整验收"。
- **`phase1/2/3_acceptance` → exit 0**:**可离线验证的逻辑硬门槛**(截断等变性 / 引擎逐笔对账 /
  CV·PBO·DSR·INV-6)通过;真实市场验收部分明确标 `NOT RUN`。
- 报告(Markdown)写到 `trading_system/reports/output/`(已 gitignore)。

---

## 9. 运行测试(宪法必须常绿)

```bash
pytest                                              # 全部(176 passed, 3 skipped)
pytest trading_system/tests/test_invariants.py -v   # 只看七不变量
pytest -W error::UserWarning                        # 把警告当错误,验证零隐患
```

| 测试文件 | 验证什么 |
|---|---|
| `test_invariants.py` | INV-1~INV-7 全部断言(INV-3 现已强校验通过) |
| `test_structure.py` | 全模块 import 干净 + 配置齐备 |
| `test_phase0_data.py` | 日历 / 双价层除权 / 披露 PIT / 交易池 / 存储增量 / 质检异常检出 / 快照解析器 |
| `test_phase1.py` | 防未来三检查(**含验证守卫确实抓得到泄漏**)/ 标签手算 / regime 指标 / 触发器 |
| `test_metrics.py` | RankIC(完全正/反相关)/ MaxDD / Calmar / 分块 |
| `test_phase2_core.py` | **引擎逐笔手工对账 5 场景** / 成本六层 / 仓位合成 |
| `test_phase2_ext.py` | 滑点压力 / 四基线 / overlay test |
| `test_phase3.py` | purged CV / PBO·DSR / LightGBM(INV-4 + 按日 group)/ Optuna / 审批 + INV-6 |
| `test_phase4.py` | OPE(IPW/DR 手算)/ reason codes / 算法厌恶护栏 / 作战手册 / 监控落盘 |
| `test_fetch_training_data.py` | 采集开关 / Tushare 降级 / BaoStock 硬失败 / 增量去重 / 双价层 / PIT 语义 |

> 3 个 skip:`fetch_training_data` 真实 baostock、`phase0` 真实 baostock、`phase0` 真实 tushare——
> 都因需网络/token 而 `NOT RUN`,带明确原因,绝不伪装通过。

---

## 10. 数据字典(双价格层 schema)

`store` 里每行(一只票一天)包含:

```text
code, trade_date,                                                  # 主键
open_raw, high_raw, low_raw, close_raw, preclose_raw, volume, amount,   # 执行层(不复权)
adj_factor, open_adj, high_adj, low_adj, close_adj,                     # 特征层(后复权)
is_suspended, is_st, is_limit_up, is_limit_down, is_one_price_limit,    # 状态位
sched_disclosure_date, has_preann, preann_sign, days_to_disclosure      # 披露季事件(PIT)
```

- **执行类**(成交价/涨跌停价/止损止盈/股数/成本/PnL)**只准用 `*_raw`**。
- **特征类**(收益/均线/波动/CGO/RPS)**只准用 `*_adj`**。
- 涨跌停价:`涨停 = round(preclose_raw × 1.1, 2)`,`跌停 = round(preclose_raw × 0.9, 2)`(ST 用 ±5%)。
- 取数走 `store.read(codes=, start=, end=, fields=)`——按 `fields` 取数,执行代码只请求 raw、特征代码只请求 adj,从出口处就把两层分开。

---

## 11. 配置文件说明

所有 v3.1 未写死、标"待自验"的阈值都在 `config/*.yaml`(带 `# TODO: Phase X 自验`),
**逻辑里禁止硬编码**。

| 文件 | 内容 |
|---|---|
| `data.yaml` | 数据源(BaoStock 唯一信息来源 / Tushare 仅财报)、store 路径、交易池规则、限频 |
| `costs.yaml` | 成本六层;已核验下限(印花税 0.05% + 经手费双向 2×0.00341% = 5.682bp)为定值,过户费/佣金待核 |
| `triggers.yaml` | L1 触发器粗桶边界(全部待自验) |
| `risk.yaml` | 单股上限档(主板 8%/特殊 5%)、连续跌停 K、波动率目标、凯利档、簇限制 |
| `exit.yaml` | 出场:2.5N 硬止损、止盈三阶梯、跟踪系数 c、最大持有 H |
| `regime.yaml` | L0 六指标权重、五阶段阈值、HiLo 参数(待自验) |
| `train.yaml` | 标签 h、embargo、purged CV、Optuna 空间、审批门槛(Phase 3) |

---

## 12. 常见问题与故障排查

**Q:`fetch_training_data` 直接退出码 2,报 `No module named 'baostock'`?**
A: 没装 baostock。BaoStock 是行情硬依赖,缺了就**故意非零退出**(不产出假数据)。
   `pip install baostock` 且保证容器/机器能联网后重试。

**Q:开了 `--enable-disclosure` 但没 token?**
A: 会打 warning 并**跳过财报采集**,披露字段置 NULL,行情照常落盘(软依赖降级)。

**Q:`phase0_acceptance` 退出码是 3,不是 0,是不是错了?**
A: 不是。3 = "合成自检通过,但真实 20+20+20 逐笔核验需实盘数据,尚未跑"。这是**诚实分层**,不是失败。

**Q:`check_env` 退出码非 0?**
A: 有阻断依赖未装。按提示 `pip install -r requirements.txt`。

**Q:监控 PNG 里中文显示成方框?**
A: 默认字体(DejaVu Sans)无中文字形,故图内轴标题用 ASCII;中文叙述在配套 Markdown 报告里。

**Q:为什么测试里有合成数据?是不是在"造数据骗自己"?**
A: 不是。合成数据是**已知性质的构造性 fixture**,用来验证**算法正确性**(对标 v3.1 附录方法),
   并明确声明**不代表因子有效性**。真实市场验收一律标 `NOT RUN`。

---

## 13. 名词表(新手必看)

| 名词 | 含义 |
|---|---|
| **不复权 / 后复权** | 不复权=实际成交价(除权日会跳);后复权=把除权影响抹平后的连续价(算收益用) |
| **PIT(point-in-time)** | "当时只知道当时的信息"——绝不用未来公告回填历史,防前视偏差 |
| **T+1** | A 股当天买入次日才能卖;本系统信号当日出、次日开盘买、最早后天卖 |
| **RankIC** | 当日"打分排名"与"未来收益排名"的秩相关;衡量选股有没有用,越高越好 |
| **ICIR** | RankIC 的均值/标准差;衡量选股能力的稳定性 |
| **MaxDD / Calmar** | 最大回撤 / 年化收益÷最大回撤;衡量风险与风险调整后收益 |
| **embargo / purged CV** | 训练集与验证集之间留隔离带、剔除标签窗重叠的样本,防时序泄漏 |
| **PBO** | 过拟合概率(组合对称交叉验证);上线要求 <30% |
| **DSR** | 去膨胀夏普——扣掉"试了很多次总有一个看着好"的运气;上线要求 >0.95 |
| **regime** | 市场状态/风格(动量期 vs 高低切期);只能事后确认,不能领先择时 |
| **HiLo 高低切** | 高位股开始跑输低位股的风格反转信号 |
| **CGO / MAX / RPS / 量比** | 浮盈水平 / 近期单日最大涨幅(博彩特征)/ 相对强弱 / 当日量比过去均量 |
| **overlay** | 覆盖层:在排序之上做的降仓/否决,必须过 overlay test(ΔMaxDD<0 且 ΔCalmar>0)才启用 |
| **2.5N / ATR** | N=ATR(波动幅度);硬止损设在入场价 - 2.5N |
| **凯利三档** | 按证据强度把单笔风险预算设为 100%/50%/0% |
| **lambdarank** | LightGBM 的 learning-to-rank 目标;本系统 group 必须按交易日 |

---

## 14. 待办与已登记边界

**需要用户提供后才能跑的"真实验收":**
- **网络 + `pip install baostock`**:真实行情拉取、Phase 0 的 20+20+20 逐笔核验、Phase 1~3 真实市场验收。
- **Tushare token**:**仅**财报(业绩预告/预约披露日)。Tushare 非信息来源——行情/日历/ST/退市一律 BaoStock。

**实现中已声明的近似 / 边界(没藏):**
- 特征是代表性 12 个;**CGO / 换手率族需流通股本**,本仓 schema 暂未含,数据补齐后再接(不拿代理冒充)。
- L0 的 **HMM 状态概率**是 v3.1 标注的"增强可选",未做。
- Phase 1 的 `y_prod` 是"固定持有 + 扣成本"的可交易标签;含完整止损/止盈状态机的版本由 Phase 2 引擎承担。
- 次新股 60 日窗目前用面板内计数近似;精确需 BaoStock `query_stock_basic` 的 `ipoDate`(已就位,待接入)。
- 采集器的网络路径离线**无法验证**,仅解析器/编排做了 mock 单测。

**已登记的不可解盲区(v3.1 第十六章):**
- regime 拐点、踩踏时点、无预告突发暴雷**本质不可提前精确预测**;系统靠事前纪律(暴露约束、单股上限、
  条件化、不加杠杆)兜底,不靠事中精准预判。

> 声明:仅用于个人交易系统工程设计与研究流程规范,**不构成投资建议**;历史数据、回测、学术与券商研究均不代表未来。
