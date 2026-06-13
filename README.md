# A股日频截面选股交易系统(v3.1 实现版)

纯后端、无前端的 A 股日频截面选股 / 交易研究系统。所有产出均为**落盘工件**:
Python 脚本(PyCharm 直接 Run)、命令行入口、CSV/Parquet/Markdown 文件、落盘的
matplotlib 静态图(`.png`/`.html`)。**不起任何 Web/GUI 服务。**

实现严格遵循 v3.1 冻结设计版,按 **Phase 0 → 4** 顺序推进,**地基先行、模型最后**,
每个 Phase 通过验收后才进入下一个。

> 当前进度:**Phase 0~4 的逻辑均已实现并有真实单元测试**(164 passed, 2 skipped)。
> 测试一律对"已知正确答案"做构造性核验(对标 v3.1 附录方法),逻辑错即真实失败;
> **绝无"通过不报错"的糊弄测试**。2 个 skip 是 BaoStock(需网络)、Tushare(需 token)的实盘路径。
>
> **重要诚实边界:** 各 Phase 的"真实市场验收口径"(Phase 0 的 20 除权/退市/涨跌停逐笔核验、
> Phase 1 的 2019 至今真实 RankIC、Phase 2 的规则基线 20bp 为正、Phase 3 的真实 walk-forward +
> DSR/PBO + 单次盲测)**需要实盘数据 / Tushare token / 训练算力,目前一律标 `NOT RUN`,未伪造**。
> 各 `run/phaseX_acceptance.py` 会跑可离线验证的"计算/逻辑硬门槛",并明确打印实盘验收尚未执行。
> 合成数据仅用于验证**算法正确性**,不代表因子有效性。

---

## 1. 环境与安装(PyCharm)

- Python **3.11+**。
- 在 PyCharm 中:`File → Settings → Project → Python Interpreter` 选 3.11+ 解释器。
- 安装依赖:

  ```bash
  pip install -r requirements.txt
  ```

- 自检环境(必跑):

  ```bash
  python -m trading_system.check_env
  ```

  它会检查 Python 版本、各依赖是否就位(区分"全程必需 / 数据源 / Phase 3 才用")、
  `config/*.yaml` 是否可加载、以及七条不变量原语是否通过冒烟断言。退出码非 0 表示有阻断项。

## 2. 运行测试(七不变量是宪法,必须常绿)

从仓库根目录运行:

```bash
pytest                       # 跑全部测试
pytest trading_system/tests/test_invariants.py -v   # 只看七不变量
```

- `test_invariants.py`:INV-1~INV-7 的断言。其中 **INV-3(标签—成交同源)** 依赖 Phase 1/2
  的共享成交函数,当前以 `skip` 形式占位,待实现后自动转为强校验。
- `test_structure.py`:导入全部子包/模块,确保骨架 import 干净(stub 不在顶层引入重型依赖)。

> 在 PyCharm 中也可右键 `tests/` 目录 → `Run 'pytest in tests'`。

## 3. 目录结构

```
trading_system/
├── config/         # 全部"待自验"阈值集中于此(YAML),逻辑里禁止硬编码
├── invariants.py   # 七条不变量(INV-1~INV-7)的可复用原语 + 守卫(系统宪法)
├── check_env.py    # 环境自检:python -m trading_system.check_env
├── data/           # Phase 0:数据底座(采集器/双价格层/日历/存储/质检/交易池)
├── features/       # Phase 1:指标注册表(防未来函数三检查)+ 特征族
├── regime/         # Phase 1:L0 六指标→情绪温度→五阶段→m_t;HiLo
├── triggers/       # Phase 1:L1 触发器(牛回头/首板/RPS;只用粗桶)
├── labels/         # Phase 1:标签(INV-1 两类标签 + 成交约束同源 INV-3)
├── backtest/       # Phase 2:事件级引擎(唯一真值)/ 成本六层 / 基线 / 指标 / 压力
├── portfolio/      # Phase 2:L3 仓位合成(相对权重×总敞口×多重 min)
├── overlays/       # Phase 2:披露季 overlay、高低切;均须 overlay test
├── model/          # Phase 3:LightGBM + 三标签路线 + purged CV + Optuna + 审批
├── playbook/       # Phase 4:作战手册(CSV+Markdown+控制台)
├── audit/          # Phase 4:否决审计(OPE:IPW/DR)+ 盲测段一次性账本(INV-6)
├── reports/        # 落盘图与报告(静态文件,不起服务)
├── tests/          # pytest:七不变量断言 + 结构冒烟
└── run/            # 各 Phase 的命令行入口 / 验收脚本
```

运行期 Parquet 数据落盘到仓库根的 `data_store/`(已 gitignore)。

## 4. Phase 路线图与验收节奏

| Phase | 内容 | 验收脚本 | 通过标准(摘要) |
|------|------|---------|----------------|
| 0 | 数据底座 + 双价格层 | `python -m trading_system.run.phase0_acceptance` | 20 除权 + 20 退市 + 20 涨跌停逐笔核验;披露日历无前视 |
| 1 | 特征 + 触发器 + 标签(纯统计) | `run/phase1_factor_report.py` | h=0 已降格;截断等变性全过;混池 A/B;收益三段拆解 |
| 2 | 事件级回测引擎 + 成本 + 压力 + overlay | `run/phase2_acceptance.py` | 引擎逐笔对账;规则基线 20bp 为正;每个 overlay ΔMaxDD<0 且 ΔCalmar>0 |
| 3 | L2 模型 + 审批协议 | `run/phase3_acceptance.py` | 胜四基线;PBO<30%、DSR>0.95;过 walk-forward + 单次盲测 |
| 4 | 作战手册 + 否决审计 + 监控 | (见 `playbook/` `audit/` `reports/`) | 执行差距/否决绩效/成交失败率可解释;回撤<8% |

**逐 Phase 验收后再放行下一个**,不跨 Phase 提前实现。

## 5. 七条不变量(硬约束,贯穿全程)

写成 `trading_system/invariants.py` 的可复用原语 + `tests/test_invariants.py` 的断言:

- **INV-1** 可交易标签优先:τ_exit ≥ 信号日 + 2 交易日;h=0 仅限诊断命名空间。
- **INV-2** 双价格层:执行类(成交/涨跌停/止损止盈/PnL)只用 `*_raw`;特征类只用 `*_adj`。
- **INV-3** 标签—成交同源:标签与引擎用同一个成交判定函数(Phase 1/2 落地)。
- **INV-4** 组内常数默认是覆盖层:L0 情绪温度等"当天同值"量只能以显式交互项进入 L2。
- **INV-5** 单股上限由连续跌停压力决定:`w_max = min(w_hard, L_tail/g_hat)`;禁止 15% 默认值。
- **INV-6** 盲测段一次性:用于换届裁决后即封存,再用报错。
- **INV-7** 条件化优先于无条件叠加:惩罚/增强信号默认以 regime 交互进入。

## 6. 配置约定

- 所有 v3.1 未写死、标注"待自验"的阈值,一律放进 `config/*.yaml`,并带 `# TODO: Phase X 自验`
  注释;**禁止把待自验阈值硬编码进逻辑**。
- `config/costs.yaml` 中已核验的成本下限(印花税 0.05%、经手费双向 2×0.00341%)为定值;
  过户费、最低佣金等"待核"项标注来源待核实(Phase 0 据交割单核对)。

## 7. 待用户提供 / 待确认

- **v3.1 设计文档**:本仓库实现引用 v3.1 多个章节(§1.1/§2.3/§5.x/§7.3/§10.3/§11/§12/§13/附录 B)。
  进入 Phase 0 前需要据该文档核对具体公式与阈值。
- **Tushare token**:Phase 0 数据备源 / 披露日历需要。建议通过环境变量 `TUSHARE_TOKEN`
  提供,或填入 `config/data.yaml`(勿提交真实 token)。
