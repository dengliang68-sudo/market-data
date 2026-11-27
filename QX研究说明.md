# QX策略版本与本次回测

基准：QX-001，上次最终推荐（含R1、次日开盘进出）。
候选：QX-002，将首次+50%立即减半改为阶梯回落减半；仅研究，未切换生产。

编号表：backtest/qx_registry.json。共228项完整版本/历史方案/已定义参数组合，其中192项为旧组合参数网格的明确映射，并非本次重新运行的228个策略。
LH与HL为QX-002的日线顺序敏感性情景；R02与R03分别为2%和3%组合风险口径，不另冒充新的策略版本。

## 复现

在GitHub研究分支research/qx-002-20261127完整代码与缓存目录下运行：

```sh
python -m unittest backtest.test_qx_profit backtest.test_qx_engine -v
python backtest/qx_compare.py
```

依赖固定核心版本：Python3.11、pandas2.2.3、numpy2.2.6、pyarrow19.0.1；另需tushare、requests、six、tabulate。只读缓存，不需要Tushare密钥。

研究结果位于reports/backtest/qx_20261127。旧基准成功逐笔复现226笔，3%风险期末558227元；12项规则与引擎测试通过。
成功运行：https://github.com/dengliang68-sudo/market-data/actions/runs/45932280793
冻结行情与原代码提交：00f762d09582dd1bb6d6a8d8da8c59c9b5b2b8b0。

本地目录是源代码与研究输出工作副本；大文件行情仍在远端冻结缓存，未完整下载至本地。
Excel由build_workbook.py生成，需openpyxl；并非回测输入。
