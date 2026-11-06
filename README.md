# market-data

股票复盘与期权交易系统共用的数据程序。数据来自 Tushare（A股、港股、期权）和 FRED（标普500、纳指100、VIX、美债收益率）。

## 运行

```bash
pip install -r requirements.txt --break-system-packages
export TS_TOKEN=<Tushare Key>        # 不要把 Key 写进仓库
python run_review.py --session pm    # 晚报（A股/港股收盘后）
python run_review.py --session am    # 早报（美股收盘后）
```

首次运行要回填约 130 个交易日的板块数据，约 6 分钟；之后只补新日期，约 30 秒。

## 输出

- `reports/YYYYMMDD_am|pm.md`：给报告用的数据摘要
- `reports/YYYYMMDD_am|pm.json`：完整结构化数据
- `data/*.parquet`：增量缓存（每次运行后提交回仓库）

## 模块

| 文件 | 内容 |
|---|---|
| `md/client.py` | Tushare 调用（重试、限速）、交易日历、按日期增量缓存、FRED 下载 |
| `md/signals.py` | 趋势雷达分级规则（观察 / 准备 / 可介入 / 退潮警示） |
| `md/cn.py` | A股大盘温度、两融、南向、申万和同花顺板块扫描、东财行业资金流、港股指数 |
| `md/us.py` | 美股指数基金面板：回撤档位、温度、过热条件 |

## 已知局限

- 美股个股和 ETF 日线：Tushare `us_daily` 频率限制太严，暂不可用；美股行业 ETF 待接 Alpha Vantage。
- 等权标普500：暂无数据源。
- 港股行业板块：暂未接入，只有恒指和恒生科技。
- "明确催化"和板块内上涨扩散仍需人工判断。
- 期权模块：待开发。
