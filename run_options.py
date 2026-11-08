"""ETF 期权数据程序（给「期权交易」项目的盘前简报用）。

用法：
    TS_TOKEN=xxx python run_options.py                # 最近一个 A股交易日收盘后的数据
    TS_TOKEN=xxx python run_options.py --date 20261106

输出：
    reports/YYYYMMDD_opt.md    数据摘要（YYYYMMDD 是数据所属的交易日；第二天早上的简报读它）
    reports/YYYYMMDD_opt.json  完整结构化数据
    data/opt_daily_*.parquet   期权日线增量缓存
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from md import opt, opt_rules  # noqa: E402
from md.client import trade_days  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="A股交易日 YYYYMMDD（默认最近交易日）")
    a = ap.parse_args()
    days = trade_days(130, end=a.date)
    p = opt.option_panel(days)
    d = p["date"]

    rules = opt_rules.evaluate(p, days)
    p["rule_signals"] = rules
    L = [f"# ETF 期权数据摘要（{d} 收盘）\n",
         f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}；数据来自 Tushare，IV 和 Delta 按 Black-Scholes 用收盘价计算"
         f"（无风险利率 Shibor 3M = {p['rate']}%），未经人工核对。\n",
         *opt_rules.to_markdown(rules),
         "\n## 品种状态\n",
         "| 品种 | 收盘 | 涨跌% | 趋势 | MA20 | MA60 | MA20 5日斜率% | 10日穿越20日线次数 | 前20日最高/最低收盘 | 量比 | 触发 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for u in p["underlyings"]:
        L.append(f"| {u['name']} | {u['close']} | {u['pct']} | **{u['state']}** | {u['ma20']} | {u['ma60']} | {u['ma20_slope5']} | "
                 f"{u['cross_ma20_10d']} | {u['high20_close']} / {u['low20_close']} | {u['vol_ratio']} | {'；'.join(u['triggers']) or '无'} |")

    L += ["\n## 隐含波动率与持仓结构\n",
          "| 品种 | 平值IV（当月 / 次月） | 参考月份 | 60日IV均值 | IV 60日分位 | IV水平 | 成交PCR | 持仓PCR | 认购最大持仓行权价（压力） | 认沽最大持仓行权价（支撑） |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for u in p["underlyings"]:
        ivs = " / ".join(f"{v}%" if v is not None else "—" for v in u["atm_iv"].values())
        L.append(f"| {u['name']} | {ivs} | {u['iv_ref_month']} | {u['iv_mean_60d']}% | {u['iv_pctile_60d']} | **{u['iv_level']}** | "
                 f"{u['pcr_vol']} | {u['pcr_oi']} | {u['max_oi_call_strike']} | {u['max_oi_put_strike']} |")

    L += ["\n## 合约日历\n", "| 品种 | 到期日 | 剩余天数 | 提示 |", "|---|---|---|---|"]
    for u in p["underlyings"]:
        for c in u["calendar"]:
            L.append(f"| {u['name']} | {c['maturity']} | {c['days_left']} | {c['note']} |")

    L.append("\n## 候选合约（Delta 0.4–0.6，已排除分红调整合约）\n")
    L.append("优先列剩余 20–45 天的月份；没有符合的月份时，列出所有剩余 ≥10 天的月份，由简报按指引取舍。"
             "只列出合约数据，是否交易要看趋势状态和触发条件：上升趋势只看认购，下降趋势只看认沽，震荡不做买方。\n")
    for u in p["underlyings"]:
        L.append(f"\n### {u['name']}（趋势：{u['state']}；月份：{'、'.join(u['candidate_months'])}）\n")
        if not u["candidates"]:
            L.append("（无符合条件的合约）\n")
            continue
        L += ["| 到期日 | 剩余天数 | 方向 | 合约代码 | 行权价 | 收盘价 | 每张权利金（元） | IV% | Delta | 持仓量 | 成交量 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in u["candidates"]:
            L.append(f"| {c['maturity']} | {c['days_left']} | {c['cp']} | {c['ts_code']} | {c['strike']} | {c['price']} | "
                     f"{c['cost_per_lot']} | {c['iv']} | {c['delta']} | {c['oi']} | {c['vol']} |")

    text = "\n".join(L) + "\n"
    (REPORTS / f"{d}_opt.md").write_text(text, encoding="utf-8")
    (REPORTS / f"{d}_opt.json").write_text(json.dumps(p, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
