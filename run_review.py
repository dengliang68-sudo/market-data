"""每日复盘数据程序。

用法：
    TS_TOKEN=xxx python run_review.py --session pm     # 晚报：A股/港股收盘后
    TS_TOKEN=xxx python run_review.py --session am     # 早报：美股收盘后

输出：
    reports/YYYYMMDD_<session>.md   给 Claude 写报告用的数据摘要（同时打印到屏幕）
    reports/YYYYMMDD_<session>.json 完整结构化数据
    data/*.parquet                  增量缓存（应提交回仓库，下次只补新日期）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from md import cn, us  # noqa: E402
from md.client import trade_days  # noqa: E402

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SCAN_DAYS = 130  # 板块扫描用的历史长度（交易日）


def fmt_tbl(df: pd.DataFrame, cols: list[str], n: int) -> str:
    if df is None or not len(df):
        return "（无）\n"
    d = df[cols].head(n)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |\n" for r in d.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["am", "pm"], required=True)
    ap.add_argument("--date", help="指定 A股交易日 YYYYMMDD（默认最近交易日）")
    a = ap.parse_args()

    out, md = {}, []
    days = trade_days(SCAN_DAYS, end=a.date)
    d = days[-1]
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    md.append(f"# 复盘数据摘要（{a.session}）\n\n生成时间 {run_at}；A股最近交易日 {d}。所有数字来自 Tushare / FRED，未经人工核对。\n")

    # 美股指数基金面板（早晚都给）
    try:
        p = us.index_fund_panel()
        out["us_index_fund"] = p
        s, n = p["spx"], p["ndx"]
        md.append("## 美股指数基金面板（FRED）\n")
        md.append(fmt_tbl(pd.DataFrame([{**s, "指数": "标普500"}, {**n, "指数": "纳指100"}]),
                          ["指数", "date", "close", "pct", "high_52w", "drawdown_52w", "vs_ma50", "vs_ma200", "ret_12m"], 2))
        md.append(f"\nVIX {p['vix']}（{p['vix_date']}）｜美债10Y {p['us10y']}%（{p['us10y_date']}）｜温度：**{p['temperature']}**\n")
        md.append(f"标普500 已触及的回撤档位：{p['drawdown_tiers_reached'] or '无'}（档位 -5/-10/-15/-20%，每档只用一次，是否已用过请对照建仓进度表）\n")
        md.append(f"过热止盈可自动判断的两项：{p['overheat_partial']}（第三项估值分位需另查）\n")
        md.append("注：FRED 通常晚一天更新。若上表日期不是最近一个美股交易日，请用新闻里的最新收盘价和上面的 52 周高点重新算回撤。等权标普500 暂无数据源。\n")
    except Exception as e:  # noqa: BLE001
        md.append(f"## 美股指数基金面板\n获取失败：{e}\n")

    if a.session == "pm":
        try:
            t = cn.market_temperature(days)
            out["cn_market"] = t
            md.append("\n## A股大盘温度\n")
            md.append(fmt_tbl(pd.DataFrame(t["indices"]), ["name", "close", "pct", "amount_yi"], 10))
            md.append(f"\n两市成交额 {t['amount_yi']} 亿，为前 20 日均值的 {t['amount_vs_20d']} 倍｜上涨 {t['up']} / 下跌 {t['down']}｜"
                      f"涨停 {t['limit_up']}、跌停 {t['limit_down']}、炸板 {t['broken_board']}｜最高连板 {t['max_board']}\n")
            md.append(f"两融余额（亿）：{t['margin_yi']}\n港股通南向净买入（亿元，买入减卖出，可能晚一天）：{t['south_net_yi']}\n")
        except Exception as e:  # noqa: BLE001
            md.append(f"\n## A股大盘温度\n获取失败：{e}\n")

    try:
        hk = cn.hk_snapshot(days)
        out["hk"] = hk
        md.append("\n## 港股指数\n" + fmt_tbl(pd.DataFrame(hk), ["name", "date", "close", "pct"], 5))
    except Exception as e:  # noqa: BLE001
        md.append(f"\n## 港股指数\n获取失败：{e}\n")

    if a.session == "pm":
        try:
            sc = cn.sector_scan(days)
            out["sector_scan"] = sc.to_dict("records")
            cols = ["src", "name", "tier", "pct", "pct5", "pct20", "vol_ratio", "dist_ma20_pct", "high60", "top10pct_3d"]
            md.append("\n## 趋势雷达（A股全部申万一二级行业 + 同花顺行业/概念，按规则自动分级）\n")
            for t in ["🟢可介入", "🟡准备", "⚪观察"]:
                sub = sc[sc.tier == t]
                md.append(f"\n### {t}（{len(sub)} 个，最多列 15 个）\n" + fmt_tbl(sub, cols, 15))
            ebb = sc[sc.tier == "🔴退潮警示"]
            ecols = cols + ["dd_from_40d_high"]
            new = ebb[ebb.ebb_type == "今日新出现"]
            old = ebb[ebb.ebb_type == "持续退潮"].sort_values("dd_from_40d_high")
            md.append(f"\n### 🔴退潮警示·今日新出现（{len(new)} 个）\n" + fmt_tbl(new, ecols, 15))
            md.append(f"\n### 🔴退潮警示·持续退潮（{len(old)} 个，按距 40 日高点回撤排序，列前 15）\n" + fmt_tbl(old, ecols, 15))
            for grp in ["科技", "医药"]:
                sub = sc[sc.group == grp].sort_values("pct", ascending=False)
                md.append(f"\n### {grp}相关板块：今日涨幅前 8 / 后 5\n" + fmt_tbl(sub, cols, 8)
                          + fmt_tbl(sub.tail(5), cols, 5))
        except Exception as e:  # noqa: BLE001
            md.append(f"\n## 趋势雷达\n获取失败：{e}\n")
        try:
            fl = cn.industry_flow(days)
            out["industry_flow"] = fl.to_dict("records")
            md.append("\n## 资金流（东财口径，近 5 日）\n")
            for ct in ["行业", "概念"]:
                sub = fl[fl.content_type == ct]
                md.append(f"\n### {ct}：5 日中至少 4 日净流入，按 5 日合计排序\n"
                          + fmt_tbl(sub[sub.days_in >= 4].sort_values("net5_yi", ascending=False),
                                    ["name", "days_in", "net5_yi", "net_today_yi", "pct_change"], 12))
                md.append(f"\n### {ct}：今日净流出最多\n"
                          + fmt_tbl(sub.sort_values("net_today_yi"), ["name", "net_today_yi", "pct_change", "net5_yi"], 6))
        except Exception as e:  # noqa: BLE001
            md.append(f"\n## 资金流\n获取失败：{e}\n")

    text = "\n".join(md)
    stem = REPORTS / f"{d if a.session == 'pm' else datetime.now().strftime('%Y%m%d')}_{a.session}"
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
