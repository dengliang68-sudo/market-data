"""基本面 / 资金面 / 情绪指标检验：这些指标能不能把赚钱和亏钱的交易区分开？

输入：reports/backtest/final_trades_ma5.csv（推荐方案在五个品种上的全部交易，单笔风险 1000 元口径）
做法：对每笔交易，取开仓前一个交易日的指标值（避免用到当天收盘后才知道的数据），
      按指标分组，比较多单、空单各自的胜率和平均盈亏；2020–2022 和 2023–2026 两段分别看。
      再把几条事先定好的简单规则（禁止某方向 / 减半仓位）放到组合里，看年化和回撤的变化。

用法：python backtest/factor_check.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from md.client import call  # noqa: E402
from backtest import opt_backtest as v1  # noqa: E402

ETF = {"300ETF": "510300.SH", "50ETF": "510050.SH", "500ETF": "510500.SH", "创业板ETF": "159915.SZ", "科创50ETF": "588000.SH"}
IDX = {"300ETF": "000300.SH", "50ETF": "000016.SH", "500ETF": "000905.SH", "创业板ETF": "399006.SZ", "科创50ETF": "000688.SH"}
START, END = "20170101", datetime.now().strftime("%Y%m%d")


def by_year(api, **kw):
    frames = []
    for y in range(int(START[:4]), int(END[:4]) + 1):
        try:
            df = call(api, start_date=f"{y}0101", end_date=f"{y}1231", **kw)
            if len(df):
                frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  {api} {kw} {y} 失败：{e}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def series(df, col, date_col="trade_date"):
    if df.empty or col not in df:
        return pd.Series(dtype=float)
    s = df[[date_col, col]].dropna().drop_duplicates(date_col)
    s[date_col] = s[date_col].astype(str)
    return pd.to_numeric(s.set_index(date_col)[col], errors="coerce").sort_index()


def asof_prev(s: pd.Series, d: str):
    """开仓日之前（不含当天）最近的值。"""
    if s is None or s.empty:
        return np.nan
    x = s[s.index < d]
    return x.iloc[-1] if len(x) else np.nan


def main():
    T = pd.read_csv(ROOT / "reports/backtest/final_trades_ma5.csv", dtype={"open": str, "close": str})
    print("交易笔数", len(T), flush=True)

    # ---------- 取指标数据 ----------
    fac = {}
    for name in ETF:
        print("取数据", name, flush=True)
        ib = by_year("index_dailybasic", ts_code=IDX[name])
        pe = series(ib, "pe_ttm")
        pe_pct = pe.rolling(750, min_periods=250).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)
        fs = by_year("fund_share", ts_code=ETF[name])
        sh = series(fs, "fd_share")
        sh5 = (sh / sh.shift(5) - 1) * 100
        sh20 = (sh / sh.shift(20) - 1) * 100
        fac[name] = {"pe_pct": pe_pct, "share5": sh5, "share20": sh20}
        print(f"  估值数据 {len(pe)} 天，份额数据 {len(sh)} 天", flush=True)

    mg = by_year("margin")
    if not mg.empty:
        rz = mg.groupby("trade_date")["rzye"].sum()
        rz.index = rz.index.astype(str)
        rz = rz.sort_index()
        margin20 = (rz / rz.shift(20) - 1) * 100
    else:
        margin20 = pd.Series(dtype=float)
    ix = by_year("index_global", ts_code="IXIC")
    nq = series(ix, "close")
    nq_vs60 = (nq / nq.rolling(60).mean() - 1) * 100

    # PCR：用本地期权缓存算每个品种每天的认沽/认购成交量比，取 5 日平均
    meta = pd.read_parquet(ROOT / "data/bt/opt_basic.parquet")
    pcr = {}
    od = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "data/bt").glob("opt_daily_*.parquet"))], ignore_index=True)
    od["trade_date"] = od["trade_date"].astype(str)
    od = od.merge(meta[["ts_code", "underlying", "call_put"]], on="ts_code")
    for name, code in ETF.items():
        x = od[od.underlying == code].groupby(["trade_date", "call_put"])["vol"].sum().unstack()
        if {"C", "P"} <= set(x.columns):
            pcr[name] = (x["P"] / x["C"]).rolling(5).mean()

    # ---------- 给每笔交易打上指标 ----------
    rows = []
    for t in T.itertuples():
        f = fac[t.etf]
        rows.append({
            "估值分位": asof_prev(f["pe_pct"], t.open),
            "份额5日变化%": asof_prev(f["share5"], t.open),
            "份额20日变化%": asof_prev(f["share20"], t.open),
            "两融20日变化%": asof_prev(margin20, t.open),
            "纳指距60日线%": asof_prev(nq_vs60, t.open),
            "PCR5日": asof_prev(pcr.get(t.etf), t.open),
        })
    F = pd.concat([T.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    F["段"] = np.where(F.open < "20230101", "2020-22", "2023-26")
    F["win"] = F.pnl > 0
    F.to_csv(ROOT / "reports/backtest/factor_trades.csv", index=False, encoding="utf-8-sig")

    L = ["# 基本面 / 资金面 / 情绪指标检验\n",
         f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}。样本：推荐方案在五个品种上的 {len(F)} 笔交易（2020-04 至今，单笔风险 1000 元口径）。"
         "每笔交易取开仓前一个交易日的指标值。'平均盈亏'单位是元。\n",
         "## 指标覆盖情况\n",
         v1.md_table(pd.DataFrame({c: [int(F[c].notna().sum())] for c in ["估值分位", "份额5日变化%", "份额20日变化%", "两融20日变化%", "纳指距60日线%", "PCR5日"]}))]

    def bucket_table(col, edges, labels):
        F["_b"] = pd.cut(F[col], edges, labels=labels)
        out = []
        for d in ["看涨", "看跌"]:
            for seg in ["全部", "2020-22", "2023-26"]:
                g = F[(F.dir == d) & ((F["段"] == seg) | (seg == "全部"))]
                for b in labels:
                    x = g[g["_b"] == b]
                    out.append({"方向": d, "时间段": seg, "分组": b, "笔数": len(x),
                                "胜率%": round(x.win.mean() * 100) if len(x) else None,
                                "平均盈亏": round(x.pnl.mean()) if len(x) else None,
                                "合计": round(x.pnl.sum()) if len(x) else 0})
        return pd.DataFrame(out)

    specs = [
        ("估值分位", [-1, 30, 70, 101], ["低（<30%）", "中", "高（>70%）"], "指数市盈率在近 3 年的分位"),
        ("份额5日变化%", [-1e9, -1, 1, 1e9], ["流出（<−1%）", "平稳", "流入（>+1%）"], "ETF 份额 5 日变化，大额流入常见于汇金等资金进场"),
        ("份额20日变化%", [-1e9, -3, 3, 1e9], ["流出（<−3%）", "平稳", "流入（>+3%）"], "ETF 份额 20 日变化"),
        ("两融20日变化%", [-1e9, -2, 2, 1e9], ["下降（<−2%）", "平稳", "上升（>+2%）"], "全市场融资余额 20 日变化"),
        ("纳指距60日线%", [-1e9, -3, 3, 1e9], ["弱（<−3%）", "中", "强（>+3%）"], "纳斯达克指数相对 60 日线的位置"),
        ("PCR5日", [-1, 0.8, 1.1, 1e9], ["低（<0.8）", "中", "高（>1.1）"], "该品种期权认沽/认购成交量比，5 日平均"),
    ]
    for col, edges, labels, desc in specs:
        if F[col].notna().sum() < 20:
            L.append(f"\n## {col}\n数据不足，跳过。\n")
            continue
        L.append(f"\n## {col}（{desc}）\n")
        L.append(v1.md_table(bucket_table(col, edges, labels)))

    # ---------- 事先定好的几条规则，放进组合里看效果 ----------
    def sim(mult, r=0.03, max_pos=4, max_same=3):
        X = F.assign(m=mult).sort_values(["open"])
        eq, peak, pending, hist = 100000.0, 100000.0, [], []
        for t in X.itertuples():
            pending.sort(key=lambda p: p[0])
            keep = []
            for p in pending:
                if p[0] <= t.open:
                    eq += p[1]
                    peak = max(peak, eq)
                    hist.append(eq)
                else:
                    keep.append(p)
            pending = keep
            if t.m <= 0 or len(pending) >= max_pos or sum(p[2] == t.dir for p in pending) >= max_same:
                continue
            pending.append((t.close, t.pnl * r * eq * t.m / 1000.0, t.dir))
        for p in sorted(pending):
            eq += p[1]
            hist.append(eq)
        h = pd.Series(hist)
        yrs = (pd.Timestamp("2026-11-06") - pd.Timestamp("2020-04-01")).days / 365
        return round(((eq / 100000) ** (1 / yrs) - 1) * 100, 1), round(((h.cummax() - h) / h.cummax()).max() * 100, 1)

    base = pd.Series(1.0, index=F.index)
    rules = {
        "基准（不加任何指标）": base,
        "估值高（>70%）时多单减半": base.where(~((F.dir == "看涨") & (F["估值分位"] > 70)), 0.5),
        "估值低（<30%）时不做空": base.where(~((F.dir == "看跌") & (F["估值分位"] < 30)), 0.0),
        "ETF份额5日大增（>+1%）时不做空": base.where(~((F.dir == "看跌") & (F["份额5日变化%"] > 1)), 0.0),
        "两融20日下降（<−2%）时多单减半": base.where(~((F.dir == "看涨") & (F["两融20日变化%"] < -2)), 0.5),
        "纳指弱（距60日线<−3%）时科创50/创业板多单减半": base.where(~((F.dir == "看涨") & F.etf.isin(["科创50ETF", "创业板ETF"]) & (F["纳指距60日线%"] < -3)), 0.5),
    }
    out = []
    for k, m in rules.items():
        c, dd = sim(m)
        c1, _ = sim(m.where(F["段"] == "2020-22", 0))
        c2, _ = sim(m.where(F["段"] == "2023-26", 0))
        out.append({"规则": k, "受影响笔数": int((m != 1).sum()), "组合年化%": c, "最大回撤%": dd, "只算2020-22交易的年化%": c1, "只算2023-26交易的年化%": c2})
    L += ["\n## 把规则放进组合（五个品种、单笔风险 3%、复利、最多 4 笔、同向最多 3 笔）\n",
          "回撤按平仓结果算。'只算某段交易的年化'是把另一段的交易全部去掉后的整体年化，用来看规则在两段是否都有帮助。\n",
          v1.md_table(pd.DataFrame(out))]
    text = "\n".join(L)
    (ROOT / "reports/backtest/factor_check.md").write_text(text, encoding="utf-8")
    print(text[:8000])


if __name__ == "__main__":
    main()
