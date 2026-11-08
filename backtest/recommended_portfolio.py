"""推荐规则（含 R1 恐慌反转）的五品种组合回测。

数据全部来自仓库里已缓存的文件，不需要 Tushare：
- data/bt/opt_basic.parquet、data/bt/opt_daily_YYYY.parquet：期权合约和日线
- reports/backtest/xval_20200101_20261106_etf_*.csv：五只 ETF 日线

用法：python backtest/recommended_portfolio.py [--risk 0.03] [--no-r1]
输出：
- reports/backtest/recommended_portfolio.md   汇总、分年收益、分信号统计
- reports/backtest/recommended_trades.csv      组合实际成交的交易
- data/opt_atm_iv_hist.csv                      五品种平值 IV 历史（每日期权报告算 250 日分位用）

组合规则：五品种共用一个账户；单笔风险 = 当前净值 × risk；最多 4 笔、同方向最多 3 笔；
不用回撤分档（只保留回撤 30% 停手的人工规则，回测里没有触发）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import kc50_backtest as K  # noqa: E402
from backtest import opt_backtest as v1  # noqa: E402

B, R = ROOT / "data" / "bt", ROOT / "reports" / "backtest"
CODES = {"300ETF": "510300.SH", "创业板ETF": "159915.SZ", "50ETF": "510050.SH", "500ETF": "510500.SH", "科创50ETF": "588000.SH"}
PRI = ("科创50ETF", "创业板ETF", "500ETF", "300ETF", "50ETF")
D0 = "20200401"


def load():
    meta = pd.read_parquet(B / "opt_basic.parquet")
    od = pd.concat([pd.read_parquet(p) for p in sorted(B.glob("opt_daily_*.parquet"))])
    od["trade_date"] = od.trade_date.astype(str)
    for c in ["open", "high", "low", "close", "settle", "vol", "oi"]:
        od[c] = pd.to_numeric(od[c], errors="coerce").fillna(0.0)
    data, ivrows = {}, []
    for name, code in CODES.items():
        e = pd.read_csv(R / f"xval_20200101_20261106_etf_{name}.csv", dtype={"trade_date": str})
        df = e[["trade_date", "open", "high", "low", "close", "vol"]].copy()
        df["prev_close"] = df.close.shift(1)
        mu = meta[meta.underlying == code]
        o = od[od.ts_code.isin(mu.ts_code)]
        u = K.frame(df)
        book = v1.Book(o, meta)
        days = list(df.trade_date)
        iv = v1.atm_iv_series(book, mu, u, days)
        ivrows += [{"trade_date": d, "code": code, "atm_iv": round(x * 100, 2)} for d, x in iv.items()]
        ivp = iv.rolling(250, min_periods=120).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True).dropna().to_dict()
        data[name] = (df, mu, o, days, ivp, o.trade_date.min(), meta)
    return data, pd.DataFrame(ivrows)


def trades(data, cfg):
    T = []
    for name, (df, mu, o, days, ivp, first, meta) in data.items():
        u = K.frame(df)
        book = v1.Book(o, meta)
        st = max(D0, first)
        i0 = max(0, next(i for i, d in enumerate(days) if d >= st) - 60)
        r = K.run_v2(cfg, days[i0:], u, book, mu, ivp)
        t = pd.DataFrame(r["trades"])
        t["etf"] = name
        T.append(t)
    T = pd.concat(T, ignore_index=True)
    T["open"], T["close"] = T.open.astype(str), T.close.astype(str)
    return T


def sim(T, r=0.03, max_pos=4, max_same=3, cap=100000.0):
    """单品种交易清单（每笔风险 1000 元）按账户规则筛选、按净值比例放大。"""
    T = T.copy()
    T["pri"] = T.etf.map({e: i for i, e in enumerate(PRI)})
    T = T.sort_values(["open", "pri"])
    eq, pending, acc, hist = cap, [], [], []
    for t in T.itertuples():
        pending.sort(key=lambda x: x[0])
        while pending and pending[0][0] <= t.open:
            c = pending.pop(0)
            eq += c[1]
            hist.append((c[0], eq))
        if len(pending) >= max_pos or sum(p[2] == t.dir for p in pending) >= max_same:
            continue
        pnl = t.pnl * r * eq / 1000.0
        pending.append((t.close, pnl, t.dir))
        acc.append({**t._asdict(), "pnl_acct": round(pnl, 1), "equity_at_open": round(eq)})
    for c in sorted(pending, key=lambda x: x[0]):
        eq += c[1]
        hist.append((c[0], eq))
    H = pd.DataFrame(hist, columns=["d", "eq"]).groupby("d").eq.last()
    yrs = (pd.Timestamp(H.index[-1]) - pd.Timestamp(D0)).days / 365
    yeq = pd.concat([pd.Series({"2019": cap}), H.groupby(H.index.str[:4]).last()])
    yret = (yeq.pct_change().dropna() * 100).round(1)
    a = H[H.index < "20230101"]
    seg1 = ((a.iloc[-1] / cap) ** (1 / ((pd.Timestamp("2022-12-31") - pd.Timestamp(D0)).days / 365)) - 1) * 100
    seg2 = ((eq / a.iloc[-1]) ** (1 / ((pd.Timestamp(H.index[-1]) - pd.Timestamp("2023-01-01")).days / 365)) - 1) * 100
    m = {"期末净值": round(eq), "全期年化%": round(((eq / cap) ** (1 / yrs) - 1) * 100, 1),
         "2020-22年化%": round(seg1, 1), "2023-26年化%": round(seg2, 1),
         "最大回撤%(按平仓)": round(((H.cummax() - H) / H.cummax()).max() * 100, 1), "成交笔数": len(acc)}
    return m, yret, pd.DataFrame(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, default=0.03)
    a = ap.parse_args()
    data, ivh = load()
    ivh.to_csv(ROOT / "data" / "opt_atm_iv_hist.csv", index=False)
    rows, yr, A_rec = [], {}, None
    for label, cfg in [("推荐规则（含R1，次日开盘进场）", K.RECOMMENDED),
                       ("推荐规则（含R1，信号当天收盘进场）", {**K.RECOMMENDED, "entry": "close"}),
                       ("不含R1（次日开盘进场）", {k: v for k, v in K.RECOMMENDED.items() if k not in ("r1", "reverse")})]:
        m, y, A = sim(trades(data, cfg), a.risk)
        rows.append({"方案": label, **m})
        yr[label] = y
        if A_rec is None:
            A_rec = A
    S = pd.DataFrame(rows)
    by = A_rec.assign(类别=A_rec.strat.str[:2]).groupby("类别").agg(笔数=("pnl_acct", "size"), 盈亏=("pnl_acct", "sum"),
                                                                 胜率=("pnl_acct", lambda s: (s > 0).mean() * 100)).round(0)
    L = [f"# 推荐规则组合回测（{D0} 至数据末，单笔风险 {a.risk*100:.0f}%）\n",
         "五品种共用 10 万账户，复利；最多 4 笔、同向最多 3 笔；滑点 = 权利金 1%，每张每边 2 元。\n",
         S.to_markdown(index=False), "\n## 分年收益%\n", pd.DataFrame(yr).T.to_markdown(),
         "\n## 推荐规则分信号（账户口径，元）\n", by.to_markdown()]
    (R / "recommended_portfolio.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    A_rec.drop(columns=["Index"], errors="ignore").to_csv(R / "recommended_trades.csv", index=False, encoding="utf-8-sig")
    print("\n".join(L))


if __name__ == "__main__":
    main()
