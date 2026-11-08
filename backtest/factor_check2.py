"""多因子检验（第二轮，2026-11-08）：参考常见多因子模型，把能用现有数据算的因子放到推荐规则（含 R1）上检验。

数据：仓库缓存的 ETF 日线和期权日线（不需要 Tushare）。
因子在信号当天收盘计算（回测按当天收盘进场，收盘时已知）。

事先定好的规则（跑之前写定，不按结果调整阈值）：
  A  期权贵：IV/RV20 ≥ 1.5 时不做趋势信号（L、S），R1 不受影响
  B  期权便宜：IV/RV20 ≤ 1.0 时趋势信号 1.5 倍仓位
  C  期限倒挂（近月 IV > 次月 IV × 1.05）：不做 L 信号；R1 1.5 倍
  D  看跌偏度高（偏度 250 日分位 ≥ 80）：不做空
  E  波动率状态：RV20 的 250 日分位 ≥ 90 时趋势信号减半，R1 不受影响
  F  横截面动量排序：仓位不够时，多单优先 20 日涨幅最高的品种、空单优先最弱的品种（替代固定顺序）
  G  横截面强弱过滤：多单（不含 R1）只做 20 日涨幅排五个品种前 3 的；空单只做后 3 的
  H  时序动量过滤：多单（不含 R1）要求 60 日涨幅 > 0；空单要求 60 日涨幅 < 0

一条规则只有在全期、2020–22、2023–26 三个口径都改善年化，才算"有效"。
用法：python backtest/factor_check2.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import kc50_backtest as K  # noqa: E402
from backtest import opt_backtest as v1  # noqa: E402
from backtest import recommended_portfolio as RP  # noqa: E402
from md.opt import implied_vol  # noqa: E402

CACHE = Path("/tmp/fc2.pkl")


def dt(s):
    return datetime.strptime(s, "%Y%m%d")


def opt_factors(df, mu, o, meta):
    """每天：近月/次月平值 IV、近月 95% 认沽和 105% 认购的 IV 差（偏度）。"""
    book = v1.Book(o, meta)
    m0 = mu[~mu.adjusted]
    closes = df.set_index("trade_date").close
    rows = []
    for d, day in book.by_day.items():
        if d not in closes.index:
            continue
        S = closes[d]
        m = m0[(m0.maturity_date > d) & (m0.ts_code.isin(day.ts_code))]
        mats = sorted(x for x in m.maturity_date.unique() if (dt(x) - dt(d)).days >= 15)
        if len(mats) < 1:
            continue
        out = {"trade_date": d}
        for tag, mt in zip(["iv1", "iv2"], mats[:2]):
            ch = m[m.maturity_date == mt]
            T = (dt(mt) - dt(d)).days / 365

            def iv_at(K_target, cp):
                c = ch[ch.call_put == cp]
                if c.empty:
                    return np.nan
                r = c.iloc[(c.exercise_price - K_target).abs().argsort()[:1]]
                p = book.price(r.ts_code.iloc[0], d)
                return implied_vol(p, S, float(r.exercise_price.iloc[0]), T, v1.R, cp) if p else np.nan
            out[tag] = np.nanmean([iv_at(S, "C"), iv_at(S, "P")])
            if tag == "iv1":
                out["skew"] = iv_at(S * 0.95, "P") - iv_at(S * 1.05, "C")
        rows.append(out)
    return pd.DataFrame(rows).set_index("trade_date").sort_index()


def build():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    data, _ = RP.load()
    T = RP.trades(data, K.RECOMMENDED)
    F = {}
    for name, (df, mu, o, days, ivp, first, meta) in data.items():
        c = df.set_index("trade_date").close.astype(float)
        f = opt_factors(df, mu, o, meta).reindex(c.index)
        lr = np.log(c).diff()
        f["rv20"] = lr.rolling(20).std() * np.sqrt(252)
        f["iv_rv"] = f.iv1 / f.rv20
        f["term"] = f.iv1 / f.iv2
        pr = lambda s: s.rolling(250, min_periods=120).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)  # noqa: E731
        f["skew_p"] = pr(f["skew"])
        f["rv_p"] = pr(f.rv20)
        f["mom20"] = c / c.shift(20) - 1
        f["mom60"] = c / c.shift(60) - 1
        F[name] = f
    mom = pd.DataFrame({n: f.mom20 for n, f in F.items()})
    rank = mom.rank(axis=1, ascending=False)   # 1 = 20 日最强
    for n in F:
        F[n]["xs_rank"] = rank[n]
    pickle.dump((T, F), open(CACHE, "wb"))
    return T, F


def attach(T, F):
    T = T.copy()
    cols = ["iv_rv", "term", "skew_p", "rv_p", "mom20", "mom60", "xs_rank"]
    for c in cols:
        T[c] = [F[e].at[d, c] if d in F[e].index else np.nan for e, d in zip(T.etf, T.open)]
    T["long"] = T.dir == "看涨"
    T["r1"] = T.strat.str.startswith("R1")
    T["trend"] = ~T.r1
    return T


def sim(T, mult=None):
    T = T.copy()
    if mult is not None:
        T["pnl"] = T.pnl * mult
        T = T[mult > 0]
    m, y, _ = RP.sim(T)
    return m, y


def _core(T, r, max_pos, max_same, cap):
    # 与 RP.sim 相同，只是保持传入的顺序
    saved = RP.PRI
    try:
        order = {i: k for k, i in enumerate(T.index)}
        T = T.assign(pri_override=[order[i] for i in T.index])
        RP_sorted = T.sort_values(["open", "pri_override"])
        # 复用 RP.sim：给每笔一个唯一 etf 优先级
        RP.PRI = tuple(f"__{i}" for i in range(len(RP_sorted)))
        X = RP_sorted.assign(etf_real=RP_sorted.etf, etf=[f"__{i}" for i in range(len(RP_sorted))])
        m, y, A = RP.sim(X, r, max_pos, max_same, cap)
    finally:
        RP.PRI = saved
    return m, y, A


def main():
    T, F = build()
    T = attach(T, F).reset_index(drop=True)
    one = pd.Series(1.0, index=T.index)
    rules = {
        "基准（推荐规则含R1）": None,
        "A 期权贵(IV/RV≥1.5)不做趋势": one.where(~(T.trend & (T.iv_rv >= 1.5)), 0),
        "B 期权便宜(IV/RV≤1.0)趋势1.5倍": one.where(~(T.trend & (T.iv_rv <= 1.0)), 1.5),
        "C 期限倒挂:不做L, R1 1.5倍": one.where(~(T.long & T.trend & (T.term > 1.05)), 0).where(~(T.r1 & (T.term > 1.05)), 1.5),
        "D 看跌偏度高(≥80分位)不做空": one.where(~(~T.long & (T.skew_p >= 80)), 0),
        "E 实现波动高(≥90分位)趋势减半": one.where(~(T.trend & (T.rv_p >= 90)), 0.5),
        "G 横截面强弱过滤": one.where(~((T.long & T.trend & (T.xs_rank > 3)) | (~T.long & (T.xs_rank < 3))), 0),
        "H 时序动量过滤(60日)": one.where(~((T.long & T.trend & (T.mom60 <= 0)) | (~T.long & (T.mom60 >= 0))), 0),
    }
    rows = []
    for k, m in rules.items():
        res, y = sim(T, m)
        rows.append({"规则": k, **res, "受影响笔数": 0 if m is None else int((m != 1).sum())})
    # F：横截面动量排序
    score = np.where(T.long, T.xs_rank, -T.xs_rank).astype(float)
    m, y, _ = _core(T.assign(_s=score).sort_values(["open", "_s"]), 0.03, 4, 3, 100000.0)
    rows.append({"规则": "F 横截面动量排序", **m, "受影响笔数": None})
    R = pd.DataFrame(rows)
    b = R.iloc[0]
    R["有效"] = [("" if i == 0 else ("是" if (r["全期年化%"] > b["全期年化%"] and r["2020-22年化%"] > b["2020-22年化%"]
                                            and r["2023-26年化%"] > b["2023-26年化%"]) else "否")) for i, r in R.iterrows()]
    # 交易层面：按因子三分位看平均盈亏（单笔风险 1000 元口径）
    L = []
    for c in ["iv_rv", "term", "skew_p", "rv_p", "mom20", "mom60"]:
        for side, g in [("多单(趋势)", T[T.long & T.trend]), ("R1", T[T.r1]), ("空单", T[~T.long])]:
            g = g.dropna(subset=[c])
            if len(g) < 15:
                continue
            q = pd.qcut(g[c], 3, labels=["低", "中", "高"], duplicates="drop")
            s = g.groupby(q, observed=True).pnl.agg(["size", "mean"]).round(0)
            L.append({"因子": c, "交易": side, "每组笔数": int(s["size"].min()),
                      **{k: int(s.at[k, "mean"]) if k in s.index else None for k in ["低", "中", "高"]}})
    out = ["# 多因子检验（第二轮）\n", R.to_markdown(index=False), "\n## 交易层面：因子三分位的平均盈亏（元/笔，风险1000口径）\n",
           pd.DataFrame(L).to_markdown(index=False)]
    (ROOT / "reports" / "backtest" / "factor_check2.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
