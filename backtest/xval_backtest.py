"""跨品种验证：把在科创50上调出来的规则，原样用到其他 ETF 期权上。

规则一条不改，只换品种。2020–2022 年的数据在调规则时没有用过，是真正的"样本外"检验。

用法：python backtest/xval_backtest.py --start 20200101 [--end 20261106]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import opt_data  # noqa: E402
from backtest import opt_backtest as v1  # noqa: E402
from backtest import kc50_backtest as K  # noqa: E402

ETFS = {"300ETF": "510300.SH", "创业板ETF": "159915.SZ", "50ETF": "510050.SH", "500ETF": "510500.SH", "科创50ETF": "588000.SH"}
CAP = 100000


def variants():
    rec = dict(trail="ma5", pullback=True, iv_skip=True, iv_long=False, iv_short=True, vol_mult=1.2, s2="off")
    H = {**rec, "s3": True, "s4": True, "bear_only": True, "trail_short": "ma5", "short_stop_pct": 0.6, "tstop_short": 5}
    M = {**H, "l4": True, "l1_relax": True, "trail": "ma20", "trail_short": "mix"}
    return [
        ("A", "A 旧规则（v1）", None),
        ("H", "H 加强做空", H),
        ("M", "M 推荐（多单20日线跟踪）", M),
        ("O", "O 多单一半5日线一半20日线", {**M, "trail": "mix"}),
        ("N", "N 多单5日线跟踪", {**M, "trail": "ma5"}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20200101")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    end = a.end or datetime.now().strftime("%Y%m%d")
    warm = str(int(a.start[:4]) - 1) + a.start[4:]
    all_days = opt_data.trade_days(warm, end)
    end = all_days[-1]
    meta = opt_data.contracts()
    od_all = opt_data.opt_daily(all_days)
    out_dir = ROOT / "reports" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"xval_{a.start}_{end}"

    rows, yearly, trades_all = [], [], []
    for name, code in ETFS.items():
        mu = meta[meta.underlying == code]
        od = od_all[od_all.ts_code.isin(mu.ts_code)]
        if od.empty:
            continue
        first_opt = od.trade_date.min()
        df = opt_data.etf_daily(code, all_days[0], end)
        df["prev_close"] = df["close"].shift(1)
        u = K.frame(df)
        u.loc[u.index >= a.start, ["open", "high", "low", "close", "vol", "ma5", "ma20", "ma60", "state", "atr"]].to_csv(out_dir / f"{tag}_etf_{name}.csv")
        book = v1.Book(od, meta)
        iv = v1.atm_iv_series(book, mu, u, all_days)
        ivp = iv.rolling(250, min_periods=120).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True).dropna().to_dict()
        st_day = max(a.start, first_opt)
        start_i = next(i for i, d in enumerate(all_days) if d >= st_day)
        sim = all_days[max(0, start_i - 60):]
        print(f"{name}: 期权数据从 {first_opt}，回测从 {st_day}", flush=True)
        for key, vname, cfg in variants():
            if cfg is None:
                c0 = {"name": vname, "unders": [code], "use_B": False, "iv_mode": "skip", "risk_tiers": True, "capital": CAP}
                old_under = v1.UNDER.get(code)
                v1.UNDER.setdefault(code, name)
                r = v1.run(c0, sim, {code: v1.underlying_frame(df)}, book, meta, {code: ivp})
                if old_under is None:
                    v1.UNDER.pop(code, None)
            else:
                r = K.run_v2(cfg, sim, u, book, mu, ivp)
            t = pd.DataFrame(r["trades"])
            st = v1.stats(r["trades"], r["equity"], CAP)
            if not t.empty:
                t["year"] = t["open"].astype(str).str[:4]
                t["etf"], t["variant"] = name, key
                trades_all.append(t)
                ins = t[t.year >= "2023"]
                outs = t[t.year < "2023"]
                lg, sh = t[t.dir == "看涨"], t[t.dir == "看跌"]
                for y, g in t.groupby("year"):
                    yearly.append({"品种": name, "方案": key, "年份": y, "盈亏": round(g.pnl.sum())})
            else:
                ins = outs = lg = sh = t
            rows.append({"品种": name, "方案": vname, "起始": st_day, "笔数": st.get("笔数", 0), "胜率%": st.get("胜率%"),
                         "盈亏比": st.get("盈亏比"), "总盈亏": st.get("总盈亏", 0), "最大回撤%": st.get("最大回撤%"),
                         "2020-22盈亏": round(outs.pnl.sum()) if len(outs) else 0,
                         "2023-26盈亏": round(ins.pnl.sum()) if len(ins) else 0,
                         "多单": f"{len(lg)}笔 {round(lg.pnl.sum()) if len(lg) else 0}",
                         "空单": f"{len(sh)}笔 {round(sh.pnl.sum()) if len(sh) else 0}",
                         "最大一笔": st.get("单笔最大盈利"), "最大亏损": st.get("单笔最大亏损")})
            print("  ", vname, st.get("总盈亏"), flush=True)

    R = pd.DataFrame(rows)
    Y = pd.DataFrame(yearly)
    L = [f"# 跨品种验证：科创50规则原样用到其他ETF（{a.start}–{end}）\n",
         f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}。10 万本金、单笔风险 1000 元、一次一笔，每个品种单独回测。"
         "规则和参数与《科创50ETF期权回测》完全相同，一条未改。2020–2022 年是调规则时没有用过的数据。\n",
         "## 汇总\n", v1.md_table(R)]
    if not Y.empty:
        P = Y.pivot_table(index=["品种", "方案"], columns="年份", values="盈亏", aggfunc="sum").fillna(0).astype(int).reset_index()
        L += ["\n## 每年盈亏\n", v1.md_table(P)]
    if trades_all:
        T = pd.concat(trades_all)
        T.to_csv(out_dir / f"{tag}_trades.csv", index=False, encoding="utf-8-sig")
        g = T.groupby(["variant", "strat"]).agg(笔数=("pnl", "size"), 胜率=("pnl", lambda s: round((s > 0).mean() * 100)),
                                                 总盈亏=("pnl", lambda s: round(s.sum()))).reset_index()
        L += ["\n## 各信号在所有品种上的合计\n", v1.md_table(g)]
    R.to_csv(out_dir / f"{tag}_summary.csv", index=False, encoding="utf-8-sig")
    Y.to_csv(out_dir / f"{tag}_yearly.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)[:6000])


if __name__ == "__main__":
    main()
