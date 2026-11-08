"""按推荐规则（研究总结 2026-11-08 版，含 R1 恐慌反转）给五个 ETF 出每日信号。

规则本身直接调用回测代码 backtest/kc50_backtest.py 的 signals_v2 和 RECOMMENDED，
保证每日信号和回测用的是同一套判断。

IV 分位口径与回测一致：参考月份（剩余 ≥15 天的最近月份）平值认购、认沽 IV 的平均，
在近 250 个交易日里的分位（至少 120 天数据）。历史保存在 data/opt_atm_iv_hist.csv，
由 backtest/recommended_portfolio.py 初始化，每天运行 run_options.py 时追加。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest import kc50_backtest as K
from .client import DATA, call

IV_HIST = DATA / "opt_atm_iv_hist.csv"
RISK = 0.03            # 单笔风险占净值
DEMO_EQUITY = 100000   # 报告里举例用的净值


def update_iv_hist(panel: dict) -> pd.DataFrame:
    h = pd.read_csv(IV_HIST, dtype={"trade_date": str}) if IV_HIST.exists() else \
        pd.DataFrame(columns=["trade_date", "code", "atm_iv"])
    new = []
    for u in panel["underlyings"]:
        ser = dict(u.get("iv_series_60d") or {})
        now = u["atm_iv"].get(u["iv_ref_month"])
        if now is not None:
            ser[panel["date"]] = now
        new += [{"trade_date": d, "code": u["code"], "atm_iv": v} for d, v in ser.items()]
    if new:
        n = pd.DataFrame(new)
        key = set(zip(n.trade_date, n.code))
        h = h[[(d, c) not in key for d, c in zip(h.trade_date, h.code)]]
        h = pd.concat([h, n], ignore_index=True).sort_values(["code", "trade_date"])
        h.to_csv(IV_HIST, index=False)
    return h


def iv_pctile(h: pd.DataFrame, code: str, d: str) -> float | None:
    s = h[(h.code == code) & (h.trade_date <= d)].sort_values("trade_date").atm_iv.astype(float).tail(250)
    if len(s) < 120 or s.iloc[-1] != s.iloc[-1]:
        return None
    return round(float((s <= s.iloc[-1]).mean() * 100), 0)


def _pick(cands: list[dict], cp: str) -> dict | None:
    """和回测 pick_contract 一致：优先剩余 20–45 天的月份，没有就取更远的下一个月份；Delta 最接近 0.5。"""
    c = [x for x in cands if x["cp"] == cp and x["vol"] > 0 and x["oi"] > 0]
    inwin = sorted({x["maturity"] for x in c if 20 <= x["days_left"] <= 45})
    later = sorted({x["maturity"] for x in c if x["days_left"] > 45})
    mt = inwin[0] if inwin else (later[0] if later else None)
    if mt is None:
        return None
    return min((x for x in c if x["maturity"] == mt), key=lambda x: abs(abs(x["delta"]) - 0.5))


def evaluate(panel: dict, days: list[str]) -> list[dict]:
    h = update_iv_hist(panel)
    out = []
    for u in panel["underlyings"]:
        df = call("fund_daily", ts_code=u["code"], start_date=days[0], end_date=days[-1]).sort_values("trade_date")
        df = df.rename(columns={"pre_close": "prev_close"})[["trade_date", "open", "high", "low", "close", "vol", "prev_close"]]
        for c in ["open", "high", "low", "close", "vol", "prev_close"]:
            df[c] = df[c].astype(float)
        f = K.frame(df.reset_index(drop=True))
        r = f.iloc[-1].copy()
        ivp = iv_pctile(h, u["code"], panel["date"])
        r["ivp"] = np.nan if ivp is None else ivp
        sigs = K.signals_v2(r, K.RECOMMENDED["pullback"], K.RECOMMENDED)
        res = []
        for s in sigs:
            item = {"strat": s["strat"], "dir": "看涨" if s["dir"] > 0 else "看跌",
                    "u_stop": round(float(s["stop"]), 3), "u_target": round(float(s["target"]), 3)}
            if s["dir"] < 0 and ivp is not None and ivp >= 70:
                item["action"] = f"放弃：IV 分位 {ivp:.0f} ≥ 70，做空信号不做"
            else:
                stop_pct = 0.4 if s["strat"].startswith(("L1", "L2", "S2")) else (0.6 if s["dir"] < 0 else 0.5)
                ct = _pick(u["candidates"], "认购" if s["dir"] > 0 else "认沽")
                item["opt_stop_pct"] = stop_pct
                if ct:
                    per = ct["price"] * 10000 * stop_pct
                    item.update(contract=ct, loss_per_lot=round(per), lots_demo=math.floor(DEMO_EQUITY * RISK / per),
                                action="开仓")
                else:
                    item["action"] = "没有符合条件的合约，按规则不开"
            res.append(item)
            if item.get("action") == "开仓":
                break   # 一个品种一次只开一笔，取优先级最高的
        watch = []
        if ivp is not None and ivp >= 90 and r.close < r.ma20 and not any(x["strat"].startswith("R1") for x in res):
            watch.append(f"R1 接近：IV 分位 {ivp:.0f}，≥95 且收盘仍在 20 日线下方即触发")
        out.append({"code": u["code"], "name": u["name"], "date": panel["date"], "close": round(float(r.close), 3),
                    "state": r.state, "iv_pctile_250d": ivp,
                    "ma5": round(float(r.ma5), 3), "ma20": round(float(r.ma20), 3), "ma60": round(float(r.ma60), 3),
                    "hi20_close": round(float(r.hi20), 3), "low10": round(float(r.low10), 3),
                    "atr": round(float(r.atr), 3), "short_allowed": bool(r.ma60_dn),
                    "ext_up": bool(r.ext_up), "ext_dn": bool(r.ext_dn), "signals": res, "watch": watch})
    return out


def to_markdown(rows: list[dict]) -> list[str]:
    L = ["\n## 规则信号（推荐规则，含 R1 恐慌反转）\n",
         "按研究总结《ETF期权买方策略研究总结》2026-11-08 版规则，由回测同一段代码自动判断，收盘价口径。"
         "品种括号里是标的 ETF 的 6 位代码（只有这 5 只 ETF 的期权在用：510050、510300、510500、588000、159915；"
         "同指数的 159919、159922、588080 等不是本系统的品种）；合约括号里是期权合约的 8 位交易代码，下单时按这个代码找。"
         "**操作：第二天开盘后尽快（9:30–9:45）限价买入表中合约；如果开盘时标的已越过标的止损位，放弃这笔。**"
         "反手（R1 遇到同品种空单）也在第二天开盘先平空、再买认购。"
         f"张数举例按净值 {DEMO_EQUITY//10000} 万、单笔风险 {RISK*100:.0f}% 计算：张数 = 净值 × {RISK*100:.0f}% ÷（权利金 × 10000 × 权利金止损比例），实际按自己的净值重算。\n",
         "| 品种 | 收盘 | 趋势 | IV 250日分位 | 今日信号 | 处理 | 标的止损 / 目标 | 合约 | 每张最大亏损 | 10万举例张数 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["signals"]:
            L.append(f"| {r['name']} | {r['close']} | {r['state']} | {r['iv_pctile_250d']} | 无 | — | — | — | — | — |")
        for s in r["signals"]:
            ct = s.get("contract")
            cts = f"{ct['name']}（{ct['ts_code']}，{ct['price']}，Delta {ct['delta']}，剩 {ct['days_left']} 天）" if ct else "—"
            L.append(f"| {r['name']} | {r['close']} | {r['state']} | {r['iv_pctile_250d']} | {s['dir']} {s['strat']} | {s['action']} | "
                     f"{s['u_stop']} / {s['u_target']} | {cts} | {s.get('loss_per_lot', '—')} | {s.get('lots_demo', '—')} |")
    L += ["\n### 关键价位与持仓跟踪\n",
          "所有按收盘价判断的出场（多单跟踪阶段收盘跌破 5 日线、空单站上 5 日/20 日线、标的收盘越过止损位、第 5 天时间止损、标的收盘到 2:1 目标平一半）都在第二天开盘执行。"
          "权利金止损和权利金 +50% 平一半是盘中价格，建议在券商设条件单。",
          "\n| 品种 | 5日线 | 20日线 | 60日线 | 前20日最高收盘（突破位） | 前10日最低（平台） | ATR | 允许做空（60日线向下） | 偏离过滤（多/空） | 观察 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['ma5']} | {r['ma20']} | {r['ma60']} | {r['hi20_close']} | {r['low10']} | {r['atr']} | "
                 f"{'是' if r['short_allowed'] else '否'} | {'是' if r['ext_up'] else '否'} / {'是' if r['ext_dn'] else '否'} | "
                 f"{'；'.join(r['watch']) or '—'} |")
    return L


SIGNAL_LOG = DATA.parent / "reports" / "opt_signal_log.csv"


def append_signal_log(rows: list[dict]) -> None:
    """把每天的规则信号追加到 reports/opt_signal_log.csv（同一天重跑会覆盖当天的记录），供试运行复盘对比实际成交。"""
    recs = []
    for r in rows:
        for s in r["signals"]:
            ct = s.get("contract") or {}
            recs.append({"signal_date": r["date"], "etf": r["name"], "code": r["code"], "dir": s["dir"],
                         "strat": s["strat"], "action": s["action"], "u_close": r["close"],
                         "u_stop": s["u_stop"], "u_target": s["u_target"], "iv_pctile_250d": r["iv_pctile_250d"],
                         "contract": ct.get("ts_code"), "contract_name": ct.get("name"), "opt_close": ct.get("price"),
                         "opt_stop_pct": s.get("opt_stop_pct"), "loss_per_lot": s.get("loss_per_lot"),
                         "lots_demo_10w": s.get("lots_demo")})
    old = pd.read_csv(SIGNAL_LOG, dtype={"signal_date": str}) if SIGNAL_LOG.exists() else pd.DataFrame()
    day = rows[0]["date"] if rows else None
    if len(old) and day:
        old = old[old.signal_date != day]
    new = pd.concat([old, pd.DataFrame(recs)], ignore_index=True)
    if len(new):
        new.to_csv(SIGNAL_LOG, index=False, encoding="utf-8-sig")

