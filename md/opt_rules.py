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
RISK = 0.02            # 单笔风险占净值（试运行期 2%，2026-11-09 起；转正后改回 0.03）
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


def _chk(items):
    """items: [(条件是否满足, 文字)] → (全部满足?, 不满足的条件文字)"""
    bad = [t for ok, t in items if not ok]
    return (not bad), bad


def explain(r, ivp, cfg) -> list[str]:
    """逐条说明每个规则为什么触发/没触发（与 signals_v2 的判断一一对应）。"""
    f = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
    c, a = r.close, r.atr
    vr = r.vol / r.vol20_prev if r.vol20_prev else float("nan")
    vm = cfg.get("vol_mult", 1.3)
    up = "是" if r.close > r.open else "否"
    L = []
    # ---- 做多 ----
    if r.ext_up:
        L.append(f"- 做多（L1–L4）：**被偏离过滤挡住**——前一天收盘离 20 日线超过 2.5 倍 ATR，或 20 日线高出 60 日线超过 10%，不追涨。")
    else:
        rules = [
            ("L4 放量大阳线", [(r.pct >= 0.03, f"涨幅 {r.pct*100:.1f}%（需 ≥3%）"), (vr >= 1.5, f"量比 {vr:.2f}（需 ≥1.5）"),
                              (c > r.ma20, f"收盘 {f(c)} 需在 20 日线 {f(r.ma20)} 上方"), (c > r.open, f"需收阳（今天收阳：{up}）")]),
            ("L2 回踩60日线起涨", [(c > r.hi20, f"收盘 {f(c)} 需突破前 20 日最高收盘 {f(r.hi20)}"), (bool(r.fresh_up), "需是第一次突破（昨天已突破过）"),
                                 (vr >= vm, f"量比 {vr:.2f}（需 ≥{vm}）"), (c > r.open, f"需收阳（今天收阳：{up}）"),
                                 (bool(r.touch60), "近 20 天需回踩过 60 日线附近"), (bool(r.held60), "近 20 天收盘需没有有效跌破 60 日线"),
                                 (bool(r.ma60_ok), "60 日线需没有明显向下"), (c > r.ma20, f"收盘需在 20 日线 {f(r.ma20)} 上方")]),
            ("L1 趋势突破", [((r.state == "上升") or (c > r.ma60 and bool(r.ma60_ok)), f"需上升趋势，或收盘在 60 日线 {f(r.ma60)} 上方且 60 日线不向下（现在：{r.state}）"),
                            (c > r.hi20, f"收盘 {f(c)} 需突破前 20 日最高收盘 {f(r.hi20)}"), (bool(r.fresh_up), "需是第一次突破"),
                            (vr >= vm, f"量比 {vr:.2f}（需 ≥{vm}）")]),
            ("L3 回调上车", [(r.state == "上升", f"需上升趋势（现在：{r.state}）"), (bool(r.touch_up), "需回踩到 20 日线附近"),
                            (c > r.open and c > r.prev_close, "需收阳且收盘高于昨收")]),
        ]
        for name, items in rules:
            ok, bad = _chk(items)
            L.append(f"- {name}：" + ("✅ 条件满足" if ok else "未触发——" + "；".join(bad)))
    ok, bad = _chk([(ivp is not None and ivp >= 95, f"IV 250 日分位 {ivp if ivp is not None else '无'}（需 ≥95）"),
                    (c < r.ma20, f"收盘 {f(c)} 需在 20 日线 {f(r.ma20)} 下方")])
    L.append(f"- R1 恐慌反转：" + ("✅ 条件满足（当天没有其他做多信号时才用）" if ok else "未触发——" + "；".join(bad)))
    # ---- 做空 ----
    if not r.ma60_dn:
        L.append(f"- 做空（S1/S4）：**不做**——60 日线 {f(r.ma60)} 不低于 20 天前，不是空头环境。")
    elif r.ext_dn:
        L.append("- 做空（S1/S4）：**被偏离过滤挡住**——前一天收盘离 20 日线超过 2.5 倍 ATR 或 20 日线低于 60 日线超过 10%，不追空。")
    else:
        bear = c < r.open and c < r.prev_close
        for name, items in [
            ("S1 跌破20日线和平台", [(c < r.ma20, f"收盘 {f(c)} 需跌破 20 日线 {f(r.ma20)}"), (c < r.low10, f"需跌破前 10 日最低 {f(r.low10)}"),
                                  (bool(r.above20_recent), "前 3 天内需还在 20 日线上方（新跌破）"),
                                  (r.ma60 < c and (c - r.ma60) >= 2 * (r.ma20 - c), "到 60 日线需有 ≥2 倍止损距离的空间")]),
            ("S4 反抽20日线受阻", [(r.state == "下降", f"需下降趋势（现在：{r.state}）"), (bool(r.touch20_dn), "近 3 天需反弹碰到 20 日线"),
                                 (bear, "需收阴且收盘低于昨收"), (c < r.ma20, "收盘需在 20 日线下方")]),
        ]:
            ok, bad = _chk(items)
            L.append(f"- {name}：" + ("✅ 条件满足" if ok else "未触发——" + "；".join(bad)))
        if ivp is not None and ivp >= 70:
            L.append(f"- 另外：IV 分位 {ivp:.0f} ≥ 70，即使出做空信号也不做（期权太贵）。")
    return L


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
        why = explain(r, ivp, K.RECOMMENDED)
        out.append({"why": why, "code": u["code"], "name": u["name"], "date": panel["date"], "close": round(float(r.close), 3),
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
    plans = [(r, s) for r in rows for s in r["signals"] if s.get("action") == "开仓" and s.get("contract")]
    if plans:
        L.append("\n### 开仓计划：止损和首次平一半的点位\n")
        L.append("权利金价格按信号日收盘价估算，实际以你的成交价为准（止损价 = 成交价 ×（1 − 止损比例），+50% 价 = 成交价 × 1.5）。\n")
        for r, s in plans:
            ct = s["contract"]
            px, sp = ct["price"], s["opt_stop_pct"]
            up = s["dir"] == "看涨"
            L += [f"**{r['name']} {s['dir']} {s['strat']}**：买 {ct['name']}（{ct['ts_code']}）",
                  f"- 止损（任一触发就全平）：权利金跌到约 {px*(1-sp):.4f}（−{int(sp*100)}%，盘中条件单）；"
                  f"或标的收盘{'跌破' if up else '站上'} {s['u_stop']}（第二天开盘平）。",
                  f"- 首次平一半（两个条件谁先到用谁，只做一次）：① 权利金涨到约 {px*1.5:.4f}（+50%，盘中条件单）；"
                  f"② 标的收盘{'到达' if up else '跌到'} {s['u_target']}（2:1 目标，第二天开盘平）。平完一半后，剩下一半的止损提到成交价（保本），"
                  f"之后{'标的收盘跌破 5 日线' if up else '一半站上 5 日线、一半站上 20 日线'}再平。"
                  + (f"**按 {DEMO_EQUITY//10000} 万 × {RISK*100:.0f}% 只能买 1 张：到这个点位时不卖，只把止损提到成交价（保本）**，之后按上面的跟踪规则一次性平掉。"
                     if s.get("lots_demo") == 1 else ""),
                  "- 为什么在这里平一半：回测 226 笔里，108 笔先到 +50%，这些交易 88% 最终赚钱；没到平一半点位的 116 笔只有 4% 赚钱。"
                  "先兑现一半锁住利润、剩余保本，这笔交易基本就不会亏了，剩下一半再去博趋势。"
                  "2:1 目标是\"赚到两倍止损距离\"的位置，实际很少先于 +50% 到达（226 笔里只有 2 笔），主要作为标的走势的参考线。",
                  ""]
    L.append("\n### 每个品种：为什么买 / 为什么不买\n")
    for r in rows:
        acts = [s for s in r["signals"] if s.get("action") == "开仓"]
        if acts:
            head = f"**结论：开仓 {acts[0]['dir']} {acts[0]['strat']}**"
        elif r["signals"]:
            head = f"**结论：有信号但不做——{r['signals'][0]['action']}**"
        else:
            head = "**结论：今天没有信号，不开仓**"
        L.append(f"**{r['name']}**（收盘 {r['close']}，趋势 {r['state']}，IV 250日分位 {r['iv_pctile_250d']}）　{head}\n")
        L += r["why"] + [""]
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

