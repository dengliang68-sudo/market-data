"""科创50ETF 期权买方策略回测（v2 规则，按 2026-11-07 用户反馈修改）。

和 v1（opt_backtest.py）的主要区别：
- 新增「回踩60日线后放量突破」做多信号，即使趋势状态被判为震荡也做
- 偏离过滤：收盘价离 20 日线太远（> 2.5 倍 ATR）或 20 日线离 60 日线太远（> 10%）时不追
- 新增「跌破20日线 + 跌破平台」做空：止损位 = 20 日线，第一目标 = 60 日线，盈亏比 ≥ 2 才做；做空信号不要求放量
- 出场：+50% 或到达 2:1 目标位（先到的那个）平一半，只执行一次；剩下的按跟踪止盈走，不再"到目标全平"
- 剩余不到 7 天时，若已进入跟踪阶段则移仓到下一个月继续持有

用法：python backtest/kc50_backtest.py --start 20230605 [--end 20261106]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import opt_data  # noqa: E402
from backtest import opt_backtest as v1  # noqa: E402

CODE = "588000.SH"
NAME = "科创50ETF"
CAP = 100000
FEE, slip, dt = v1.FEE, v1.slip, v1.dt
EXT_ATR, EXT_GAP = 2.5, 0.10


# ---------------- 指标 ----------------
def frame(df: pd.DataFrame) -> pd.DataFrame:
    u = v1.underlying_frame(df)
    c, h, l = u["close"], u["high"], u["low"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    u["atr"] = tr.rolling(20).mean()
    u["ma5"] = c.rolling(5).mean()
    u["ma10"] = c.rolling(10).mean()
    u["low10"] = l.rolling(10).min().shift(1)       # 平台低点：前 10 日最低价
    u["high10"] = h.rolling(10).max().shift(1)
    u["touch60"] = (l <= u["ma60"] * 1.02).astype(float).rolling(20).max() > 0      # 近20天回踩到60日线附近
    u["held60"] = (c / u["ma60"]).rolling(20).min() >= 0.96                          # 近20天收盘没有有效跌破60日线
    u["ma60_ok"] = u["ma60"] >= u["ma60"].shift(10) * 0.98                           # 60日线没有明显向下
    u["fresh_up"] = c.shift(1) <= u["hi20"].shift(1)                                  # 昨天还没突破：今天是第一次突破
    u["fresh_dn"] = c.shift(1) >= u["lo20"].shift(1)
    u["ma60_up"] = u["ma60"] > u["ma60"].shift(20)
    u["pct"] = c / c.shift(1) - 1
    u["ma60_dn"] = u["ma60"] < u["ma60"].shift(20)                                    # 60日线比20天前低：空头环境
    u["touch60_dn"] = (h >= u["ma60"] * 0.99).astype(float).rolling(3).max() > 0      # 近3天反弹碰到60日线
    u["touch20_dn"] = (h >= u["ma20"] * 0.99).astype(float).rolling(3).max() > 0
    u["high5"] = h.rolling(5).max()
    u["above20_recent"] = (c.shift(1) >= u["ma20"].shift(1)).astype(float).rolling(3).max() > 0
    u["below20_recent"] = (c.shift(1) <= u["ma20"].shift(1)).astype(float).rolling(3).max() > 0
    # 偏离过滤按"突破前一天"的位置算：当天的大阳线/大阴线本身不算偏离
    gap = u["ma20"] / u["ma60"] - 1
    u["ext_up"] = (((c - u["ma20"]) > EXT_ATR * u["atr"]) | (gap > EXT_GAP)).shift(1).fillna(False).astype(bool)
    u["ext_dn"] = (((u["ma20"] - c) > EXT_ATR * u["atr"]) | (gap < -EXT_GAP)).shift(1).fillna(False).astype(bool)
    return u


def signals_v2(u: pd.Series, use_pullback: bool, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or {}
    if any(pd.isna(u[k]) for k in ["ma60", "hi20", "vol20_prev", "atr", "low10"]):
        return []
    c, a = u.close, u.atr
    vol_ok = u.vol >= cfg.get("vol_mult", 1.3) * u.vol20_prev
    out = []
    # ---- 做多 ----
    if not u.ext_up:
        vr = u.vol / u.vol20_prev
        if cfg.get("l4") and u.pct >= 0.03 and vr >= 1.5 and c > u.ma20 and c > u.open:
            stop = min(u.low, c - a)
            out.append(dict(dir=1, strat="L4放量大阳线", stop=stop, target=c + 2 * (c - stop)))
        elif c > u.hi20 and u.fresh_up and vol_ok and c > u.open and u.touch60 and u.held60 and u.ma60_ok and c > u.ma20:
            stop = min(u.ma20, c - a)
            out.append(dict(dir=1, strat="L2回踩60日线起涨", stop=stop, target=c + 2 * (c - stop)))
        elif (u.state == "上升" or (cfg.get("l1_relax") and c > u.ma60 and u.ma60_ok)) and c > u.hi20 and u.fresh_up and vol_ok:
            stop = min(u.hi20, c - a)
            out.append(dict(dir=1, strat="L1趋势突破", stop=stop, target=c + 2 * (c - stop)))
        elif use_pullback and cfg.get("l3_new") and u.ma60_up and u.touch_up and c > u.open and c > u.prev_close and c > u.ma20:
            stop = min(u.low5, c - a)
            out.append(dict(dir=1, strat="L3回踩20日线", stop=stop, target=c + 2 * (c - stop)))
        elif use_pullback and not cfg.get("l3_new") and u.state == "上升" and u.touch_up and c > u.open and c > u.prev_close:
            stop, target = min(u.low5, c - a), u.max20
            if target - c >= 2 * (c - stop) > 0:
                out.append(dict(dir=1, strat="L3回调上车", stop=stop, target=target))
    # ---- 做空（不要求放量）----
    if not u.ext_dn and (not cfg.get("bear_only") or u.ma60_dn):
        bear_bar = c < u.open and c < u.prev_close
        if c < u.ma20 and c < u.low10 and u.above20_recent and u.ma60 < c and (c - u.ma60) >= 2 * (u.ma20 - c):
            out.append(dict(dir=-1, strat="S1跌破20日线和平台", stop=u.ma20, target=u.ma60))
        elif cfg.get("s3", False) and u.ma20 < u.ma60 and u.touch60_dn and bear_bar and c < u.ma60:
            stop = max(u.high5, u.ma60, c + a)
            out.append(dict(dir=-1, strat="S3反抽60日线受阻", stop=stop, target=c - 2 * (stop - c)))
        elif cfg.get("s4", False) and u.state == "下降" and u.touch20_dn and bear_bar and c < u.ma20:
            stop = max(u.high5, u.ma20, c + a)
            out.append(dict(dir=-1, strat="S4反抽20日线受阻", stop=stop, target=c - 2 * (stop - c)))
        elif cfg.get("s2", "any") != "off" and c < u.lo20 and u.fresh_dn and (c < u.ma20 or c < u.ma60) \
                and (u.state == "下降" if cfg.get("s2") == "down" else u.state != "上升"):
            stop = max(u.lo20, c + a)
            out.append(dict(dir=-1, strat="S2跌破前低", stop=stop, target=c - 2 * (stop - c)))
    # ---- R1 恐慌反转做多（2026-11-08 加入）：期权 IV 处于近一年极高分位、标的在 20 日线下方 ----
    r1 = cfg.get("r1")
    ivp_now = u.get("ivp", np.nan) if hasattr(u, "get") else np.nan
    if r1 and not any(s["dir"] > 0 for s in out) and pd.notna(ivp_now) and ivp_now >= r1.get("iv", 95) \
            and c < u.ma20 and (not r1.get("bull") or c > u.open):
        stop = c - r1.get("k", 1.5) * a
        out.append(dict(dir=1, strat="R1恐慌反转", stop=stop, target=c + 2 * (c - stop)))
    return out


# 当前推荐规则（研究总结文档《ETF期权买方策略研究总结》，2026-11-08 版）
RECOMMENDED = dict(pullback=True, iv_skip=True, iv_long=False, iv_short=True, vol_mult=1.2, s2="off",
                   s3=False, s4=True, bear_only=True, trail="ma5", trail_short="mix", short_stop_pct=0.6,
                   tstop_short=5, l4=True, l1_relax=True, tiers_off=True,
                   r1={"iv": 95, "k": 1.5}, reverse=("R1",))


# ---------------- 模拟 ----------------
@dataclass
class P:
    dir: int
    strat: str
    leg: str
    lots: int
    entry: float
    stop_pct: float
    u_entry: float
    u_stop: float
    u_target: float
    open_i: int
    maturity: str
    risk: float
    delta: float
    dl: int
    ivp: float | None
    lots_open: int = 0
    partial: bool = False
    be: bool = False
    peak: float = 0.0
    pending: str = ""
    holiday: bool = False
    realized: float = 0.0
    exits: list = field(default_factory=list)
    rolls: list = field(default_factory=list)


def run_v2(cfg, days, u, book, meta_u, ivp):
    if cfg.get("r1") and "ivp" not in u.columns:
        u = u.copy()
        u["ivp"] = pd.Series(ivp, dtype=float).reindex(u.index)
    pos: P | None = None
    trades, eq_curve, decisions = [], [], []
    realized, peak_eq, tier, pause_until = 0.0, CAP, "正常", -1
    month = defaultdict(int)
    tier_log = []

    def px(code, d, k="close"):
        return book.price(code, d, k)

    def close(p: P, n, price, d, why):
        nonlocal realized
        n = min(n, p.lots_open)
        if n <= 0:
            return
        ex = price - slip(price)
        pnl = (ex - p.entry) * 10000 * n - FEE * n
        p.realized += pnl
        realized += pnl
        p.lots_open -= n
        p.exits.append((d, why, n, round(price, 4)))

    def finish(p: P, d):
        trades.append({"code": NAME, "dir": "看涨" if p.dir > 0 else "看跌", "level": "A", "strat": p.strat,
                       "open": days[p.open_i], "close": d, "lots": p.lots, "entry": round(p.entry, 4),
                       "risk": round(p.risk), "pnl": round(p.realized, 1),
                       "exits": "；".join(f"{x[0]} {x[1]}×{x[2]}" for x in p.exits),
                       "exit_detail": json.dumps(p.exits, ensure_ascii=False), "maturity": p.maturity,
                       "u_entry": round(p.u_entry, 3), "u_stop": round(p.u_stop, 3), "u_target": round(p.u_target, 3),
                       "legs": f"买{book.meta.at[p.leg, 'name'] if 'name' in book.meta.columns and p.leg in book.meta.index else p.leg}"
                               + ("；移仓：" + "，".join(p.rolls) if p.rolls else ""),
                       "entry_delta": round(p.delta, 2), "entry_days_left": p.dl, "iv_pct": p.ivp})

    def decide(d, s, outcome):
        decisions.append({"date": d, "code": NAME, "dir": "看涨" if s["dir"] > 0 else "看跌", "level": "A",
                          "strat": s["strat"], "u_close": round(float(u.at[d, "close"]), 3),
                          "stop": round(float(s["stop"]), 3), "target": round(float(s["target"]), 3),
                          "iv_pct": ivp.get(d), "outcome": outcome})

    for i, d in enumerate(days):
        if d not in u.index:
            continue
        r = u.loc[d]
        gap_next = (dt(days[i + 1]) - dt(d)).days if i + 1 < len(days) else 1
        # ---------- 持仓管理 ----------
        if pos:
            p = pos
            o, hi, lo, cl = px(p.leg, d, "open"), px(p.leg, d, "high"), px(p.leg, d, "low"), px(p.leg, d)
            if None not in (o, hi, lo, cl):
                if p.pending:
                    close(p, p.lots_open, o, d, p.pending)
                if p.lots_open:
                    sp = p.entry if p.be else p.entry * (1 - p.stop_pct)
                    if lo <= sp:
                        close(p, p.lots_open, min(o, sp), d, "保本止损" if p.be else "权利金止损")
                if p.lots_open and not p.partial:
                    hit50 = hi >= p.entry * 1.5
                    hit_t = (r.close >= p.u_target) if p.dir > 0 else (r.close <= p.u_target)
                    if hit50 or hit_t:
                        p.partial = True
                        why = "+50%平一半" if hit50 else "到达2:1目标平一半"
                        price = max(o, p.entry * 1.5) if hit50 else cl
                        if p.lots_open >= 2:
                            close(p, p.lots_open // 2, price, d, why)
                        else:
                            p.exits.append((d, why.replace("平一半", "（只有1张，止损提到保本）"), 0, round(price, 4)))
                        p.be = True
                        p.peak = cl
                if p.lots_open and p.partial:
                    p.peak = max(p.peak, cl)
                    mode = cfg.get("trail_short", cfg["trail"]) if p.dir < 0 else cfg["trail"]
                    out = False
                    if mode.startswith("opt"):
                        k = int(mode[3:]) / 100
                        out = cl <= p.peak * (1 - k)
                        why = f"期权从最高回落{int(k*100)}%"
                    elif mode == "ma5":
                        out = (r.close < r.ma5) if p.dir > 0 else (r.close > r.ma5)
                        why = "标的跌破5日线" if p.dir > 0 else "标的站上5日线"
                    elif mode == "adapt":
                        strong = (r.state == "上升") if p.dir > 0 else (r.state == "下降")
                        line = r.ma20 if strong else r.ma5
                        out = (r.close < line) if p.dir > 0 else (r.close > line)
                        why = ("趋势强，标的跌破20日线" if strong else "趋势转弱，标的跌破5日线") if p.dir > 0 else \
                              ("趋势强，标的站上20日线" if strong else "趋势转弱，标的站上5日线")
                    elif mode.startswith("chand"):
                        k = float(mode[5:] or 3)
                        ext = getattr(p, "u_ext", r.close)
                        ext = max(ext, r.close) if p.dir > 0 else min(ext, r.close)
                        p.u_ext = ext
                        out = (r.close < ext - k * r.atr) if p.dir > 0 else (r.close > ext + k * r.atr)
                        why = f"标的从最高回落超过{k:g}倍ATR" if p.dir > 0 else f"标的从最低反弹超过{k:g}倍ATR"
                    elif mode == "mix10":
                        b5 = (r.close < r.ma5) if p.dir > 0 else (r.close > r.ma5)
                        b10 = (r.close < r.ma10) if p.dir > 0 else (r.close > r.ma10)
                        if b10:
                            out, why = True, ("标的跌破10日线" if p.dir > 0 else "标的站上10日线")
                        elif b5 and not getattr(p, "mix_half", False) and p.lots_open >= 2:
                            p.mix_half = True
                            close(p, p.lots_open // 2, cl, d, "标的跌破5日线（平剩余的一半）" if p.dir > 0 else "标的站上5日线（平剩余的一半）")
                    elif mode == "mix":
                        # 剩余仓位一半跌破5日线出，另一半跌破20日线出
                        b5 = (r.close < r.ma5) if p.dir > 0 else (r.close > r.ma5)
                        b20 = (r.close < r.ma20) if p.dir > 0 else (r.close > r.ma20)
                        if b20:
                            out, why = True, ("标的跌破20日线" if p.dir > 0 else "标的站上20日线")
                        elif b5 and not getattr(p, "mix_half", False) and p.lots_open >= 2:
                            p.mix_half = True
                            close(p, p.lots_open // 2, cl, d, "标的跌破5日线（平剩余的一半）" if p.dir > 0 else "标的站上5日线（平剩余的一半）")
                    elif mode == "ma20":
                        out = (r.close < r.ma20) if p.dir > 0 else (r.close > r.ma20)
                        why = "标的跌破20日线" if p.dir > 0 else "标的站上20日线"
                    elif mode == "ma10":
                        ma10 = r.ma10
                        out = (r.close < ma10) if p.dir > 0 else (r.close > ma10)
                        why = "标的跌破10日线" if p.dir > 0 else "标的站上10日线"
                    if out:
                        close(p, p.lots_open, cl, d, why)
                if p.lots_open:
                    broke = (r.close < p.u_stop) if p.dir > 0 else (r.close > p.u_stop)
                    prog = (r.close - p.u_entry) * p.dir / max(abs(p.u_target - p.u_entry), 1e-9)
                    left = (dt(p.maturity) - dt(d)).days
                    if left < 7:
                        if p.partial and cfg.get("roll", True):
                            # 移仓：平掉当前合约，换下一个月 Delta≈0.5 的合约，张数不变
                            n = p.lots_open
                            close(p, n, cl, d, "到期前移仓（平旧）")
                            ct = v1.pick_contract(book, meta_u, d, "C" if p.dir > 0 else "P", r.close, False)
                            if ct:
                                newpx = ct["long_px"] + slip(ct["long_px"])
                                p.realized -= FEE * n
                                realized_adj = FEE * n
                                p.rolls.append(f"{d} 换{book.meta.at[ct['long'], 'name'] if 'name' in book.meta.columns else ct['long']}")
                                p.leg, p.maturity, p.lots_open = ct["long"], ct["maturity"], n
                                # 新合约以新价格为成本，跟踪最高价重置；权利金止损恢复为正常比例
                                p.entry, p.peak, p.be = newpx, ct["long_px"], False
                                realized -= realized_adj
                    elif not p.partial and i - p.open_i == (cfg.get("tstop_short", 5) if p.dir < 0 else 5) and prog < 0.5:
                        close(p, p.lots_open, cl, d, "时间止损")
                    elif broke and not p.pending:
                        p.pending = "逻辑止损（次日开盘）"
                    if p.lots_open and gap_next > 5 and not p.holiday:
                        p.holiday = True
                        close(p, p.lots_open - p.lots_open // 2 if p.lots_open >= 2 else p.lots_open, cl, d, "长假前减仓")
            if p.lots_open <= 0:
                finish(p, d)
                pos = None

        # ---------- 净值与档位 ----------
        unreal = 0.0
        if pos:
            cl = px(pos.leg, d)
            if cl:
                unreal = (cl - pos.entry) * 10000 * pos.lots_open
        eq = CAP + realized + unreal
        eq_curve.append((d, round(eq, 1)))
        # 风控回撤按已平仓的盈亏算：持仓中浮盈的回吐不算亏损（跟踪止盈本来就要回吐一部分）
        eq_risk = CAP + realized
        peak_eq = max(peak_eq, eq_risk)
        dd = (peak_eq - eq_risk) / CAP
        week_end = i + 1 >= len(days) or dt(days[i + 1]).isocalendar()[1] != dt(d).isocalendar()[1]
        if week_end and i >= pause_until and not cfg.get("tiers_off"):
            kk = cfg.get("risk_k", 1)
            new = "正常" if dd < 0.05 * kk else "减速" if dd < 0.07 * kk else "保护" if dd < 0.10 * kk else "停手"
            if new == "停手" and tier != "停手":
                if pos:
                    c2 = px(pos.leg, d)
                    if c2:
                        close(pos, pos.lots_open, c2, d, "回撤10%熔断")
                    finish(pos, d)
                    pos = None
                pause_until, peak_eq = i + 10, eq_risk
            if new != tier:
                tier_log.append((d, tier, new, round(dd * 100, 1)))
            tier = new
        if tier == "停手" and i >= pause_until:
            tier = "保护"

        # ---------- 开仓 ----------
        if i < 60 or i >= len(days) - 1:
            continue
        sigs = signals_v2(r, cfg["pullback"], cfg)
        for s in sigs:
            if tier == "停手":
                decide(d, s, "放弃：风控停手档")
                continue
            if pos and cfg.get("reverse") and pos.dir != s["dir"] and s["strat"][:2] in cfg["reverse"]:
                c_old = book.price(pos.leg, d)
                if c_old:
                    close(pos, pos.lots_open, c_old, d, "反向信号，反手")
                    finish(pos, d)
                    pos = None
            if pos:
                decide(d, s, "放弃：已有持仓")
                continue
            if month[d[:6]] >= 6:
                decide(d, s, "放弃：本月已开6笔")
                continue
            iv = ivp.get(d)
            iv_rule = cfg.get("iv_long" if s["dir"] > 0 else "iv_short", cfg["iv_skip"])
            if iv_rule and iv is not None and iv >= 70:
                decide(d, s, "放弃：IV偏高")
                continue
            ct = v1.pick_contract(book, meta_u, d, "C" if s["dir"] > 0 else "P", r.close, False)
            if ct is None:
                decide(d, s, "放弃：没有符合条件的合约")
                continue
            entry = ct["long_px"] + slip(ct["long_px"])
            stop_pct = 0.4 if s["strat"].startswith(("L1", "L2", "S2")) else 0.5
            if s["dir"] < 0 and cfg.get("short_stop_pct"):
                stop_pct = cfg["short_stop_pct"]
            risk = {"正常": 1000, "减速": 500, "保护": 300}[tier] * cfg.get("risk_k", 1)
            lots = math.floor(risk / (entry * 10000 * stop_pct))
            if lots < 1:
                decide(d, s, "放弃：一张的风险就超过预算")
                continue
            pos = P(dir=s["dir"], strat=s["strat"], leg=ct["long"], lots=lots, entry=entry, stop_pct=stop_pct,
                    u_entry=r.close, u_stop=s["stop"], u_target=s["target"], open_i=i, maturity=ct["maturity"],
                    risk=lots * entry * 10000 * stop_pct, delta=ct["delta"], dl=ct["days_left"],
                    ivp=None if iv is None else round(iv), lots_open=lots)
            pos.realized -= FEE * lots
            realized -= FEE * lots
            month[d[:6]] += 1
            decide(d, s, f"开仓：{lots}张")
            break
    if pos:
        c2 = book.price(pos.leg, days[-1])
        if c2:
            close(pos, pos.lots_open, c2, days[-1], "回测结束")
        finish(pos, days[-1])
    return {"trades": trades, "equity": eq_curve, "decisions": decisions, "tiers": tier_log, "capital": CAP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20230605")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    end = a.end or datetime.now().strftime("%Y%m%d")
    warm = str(int(a.start[:4]) - 1) + a.start[4:]
    all_days = opt_data.trade_days(warm, end)
    end = all_days[-1]
    meta = opt_data.contracts()
    od = opt_data.opt_daily([d for d in all_days if d >= "20230601"])
    df = opt_data.etf_daily(CODE, all_days[0], end)
    df["prev_close"] = df["close"].shift(1)
    u = frame(df)
    book = v1.Book(od, meta)
    meta_u = meta[meta.underlying == CODE]
    iv = v1.atm_iv_series(book, meta_u, u, all_days)
    ivp = iv.rolling(250, min_periods=120).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True).dropna().to_dict()
    sim = all_days[all_days.index(next(d for d in all_days if d >= a.start)) - 60:]

    rec = dict(trail="ma5", pullback=True, iv_skip=True, iv_long=False, iv_short=True, vol_mult=1.2, s2="off")
    H = {**rec, "s3": True, "s4": True, "bear_only": True, "trail_short": "ma5", "short_stop_pct": 0.6, "tstop_short": 5}
    M = {**H, "l4": True, "l1_relax": True, "trail": "ma20", "trail_short": "mix"}
    variants = [
        {"key": "A", "name": "A 旧规则（v1）", "old": True},
        {"key": "H", "name": "H 上一版：加强做空", **H},
        {"key": "M", "name": "M 新推荐：加放量大阳线、放宽突破，多单跌破20日线才走", **M},
        {"key": "N", "name": "N 同M，多单跌破5日线就走", **{**M, "trail": "ma5"}},
        {"key": "O", "name": "O 同M，多单一半跌破5日线走、一半跌破20日线走", **{**M, "trail": "mix"}},
        {"key": "P", "name": "P 同M，不要放量大阳线信号", **{**M, "l4": False}},
        {"key": "Q", "name": "Q 同M，突破仍要求上升趋势", **{**M, "l1_relax": False}},
    ]
    res = []
    for v in variants:
        print("回测", v["name"], flush=True)
        if v.get("old"):
            cfg = {"name": v["name"], "unders": [CODE], "use_B": False, "iv_mode": "skip", "risk_tiers": True, "capital": CAP}
            r = v1.run(cfg, sim, {CODE: v1.underlying_frame(df)}, book, meta, {CODE: ivp})
        else:
            r = run_v2(v, sim, u, book, meta_u, ivp)
        r["stats"] = v1.stats(r["trades"], r["equity"], CAP)
        r["v"] = v
        res.append(r)

    out = ROOT / "reports" / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"kc50_{a.start}_{end}"
    L = [f"# 科创50ETF 期权策略回测：新旧规则对比（{a.start}–{end}）\n",
         f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}。本金 10 万，单笔风险 1000 元（减速档 500 元），只做科创50ETF期权，一次只持有一笔。\n",
         "## 新规则（v2）要点\n",
         "- 做多 L1 趋势突破：上升趋势 + 收盘首次创 20 日新高 + 放量（推荐方案 1.2 倍）；止损 = min(突破位, 收盘 − 1ATR)；目标 = 2:1",
         "- 做多 L2 回踩60日线起涨：近 20 天回踩到 60 日线附近（最低价 ≤ MA60×1.02）且收盘没跌破（≥ MA60×0.96）、60 日线不明显向下，当天阳线放量、首次创 20 日新高、站上 20 日线；不要求趋势状态为上升；止损 = min(MA20, 收盘 − 1ATR)；目标 = 2:1",
         "- 做多 L3 回调上车（旧规则②）：止损 = min(近5日最低, 收盘 − 1ATR)，目标 = 前高，盈亏比 ≥ 2",
         f"- 偏离过滤（按信号前一天判断）：收盘离 20 日线超过 {EXT_ATR} 倍 ATR，或 20 日线离 60 日线超过 {int(EXT_GAP*100)}%，不追（做空对称）",
         "- IV 过滤（推荐方案）：只对做空生效；做多不过滤，因为科创50大涨时 IV 往往同步冲高",
         "- 做空 S1 跌破20日线和平台：前 3 天内还在 20 日线上方，今天收盘跌破 20 日线，并跌破前 10 日最低价；止损 = 20 日线；目标 = 60 日线；盈亏比 ≥ 2；不要求放量",
         "- 做空 S2 跌破前低（推荐方案不做，E 方案对照）：收盘首次跌破 20 日最低收盘，且在 20 日线或 60 日线下方，趋势状态不是上升；止损 = max(前低, 收盘 + 1ATR)；目标 = 2:1；不要求放量",
         "- 出场：逻辑止损（次日开盘）、权利金止损（L1/L2/S2 −40%，其他 −50%）、未进入跟踪阶段时 5 日时间止损；"
         "+50% 或到达 2:1 目标（先到者）平一半、只执行一次、剩余止损提到保本；剩余仓位按各方案的跟踪规则出场；剩余不到 7 天：已进入跟踪阶段就移仓到下个月，否则平仓；长假前减半",
         "- 做空 S3 反抽60日线受阻：20 日线在 60 日线下方，近 3 天最高价碰到 60 日线附近，当天阴线且收盘低于前一天、在 60 日线下方；止损 = max(近5日最高, 60日线, 收盘 + 1ATR)；目标 = 2:1",
         "- 做空 S4 反抽20日线受阻：下降趋势中近 3 天最高价碰到 20 日线附近，当天阴线收低、在 20 日线下方；止损 = max(近5日最高, 20日线, 收盘 + 1ATR)；目标 = 2:1",
         "- 做多 L4 放量大阳线（M 方案起）：当天涨幅 ≥ 3%、成交量 ≥ 20 日均量 1.5 倍、收阳并站上 20 日线；止损 = min(当天最低, 收盘 − 1ATR)；目标 = 2:1",
         "- L1 突破放宽（M 方案起）：不再要求趋势状态为上升，改为收盘在 60 日线上方且 60 日线不明显向下",
         "- 跟踪止盈（M 方案）：多单剩余仓位标的跌破 20 日线才清仓；空单剩余仓位一半站上 5 日线平、另一半站上 20 日线平",
         "- 空头环境过滤（H 方案起）：只在 60 日线低于 20 天前时做空；做空权利金止损 −60%",
         "- 风控：每周按已平仓盈亏计算的回撤分档（浮盈回吐不算）：正常 1000 元、减速 500 元、保护 300 元；停手档平仓停 10 个交易日。每月最多 6 笔。\n",
         "## 结果对比\n",
         v1.md_table(pd.DataFrame([{"方案": r["v"]["name"], **r["stats"]} for r in res]))]
    for r in res:
        L.append(f"\n## {r['v']['name']}\n")
        L.append("### 按信号\n" + v1.md_table(v1.group_stats(r["trades"], "strat")))
        L.append("\n### 按方向\n" + v1.md_table(v1.group_stats(r["trades"], "dir")))
        t = pd.DataFrame(r["trades"])
        if not t.empty:
            t["year"] = t["open"].astype(str).str[:4]
            L.append("\n### 按年份\n" + v1.md_table(v1.group_stats(t.to_dict("records"), "year")))
        dec = pd.DataFrame(r.get("decisions", []))
        if not dec.empty:
            L.append("\n### 信号处理结果\n" + v1.md_table(dec["outcome"].str.replace(r"\d+张", "N张", regex=True).value_counts().reset_index()))
        if r["tiers"]:
            L.append("\n风控档位变化：" + "；".join(f"{x[0]}：{x[1]}→{x[2]}（回撤{x[3]}%）" for x in r["tiers"]) + "\n")
        L.append("\n### 交易明细\n" + v1.md_table(t.drop(columns=["exit_detail", "code", "level"], errors="ignore") if not t.empty else t))
    (out / f"{tag}.md").write_text("\n".join(L), encoding="utf-8")
    for r in res:
        k = r["v"]["key"]
        pd.DataFrame(r["trades"]).to_csv(out / f"{tag}_{k}_trades.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(r["equity"], columns=["date", "equity"]).to_csv(out / f"{tag}_{k}_equity.csv", index=False)
        pd.DataFrame(r.get("decisions", [])).to_csv(out / f"{tag}_{k}_signals.csv", index=False, encoding="utf-8-sig")
    cols = ["open", "high", "low", "close", "vol", "ma5", "ma20", "ma60", "state", "atr", "ext_up", "ext_dn"]
    u.loc[u.index >= a.start, cols].to_csv(out / f"{tag}_etf.csv")
    print("\n".join(L)[:5000])


if __name__ == "__main__":
    main()
