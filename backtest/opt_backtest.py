"""ETF 期权买方策略回测：按《ETF期权买方交易指引（5万本金版）》的规则逐日模拟。

用法（需要环境变量 TS_TOKEN）：
    python backtest/opt_backtest.py --start 20230101 --end 20261106

数据：日线级别（Tushare）。信号在收盘时判断、按当日收盘价进场；盘中的止损和止盈用期权当日最高/最低价近似。
所有假设写在报告开头。
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
from md.opt import bs_delta, implied_vol  # noqa: E402

UNDER = {"510300.SH": "300ETF", "510500.SH": "500ETF", "510050.SH": "50ETF",
         "159915.SZ": "创业板ETF", "588000.SH": "科创50ETF"}
PRIORITY = list(UNDER)          # 同一天同方向多个信号时，按这个顺序取第一个（指引：不确定就用300ETF）
CAPITAL = 50000
R = 0.015                        # 无风险利率近似
FEE = 2.0                        # 每张每边费用（佣金+经手费），元
SLIP_PCT, SLIP_MIN = 0.01, 0.0001  # 滑点：权利金的 1%，至少 1 个最小变动单位


def dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def slip(px: float) -> float:
    return max(px * SLIP_PCT, SLIP_MIN)


# ---------------- 标的指标与信号 ----------------
def underlying_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().set_index("trade_date")
    c, v = df["close"], df["vol"]
    df["ma20"], df["ma60"] = c.rolling(20).mean(), c.rolling(60).mean()
    df["vol20_prev"] = v.rolling(20).mean().shift(1)
    df["hi20"] = c.rolling(20).max().shift(1)       # 前 20 日最高收盘（不含当天）
    df["lo20"] = c.rolling(20).min().shift(1)
    df["ma20_5"] = df["ma20"].shift(5)
    side = np.sign(c - df["ma20"])
    df["cross10"] = (side.diff().abs() > 0).astype(int).rolling(10).sum()
    up = (c > df.ma20) & (df.ma20 > df.ma60) & (df.ma20 > df.ma20_5)
    dn = (c < df.ma20) & (df.ma20 < df.ma60) & (df.ma20 < df.ma20_5)
    st = np.where(up, "上升", np.where(dn, "下降", "震荡"))
    st = np.where(df["cross10"] >= 3, "震荡", st)
    df["state"] = st
    df["low5"] = df["low"].rolling(5).min()
    df["high5"] = df["high"].rolling(5).max()
    df["touch_up"] = (df["low"] <= df["ma20"] * 1.01).astype(float).rolling(3).max() > 0
    df["touch_dn"] = (df["high"] >= df["ma20"] * 0.99).astype(float).rolling(3).max() > 0
    df["max20"] = c.rolling(20).max()
    df["min20"] = c.rolling(20).min()
    return df


def signals(u: pd.Series) -> list[dict]:
    """返回当天的信号列表：方向、级别、策略、标的止损位、目标位。"""
    out = []
    c = u.close
    if any(pd.isna(u[k]) for k in ["ma60", "hi20", "vol20_prev"]):
        return out
    vol_ok = u.vol >= 1.3 * u.vol20_prev
    # ① 突破追势
    if c > u.hi20 and vol_ok:
        dist = max(c - u.hi20, 0.01 * c)
        lvl = "A" if u.state == "上升" else ("B" if u.state == "震荡" else None)
        if lvl:
            out.append(dict(dir=1, level=lvl, strat="①突破", stop=u.hi20, target=c + 2 * dist))
    if c < u.lo20 and vol_ok:
        dist = max(u.lo20 - c, 0.01 * c)
        lvl = "A" if u.state == "下降" else ("B" if u.state == "震荡" else None)
        if lvl:
            out.append(dict(dir=-1, level=lvl, strat="①突破", stop=u.lo20, target=c - 2 * dist))
    # ② 回调上车（只在趋势明确时）
    if u.state == "上升" and u.touch_up and c > u.open and c > u.prev_close:
        stop, target = u.low5, u.max20
        if target - c >= 2 * (c - stop) > 0:
            out.append(dict(dir=1, level="A", strat="②回调", stop=stop, target=target))
    if u.state == "下降" and u.touch_dn and c < u.open and c < u.prev_close:
        stop, target = u.high5, u.min20
        if c - target >= 2 * (stop - c) > 0:
            out.append(dict(dir=-1, level="A", strat="②回调", stop=stop, target=target))
    return out


# ---------------- 持仓 ----------------
@dataclass
class Pos:
    code: str
    dir: int
    level: str
    strat: str
    legs: list            # [(ts_code, side)]  side=+1 买入, -1 卖出
    lots: int
    entry: float          # 每份净权利金（元/份，正数）
    stop_pct: float
    u_entry: float
    u_stop: float
    u_target: float
    open_i: int
    maturity: str
    risk: float
    spread: bool = False
    short_strike: float = 0.0
    half_taken: bool = False
    be_stop: bool = False
    max_close: float = 0.0
    pending_exit: str = ""
    holiday_cut: bool = False
    realized: float = 0.0
    lots_open: int = 0
    exits: list = field(default_factory=list)


class Book:
    def __init__(self, od: pd.DataFrame, meta: pd.DataFrame):
        self.px = {(r.ts_code, r.trade_date): (r.open, r.high, r.low, r.close, r.settle)
                   for r in od.itertuples(index=False)}
        self.by_day = {d: g for d, g in od.groupby("trade_date")}
        self.meta = meta.set_index("ts_code")

    def price(self, code: str, d: str, kind: str = "close") -> float | None:
        p = self.px.get((code, d))
        if p is None:
            return None
        o, h, l, c, s = p
        eff = c if c > 0 else s
        if kind == "close":
            return eff or None
        if kind == "open":
            return (o if o > 0 else eff) or None
        if kind == "high":
            return (h if h > 0 else eff) or None
        if kind == "low":
            return (l if l > 0 else eff) or None


def pick_contract(book: Book, meta_u: pd.DataFrame, d: str, cp: str, S: float, spread: bool):
    day = book.by_day.get(d)
    if day is None:
        return None
    m = meta_u[(meta_u.call_put == cp) & (~meta_u.adjusted) & (meta_u.list_date <= d) & (meta_u.maturity_date > d)]
    m = m[m.ts_code.isin(day.ts_code)]
    if m.empty:
        return None
    dl = {mt: (dt(mt) - dt(d)).days for mt in m.maturity_date.unique()}
    inwin = sorted([mt for mt, x in dl.items() if 20 <= x <= 45])
    later = sorted([mt for mt, x in dl.items() if x > 45])
    mt = inwin[0] if inwin else (later[0] if later else None)  # 指引：20–45 天，一般是次月
    if mt is None:
        return None
    T = dl[mt] / 365
    ch = m[m.maturity_date == mt].copy()
    best = None
    for r in ch.itertuples():
        p = book.price(r.ts_code, d)
        live = day[day.ts_code == r.ts_code]
        if not p or live.empty or live.vol.iloc[0] <= 0 or live.oi.iloc[0] <= 0:
            continue
        sig = implied_vol(p, S, r.exercise_price, T, R, cp)
        de = bs_delta(S, r.exercise_price, T, R, sig, cp)
        if np.isfinite(de) and 0.4 <= abs(de) <= 0.6:
            score = abs(abs(de) - 0.5)
            if best is None or score < best[0]:
                best = (score, r.ts_code, r.exercise_price, p, de)
    if best is None:
        return None
    _, code, k, p, de = best
    res = {"maturity": mt, "long": code, "long_k": k, "long_px": p, "delta": de, "days_left": dl[mt]}
    if spread:
        ks = sorted(ch.exercise_price.unique())
        i = ks.index(k)
        j = i + 2 if cp == "C" else i - 2
        if not (0 <= j < len(ks)):
            return None
        sk = ks[j]
        srow = ch[ch.exercise_price == sk]
        sc = srow.ts_code.iloc[0]
        sp = book.price(sc, d)
        if not sp:
            return None
        res.update(short=sc, short_k=sk, short_px=sp)
    return res


def atm_iv_series(book: Book, meta_u: pd.DataFrame, u: pd.DataFrame, days: list[str]) -> pd.Series:
    out = {}
    for d in days:
        if d not in u.index or d not in book.by_day:
            continue
        S = u.at[d, "close"]
        day = book.by_day[d]
        m = meta_u[(~meta_u.adjusted) & (meta_u.maturity_date > d) & (meta_u.ts_code.isin(day.ts_code))]
        mats = sorted(x for x in m.maturity_date.unique() if (dt(x) - dt(d)).days >= 15)
        if not mats:
            continue
        mt = mats[0]
        ch = m[m.maturity_date == mt]
        if ch.empty:
            continue
        k = ch.iloc[(ch.exercise_price - S).abs().argsort()[:1]].exercise_price.iloc[0]
        T = (dt(mt) - dt(d)).days / 365
        ivs = []
        for cp in ["C", "P"]:
            row = ch[(ch.exercise_price == k) & (ch.call_put == cp)]
            if len(row):
                p = book.price(row.ts_code.iloc[0], d)
                if p:
                    ivs.append(implied_vol(p, S, k, T, R, cp))
        ivs = [x for x in ivs if np.isfinite(x)]
        if ivs:
            out[d] = float(np.mean(ivs))
    s = pd.Series(out).sort_index()
    return s


# ---------------- 模拟 ----------------
def run(cfg: dict, days: list[str], U: dict, book: Book, meta: pd.DataFrame, ivp: dict) -> dict:
    """cfg: name, unders, use_B, iv_mode(spread/skip/off), risk_tiers, capital"""
    cap = cfg.get("capital", CAPITAL)
    k = cap / 50000            # 风险额度按本金等比例放大（指引：单笔 = 本金 1%）
    positions: list[Pos] = []
    closed: list[dict] = []
    realized = 0.0
    equity_curve = []
    peak = cap
    tier = "正常"
    pause_until = -1
    month_count = defaultdict(int)
    skipped = defaultdict(int)
    skipped_by = defaultdict(lambda: defaultdict(int))
    tier_log = []
    meta_by_u = {c: meta[meta.underlying == c] for c in cfg["unders"]}

    def pos_value(p: Pos, d: str, kind="close") -> float | None:
        v = 0.0
        for code, side in p.legs:
            x = book.price(code, d, kind)
            if x is None:
                return None
            v += side * x
        return v

    def close_lots(p: Pos, n: int, px_per: float, d: str, reason: str):
        nonlocal realized
        n = min(n, p.lots_open)
        if n <= 0:
            return
        legs = len(p.legs)
        exit_px = px_per - slip(abs(px_per)) * legs
        pnl = (exit_px - p.entry) * 10000 * n - FEE * legs * n
        p.realized += pnl
        realized += pnl
        p.lots_open -= n
        p.exits.append((d, reason, n, round(px_per, 4)))

    def finish(p: Pos, d: str):
        closed.append({
            "code": UNDER[p.code], "dir": "看涨" if p.dir > 0 else "看跌", "level": p.level,
            "strat": "③价差" if p.spread else p.strat, "open": days[p.open_i], "close": d,
            "lots": p.lots, "entry": round(p.entry, 4), "risk": round(p.risk, 0),
            "pnl": round(p.realized, 1),
            "exits": "；".join(f"{x[0]} {x[1]}×{x[2]}" for x in p.exits),
            "maturity": p.maturity,
        })

    for i, d in enumerate(days):
        # ---- 1. 管理已有持仓 ----
        nxt_gap = (dt(days[i + 1]) - dt(d)).days if i + 1 < len(days) else 1
        for p in list(positions):
            u = U[p.code]
            if d not in u.index:
                continue
            uc = u.at[d, "close"]
            if p.pending_exit:
                v = pos_value(p, d, "open")
                if v is not None:
                    close_lots(p, p.lots_open, v, d, p.pending_exit)
            if p.lots_open > 0 and not p.spread:
                lo, hi = book.price(p.legs[0][0], d, "low"), book.price(p.legs[0][0], d, "high")
                op, cl = book.price(p.legs[0][0], d, "open"), book.price(p.legs[0][0], d, "close")
                if None in (lo, hi, op, cl):
                    continue
                stop_px = p.entry if p.be_stop else p.entry * (1 - p.stop_pct)
                if lo <= stop_px:
                    close_lots(p, p.lots_open, min(op, stop_px), d, "保本止损" if p.be_stop else "权利金止损")
                elif not p.half_taken and hi >= p.entry * 1.5:
                    p.half_taken = True
                    if p.lots_open >= 2:
                        close_lots(p, p.lots_open // 2, max(op, p.entry * 1.5), d, "+50%平一半")
                    else:
                        p.be_stop = True
                if p.lots_open > 0:
                    p.max_close = max(p.max_close, cl)
                    if p.half_taken and cl <= 0.7 * p.max_close:
                        close_lots(p, p.lots_open, cl, d, "高点回撤30%")
            if p.lots_open > 0 and p.spread:
                v = pos_value(p, d)
                if v is not None and v <= p.entry * 0.5:
                    close_lots(p, p.lots_open, v, d, "价差价值-50%")
            if p.lots_open > 0:
                v = pos_value(p, d)
                if v is None:
                    continue
                held = i - p.open_i
                hit_target = (uc >= p.u_target) if p.dir > 0 else (uc <= p.u_target)
                if p.spread:
                    hit_target = (uc >= p.short_strike) if p.dir > 0 else (uc <= p.short_strike)
                broke = (uc < p.u_stop) if p.dir > 0 else (uc > p.u_stop)
                prog = (uc - p.u_entry) * p.dir / max(abs(p.u_target - p.u_entry), 1e-9)
                left = (dt(p.maturity) - dt(d)).days
                if hit_target:
                    close_lots(p, p.lots_open, v, d, "到达目标")
                elif left < 7:
                    close_lots(p, p.lots_open, v, d, "剩余<7天平仓")
                elif held == 5 and prog < 0.5:
                    close_lots(p, p.lots_open, v, d, "时间止损")
                elif broke:
                    p.pending_exit = "逻辑止损（次日开盘）"
                if p.lots_open > 0 and nxt_gap > 5 and not p.holiday_cut:
                    p.holiday_cut = True
                    close_lots(p, max(1, p.lots_open - p.lots_open // 2) if p.lots_open >= 2 else p.lots_open,
                               v, d, "长假前减仓")
            if p.lots_open <= 0:
                positions.remove(p)
                finish(p, d)

        # ---- 2. 估值与风控档位 ----
        unreal = 0.0
        for p in positions:
            v = pos_value(p, d)
            if v is not None:
                unreal += (v - p.entry) * 10000 * p.lots_open
        eq = cap + realized + unreal
        peak = max(peak, eq)
        dd = (peak - eq) / cap
        equity_curve.append((d, round(eq, 1)))
        is_week_end = i + 1 >= len(days) or dt(days[i + 1]).isocalendar()[1] != dt(d).isocalendar()[1]
        if cfg["risk_tiers"] and is_week_end and i >= pause_until:
            new = "正常" if dd < 0.05 else "减速" if dd < 0.07 else "保护" if dd < 0.10 else "停手"
            if new == "停手" and tier != "停手":
                for p in list(positions):
                    v = pos_value(p, d)
                    if v is not None:
                        close_lots(p, p.lots_open, v, d, "回撤10%熔断")
                    positions.remove(p)
                    finish(p, d)
                pause_until = i + 10
                peak = eq  # 停手两周复盘后，以当前净值为新起点
            if new != tier:
                tier_log.append((d, tier, new, round(dd * 100, 1)))
            tier = new if new != "停手" else "停手"
        if tier == "停手" and i >= pause_until:
            tier = "保护"
        if i < 60:
            continue

        # ---- 3. 新开仓 ----
        if tier == "停手" or i >= len(days) - 1:
            continue
        risk = ({"正常": 500, "减速": 250, "保护": 150}[tier] if cfg["risk_tiers"] else 500) * k
        max_pos = 1 if tier == "减速" else 3
        ym = d[:6]
        cands = []
        for code in sorted(cfg["unders"], key=PRIORITY.index):
            u = U[code]
            if d not in u.index:
                continue
            for s in signals(u.loc[d]):
                if s["level"] == "B" and not cfg["use_B"]:
                    continue
                if s["level"] == "B" and tier != "正常":
                    continue
                cands.append((code, s))
        for code, s in cands:
            if month_count[ym] >= 6:
                skipped["月度6笔上限"] += 1
                break
            if len(positions) >= max_pos:
                skipped["持仓已满"] += 1
                break
            if any(p.dir == s["dir"] for p in positions):
                skipped["同方向已有持仓"] += 1
                continue
            mode = cfg.get("iv_mode", "spread" if cfg.get("iv_filter", True) else "off")
            iv_hi = mode != "off" and ivp.get(code, {}).get(d, 50) >= 70
            if iv_hi and mode == "skip":
                skipped["IV偏高不做"] += 1
                skipped_by[UNDER[code]]["IV偏高不做"] += 1
                continue
            spread = (iv_hi and mode == "spread") or tier == "保护"
            rr = risk * (0.5 if s["level"] == "B" else 1)
            S = U[code].at[d, "close"]
            cp = "C" if s["dir"] > 0 else "P"
            ct = pick_contract(book, meta_by_u[code], d, cp, S, spread)
            if ct is None:
                skipped["无合适合约"] += 1
                continue
            if spread:
                net = ct["long_px"] - ct["short_px"]
                entry = net + 2 * slip(net)
                stop_pct = 0.5
                legs = [(ct["long"], 1), (ct["short"], -1)]
            else:
                entry = ct["long_px"] + slip(ct["long_px"])
                stop_pct = 0.4 if s["strat"] == "①突破" else 0.5
                legs = [(ct["long"], 1)]
            if entry <= 0:
                skipped["无合适合约"] += 1
                continue
            lots = math.floor(rr / (entry * 10000 * stop_pct))
            if lots < 1:
                skipped["一张的风险就超过预算"] += 1
                skipped_by[UNDER[code]]["一张的风险就超过预算"] += 1
                continue
            p = Pos(code=code, dir=s["dir"], level=s["level"], strat=s["strat"], legs=legs, lots=lots,
                    entry=entry, stop_pct=stop_pct, u_entry=S, u_stop=s["stop"], u_target=s["target"],
                    open_i=i, maturity=ct["maturity"], risk=lots * entry * 10000 * stop_pct, spread=spread,
                    short_strike=ct.get("short_k", 0.0), max_close=ct["long_px"], lots_open=lots)
            entry_fee = FEE * len(legs) * lots
            p.realized -= entry_fee
            realized -= entry_fee
            positions.append(p)
            month_count[ym] += 1

    # 回测结束时按收盘价了结
    for p in list(positions):
        v = pos_value(p, days[-1])
        if v is not None:
            close_lots(p, p.lots_open, v, days[-1], "回测结束")
        finish(p, days[-1])
    return {"trades": closed, "equity": equity_curve, "skipped": dict(skipped), "tiers": tier_log,
            "skipped_by": {u: dict(v) for u, v in skipped_by.items()}, "capital": cap}


# ---------------- 统计 ----------------
def stats(trades: list[dict], equity: list, cap: float = CAPITAL) -> dict:
    t = pd.DataFrame(trades)
    if t.empty:
        return {"笔数": 0}
    w, l = t[t.pnl > 0], t[t.pnl <= 0]
    eq = pd.Series([e for _, e in equity])
    mdd = (eq.cummax() - eq).max()
    streak = cur = 0
    for x in t.pnl:
        cur = cur + 1 if x <= 0 else 0
        streak = max(streak, cur)
    return {
        "笔数": len(t), "胜率%": round(len(w) / len(t) * 100, 1),
        "平均盈利": round(w.pnl.mean(), 0) if len(w) else 0, "平均亏损": round(l.pnl.mean(), 0) if len(l) else 0,
        "盈亏比": round(w.pnl.mean() / -l.pnl.mean(), 2) if len(w) and len(l) and l.pnl.mean() < 0 else None,
        "利润因子": round(w.pnl.sum() / -l.pnl.sum(), 2) if len(l) and l.pnl.sum() < 0 else None,
        "总盈亏": round(t.pnl.sum(), 0), "收益率%": round(t.pnl.sum() / cap * 100, 1),
        "最大回撤(元)": round(mdd, 0), "最大回撤%": round(mdd / cap * 100, 1),
        "最长连亏": streak, "单笔最大亏损": round(t.pnl.min(), 0), "单笔最大盈利": round(t.pnl.max(), 0),
    }


def group_stats(trades: list[dict], key: str) -> pd.DataFrame:
    t = pd.DataFrame(trades)
    if t.empty:
        return t
    g = t.groupby(key).agg(笔数=("pnl", "size"), 胜率=("pnl", lambda s: round((s > 0).mean() * 100, 1)),
                           总盈亏=("pnl", lambda s: round(s.sum(), 0)), 平均每笔=("pnl", lambda s: round(s.mean(), 0)))
    return g.reset_index()


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "（无）\n"
    cols = list(df.columns)
    s = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.values:
        s += "| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default=None)
    ap.add_argument("--set", default="base", help="base=5万本金五个方案；100k=10万本金方案")
    a = ap.parse_args()
    end = a.end or datetime.now().strftime("%Y%m%d")
    warm = str(int(a.start[:4]) - 1) + a.start[4:]           # 预热一年：均线和 IV 分位要用
    all_days = opt_data.trade_days(warm, end)
    end = all_days[-1]
    print(f"下载数据 {all_days[0]}–{end} ...", flush=True)
    meta = opt_data.contracts()
    od = opt_data.opt_daily(all_days)
    U = {}
    for code in UNDER:
        df = opt_data.etf_daily(code, all_days[0], end)
        df["prev_close"] = df["close"].shift(1)
        U[code] = underlying_frame(df)
    book = Book(od, meta)
    print("计算历史平值 IV ...", flush=True)
    ivp = {}
    iv_raw = {}
    for code in UNDER:
        s = atm_iv_series(book, meta[meta.underlying == code], U[code], all_days)
        iv_raw[code] = s
        pct = s.rolling(250, min_periods=120).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)
        ivp[code] = pct.dropna().to_dict()

    days = [d for d in all_days if d >= a.start]
    # 预留 60 天让第一天就有完整均线：把预热期的最后 60 天也放进模拟（这些天不开仓）
    sim_days = all_days[all_days.index(days[0]) - 60:]
    ALL5 = list(UNDER)
    if a.set == "100k":
        variants = [
            {"name": "V6 10万本金·5个品种·IV高不做", "unders": ALL5, "use_B": False, "iv_mode": "skip", "risk_tiers": True, "capital": 100000},
            {"name": "V7 10万本金·去掉50ETF·IV高不做", "unders": [c for c in ALL5 if c != "510050.SH"], "use_B": False, "iv_mode": "skip", "risk_tiers": True, "capital": 100000},
            {"name": "V8 10万本金·只做300ETF和500ETF·IV高不做", "unders": ["510300.SH", "510500.SH"], "use_B": False, "iv_mode": "skip", "risk_tiers": True, "capital": 100000},
            {"name": "V9 10万本金·5个品种·IV高用价差（V1放大对照）", "unders": ALL5, "use_B": False, "iv_mode": "spread", "risk_tiers": True, "capital": 100000},
        ]
    else:
        variants = [
            {"name": "V1 完整规则（A级信号，IV高用价差，风控分档，5个品种）", "unders": ALL5, "use_B": False, "iv_mode": "spread", "risk_tiers": True},
            {"name": "V2 只做300ETF（其他同V1）", "unders": ["510300.SH"], "use_B": False, "iv_mode": "spread", "risk_tiers": True},
            {"name": "V3 加上B级信号（半份风险）", "unders": ALL5, "use_B": True, "iv_mode": "spread", "risk_tiers": True},
            {"name": "V4 不用IV过滤（全部单腿买入）", "unders": ALL5, "use_B": False, "iv_mode": "off", "risk_tiers": True},
            {"name": "V5 不用风控分档（固定每笔500元）", "unders": ALL5, "use_B": False, "iv_mode": "spread", "risk_tiers": False},
        ]
    results = []
    for v in variants:
        print("回测", v["name"], flush=True)
        r = run(v, sim_days, U, book, meta, ivp)
        r["stats"] = stats(r["trades"], r["equity"], r["capital"])
        r["cfg"] = v
        results.append(r)

    out_dir = ROOT / "reports" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"opt_{a.start}_{end}" + ("" if a.set == "base" else f"_{a.set}")
    L = [f"# ETF 期权买方策略回测（{a.start}–{end}，方案组：{a.set}）\n",
         f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M')}。规则来自《ETF期权买方交易指引（5万本金版）》，数据来自 Tushare。\n",
         "## 回测假设\n",
         "- 日线级别：收盘判断信号、按当日收盘价进场；权利金止损和+50%止盈用期权当日最低/最高价判断，按触发价成交（跳空时按开盘价）；逻辑止损按次日开盘价平仓。",
         "- 信号：A级 = 趋势明确 + ①放量（≥20日均量1.3倍）突破20日高/低点 或 ②回调到20日线后转向（阳线/阴线收盘、盈亏比≥2）。"
         "\"重大政策/数据后第一根大阳/大阴线\"无法用历史数据判断，未纳入。B级 = 放量突破但趋势状态为震荡。",
         "- 合约：剩余20–45天的月份，没有就取次月；Delta 0.4–0.6 中最接近0.5的；排除分红调整合约和当天无成交、无持仓的合约。",
         "- IV过滤：平值IV处在近一年（250个交易日）的70%分位以上时改用价差（买平值、卖同月虚二档），否则单腿买入。",
         "- 出场：逻辑止损、权利金止损（①-40%、②-50%、③价差-50%）、5个交易日未走出目标一半的时间止损、+50%平一半（1张时止损提到保本）、剩余从最高收盘回撤30%出、"
         "到达目标位（价差为到达卖出行权价）、剩余不到7天平仓、长假（休市超过5天）前减半。",
         "- 本金：方案名里写了 10 万的按 10 万计，单笔风险和各档位额度按本金等比例放大（单笔 = 本金 1%，即 1000 元；减速档 500 元；保护档 300 元）；'IV高不做'表示平值 IV 在近一年 70% 分位以上时放弃该信号。",
         "- 仓位：张数 = 单笔风险 ÷（权利金 × 10000 × 止损比例）向下取整，算出来不到1张就放弃；每月最多6笔新开仓；同一方向同时只持1笔；最多3笔。",
         "- 风控分档（每周末按净值从高点的回撤）：<5% 每笔500元；5–7% 250元、只做A级、最多1笔；7–10% 150元、只做价差；≥10% 全部平仓、停两周，之后从\"保护\"档重新开始，回撤基准重置。",
         f"- 成本：每张每边 {FEE} 元费用；滑点为权利金的 1%（至少 0.0001）每边每腿。无风险利率按 {R*100}% 估算 IV 和 Delta。",
         "- 未模拟：单日亏750元/单周亏1500元暂停、限价单是否成交、买卖价差是否超过3%（历史数据没有盘口）。\n",
         "## 各方案结果对比\n"]
    rows = []
    for r in results:
        rows.append({"方案": r["cfg"]["name"], **r["stats"]})
    L.append(md_table(pd.DataFrame(rows)))
    for r in results:
        L.append(f"\n## {r['cfg']['name']}\n")
        L.append("### 按策略\n" + md_table(group_stats(r["trades"], "strat")))
        L.append("\n### 按品种\n" + md_table(group_stats(r["trades"], "code")))
        L.append("\n### 按方向\n" + md_table(group_stats(r["trades"], "dir")))
        t = pd.DataFrame(r["trades"])
        if not t.empty:
            t["year"] = t["open"].str[:4]
            L.append("\n### 按年份\n" + md_table(group_stats(t.to_dict("records"), "year")))
            ex = t["exits"].str.extract(r"\d{8} ([^×]+)×")[0].value_counts().reset_index()
            ex.columns = ["首个出场原因", "笔数"]
            L.append("\n### 出场原因\n" + md_table(ex))
        L.append(f"\n放弃的信号：{r['skipped']}\n")
        L.append(f"按品种的放弃原因：{r['skipped_by']}\n")
        if r["tiers"]:
            L.append("风控档位变化：" + "；".join(f"{a}：{b}→{c}（回撤{d}%）" for a, b, c, d in r["tiers"]) + "\n")
    v1 = results[0]
    L.append("\n## V1 全部交易明细\n" + md_table(pd.DataFrame(v1["trades"])))
    text = "\n".join(L)
    (out_dir / f"{tag}.md").write_text(text, encoding="utf-8")
    for r in results:
        key = r["cfg"]["name"].split()[0]
        pd.DataFrame(r["trades"]).to_csv(out_dir / f"{tag}_{key}_trades.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(r["equity"], columns=["date", "equity"]).to_csv(out_dir / f"{tag}_{key}_equity.csv", index=False)
    ivdf = pd.DataFrame({UNDER[c]: s for c, s in iv_raw.items()})
    ivdf.to_csv(out_dir / f"{tag}_atm_iv.csv")
    print(text[:6000])


if __name__ == "__main__":
    main()
