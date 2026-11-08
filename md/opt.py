"""ETF 期权模块：给「期权交易」项目的盘前简报用。

对应用户的交易指引：
- 趋势状态：上升 = 收盘 > MA20 > MA60 且 MA20 向上；下降 = 收盘 < MA20 < MA60 且 MA20 向下；
  其他、或近 10 天反复穿越 20 日线（≥3 次）= 震荡
- 触发：放量突破 20 日高/低点；回调到 20 日线后转向；大阳/大阴线（需结合新闻判断是否"重大政策/数据后第一根"）
- 合约：剩余 20–45 天、Delta 0.4–0.6、不买分红调整合约（合约单位不是 10000 的）
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from .client import DATA, cached_by_date, call

UNDERLYINGS = {
    "510050.SH": "50ETF",
    "510300.SH": "300ETF",
    "510500.SH": "500ETF",
    "588000.SH": "科创50ETF",
    "159915.SZ": "创业板ETF",
}


# ---------- Black-Scholes（欧式，ETF 期权为欧式） ----------
def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, sig, cp):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if cp == "C":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S, K, T, r, sig, cp):
    if T <= 0 or sig <= 0 or not np.isfinite(sig):
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return _ncdf(d1) if cp == "C" else _ncdf(d1) - 1


def implied_vol(price, S, K, T, r, cp):
    if not (price and price > 0 and T > 0):
        return np.nan
    intrinsic = max(0.0, (S - K * math.exp(-r * T)) if cp == "C" else (K * math.exp(-r * T) - S))
    if price <= intrinsic + 1e-6:
        return np.nan
    lo, hi = 0.01, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, r, mid, cp) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------- 数据 ----------
def _rate(d: str) -> float:
    """无风险利率：Shibor 3 个月，取不到用 1.5%。"""
    try:
        s = call("shibor", start_date=str(int(d) - 100), end_date=d).sort_values("date")
        return float(s["3m"].iloc[-1]) / 100
    except Exception:  # noqa: BLE001
        return 0.015


def _etf_hist(code: str, days: list[str]) -> pd.DataFrame:
    df = call("fund_daily", ts_code=code, start_date=days[0], end_date=days[-1])
    return df.sort_values("trade_date").reset_index(drop=True)


def _contracts() -> pd.DataFrame:
    frames = []
    for ex in ["SSE", "SZSE"]:
        b = call("opt_basic", exchange=ex)
        frames.append(b)
    b = pd.concat(frames, ignore_index=True)
    b["underlying"] = b["opt_code"].str.replace("OP", "", regex=False)
    b["adjusted"] = pd.to_numeric(b["per_unit"], errors="coerce") != 10000
    return b


def _opt_daily(days: list[str]) -> pd.DataFrame:
    sse = cached_by_date("opt_daily_sse", "opt_daily", days, keep=80, exchange="SSE")
    szse = cached_by_date("opt_daily_szse", "opt_daily", days, keep=80, exchange="SZSE")
    return pd.concat([sse, szse], ignore_index=True)


# ---------- 计算 ----------
def trend_state(h: pd.DataFrame) -> dict:
    c, v = h["close"].astype(float), h["vol"].astype(float)
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    vol20 = v.rolling(20).mean()
    last = len(c) - 1
    up = c[last] > ma20[last] > ma60[last] and ma20[last] > ma20[last - 5]
    down = c[last] < ma20[last] < ma60[last] and ma20[last] < ma20[last - 5]
    side = np.sign(c - ma20)
    crosses = int((side.iloc[-10:].diff().abs() > 0).sum())
    state = "上升" if up else "下降" if down else "震荡"
    if crosses >= 3:
        state = "震荡"

    hi20 = c.iloc[-21:-1].max()   # 不含今天的前 20 日最高收盘
    lo20 = c.iloc[-21:-1].min()
    hh20 = h["high"].astype(float).iloc[-21:-1].max()
    ll20 = h["low"].astype(float).iloc[-21:-1].min()
    vr = v[last] / vol20[last - 1]
    pct = (c[last] / c[last - 1] - 1) * 100

    triggers = []
    if c[last] > hi20 and vr >= 1.3:
        triggers.append("放量突破20日高点")
    if c[last] < lo20 and vr >= 1.3:
        triggers.append("放量跌破20日低点")
    lows3 = h["low"].astype(float).iloc[-3:]
    highs3 = h["high"].astype(float).iloc[-3:]
    closes3 = c.iloc[-4:-1]
    if state == "上升" and (lows3 <= ma20.iloc[-3:] * 1.01).any() and (closes3 <= ma20.iloc[-4:-1] * 1.015).any() \
            and c[last] > c[last - 1]:
        triggers.append("回调到20日线后转向上")
    if state == "下降" and (highs3 >= ma20.iloc[-3:] * 0.99).any() and (closes3 >= ma20.iloc[-4:-1] * 0.985).any() \
            and c[last] < c[last - 1]:
        triggers.append("反弹到20日线后转向下")
    if pct >= 2:
        triggers.append(f"大阳线（+{pct:.1f}%）")
    if pct <= -2:
        triggers.append(f"大阴线（{pct:.1f}%）")

    f = lambda x, n=3: round(float(x), n)  # noqa: E731
    return {
        "close": f(c[last]), "pct": f(pct, 2),
        "ma20": f(ma20[last]), "ma60": f(ma60[last]),
        "ma20_slope5": f((ma20[last] / ma20[last - 5] - 1) * 100, 2),
        "cross_ma20_10d": crosses,
        "high20_close": f(hi20), "low20_close": f(lo20),
        "high20": f(hh20), "low20": f(ll20),
        "vol_ratio": f(vr, 2), "state": state, "triggers": triggers,
    }


def _atm_iv(chain: pd.DataFrame, S: float, T: float, r: float) -> float:
    """平值隐含波动率：最接近标的价格的行权价，认购和认沽 IV 的平均。"""
    if chain.empty:
        return np.nan
    k = chain.iloc[(chain["exercise_price"] - S).abs().argsort()[:1]]["exercise_price"].iloc[0]
    ivs = []
    for cp in ["C", "P"]:
        row = chain[(chain.exercise_price == k) & (chain.call_put == cp)]
        if len(row):
            px = float(row["close"].iloc[0]) or float(row["settle"].iloc[0])
            ivs.append(implied_vol(px, S, k, T, r, cp))
    ivs = [x for x in ivs if np.isfinite(x)]
    return float(np.mean(ivs)) if ivs else np.nan


def _months(meta_u: pd.DataFrame, d: str) -> list[str]:
    m = sorted(meta_u.loc[meta_u.maturity_date >= d, "maturity_date"].unique())
    return m[:2]  # 当月、次月


def option_panel(days: list[str]) -> dict:
    d = days[-1]
    dt = datetime.strptime(d, "%Y%m%d")
    r = _rate(d)
    meta = _contracts()
    od = _opt_daily(days[-61:])
    od["trade_date"] = od["trade_date"].astype(str)
    full = od.merge(meta[["ts_code", "underlying", "call_put", "exercise_price", "maturity_date", "adjusted", "name"]],
                    on="ts_code", how="inner")
    full["exercise_price"] = full["exercise_price"].astype(float)

    out = {"date": d, "rate": round(r * 100, 2), "underlyings": []}
    for code, nm in UNDERLYINGS.items():
        h = _etf_hist(code, days)
        if len(h) < 65:
            continue
        ts_ = trend_state(h)
        S = ts_["close"]
        mu = meta[(meta.underlying == code)]
        months = _months(mu, d)
        today = full[(full.underlying == code) & (full.trade_date == d)]

        # 到期日历
        cal = []
        for m in months:
            days_left = (datetime.strptime(m, "%Y%m%d") - dt).days
            cal.append({"maturity": m, "days_left": days_left,
                        "note": "不新开" if days_left < 10 else ("适合" if 20 <= days_left <= 45 else "")})

        # 平值 IV：当月、次月；以及次月 IV 的 60 日历史分位
        iv = {}
        for m in months:
            T = max((datetime.strptime(m, "%Y%m%d") - dt).days, 1) / 365
            ch = today[(today.maturity_date == m) & (~today.adjusted)]
            iv[m] = _atm_iv(ch, S, T, r)
        iv_hist, iv_dates = [], []
        closes = h.set_index("trade_date")["close"].astype(float)
        for hd in days[-60:]:
            if hd not in closes.index:
                continue
            hdt = datetime.strptime(hd, "%Y%m%d")
            sub = full[(full.underlying == code) & (full.trade_date == hd) & (~full.adjusted)]
            mats = sorted(m for m in sub.maturity_date.unique() if (datetime.strptime(m, "%Y%m%d") - hdt).days >= 15)
            if not mats:
                continue
            m0 = mats[0]
            T = (datetime.strptime(m0, "%Y%m%d") - hdt).days / 365
            iv_hist.append(_atm_iv(sub[sub.maturity_date == m0], closes[hd], T, r))
            iv_dates.append(hd)
        iv_series = {dd: round(x * 100, 2) for dd, x in zip(iv_dates, iv_hist) if np.isfinite(x)}
        iv_hist = [x for x in iv_hist if np.isfinite(x)]
        ref_m = next((c["maturity"] for c in cal if c["days_left"] >= 15), months[-1] if months else None)
        iv_now = iv.get(ref_m, np.nan)
        pctile = (float(np.mean(np.array(iv_hist) <= iv_now)) * 100) if iv_hist and np.isfinite(iv_now) else np.nan
        iv_level = ("偏低" if pctile < 30 else "偏高" if pctile > 70 else "中") if np.isfinite(pctile) else "未获取"

        # PCR、最大持仓行权价（当月+次月，不含调整合约）
        near = today[today.maturity_date.isin(months)]
        cv, pv = near.loc[near.call_put == "C", "vol"].sum(), near.loc[near.call_put == "P", "vol"].sum()
        co, po = near.loc[near.call_put == "C", "oi"].sum(), near.loc[near.call_put == "P", "oi"].sum()
        std = near[~near.adjusted]
        oi_by_k = std.groupby(["call_put", "exercise_price"])["oi"].sum().reset_index()
        maxc = oi_by_k[oi_by_k.call_put == "C"].nlargest(1, "oi")
        maxp = oi_by_k[oi_by_k.call_put == "P"].nlargest(1, "oi")

        # 候选合约：剩余 20–45 天的月份；没有的话，把剩余 ≥10 天的月份都列出来，由简报判断
        cand_ms = [c["maturity"] for c in cal if 20 <= c["days_left"] <= 45] \
            or [c["maturity"] for c in cal if c["days_left"] >= 10]
        cands = []
        for cand_m in cand_ms:
            T = (datetime.strptime(cand_m, "%Y%m%d") - dt).days / 365
            ch = today[(today.maturity_date == cand_m) & (~today.adjusted)]
            for row in ch.itertuples():
                px = float(row.close) or float(row.settle)
                sig = implied_vol(px, S, row.exercise_price, T, r, row.call_put)
                dl = bs_delta(S, row.exercise_price, T, r, sig, row.call_put)
                if np.isfinite(dl) and 0.4 <= abs(dl) <= 0.6:
                    cands.append({"maturity": cand_m, "days_left": (datetime.strptime(cand_m, "%Y%m%d") - dt).days,
                                  "ts_code": row.ts_code, "name": row.name, "cp": "认购" if row.call_put == "C" else "认沽",
                                  "strike": row.exercise_price, "price": px, "iv": round(sig * 100, 1),
                                  "delta": round(dl, 2), "oi": int(row.oi), "vol": int(row.vol),
                                  "cost_per_lot": round(px * 10000, 0)})
        out["underlyings"].append({
            "code": code, "name": f"{nm}（{code[:6]}）", **ts_,
            "calendar": cal,
            "atm_iv": {m: (round(v * 100, 1) if np.isfinite(v) else None) for m, v in iv.items()},
            "iv_ref_month": ref_m, "iv_pctile_60d": round(pctile, 0) if np.isfinite(pctile) else None,
            "iv_mean_60d": round(float(np.mean(iv_hist)) * 100, 1) if iv_hist else None,
            "iv_level": iv_level, "iv_series_60d": iv_series,
            "pcr_vol": round(pv / cv, 2) if cv else None, "pcr_oi": round(po / co, 2) if co else None,
            "max_oi_call_strike": float(maxc.exercise_price.iloc[0]) if len(maxc) else None,
            "max_oi_put_strike": float(maxp.exercise_price.iloc[0]) if len(maxp) else None,
            "candidate_months": cand_ms,
            "candidates": sorted(cands, key=lambda x: (x["maturity"], x["cp"], x["strike"])),
        })
    return out
