"""板块趋势雷达：按《复盘系统说明》第五节的规则给每个指数分级。

输入是一张长表（trade_date, ts_code, name, close, pct_change, vol），
输出每个指数在最新交易日的指标和等级。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["🟢可介入", "🟡准备", "⚪观察", "🔴退潮警示", ""]


def _one(g: pd.DataFrame, rank_top: pd.Series) -> dict | None:
    g = g.sort_values("trade_date")
    if len(g) < 65:
        return None
    c = g["close"].astype(float).values
    v = g["vol"].astype(float).values
    pct = g["pct_change"].astype(float).values
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    vol20 = pd.Series(v).rolling(20).mean().values
    last = -1

    # 观察级条件
    top3 = bool(rank_top.loc[g.index[-3:]].all())                      # 连续 3 日涨幅前 10%
    vol_ratio = v[last] / vol20[last - 1] if vol20[last - 1] else np.nan   # 今日量 / 前 20 日均量
    vol_burst = bool(vol_ratio >= 1.5)
    obs_hits = int(top3) + int(vol_burst)  # 第三条"明确催化"需要人工判断

    # 准备级条件
    above = bool(c[last] > ma20[last] and c[last] > ma60[last] and ma20[last] > ma20[last - 5])
    high60 = bool(c[last] >= np.nanmax(c[-60:]))
    prep_hits = int(above) + int(high60)

    # 可介入：近 15 日内出现过 60 日新高 → 之后缩量回调不破 20 日线 → 今天放量上涨
    go = False
    win = 15
    for k in range(len(c) - win, len(c) - 2):
        if c[k] >= np.nanmax(c[max(0, k - 59): k + 1]) and v[k] >= 1.3 * vol20[k - 1]:
            after = slice(k + 1, len(c) - 1)
            pulled = c[after].size > 0 and np.min(c[after]) < c[k]
            held = c[after].size > 0 and np.all(c[after] >= ma20[after] * 0.995)
            shrink = c[after].size > 0 and np.mean(v[after]) < v[k]
            today_up = pct[last] > 0 and v[last] > np.mean(v[after]) * 1.2
            if pulled and held and shrink and today_up:
                go = True
                break

    # 退潮：板块处在高位（近 20 日内创过 60 日新高），今天放量跌破 20 日线，或放量大跌
    was_strong = bool(np.nanmax(c[-20:-1]) >= np.nanmax(c[-60:-20]))
    broke = c[last] < ma20[last] and v[last] > 1.2 * vol20[last - 1]
    big_drop = pct[last] <= -4 and v[last] > 1.2 * vol20[last - 1]
    # 持续退潮：近 40 日内创过约半年新高，现已跌破 20 日线、20 日线拐头向下、距 40 日高点回撤超过 10%
    hi40 = np.nanmax(c[-40:])
    ongoing = (hi40 >= np.nanmax(c[-120:]) * 0.98 and c[last] < ma20[last]
               and ma20[last] < ma20[last - 5] and c[last] / hi40 - 1 <= -0.10)
    fresh = was_strong and (broke or big_drop)
    ebb = fresh or ongoing

    if ebb:
        tier = "🔴退潮警示"
    elif obs_hits >= 1 and prep_hits >= 2 and go:
        tier = "🟢可介入"
    elif obs_hits >= 1 and prep_hits >= 2:
        tier = "🟡准备"
    elif obs_hits >= 2 or (obs_hits >= 1 and prep_hits >= 1):
        tier = "⚪观察"
    else:
        tier = ""

    return {
        "ts_code": g["ts_code"].iloc[-1],
        "name": g["name"].iloc[-1],
        "close": round(c[last], 2),
        "pct": round(pct[last], 2),
        "pct5": round((c[last] / c[-6] - 1) * 100, 2),
        "pct20": round((c[last] / c[-21] - 1) * 100, 2),
        "vol_ratio": round(float(vol_ratio), 2) if np.isfinite(vol_ratio) else None,
        "above_ma20_60": above,
        "high60": high60,
        "top10pct_3d": top3,
        "dist_ma20_pct": round((c[last] / ma20[last] - 1) * 100, 2),
        "dd_from_40d_high": round((c[last] / hi40 - 1) * 100, 2),
        "tier": tier,
        "ebb_type": ("今日新出现" if fresh else "持续退潮") if tier == "🔴退潮警示" else "",
    }


def scan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["close"]).copy()
    df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
    # 每日涨幅是否排进前 10%
    df["rank_top"] = df.groupby("trade_date")["pct_change"].rank(pct=True, ascending=False) <= 0.10
    rows = []
    for _, g in df.groupby("ts_code"):
        r = _one(g, df.loc[g.index, "rank_top"])
        if r:
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["tier_order"] = out["tier"].map({t: i for i, t in enumerate(TIERS)})
        out = out.sort_values(["tier_order", "pct5"], ascending=[True, False]).drop(columns="tier_order")
    return out
