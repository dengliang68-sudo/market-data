"""美股指数基金模块：温度计、回撤加投档位、卖出规则检查。数据来自 FRED（标普500、纳指100、VIX、美债10Y）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .client import fred

TIERS = [-5, -10, -15, -20]  # 回撤加投档位（距 52 周高点 %）


def _index_stats(s: pd.Series) -> dict:
    s = s.dropna()
    last = s.iloc[-1]
    hi52 = s[s.index >= s.index[-1] - pd.Timedelta(days=365)].max()
    ma50 = s.rolling(50).mean().iloc[-1]
    ma200 = s.rolling(200).mean().iloc[-1]
    y1 = s[s.index <= s.index[-1] - pd.Timedelta(days=365)]
    ret12m = (last / y1.iloc[-1] - 1) * 100 if len(y1) else np.nan
    return {
        "date": s.index[-1].strftime("%Y-%m-%d"),
        "close": round(last, 2),
        "pct": round((last / s.iloc[-2] - 1) * 100, 2),
        "high_52w": round(hi52, 2),
        "drawdown_52w": round((last / hi52 - 1) * 100, 2),
        "vs_ma50": round((last / ma50 - 1) * 100, 2),
        "vs_ma200": round((last / ma200 - 1) * 100, 2),
        "ret_12m": round(ret12m, 1) if np.isfinite(ret12m) else None,
    }


def index_fund_panel() -> dict:
    df = fred(["SP500", "NASDAQ100", "VIXCLS", "DGS10"], years=3)
    spx, ndx = _index_stats(df["SP500"]), _index_stats(df["NASDAQ100"])
    vix = df["VIXCLS"].dropna()
    t10 = df["DGS10"].dropna()

    dd = spx["drawdown_52w"]
    hit = [t for t in TIERS if dd <= t]
    if dd <= -10 or vix.iloc[-1] >= 25:
        temp = "偏冷"
    elif dd >= -2 and spx["vs_ma50"] >= 4 and vix.iloc[-1] < 15:
        temp = "偏热"
    else:
        temp = "中性"

    # 卖出规则第 3 条（过热小幅止盈）中能自动判断的两项；估值分位需另行查证
    overheat_partial = {
        "12个月涨幅>30%": bool(spx["ret_12m"] is not None and spx["ret_12m"] > 30),
        "跌破200日线": bool(spx["vs_ma200"] < 0),
    }
    return {
        "spx": spx, "ndx": ndx,
        "vix": round(float(vix.iloc[-1]), 2), "vix_date": vix.index[-1].strftime("%Y-%m-%d"),
        "us10y": round(float(t10.iloc[-1]), 2), "us10y_date": t10.index[-1].strftime("%Y-%m-%d"),
        "temperature": temp,
        "drawdown_tiers_reached": hit,
        "overheat_partial": overheat_partial,
    }
