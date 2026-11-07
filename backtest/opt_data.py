"""回测用历史数据：ETF 日线、期权合约表、期权日线（按年缓存到 data/bt/）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from md.client import DATA, call  # noqa: E402

BT = DATA / "bt"
BT.mkdir(exist_ok=True)


def trade_days(start: str, end: str) -> list[str]:
    cal = call("trade_cal", exchange="SSE", start_date=start, end_date=end, is_open="1")
    return sorted(cal["cal_date"].astype(str).tolist())


def etf_daily(code: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    s = start
    while s <= end:  # fund_daily 单次最多返回约 2000 行，分段取
        e = str(min(int(end), int(s[:4]) * 10000 + 1231))
        frames.append(call("fund_daily", ts_code=code, start_date=s, end_date=e))
        s = str(int(s[:4]) + 1) + "0101"
    df = pd.concat(frames).drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    return df


def contracts() -> pd.DataFrame:
    path = BT / "opt_basic.parquet"
    frames = [call("opt_basic", exchange=ex) for ex in ["SSE", "SZSE"]]
    b = pd.concat(frames, ignore_index=True)
    b["underlying"] = b["opt_code"].str.replace("OP", "", regex=False)
    b["adjusted"] = pd.to_numeric(b["per_unit"], errors="coerce") != 10000
    b["exercise_price"] = b["exercise_price"].astype(float)
    b.to_parquet(path, index=False)
    return b


def opt_daily(days: list[str]) -> pd.DataFrame:
    """按年缓存。已缓存的年份只补缺失日期。"""
    out = []
    years = sorted({d[:4] for d in days})
    for y in years:
        yd = [d for d in days if d.startswith(y)]
        path = BT / f"opt_daily_{y}.parquet"
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["trade_date"])
        have = set(old["trade_date"].astype(str)) if len(old) else set()
        new = []
        for d in yd:
            if d in have:
                continue
            for ex in ["SSE", "SZSE"]:
                df = call("opt_daily", trade_date=d, exchange=ex)
                if len(df):
                    new.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "settle", "vol", "oi"]])
            if d.endswith(("0331", "0630", "0930", "1231")) or d == yd[-1]:
                print(f"  {y}: 已下载到 {d}", flush=True)
        if new:
            old = pd.concat([old] + new, ignore_index=True)
            old["trade_date"] = old["trade_date"].astype(str)
            old.to_parquet(path, index=False)
        out.append(old[old["trade_date"].astype(str).isin(yd)])
    df = pd.concat(out, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    for c in ["open", "high", "low", "close", "settle", "vol", "oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df
