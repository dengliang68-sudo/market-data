"""Tushare / FRED 数据访问层：带重试、限速和本地缓存。"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import tushare as ts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

_pro = None


def pro():
    global _pro
    if _pro is None:
        token = os.environ.get("TS_TOKEN")
        if not token:
            raise RuntimeError("缺少环境变量 TS_TOKEN（Tushare Key）")
        _pro = ts.pro_api(token)
    return _pro


def call(api: str, retries: int = 4, **kw) -> pd.DataFrame:
    """调用 Tushare 接口，遇到频率限制或网络错误自动等待重试。"""
    last = None
    for i in range(retries):
        try:
            df = getattr(pro(), api)(**kw)
            time.sleep(0.12)
            return df if df is not None else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 65 if "频率" in str(e) or "超限" in str(e) else 3 * (i + 1)
            time.sleep(wait)
    raise RuntimeError(f"{api} 调用失败：{last}")


def trade_days(n: int, end: str | None = None, exchange: str = "SSE") -> list[str]:
    """返回截至 end（含）最近 n 个交易日，升序。"""
    end = end or datetime.now().strftime("%Y%m%d")
    start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=int(n * 1.6) + 30)).strftime("%Y%m%d")
    cal = call("trade_cal", exchange=exchange, start_date=start, end_date=end, is_open="1")
    days = sorted(cal["cal_date"].tolist())
    return days[-n:]


def cached_by_date(name: str, api: str, days: list[str], keep: int = 320, **kw) -> pd.DataFrame:
    """按交易日增量缓存：已有日期不再请求，只补缺失日期。缓存保留最近 keep 个交易日。"""
    path = DATA / f"{name}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["trade_date"])
    have = set(old["trade_date"].astype(str)) if len(old) else set()
    # 最近 2 个交易日总是重新拉，防止盘后数据修正
    need = [d for d in days if d not in have] + [d for d in days[-2:] if d in have]
    frames = [old[~old["trade_date"].astype(str).isin(days[-2:])]] if len(old) else []
    for d in need:
        df = call(api, trade_date=d, **kw)
        if len(df):
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else old
    out["trade_date"] = out["trade_date"].astype(str)
    keep_days = sorted(out["trade_date"].unique())[-keep:]
    out = out[out["trade_date"].isin(keep_days)].reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def fred(series: list[str], years: int = 2) -> pd.DataFrame:
    """从 FRED 下载日度序列（无需 Key），返回以日期为索引的宽表。"""
    cols = []
    for s in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        cols.append(df.set_index("date")[[s]].apply(pd.to_numeric, errors="coerce"))
    out = pd.concat(cols, axis=1).sort_index()
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    return out[out.index >= cutoff]
