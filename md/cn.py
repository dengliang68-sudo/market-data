"""A股、港股：大盘温度、板块扫描、资金流。"""
from __future__ import annotations

import pandas as pd

from .client import DATA, cached_by_date, call
from .signals import scan

TECH_KW = ["半导体", "芯片", "集成电路", "光模块", "CPO", "PCB", "印制电路", "算力", "人工智能", "AI", "机器人",
           "服务器", "存储", "液冷", "消费电子", "软件", "云计算", "通信设备", "电子", "计算机", "通信",
           "商业航天", "低空", "量子", "智能驾驶", "数据中心", "光通信"]
PHARMA_KW = ["医药", "创新药", "CXO", "医疗", "生物", "化学制药", "中药", "减肥药", "GLP", "疫苗", "器械"]


def _tag(name: str) -> str:
    if any(k in name for k in PHARMA_KW):
        return "医药"
    if any(k in name for k in TECH_KW):
        return "科技"
    return "其他"


def market_temperature(days: list[str]) -> dict:
    d = days[-1]
    idx_codes = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指",
                 "000688.SH": "科创50", "000300.SH": "沪深300"}
    idx = []
    for code, nm in idx_codes.items():
        df = call("index_daily", ts_code=code, start_date=days[-25], end_date=d).sort_values("trade_date")
        if len(df):
            idx.append({"name": nm, "close": round(df.close.iloc[-1], 2), "pct": round(df.pct_chg.iloc[-1], 2),
                        "amount_yi": round(df.amount.iloc[-1] / 1e5, 0)})  # 千元 → 亿元
    # 两市成交额：上证 + 深证成指的成交额
    sh = call("index_daily", ts_code="000001.SH", start_date=days[-25], end_date=d).set_index("trade_date")["amount"]
    sz = call("index_daily", ts_code="399106.SZ", start_date=days[-25], end_date=d).set_index("trade_date")["amount"]  # 深证综指=深市全部
    tot = (sh + sz).sort_index() / 1e5
    amount_today = round(tot.iloc[-1], 0)
    amount_ratio = round(tot.iloc[-1] / tot.iloc[-21:-1].mean(), 2)

    daily = call("daily", trade_date=d)
    up, down = int((daily.pct_chg > 0).sum()), int((daily.pct_chg < 0).sum())
    lim = call("limit_list_d", trade_date=d)
    zt = int((lim.limit == "U").sum()) if len(lim) else 0
    dt = int((lim.limit == "D").sum()) if len(lim) else 0
    zb = int((lim.limit == "Z").sum()) if len(lim) else 0
    top_board = int(lim.loc[lim.limit == "U", "limit_times"].max()) if zt else 0

    mg = call("margin", start_date=days[-6], end_date=d)
    mg_tot = mg.groupby("trade_date")["rzrqye"].sum().sort_index() / 1e8
    gg = call("ggt_daily", start_date=days[-6], end_date=d).sort_values("trade_date")

    return {
        "date": d, "indices": idx,
        "amount_yi": amount_today, "amount_vs_20d": amount_ratio,
        "up": up, "down": down, "limit_up": zt, "limit_down": dt, "broken_board": zb, "max_board": top_board,
        "margin_yi": {k: round(v, 0) for k, v in mg_tot.items()},
        # 港股通（南向）买入减卖出，单位亿元，ggt_daily 口径，通常比 A股晚一天更新
        "south_net_yi": [(r.trade_date, round(float(r.buy_amount) - float(r.sell_amount), 1)) for r in gg.itertuples()],
    }


def sector_scan(days: list[str]) -> pd.DataFrame:
    """申万一、二级行业 + 同花顺行业/概念指数，统一扫描。"""
    cls = call("index_classify", src="SW2021")
    sw_codes = set(cls.loc[cls.level.isin(["L1", "L2"]), "index_code"])
    sw = cached_by_date("sw_daily", "sw_daily", days, keep=180)
    sw = sw[sw.ts_code.isin(sw_codes)][["trade_date", "ts_code", "name", "close", "pct_change", "vol"]]
    sw["src"] = "申万"

    meta_path = DATA / "ths_index.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
    else:
        meta = call("ths_index", exchange="A")
        meta.to_parquet(meta_path, index=False)
    meta = meta[meta.type.isin(["N", "I"])]
    ths = cached_by_date("ths_daily", "ths_daily", days, keep=180)
    ths = ths.merge(meta[["ts_code", "name", "type"]], on="ts_code")
    ths = ths[["trade_date", "ts_code", "name", "close", "pct_change", "vol", "type"]]
    ths["src"] = ths["type"].map({"N": "同花顺概念", "I": "同花顺行业"})

    frames = []
    for src, df in [("申万", sw), ("同花顺", ths)]:
        df = df[df.trade_date.isin(days)]
        res = scan(df.drop(columns=[c for c in ["type", "src"] if c in df]))
        if len(res):
            res = res.merge(df.drop_duplicates("ts_code")[["ts_code", "src"]], on="ts_code")
            frames.append(res)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["name"], keep="first")  # 同名指数只留一个（申万优先）
    out["group"] = out["name"].map(_tag)
    return out


def industry_flow(days: list[str]) -> pd.DataFrame:
    """东财行业/概念资金流：最近 5 日净流入天数和合计。"""
    df = cached_by_date("moneyflow_ind_dc", "moneyflow_ind_dc", days[-5:], keep=30)
    df = df[df.trade_date.isin(days[-5:])]
    g = df.groupby(["content_type", "name"]).agg(
        days_in=("net_amount", lambda s: int((s > 0).sum())),
        net5_yi=("net_amount", lambda s: round(s.sum() / 1e8, 1)),
    ).reset_index()
    today = df[df.trade_date == days[-1]][["content_type", "name", "net_amount", "pct_change"]]
    today = today.assign(net_today_yi=(today.net_amount / 1e8).round(1)).drop(columns="net_amount")
    return g.merge(today, on=["content_type", "name"], how="left")


def hk_snapshot(days: list[str]) -> list[dict]:
    out = []
    for code, nm in {"HSI": "恒生指数", "HKTECH": "恒生科技"}.items():
        df = call("index_global", ts_code=code, start_date=days[-10], end_date=days[-1]).sort_values("trade_date")
        if len(df):
            out.append({"name": nm, "date": df.trade_date.iloc[-1], "close": round(df.close.iloc[-1], 2),
                        "pct": round(df.pct_chg.iloc[-1], 2)})
    return out
