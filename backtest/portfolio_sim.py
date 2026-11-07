# 组合层面模拟：在逐品种交易清单基础上，按账户统一处理持仓上限、同向上限、单笔风险比例、复利和风控分档。
# 交易清单由 portfolio_gen.py 生成（本地数据缓存版）。
import pandas as pd, numpy as np, pickle, itertools
TL=pickle.load(open("/tmp/tl.pkl","rb"))
D0,D1=pd.Timestamp("2020-04-01"),pd.Timestamp("2026-11-06")
def sim(T, r=0.02, compound=True, max_pos=3, max_same=2, pri=("科创50ETF","创业板ETF","500ETF","300ETF","50ETF"), excl=(), tiers=True):
    T=T[~T.etf.isin(excl)].copy(); T["pri"]=T.etf.map({e:i for i,e in enumerate(pri)}); T=T.sort_values(["open","pri"])
    eq=100000.0; peak=eq; open_=[]; acc=[]; pause_until=None; hist=[]
    events=[]
    rows=list(T.itertuples())
    # 先把所有平仓事件按日期处理：逐个开仓日推进
    pending=[]  # (close_date, pnl_scaled, dir)
    def settle(upto):
        nonlocal eq, peak, pending
        pending.sort(key=lambda x:x[0])
        keep=[]
        for c in pending:
            if c[0] <= upto:
                eq += c[1]; peak=max(peak,eq); hist.append((c[0],eq))
            else: keep.append(c)
        pending=keep
    for t in rows:
        settle(t.open)  # 当天之前（含当天收盘前）平掉的先结算
        act=[p for p in pending if p[0] > t.open]
        if len(act)>=max_pos or sum(p[2]==t.dir for p in act)>=max_same: continue
        dd=(peak-eq)/peak
        f=1.0
        if tiers:
            k=r/0.01
            if pause_until and t.open < pause_until: continue
            if dd>=0.10*k:
                pause_until=(pd.Timestamp(t.open)+pd.Timedelta(days=14)).strftime("%Y%m%d"); peak=eq; continue
            f=1.0 if dd<0.05*k else 0.5 if dd<0.07*k else 0.3
        basis= eq if compound else 100000
        scale= r*basis*f/1000.0
        pending.append((t.close, t.pnl*scale, t.dir)); acc.append((t.open,t.close,t.etf,t.strat,t.dir,t.pnl*scale))
    settle("99999999")
    A=pd.DataFrame(acc,columns=["open","close","etf","strat","dir","pnl"])
    H=pd.DataFrame(hist,columns=["d","eq"]).groupby("d").eq.last()
    yrs=(D1-D0).days/365
    cagr=((eq/100000)**(1/yrs)-1)*100
    mdd=((H.cummax()-H)/H.cummax()).max()*100 if len(H) else 0
    yeq=H.groupby(H.index.str[:4]).last(); yeq=pd.concat([pd.Series({"2019":100000.0}),yeq]); yret=(yeq.pct_change().dropna()*100).round(1)
    o=H[H.index<"20230101"]; oos=(((o.iloc[-1]/100000)**(1/((pd.Timestamp("2022-12-31")-D0).days/365))-1)*100) if len(o) else 0
    ex=((np.prod([1+x/100 for x in yret.drop(yret.idxmax())]))**(1/(yrs-1))-1)*100
    return dict(年化=round(cagr,1),样本外20_22=round(oos,1),去掉最好一年=round(ex,1),最大回撤=round(mdd,1),最差年=yret.min(),亏损年数=int((yret<0).sum()),笔数=len(A)), yret, A
if __name__=="__main__":
    rows=[]
    for (tr,s4),T in TL.items():
        for r,comp,mp,ms in itertools.product([0.02,0.025,0.03],[False,True],[3,4],[2,3]):
            if ms>mp: continue
            m,_,_=sim(T,r,comp,mp,ms)
            rows.append({"止盈":tr,"S4":s4,"单笔风险":r,"复利":comp,"最多持仓":mp,"同向最多":ms,**m})
    R=pd.DataFrame(rows); R.to_csv("/tmp/pgrid.csv",index=False)
    print(R.sort_values("年化",ascending=False).head(20).to_string())
    for c in ["止盈","S4","单笔风险","复利","最多持仓","同向最多"]: print(c, R.groupby(c)[["年化","样本外20_22","最大回撤"]].mean().round(1).to_dict("index"))
