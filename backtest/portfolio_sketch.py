# 组合回测草稿：五个品种同一账户、最多3笔、同方向最多2笔（按逐品种回测的交易清单贪心筛选，回撤按平仓盈亏计）。本地数据缓存版，用于快速比较。
import sys; sys.path.insert(0,'/home/claude/market-data')
import pandas as pd, numpy as np, pickle
from backtest import opt_backtest as v1
from backtest import kc50_backtest as K
meta,data=pickle.load(open("/tmp/d5.pkl","rb"))
rec=dict(pullback=True, iv_skip=True, iv_long=False, iv_short=True, vol_mult=1.2, s2="off")
base={**rec,"s3":False,"s4":True,"bear_only":True,"trail_short":"mix","short_stop_pct":0.6,"tstop_short":5,"l4":True,"l1_relax":True}
PRI=["科创50ETF","创业板ETF","500ETF","300ETF","50ETF"]
rows=[]; yrs={}
for tr,lab in [("ma5","5日线"),("mix","一半5日线一半20日线"),("ma20","20日线")]:
  for k in [1,2,3]:
    T=[]
    for name in PRI:
        df,mu,o,days,ivp,first=data[name]; u=K.frame(df); book=v1.Book(o,meta)
        st=max("20200401",first); i0=max(0,next(i for i,d in enumerate(days) if d>=st)-60)
        r=K.run_v2({**base,"trail":tr,"risk_k":k},days[i0:],u,book,mu,ivp); t=pd.DataFrame(r["trades"]); t["etf"]=name; t["pri"]=PRI.index(name); T.append(t)
    T=pd.concat(T); T["open"]=T.open.astype(str); T["close"]=T.close.astype(str); T=T.sort_values(["open","pri"])
    acc=[]; 
    for r_ in T.itertuples():
        act=[a for a in acc if a.open<=r_.open<a.close]
        if len(act)>=3 or sum(a.dir==r_.dir for a in act)>=2: continue
        acc.append(r_)
    A=pd.DataFrame(acc); A["y"]=A.open.str[:4]
    eq=100000+A.sort_values("close").groupby("close").pnl.sum().cumsum()
    dd=((eq.cummax()-eq)/eq.cummax()).max()*100
    yrs_n=(pd.to_datetime("20261106")-pd.to_datetime("20200401")).days/365
    tot=A.pnl.sum(); cagr=(((100000+tot)/100000)**(1/yrs_n)-1)*100
    y=A.groupby("y").pnl.sum().round().astype(int)
    rows.append({"多单止盈":lab,"单笔风险":f"{k}%","容忍回撤":f"{10*k}%","笔数":len(A),"总盈亏":round(tot),"年化%":round(cagr,1),"最大回撤%(按平仓)":round(dd,1),"亏损年数":int((y<0).sum()),"去掉2024年的年化%":round((((100000+tot-y.get('2024',0))/100000)**(1/(yrs_n-1))-1)*100,1)})
    yrs[(lab,k)]=y.to_dict()
print(pd.DataFrame(rows).to_string())
for kk,v in yrs.items(): print(kk,v)
