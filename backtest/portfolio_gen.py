# 生成组合模拟用的逐品种交易清单（单笔风险固定1000元、不分档）。
# 生成每个品种、每种规则组合下的原始交易清单（单笔风险固定 1000 元，不分档），供组合层面再筛选和放大
import sys; sys.path.insert(0,'/home/claude/market-data')
import pandas as pd, pickle, itertools
from backtest import opt_backtest as v1
from backtest import kc50_backtest as K
meta,data=pickle.load(open("/tmp/d5.pkl","rb"))
rec=dict(pullback=True, iv_skip=True, iv_long=False, iv_short=True, vol_mult=1.2, s2="off")
base={**rec,"s3":False,"s4":True,"bear_only":True,"trail_short":"mix","short_stop_pct":0.6,"tstop_short":5,"l4":True,"l1_relax":True,"tiers_off":True}
out={}
for tr in ["mix","ma20","adapt","ma5"]:
    for s4 in [True,False]:
        T=[]
        for name,(df,mu,o,days,ivp,first) in data.items():
            u=K.frame(df); book=v1.Book(o,meta)
            st=max("20200401",first); i0=max(0,next(i for i,d in enumerate(days) if d>=st)-60)
            r=K.run_v2({**base,"trail":tr,"s4":s4},days[i0:],u,book,mu,ivp); t=pd.DataFrame(r["trades"]); t["etf"]=name; T.append(t)
        T=pd.concat(T); T["open"]=T.open.astype(str); T["close"]=T.close.astype(str)
        out[(tr,s4)]=T; print(tr,s4,len(T),round(T.pnl.sum()))
pickle.dump(out,open("/tmp/tl.pkl","wb"))
