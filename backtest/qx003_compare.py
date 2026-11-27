"""QX-003: all discretionary exits confirmed at close and filled next open."""
from pathlib import Path
import sys
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import kc50_backtest as K, recommended_portfolio as RP, opt_backtest as V
from backtest.qx_compare import blocked_network, clean_trades, marked_curve
from md import client

OUT = ROOT / 'reports/backtest/qx003_20261127'
RULE = {**K.RECOMMENDED, 'close_only': True, 'first_profit': 'step_close', 'research_trace': True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client.call = blocked_network
    data, _ = RP.load()
    sets = {'QX-001': [], 'QX-003': []}
    for name, (df, mu, od, days, ivp, first, meta) in data.items():
        u, book = K.frame(df), V.Book(od, meta)
        st = max(RP.D0, first)
        i0 = max(0, next(i for i, d in enumerate(days) if d >= st) - 60)
        for key, cfg in [('QX-001', {**K.RECOMMENDED, 'research_trace': True}), ('QX-003', RULE)]:
            r = K.run_v2(cfg, days[i0:], u, book, mu, ivp)
            t = pd.DataFrame(r['trades'])
            t['etf'] = name
            sets[key].append(t)
            print('ENGINE', key, name, len(t), round(t.pnl.sum(), 1), flush=True)
    all_days = sorted(set(d for item in data.values() for d in item[3] if d >= RP.D0))
    metrics, annual, by_signal = [], [], []
    prior = pd.read_csv(ROOT / 'reports/backtest/recommended_trades.csv', dtype={'open':str,'close':str})
    for key, dfs in sets.items():
        raw = pd.concat(dfs, ignore_index=True)
        raw['open'], raw['close'] = raw.open.astype(str), raw.close.astype(str)
        clean_trades(raw).to_csv(OUT / f'{key}_raw_trades.csv', index=False, encoding='utf-8-sig')
        for risk in [.02, .03]:
            m, yr, accepted = RP.sim(raw, risk)
            if key == 'QX-001' and risk == .03:
                cols = ['etf','open','close','strat','lots','pnl','pnl_acct','equity_at_open']
                pd.testing.assert_frame_equal(accepted[cols].reset_index(drop=True),prior[cols].reset_index(drop=True),
                                              check_dtype=False,atol=.11,rtol=0)
                assert m['期末净值'] == 558227 and m['成交笔数'] == 226
                print('BASELINE_REPRODUCED', flush=True)
            daily = marked_curve(accepted, all_days, risk)
            eq = daily.equity_marked
            loss_fraction = -accepted.pnl_acct / accepted.equity_at_open
            m.update({'版本': key, '单笔风险预算%': risk*100,
                      '最大回撤含日末浮盈亏%': round(float(((eq.cummax()-eq)/eq.cummax()).max()*100),2),
                      '最差单笔账户亏损%': round(float(loss_fraction.max()*100),2),
                      '亏损超过单笔风险预算次数': int((loss_fraction > risk + .000001).sum()),
                      '首次阶梯减半次数': int(accepted.exits.str.contains('阶梯回落').sum())})
            metrics.append(m)
            for y, ret in yr.items():
                annual.append({'版本':key,'单笔风险预算%':risk*100,'年份':y,'收益率%':float(ret)})
            for sig,g in accepted.groupby(accepted.strat.str[:2]):
                by_signal.append({'版本':key,'单笔风险预算%':risk*100,'信号':sig,'笔数':len(g),
                                  '盈亏元':round(float(g.pnl_acct.sum()),2),'胜率%':round(float((g.pnl_acct>0).mean()*100),2)})
            tag = f'{key}-R{int(risk*100):02}'
            accepted = clean_trades(accepted)
            accepted['单笔账户损失%'] = loss_fraction*100
            accepted.to_csv(OUT / f'{tag}_trades.csv',index=False,encoding='utf-8-sig')
            daily.to_csv(OUT / f'{tag}_daily.csv',index=False,encoding='utf-8-sig')
            print('RESULT',json.dumps(m,ensure_ascii=False),flush=True)
    s,a,b = pd.DataFrame(metrics),pd.DataFrame(annual),pd.DataFrame(by_signal)
    s.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    a.to_csv(OUT/'annual.csv',index=False,encoding='utf-8-sig')
    b.to_csv(OUT/'by_signal.csv',index=False,encoding='utf-8-sig')
    content={'baseline_reproduced':True,'metrics':metrics,'annual':annual,'by_signal':by_signal,'config':RULE}
    (OUT/'results.json').write_text(json.dumps(content,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data/bt').glob('*.parquet'))}
    (OUT/'provenance.json').write_text(json.dumps({'data_end':max(all_days),'input_sha256':hashes,'config':RULE},indent=2,ensure_ascii=False),encoding='utf-8')
    old=pd.read_csv(ROOT/'reports/backtest/qx_20261127/summary.csv')
    lines=['# QX-003收盘确认、次日开盘执行回测','',
           '初始10万元；沿用2020-04-01至2026-11-06历史缓存。QX-001原基准226笔逐笔复现通过。','',
           '## QX-003规则','',
           '- 权利金止损、保本止损：仅以收盘价确认，次日开盘执行，不使用期权盘中最高或最低价。',
           '- 阶梯首次减半：历次收盘价最高涨幅达到50%后启动；60%→保护55%，70%→65%，依此类推。收盘严格跌破保护线，次日开盘减半。1张不卖，仅进入原保本/跟踪阶段。',
           '- 标的达到2:1目标、均线跟踪、逻辑止损、时间止损均收盘确认、次日开盘执行。',
           '- 移仓也改成收盘选合约、次日开盘平旧买新；未进入跟踪阶段的近到期持仓安排次日退出。长假减仓按已知交易日历提前一晚安排，在节前最后交易日开盘执行。',
           '- 信号、选合约、风险预算、月度最多6次、组合最多4笔/同向3笔、滑点费用及原组合缩放方式不变。',
           '- 样本末尾统一按末日收盘强制结算，仅作回测终值处理，不是日常收盘后能按已知收盘价下单。','',
           '## 结果','',s.to_markdown(index=False),'','## 分年收益（平仓结转口径）','',
           a.pivot(index=['版本','单笔风险预算%'],columns='年份',values='收益率%').reset_index().to_markdown(index=False),'',
           '## QX-002旧对照结果（含日线盘中路径假设）','',old.to_markdown(index=False),'',
           '## 解释限制','',
           '本轮2%/3%为仓位计算的计划风险，不是单笔亏损硬上限。次日开盘可能跳空；最差单笔及超预算次数见上表。',
           '回撤包含日末浮盈亏，仍不是盘中最大回撤。组合沿用按原始1000元风险交易路径比例缩放的方法；不是逐单按实盘资金重新计算整数张，亦不是固定每笔4000元。',
           '开盘报价缺失时旧数据读取器可能以收盘/结算价回退；历史合约条款、移仓等原模型近似仍存在。相同历史窗口不是新样本外验证。',
           '不盯盘指收盘后检查、次日开盘执行；仍需每天检查和安排委托。研究分支未替换主分支信号或现有持仓规则。']
    (OUT/'QX-003_不用盯盘回测报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('QX003_COMPLETE',flush=True)


if __name__=='__main__':
    main()
