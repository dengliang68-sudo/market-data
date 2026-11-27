"""Frozen QX-001 / QX-002 comparison. Reads repository caches only.

Does not change production signals, positions, schedules or old research reports.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest import kc50_backtest as K
from backtest import recommended_portfolio as RP
from backtest import opt_backtest as V
from md import client

OUT = ROOT / 'reports/backtest/qx_20261127'
BASE_SOURCE = '00f762d09582dd1bb6d6a8d8da8c59c9b5b2b8b0'


def blocked_network(*a, **kw):
    raise RuntimeError('This experiment is offline: only frozen cached market data are allowed')


def clean_trades(t):
    return t.drop(columns=['trace', 'Index'], errors='ignore')


def marked_curve(accepted, days, risk):
    curve = pd.Series(100000., index=days)
    exposure = pd.Series(0., index=days)
    cost = pd.Series(0., index=days)
    for t in accepted.itertuples():
        scale = risk * t.equity_at_open / 1000
        trace = pd.DataFrame(t.trace).drop_duplicates('date', keep='last').set_index('date')
        pnl = trace['pnl'].reindex(days).ffill().fillna(0.)
        # Terminal PnL uses the identical rounded per-trade figure as RP.sim.
        pnl.loc[pnl.index >= t.close] = t.pnl
        curve += pnl * scale
        exposure += trace['market_value'].reindex(days).fillna(0.) * scale
        cost += trace['cost'].reindex(days).fillna(0.) * scale
    return pd.DataFrame({'date': days, 'equity_marked': curve.to_numpy(),
                         'option_value': exposure.to_numpy(), 'remaining_cost': cost.to_numpy()})


def table(df):
    return df.to_markdown(index=False, floatfmt='.2f')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    client.call = blocked_network
    print('Loading frozen cached data and rebuilding IV histories ...', flush=True)
    data, _ = RP.load()
    print('DATA LOADED', flush=True)
    configs = {
        'QX-001': {**K.RECOMMENDED, 'research_trace': True},
        'QX-002-LH': {**K.RECOMMENDED, 'research_trace': True,
                      'first_profit': 'step_pullback', 'profit_bar_order': 'low_first'},
        'QX-002-HL': {**K.RECOMMENDED, 'research_trace': True,
                      'first_profit': 'step_pullback', 'profit_bar_order': 'high_first'},
    }
    buckets = {k: [] for k in configs}
    asset_rows = []
    for name, (df, mu, o, days, ivp, first, meta) in data.items():
        u = K.frame(df)
        book = V.Book(o, meta)
        st = max(RP.D0, first)
        i0 = max(0, next(i for i, d in enumerate(days) if d >= st) - 60)
        for key, cfg in configs.items():
            result = K.run_v2(cfg, days[i0:], u, book, mu, ivp)
            t = pd.DataFrame(result['trades'])
            t['etf'] = name
            buckets[key].append(t)
            asset_rows.append({'版本情景': key, '品种': name, '原始交易笔数': len(t),
                               '固定风险原始盈亏': float(t.pnl.sum()),
                               '阶梯首次减半笔数': int(t.exits.str.contains('阶梯回落').sum())})
            print('ENGINE', name, key, len(t), round(t.pnl.sum(), 1), flush=True)
    all_days = sorted(set(d for item in data.values() for d in item[3] if d >= RP.D0))
    metrics, annuals, signals = [], [], []
    outputs = {}
    baseline_passed = False
    prior = pd.read_csv(ROOT / 'reports/backtest/recommended_trades.csv', dtype={'open': str, 'close': str})
    for key, dfs in buckets.items():
        raw = pd.concat(dfs, ignore_index=True)
        raw['open'], raw['close'] = raw.open.astype(str), raw.close.astype(str)
        clean_trades(raw).to_csv(OUT / f'{key}_raw_trades.csv', index=False, encoding='utf-8-sig')
        for risk in [.03, .02]:
            m, yearly, accepted = RP.sim(raw, risk)
            tag = f'{key}-R{int(risk*100):02}'
            if key == 'QX-001' and risk == .03:
                fields = ['etf', 'open', 'close', 'strat', 'lots', 'pnl', 'pnl_acct', 'equity_at_open']
                pd.testing.assert_frame_equal(accepted[fields].reset_index(drop=True),
                                              prior[fields].reset_index(drop=True), check_dtype=False,
                                              atol=.11, rtol=0)
                assert m['期末净值'] == 558227 and m['成交笔数'] == 226, m
                baseline_passed = True
                print('BASELINE REPRODUCTION PASSED: 226 trades, ending equity 558227', flush=True)
            daily = marked_curve(accepted, all_days, risk)
            eq = daily.equity_marked
            dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
            m.update({'版本情景': key, '单笔风险%': risk * 100, '最大回撤%(含日末浮盈亏)': round(dd, 2),
                      '最高收盘市值仓位%': round(float((daily.option_value / eq).max() * 100), 2),
                      '最高剩余成本仓位%': round(float((daily.remaining_cost / eq).max() * 100), 2),
                      '首次阶梯减半笔数': int(accepted.exits.str.contains('阶梯回落').sum()),
                      '原目标减半笔数': int(accepted.exits.str.contains('到达2:1目标').sum())})
            metrics.append(m)
            for year, ret in yearly.items():
                annuals.append({'版本情景': key, '单笔风险%': risk * 100, '年份': year, '收益率%': float(ret)})
            for sig, group in accepted.groupby(accepted.strat.str[:2]):
                signals.append({'版本情景': key, '单笔风险%': risk * 100, '信号': sig, '笔数': len(group),
                                '盈亏元': round(float(group.pnl_acct.sum()), 2),
                                '胜率%': round(float((group.pnl_acct > 0).mean() * 100), 2)})
            clean_trades(accepted).to_csv(OUT / f'{tag}_trades.csv', index=False, encoding='utf-8-sig')
            daily.to_csv(OUT / f'{tag}_daily.csv', index=False, encoding='utf-8-sig')
            outputs[tag] = m
            print('RESULT', tag, json.dumps(m, ensure_ascii=False), flush=True)
    summary, annual, by_signal = pd.DataFrame(metrics), pd.DataFrame(annuals), pd.DataFrame(signals)
    summary.to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')
    annual.to_csv(OUT / 'annual.csv', index=False, encoding='utf-8-sig')
    by_signal.to_csv(OUT / 'by_signal.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(asset_rows).to_csv(OUT / 'by_asset.csv', index=False, encoding='utf-8-sig')
    manifest = {'baseline_source_commit': BASE_SOURCE, 'baseline_reproduced': baseline_passed,
                'study_end': max(all_days), 'rules': configs, 'python': sys.version,
                'pandas': pd.__version__, 'numpy': np.__version__, 'inputs': {}}
    for path in sorted((ROOT / 'data/bt').glob('*.parquet')):
        if path.name.startswith(('opt_daily_', 'opt_basic')):
            manifest['inputs'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'results.json').write_text(json.dumps({'baseline_reproduced': baseline_passed, 'metrics': outputs,
                                               'annual': annuals, 'by_signal': signals}, ensure_ascii=False, indent=2), encoding='utf-8')
    cols = ['版本情景', '单笔风险%', '期末净值', '全期年化%', '最大回撤%(按平仓)',
            '最大回撤%(含日末浮盈亏)', '成交笔数', '最高收盘市值仓位%']
    lines = ['# QX-001与QX-002期权策略对照回测', '',
             '研究日期：2026-11-27。历史窗口沿用旧报告缓存（2020-04-01至2026-11-06；各品种从可用日期开始）。',
             f'冻结来源提交：{BASE_SOURCE}。原推荐版现统一编号QX-001；本次微调编号QX-002。这些编号本次新建，并非原仓库已有。', '',
             '## 规则差异', '',
             '|历史最高权利金涨幅|QX-001|QX-002首次减半保护线|',
             '|---|---|---|', '|达到50%|立即减半|跌破+50%时减半|',
             '|达到60%|已完成首次减半|跌破+55%时减半|',
             '|达到70%|已完成首次减半|跌破+65%时减半|',
             '|达到80%|已完成首次减半|跌破+75%时减半|', '',
             '涨幅相对于期权开仓权利金（沿用原回测含开仓滑点的价格），是百分点上移，不是每档复利。',
             '保护线只上移；首次减半只做一次；1张不卖，只转入原规则的保本及跟踪阶段。跳空跌破按开盘价减半，扣原有滑点和费用。',
             '只替换权利金+50%减半分支；标的到达2:1目标减半仍保留，可以先于阶梯保护触发。首次真正减半/目标触发前，原时间止损、逻辑止损、权利金止损继续有效；仅启动阶梯保护不提前切换保本/均线跟踪阶段。',
             '其他信号、选合约、移仓、月度开仓次数、最多4笔/同向3笔及原组合缩放算法保持不变。', '',
             '## 基准复现', '',
             f'旧推荐版逐笔记录复现通过：{baseline_passed}；3%口径226笔，期末558227元，年化29.7%，按平仓回撤19.5%。先通过复现再报告新版。', '',
             '## 结果', '', table(summary[cols]), '',
             'LH＝假设当日开盘→最低→最高→收盘；HL＝开盘→最高→最低→收盘。两者是日线顺序敏感性场景，不是严格上下界，也不是分钟数据回测。', '',
             '## 分年收益（沿用原平仓结转口径）', '',
             table(annual.pivot(index=['版本情景', '单笔风险%'], columns='年份', values='收益率%').reset_index()), '',
             '## 首次减半归因', '',
             table(summary[['版本情景', '单笔风险%', '首次阶梯减半笔数', '原目标减半笔数']]), '',
             '## 按品种的原始固定风险路径', '', table(pd.DataFrame(asset_rows)), '',
             '## 结果适用边界', '',
             '1. 为可比性，本轮沿用旧回测模型：单品种按1000元风险整数张生成交易，再按组合净值比例缩放盈亏；不是按真实组合资金重算整数张、逐笔扣款的实盘撮合。2%风险也不等同于固定4000元开仓上限。',
             '2. 日线内原权利金止损仍优先于首次减半；HL/LH只检验新增阶梯保护分支的高低价顺序。不能据此保证盘中可成交收益。',
             '3. 新增含浮盈亏回撤为日末盯市估计，仓位指标也是收盘口径；组合开仓缩放仍用原来已平仓权益。没有盘中完整净值，不能当盘中最大回撤或总体仓位绝对峰值。',
             '4. 原模型仍有期末强制平仓、移仓近似、历史合约单位/除息处理及开盘价缺失回退等限制；本轮未借修改止盈顺带改这些假设。',
             '5. 新规则是在已看过历史结果后提出；同一历史窗口仅作比较，不能称新的独立样本外验证。两种场景都优于基准才有更强研究价值，仍需分钟数据与后续模拟验证。', '',
             '代码、完整逐笔、逐日净值、分年、分信号和输入哈希一并保留。生产主分支和实盘规则未替换。']
    (OUT / 'QX-001_QX-002_回测对照报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('QX_RESEARCH_COMPLETE', json.dumps(outputs, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
