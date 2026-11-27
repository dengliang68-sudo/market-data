from pathlib import Path
import json
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.axis import DateAxis

ROOT=Path(__file__).resolve().parent
NEW=ROOT/'reports/backtest/qx003_20261127'
OLD=ROOT/'reports/backtest/qx_20261127'
DEST=NEW/'QX-003_不用盯盘策略回测.xlsx'
wb=Workbook()
ws=wb.active
ws.title='结论与执行规则'
ws.merge_cells('A1:H1')
ws['A1']='QX-003｜收盘检查 · 次日开盘执行'
ws['A1'].font=Font(name='微软雅黑',size=20,bold=True,color='FFFFFF')
ws['A1'].fill=PatternFill('solid',fgColor='16324F')
ws.row_dimensions[1].height=38
ws.merge_cells('A2:H2')
ws['A2']='2026-11-27研究｜历史缓存2020-04-01至2026-11-06｜初始10万元｜QX-001逐笔复现'
ws.append([])
ws.append(['版本','风险预算','期末资金元','年化收益','日末盯市回撤','交易笔数','最差单笔亏损','超风险预算次数'])
new=pd.read_csv(NEW/'summary.csv')
old=pd.read_csv(OLD/'summary.csv')
rows=[]
for risk in [2,3]:
    for key in ['QX-001','QX-002-HL','QX-002-LH','QX-003']:
        if key in ['QX-001','QX-003']:
            m=new[(new['版本']==key)&(new['单笔风险预算%']==risk)].iloc[0]
            row=[key,risk/100,int(m['期末净值']),m['全期年化%']/100,m['最大回撤含日末浮盈亏%']/100,
                 int(m['成交笔数']),m['最差单笔账户亏损%']/100,int(m['亏损超过单笔风险预算次数'])]
        else:
            m=old[(old['版本情景']==key)&(old['单笔风险%']==risk)].iloc[0]
            t=pd.read_csv(OLD/f'{key}-R{risk:02}_trades.csv')
            loss=-t.pnl_acct/t.equity_at_open
            row=[key,risk/100,int(m['期末净值']),m['全期年化%']/100,m['最大回撤%(含日末浮盈亏)']/100,
                 int(m['成交笔数']),float(loss.max()),int((loss>risk/100+.000001).sum())]
        rows.append(row)
        ws.append(row)
for rr in ws.iter_rows(min_row=5,max_row=12):
    for i in [1,3,4,6]: rr[i].number_format='0.00%'
    rr[2].number_format='#,##0'
notes=[
('版本','QX-001原基准；QX-002为盘中阶梯、LH/HL为日内路径假设；QX-003为本次收盘确认版。'),
('初始止损','以期权收盘价判断原40%/50%/60%止损，次日开盘执行；盘中跌破、收盘收回不卖。'),
('首次阶梯','只记录历史最高收盘价。50%启动、60%保护55%、70%保护65%，收盘严格跌破后次日开盘减半。'),
('剩余仓位','首次减半后沿用保本与均线跟踪；保本也只看收盘。仅1张则不减半、转入保本/跟踪阶段。'),
('其他退出','2:1目标、均线、逻辑、时间止损都在收盘确认。移仓收盘选约、次日开盘换仓；节前减仓前一晚安排。'),
('时间要求','免盯盘仍需每天收盘后检查、安排次日开盘委托；若软件不支持预设委托，仍需开盘时操作。'),
('风险预算','2%/3%是计算仓位的计划风险，不是亏损硬上限。表中最差单笔和超预算次数按最终整笔账户盈亏计。'),
('比较边界','沿用原1000元固定风险路径再按账户权益缩放模型，不是按实盘整数张重撮合，也不是每笔固定4000元。'),
('日线限制','无盘中最高回撤、无盘口成交保证；历史条款/移仓与缺报价回退等原模型近似仍在。样本末日收盘强制估值。'),
('状态','本次只运行研究分支，不切换生产信号、自动任务或现有持仓。'),
]
for i,(a,b) in enumerate(notes,start=14):
    ws.cell(i,1,a)
    ws.merge_cells(start_row=i,start_column=2,end_row=i,end_column=8)
    ws.cell(i,2,b).alignment=Alignment(wrap_text=True,vertical='center')
    ws.row_dimensions[i].height=37

def add(name,df):
    s=wb.create_sheet(name)
    s.append([str(x) for x in df.columns])
    for rr in df.itertuples(index=False,name=None):
        s.append([None if pd.isna(x) else x.item() if hasattr(x,'item') else x for x in rr])
    return s

add('完整结果',new)
annual=pd.read_csv(NEW/'annual.csv')
add('分年收益百分点',annual.pivot(index=['版本','单笔风险预算%'],columns='年份',values='收益率%').reset_index())
add('分信号统计',pd.read_csv(NEW/'by_signal.csv'))
curve=None
for risk in [2,3]:
    for key in ['QX-001','QX-003']:
        tag=f'{key}-R{risk:02}'
        t=pd.read_csv(NEW/f'{tag}_trades.csv').drop(columns=['code','level','pri'],errors='ignore')
        t=t.rename(columns={'etf':'实际品种','open':'开仓日','close':'平仓日','strat':'信号','lots':'原始固定风险张数',
                            'entry':'期权入场成本价','pnl':'原始固定风险盈亏','pnl_acct':'组合缩放盈亏',
                            'equity_at_open':'开仓时账户结转权益','exits':'退出记录'})
        add(tag+'逐笔',t)
        add(tag+'超预算亏损',t[t['单笔账户损失%']>risk+.0001])
        d=pd.read_csv(NEW/f'{tag}_daily.csv')
        d['date']=pd.to_datetime(d.date.astype(str))
        add(tag+'逐日',d)
        if risk==2:
            ss=d.set_index('date').equity_marked/100000
            ss.name=key
            curve=ss.to_frame() if curve is None else curve.join(ss)
for key in ['QX-002-HL','QX-002-LH']:
    d=pd.read_csv(OLD/f'{key}-R02_daily.csv')
    ss=pd.Series(d.equity_marked.to_numpy()/100000,index=pd.to_datetime(d.date.astype(str)),name=key)
    curve=curve.join(ss)
s=add('2%风险净值图数据',curve.iloc[::5].reset_index())
for rr in s.iter_rows(min_row=2): rr[0].number_format='yyyy-mm'
chart=LineChart()
chart.title='单笔风险预算2%：日末盯市净值'
chart.x_axis=DateAxis(crossAx=100)
chart.x_axis.number_format='yy-mm'
chart.y_axis.title='初始净值1'
chart.add_data(Reference(s,min_col=2,max_col=5,min_row=1,max_row=s.max_row),titles_from_data=True)
chart.set_categories(Reference(s,min_col=1,min_row=2,max_row=s.max_row))
chart.width,chart.height=26,12
ws.add_chart(chart,'A26')
for sheet in wb:
    sheet.sheet_view.showGridLines=False
    sheet.freeze_panes='B5' if sheet==ws else 'B2'
    if sheet!=ws: sheet.auto_filter.ref=sheet.dimensions
    for cell in sheet[4 if sheet==ws else 1]:
        cell.fill=PatternFill('solid',fgColor='16324F')
        cell.font=Font(name='微软雅黑',bold=True,color='FFFFFF',size=10)
        cell.alignment=Alignment(wrap_text=True)
    for row in sheet.iter_rows(min_row=5 if sheet==ws else 2):
        for c in row:
            if not c.has_style: c.font=Font(name='微软雅黑',size=10)
            if isinstance(c.value,float) and c.number_format=='General': c.number_format='#,##0.00;[Red](#,##0.00)'
    for col in sheet.columns:
        if not hasattr(col[0],'column_letter'): continue
        sheet.column_dimensions[col[0].column_letter].width=min(48,max(16,max(len(str(c.value or '')) for c in list(col)[:40])*1.3))
for col in 'ABCDEFGH': ws.column_dimensions[col].width=19
wb.save(DEST)
check=load_workbook(DEST,read_only=True)
assert check['结论与执行规则']['C5'].value==340308
assert len(check.sheetnames)==17
print(json.dumps({'path':str(DEST),'sheets':len(check.sheetnames),'bytes':DEST.stat().st_size},ensure_ascii=False))
