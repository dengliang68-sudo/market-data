from pathlib import Path
import json
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.axis import DateAxis
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.workbook.properties import CalcProperties

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'reports/backtest/qx_20261127'
DEST = DATA / 'QX-001_QX-002_期权策略回测对照.xlsx'
BLUE, TEAL, LIGHT = '16324F', '007F86', 'EAF3F6'
wb = Workbook()
ws = wb.active
ws.title = '结论与规则'
ws.merge_cells('A1:H1')
ws['A1'] = '期权策略回测｜QX-001 → QX-002'
ws['A1'].font = Font(name='微软雅黑', size=20, bold=True, color='FFFFFF')
ws['A1'].fill = PatternFill('solid', fgColor=BLUE)
ws.row_dimensions[1].height = 38
ws.merge_cells('A2:H2')
ws['A2'] = '研究日期2026-11-27｜历史缓存2020-04-01至2026-11-06｜初始10万元｜原基准226笔复现通过'
headers = ['版本/日内情景', '单笔风险', '期末资金（元）', '年化', '平仓回撤', '含日末浮盈亏回撤', '交易笔数', '最高收盘市值仓位']
ws.append([])
ws.append(headers)
result = json.loads((DATA / 'results.json').read_text(encoding='utf-8'))
for tag in ['QX-001-R02', 'QX-002-HL-R02', 'QX-002-LH-R02', 'QX-001-R03', 'QX-002-HL-R03', 'QX-002-LH-R03']:
    m = result['metrics'][tag]
    ws.append([m['版本情景'], m['单笔风险%']/100, m['期末净值'], m['全期年化%']/100,
               m['最大回撤%(按平仓)']/100, m['最大回撤%(含日末浮盈亏)']/100,
               m['成交笔数'], m['最高收盘市值仓位%']/100])
for row in ws.iter_rows(min_row=5, max_row=10):
    for j in [1, 3, 4, 5, 7]:
        row[j].number_format = '0.00%'
    row[2].number_format = '#,##0'
notes = [
    ('编号', 'QX-001＝上次最终推荐基准；QX-002＝本次阶梯回落首次减半。编号本次新建，旧V1/A/B等保留映射。'),
    ('收益初判', '两种日内顺序情景均优于旧版，2%风险年化20.4%→25.6%—27.1%；属于历史模型对照。'),
    ('日内顺序', 'LH＝开盘→最低→最高→收盘；HL＝开盘→最高→最低→收盘。它们不是严格上下界。'),
    ('50%档', '最高涨幅达到50%时不卖；之后跌破+50%才减半。'),
    ('后续阶梯', '最高涨幅60%→保护线55%；70%→65%；80%→75%；每再涨10个百分点，保护线上移10个百分点。'),
    ('计量基准', '涨幅以期权开仓权利金计算，沿用原模型含开仓滑点价格；不是标的ETF涨幅，也不是分档复利。'),
    ('执行约定', '保护线只上移；首次减半只做一次；只有1张时不卖，转入原保本和跟踪阶段；跳空跌破按开盘价成交并扣滑点费用。'),
    ('保留分支', '标的达到2:1目标减半仍然保留；首次实际减半前，原时间/逻辑/权利金止损继续有效。仅启动阶梯不提前切换均线跟踪。'),
    ('资金口径', '原模型先按固定1000元风险生成整数张，再按组合权益放大盈亏。2%风险不等于固定4000元开仓上限。'),
    ('明细张数', '逐笔页的张数为原始固定风险模型张数，组合盈亏按比例缩放；不能直接照抄张数下单。'),
    ('仓位口径', '收盘市值仓位包括持仓浮盈。它可显著高于开仓成本比例；本表不代表盘中绝对最高仓位。'),
    ('模型边界', '沿用原日线止损优先处理、移仓及期末平仓等近似。未采用分钟行情；相同历史窗口不是新的独立样本外证明。'),
    ('策略状态', '仅保存在研究分支，未切换主分支生产信号、定时任务或现有持仓。'),
]
for i, (title, text) in enumerate(notes, start=12):
    ws.cell(i, 1, title)
    ws.merge_cells(start_row=i, start_column=2, end_row=i, end_column=8)
    ws.cell(i, 2, text)
    ws.cell(i, 2).alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[i].height = 34 if i not in (19, 20, 23) else 44


def add_df(name, df):
    s = wb.create_sheet(name)
    s.append(list(df.columns))
    for row in df.itertuples(index=False, name=None):
        vals = []
        for x in row:
            if isinstance(x, (dict, list)):
                x = json.dumps(x, ensure_ascii=False)
            elif pd.isna(x):
                x = None
            elif hasattr(x, 'item'):
                x = x.item()
            vals.append(x)
        s.append(vals)
    return s


add_df('完整指标', pd.read_csv(DATA / 'summary.csv'))
annual = pd.read_csv(DATA / 'annual.csv').pivot(index=['版本情景', '单笔风险%'], columns='年份', values='收益率%').reset_index()
add_df('分年收益百分点', annual)
add_df('分信号', pd.read_csv(DATA / 'by_signal.csv'))
add_df('分品种原始路径', pd.read_csv(DATA / 'by_asset.csv'))
registry = json.loads((ROOT / 'backtest/qx_registry.json').read_text(encoding='utf-8'))
add_df('策略编号表', pd.DataFrame(registry['versions'] + registry['legacy']).fillna(''))
cn = {'etf': '实际品种', 'open': '开仓日', 'close': '最终平仓日', 'strat': '信号',
      'dir': '方向', 'lots': '原始1000风险张数', 'entry': '期权入场成本价',
      'risk': '原始计划风险元', 'pnl': '原始固定风险盈亏元', 'pnl_acct': '组合缩放盈亏元',
      'equity_at_open': '开仓时结转权益元', 'exits': '退出记录', 'exit_detail': '退出明细JSON',
      'legs': '合约与移仓', 'maturity': '到期日', 'u_entry': '标的入场价',
      'u_stop': '标的止损', 'u_target': '标的目标', 'entry_delta': '入场Delta',
      'entry_days_left': '入场余日', 'iv_pct': 'IV分位'}
combined = None
for tag in result['metrics']:
    t = pd.read_csv(DATA / f'{tag}_trades.csv')
    t = t.drop(columns=['code', 'level', 'pri'], errors='ignore')
    first = ['etf', 'open', 'close', 'strat', 'dir', 'lots', 'entry', 'pnl_acct', 'equity_at_open', 'exits']
    t = t[[x for x in first if x in t] + [x for x in t if x not in first]]
    add_df(tag + '逐笔', t.rename(columns=cn))
    daily = pd.read_csv(DATA / f'{tag}_daily.csv')
    daily['date'] = pd.to_datetime(daily['date'].astype(str))
    if tag.endswith('R02'):
        ss = daily.set_index('date')['equity_marked'] / 100000
        ss.name = tag
        combined = ss.to_frame() if combined is None else combined.join(ss)
    daily = daily.rename(columns={'date':'日期', 'equity_marked':'含日末浮盈亏权益元', 'option_value':'期权收盘市值元', 'remaining_cost':'余仓成本元'})
    dws = add_df(tag + '逐日', daily)
    for row in dws.iter_rows(min_row=2):
        row[0].number_format = 'yyyy-mm-dd'

curve_ws = add_df('2%风险净值图数据', combined.iloc[::5].reset_index())
for row in curve_ws.iter_rows(min_row=2):
    row[0].number_format = 'yyyy-mm'
chart = LineChart()
chart.title = '单笔风险2%：日末盯市净值对照'
chart.y_axis.title = '净值（初始=1）'
chart.x_axis = DateAxis(crossAx=100)
chart.x_axis.number_format = 'yy-mm'
chart.x_axis.majorTimeUnit = 'years'
chart.add_data(Reference(curve_ws, min_col=2, max_col=4, min_row=1, max_row=curve_ws.max_row), titles_from_data=True)
chart.set_categories(Reference(curve_ws, min_col=1, min_row=2, max_row=curve_ws.max_row))
chart.width, chart.height = 26, 12
for series, color in zip(chart.series, ['6E7F91', '007F86', 'E59A24']):
    series.graphicalProperties.line.solidFill = color
    series.graphicalProperties.line.width = 20000
ws.add_chart(chart, 'A27')

for s in wb:
    s.sheet_view.showGridLines = False
    s.freeze_panes = 'B5' if s == ws else 'B2'
    if s != ws:
        s.auto_filter.ref = s.dimensions
        s.row_dimensions[1].height = 28
    for cell in s[4 if s == ws else 1]:
        cell.fill = PatternFill('solid', fgColor=BLUE)
        cell.font = Font(name='微软雅黑', bold=True, color='FFFFFF', size=10)
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    for row in s.iter_rows(min_row=5 if s == ws else 2):
        for c in row:
            if not c.has_style:
                c.font = Font(name='微软雅黑', size=10, color='23364A')
            if row[0].row % 2 == 0 and s != ws:
                c.fill = PatternFill('solid', fgColor='F1F6F9')
            if isinstance(c.value, float) and c.number_format == 'General':
                c.number_format = '#,##0.00;[Red](#,##0.00)'
    for col in s.columns:
        letter = col[0].column_letter if hasattr(col[0], 'column_letter') else None
        if letter:
            samples = [str(c.value or '') for c in list(col)[:60]]
            s.column_dimensions[letter].width = min(50, max(15, max(map(len, samples), default=10) * 1.35))
if ws.column_dimensions['A'].width > 24:
    ws.column_dimensions['A'].width = 24
for c in 'BCDEFGH':
    ws.column_dimensions[c].width = 18
wb.calculation = CalcProperties(fullCalcOnLoad=True)
wb.save(DEST)
check = load_workbook(DEST, read_only=True, data_only=False)
assert len(check.sheetnames) == 19, check.sheetnames
assert check['结论与规则']['C5'].value == 340308
assert check['QX-001-R03逐笔'].max_row == 227
print(json.dumps({'path': str(DEST), 'sheets': len(check.sheetnames), 'size_bytes': DEST.stat().st_size}, ensure_ascii=False))
