"""Integration checks against the actual engine, without market/network mocks."""
import unittest
from unittest.mock import patch
import pandas as pd
from backtest import kc50_backtest as K


class Book:
    def __init__(self, days):
        self.meta = pd.DataFrame({'name': ['fixture']}, index=['X'])
        self.bars = {d: (.1, .14, .1, .14) for d in days}
        self.bars[days[62]] = (.15, .16, .15, .16)
        self.bars[days[63]] = (.16, .17, .16, .17)
        self.bars[days[64]] = (.17, .171, .16, .17)

    def price(self, code, day, kind='close'):
        return self.bars[day][{'open': 0, 'high': 1, 'low': 2, 'close': 3}[kind]]


def run_fixture(step=True, risk_k=1, target=100):
    days = pd.bdate_range('2026-01-05', periods=65).strftime('%Y%m%d').tolist()
    u = pd.DataFrame({'open': 10., 'close': 10., 'ma5': 9., 'ma20': 9., 'state': '上升', 'atr': .1}, index=days)
    sig = dict(dir=1, strat='L1趋势突破', stop=1., target=target)
    contract = dict(long='X', maturity='20261231', delta=.5, days_left=40)
    cfg = {**K.RECOMMENDED, 'risk_k': risk_k}
    if step:
        cfg.update(first_profit='step_pullback', profit_bar_order='low_first')
    with patch.object(K, 'signals_v2', side_effect=lambda r, *_: [sig] if r.name == days[60] else []), \
         patch.object(K.v1, 'pick_contract', return_value=contract), patch.object(K, 'slip', return_value=0.):
        return K.run_v2(cfg, days, u, Book(days), pd.DataFrame(), {})['trades'][0]


class EngineTests(unittest.TestCase):
    def test_delays_first_half_and_only_once(self):
        old, new = run_fixture(False), run_fixture(True)
        old_half = [e for e in __import__('json').loads(old['exit_detail']) if e[2] > 0][0]
        new_half = [e for e in __import__('json').loads(new['exit_detail']) if '阶梯回落' in e[1]]
        self.assertEqual(len(new_half), 1)
        self.assertGreater(new_half[0][0], old_half[0])
        self.assertAlmostEqual(new_half[0][3], .165)
        self.assertEqual(new_half[0][2], 1)

    def test_one_lot_does_not_sell_half(self):
        new = run_fixture(True, risk_k=.4)
        half = [e for e in __import__('json').loads(new['exit_detail']) if '阶梯回落' in e[1]]
        self.assertEqual(len(half), 1)
        self.assertEqual(half[0][2], 0)
        self.assertEqual(new['lots'], 1)

    def test_underlying_target_branch_is_preserved(self):
        new = run_fixture(True, target=10.)
        self.assertIn('到达2:1目标', new['exits'])
        self.assertNotIn('阶梯回落', new['exits'])


if __name__ == '__main__':
    unittest.main()
