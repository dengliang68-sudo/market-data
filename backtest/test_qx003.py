import json
import unittest
from unittest.mock import patch
import pandas as pd
from backtest import kc50_backtest as K


def fixture(bars, *, close_only=True):
    days = pd.bdate_range('2026-01-05', periods=68).strftime('%Y%m%d').tolist()
    u = pd.DataFrame({'open': 10., 'close': 10., 'ma5': 9., 'ma20': 9., 'state': '上升', 'atr': .1}, index=days)
    class Book:
        meta = pd.DataFrame({'name': ['fixture']}, index=['X'])
        def price(self, code, day, kind='close'):
            if close_only and kind in ('high', 'low'):
                raise AssertionError('QX-003 must not read intraday option high or low')
            i = days.index(day)
            return bars.get(i, (.1, .14, .01, .14))[{'open':0, 'high':1, 'low':2, 'close':3}[kind]]
    sig = dict(dir=1, strat='L1趋势突破', stop=1., target=100.)
    contract = dict(long='X', maturity='20261231', delta=.5, days_left=40)
    cfg = {**K.RECOMMENDED, 'close_only': close_only, 'first_profit': 'step_close'}
    with patch.object(K, 'signals_v2', side_effect=lambda r, *_: [sig] if r.name == days[60] else []), \
         patch.object(K.v1, 'pick_contract', return_value=contract), patch.object(K, 'slip', return_value=0.):
        result = K.run_v2(cfg, days, u, Book(), pd.DataFrame(), {})
    return result['trades'][0], days


class CloseOnlyTests(unittest.TestCase):
    def test_ignores_intraday_high_and_low_and_uses_next_open(self):
        t, days = fixture({62:(.14,.9,.01,.18),63:(.18,.9,.01,.17),64:(.155,.9,.01,.17)})
        half = [x for x in json.loads(t['exit_detail']) if '阶梯回落' in x[1]]
        self.assertEqual(len(half), 1)
        self.assertEqual(half[0][0], days[64])
        self.assertAlmostEqual(half[0][3], .155)
        self.assertIn('+75%', half[0][1])

    def test_stop_is_close_confirmed_and_gap_loss_is_not_capped(self):
        t, days = fixture({62:(.14,.5,.01,.05),63:(.04,.1,.01,.08)})
        ex = json.loads(t['exit_detail'])[0]
        self.assertEqual(ex[0], days[63])
        self.assertIn('收盘确认权利金止损', ex[1])
        self.assertAlmostEqual(ex[3], .04)
        self.assertLess(t['pnl'], -t['risk'])

    def test_intraday_profit_peak_does_not_arm_rule(self):
        t, _ = fixture({62:(.14,.9,.01,.14),63:(.14,.8,.01,.14)})
        self.assertNotIn('阶梯回落', t['exits'])

    def test_exact_floor_touch_does_not_trigger(self):
        t, days = fixture({62:(.14,.9,.01,.18),63:(.18,.9,.01,.175),64:(.175,.9,.01,.18)})
        half = [x for x in json.loads(t['exit_detail']) if '阶梯回落' in x[1]]
        self.assertTrue(all(x[0] > days[64] for x in half))


if __name__ == '__main__':
    unittest.main()
