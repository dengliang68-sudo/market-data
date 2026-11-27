import unittest

from backtest.qx_profit import first_pullback, profit_floor


class ProfitRuleTests(unittest.TestCase):
    def test_user_examples_and_boundary(self):
        for peak, floor in [(1.49, None), (1.5, 1.5), (1.59, 1.5),
                            (1.6, 1.55), (1.7, 1.65), (1.8, 1.75), (2.0, 1.95)]:
            actual = profit_floor(1, peak)
            if floor is None:
                self.assertIsNone(actual)
            else:
                self.assertAlmostEqual(actual, floor)

    def test_scaled_option_premium(self):
        self.assertAlmostEqual(profit_floor(.0943, .0943 * 1.7), .0943 * 1.65)

    def test_no_sale_when_continues_up(self):
        for order in ("low_first", "high_first"):
            _, fill, _ = first_pullback(1, 1, 1.56, 1.6, 1.56, 1.6, order)
            self.assertIsNone(fill)

    def test_daily_order_is_material(self):
        a = first_pullback(1, 1, 1.4, 1.8, 1.35, 1.8, "low_first")
        b = first_pullback(1, 1, 1.4, 1.8, 1.35, 1.8, "high_first")
        self.assertIsNone(a[1])
        self.assertAlmostEqual(b[1], 1.75)

    def test_prior_day_floor_handles_gap_without_fictional_fill(self):
        _, fill, floor = first_pullback(1, 1.8, 1.6, 1.9, 1.4, 1.8)
        self.assertAlmostEqual(fill, 1.6)
        self.assertAlmostEqual(floor, 1.75)

    def test_no_floor_lowering_after_pullback(self):
        peak, fill, _ = first_pullback(1, 1.8, 1.85, 1.86, 1.76, 1.8)
        self.assertIsNone(fill)
        self.assertGreaterEqual(peak, 1.8)
        _, fill, floor = first_pullback(1, peak, 1.8, 1.82, 1.7, 1.71)
        self.assertAlmostEqual(fill, 1.75)
        self.assertAlmostEqual(floor, 1.75)

    def test_equal_touch_is_not_break(self):
        self.assertIsNone(first_pullback(1, 1.8, 1.8, 1.81, 1.75, 1.8)[1])

    def test_peak_does_not_include_future_price_after_sale(self):
        peak, fill, _ = first_pullback(1, 1.6, 1.6, 2.5, 1.5, 2.4, "low_first")
        self.assertAlmostEqual(peak, 1.6)
        self.assertAlmostEqual(fill, 1.55)

    def test_invalid_ohlc_rejected(self):
        with self.assertRaises(ValueError):
            first_pullback(1, 1, 1.4, 1.3, 1.2, 1.4)


if __name__ == "__main__":
    unittest.main()
