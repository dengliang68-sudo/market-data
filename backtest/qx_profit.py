"""QX-002 first-half trailing profit rule; pure, price-path based functions.

All gains use the original option entry premium, not ETF gains or compounding.
Daily OHLC paths are sensitivity assumptions, never observed tick sequences.
"""
from decimal import Decimal, ROUND_FLOOR


def profit_floor(entry: float, peak: float) -> float | None:
    """50..59.99% peak -> 50%; 60..69.99% -> 55%; 70..79.99% -> 65%."""
    if entry <= 0:
        raise ValueError("entry must be positive")
    gain = Decimal(str(peak)) / Decimal(str(entry)) - 1
    # Only absorb binary floating point noise at an exact 10-point boundary.
    gain += Decimal("0.000000000001")
    if gain < Decimal("0.5"):
        return None
    rung = int(((gain - Decimal("0.5")) / Decimal("0.1")).to_integral_value(rounding=ROUND_FLOOR))
    floor = Decimal("0.5") if rung == 0 else Decimal("0.45") + Decimal("0.1") * rung
    return entry * (1 + float(floor))


def first_pullback(entry, previous_peak, o, hi, lo, cl, order="low_first"):
    """Return (observed_peak_until_fill, fill_or_None, floor_at_fill_or_None).

    At a gap below an already armed floor, fill at the opening price.
    Intraday downward crossings fill at the floor before the engine's slippage.
    The rule triggers strictly below a floor, not on an equal-price touch.
    """
    if not (0 < lo <= min(o, cl) <= max(o, cl) <= hi):
        raise ValueError("invalid positive OHLC bar")
    if order not in ("low_first", "high_first"):
        raise ValueError("unknown OHLC order")
    peak = max(float(previous_peak), float(entry))
    floor = profit_floor(entry, peak)
    if floor is not None and o < floor - 1e-12 * entry:
        return peak, float(o), floor
    peak = max(peak, o)
    points = [o, lo, hi, cl] if order == "low_first" else [o, hi, lo, cl]
    for a, b in zip(points, points[1:]):
        floor = profit_floor(entry, peak)
        if b < a and floor is not None and b < floor - 1e-12 * entry:
            return peak, float(min(a, floor)), floor
        peak = max(peak, b)
    return peak, None, None
