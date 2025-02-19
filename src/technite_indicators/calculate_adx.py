import os

def calculateADX(candles, period):
    if len(candles) < period + 1:
        return None
    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i - 1]
        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"]),
        )
        trs.append(tr)
        up_move = curr["high"] - prev["high"]
        down_move = prev["low"] - curr["low"]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    atr = sum(trs[:period])
    smooth_plus = sum(plus_dm[:period])
    smooth_minus = sum(minus_dm[:period])
    dxs = []
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
        smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period
        plus_di = 100 * (smooth_plus / (atr + 1e-10))
        minus_di = 100 * (smooth_minus / (atr + 1e-10))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        dxs.append(dx)
    if len(dxs) < period:
        return None
    return sum(dxs[:period]) / period
