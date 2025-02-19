import os

def calculateEMA(prices, period):
    ema_array = []
    multiplier = 2 / (period + 1)
    for i, price in enumerate(prices):
        if i < period - 1:
            ema_array.append(None)
        elif i == period - 1:
            initial_sma = sum(prices[:period]) / period
            ema_array.append(initial_sma)
        else:
            prev_ema = ema_array[i - 1]
            ema_array.append((price - prev_ema) * multiplier + prev_ema)
    return ema_array
