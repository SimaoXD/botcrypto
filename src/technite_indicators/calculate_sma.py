import os
def calculateSMA(prices, period):
    if len(prices) < period:
        return None
    recent_prices = prices[-period:]
    return sum(recent_prices) / period
