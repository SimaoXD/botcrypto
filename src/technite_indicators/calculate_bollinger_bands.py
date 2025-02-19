import math
import os

def calculateBollingerBands(prices, period=20, multiplier=2):
    if not isinstance(prices, list) or len(prices) < period:
        return None
    recent_prices = prices[-period:]
    middle = sum(recent_prices) / period
    variance = sum((p - middle) ** 2 for p in recent_prices) / period
    epsilon = 1e-10
    std_dev = math.sqrt(variance + epsilon)
    return {
        "middle": middle,
        "upper": middle + multiplier * std_dev,
        "lower": middle - multiplier * std_dev,
    }