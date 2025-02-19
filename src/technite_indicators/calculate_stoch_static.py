import os


def calculateStochastic(highs, lows, closes, period=14):
    if len(closes) < period:
        return None
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    highest = max(recent_highs)
    lowest = min(recent_lows)
    last_close = closes[-1]
    if highest == lowest:
        print("Mercado em consolidação para Estocástico.")
        return 50
    return ((last_close - lowest) / (highest - lowest)) * 100
