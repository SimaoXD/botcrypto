import os
def calculateRSI(prices, period):
    epsilon = 1e-10
    if len(prices) < period + 1:
        return None
    gains = 0
    losses = 0
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)
    avg_gain = gains / period
    avg_loss = losses / period
    rs = avg_gain / (avg_loss + epsilon)
    rsi = 100 - 100 / (1 + rs)
    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = change if change > 0 else 0
        loss = abs(change) if change < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / (avg_loss + epsilon)
        rsi = 100 - 100 / (1 + rs)
    return rsi