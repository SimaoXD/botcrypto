from .calculate_ema import calculateEMA


def calculateMACD(prices, fastPeriod=12, slowPeriod=26, signalPeriod=9):
    if len(prices) < slowPeriod:
        return None
    fast_ema = calculateEMA(prices, fastPeriod)
    slow_ema = calculateEMA(prices, slowPeriod)
    macd_line = []
    for i in range(len(prices)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line.append(fast_ema[i] - slow_ema[i])
        else:
            macd_line.append(None)
    valid_macd = [val for val in macd_line[slowPeriod - 1 :] if val is not None]
    signal_valid = calculateEMA(valid_macd, signalPeriod)
    signal_line = [None] * len(prices)
    signal_start = slowPeriod - 1 + signalPeriod - 1
    for i in range(signal_start, len(prices)):
        idx = i - signal_start
        if idx < len(signal_valid):
            signal_line[i] = signal_valid[idx]
    histogram = []
    for i in range(len(prices)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    if sum(1 for val in histogram if val is not None) < 1:
        return None
    return {"macdLine": macd_line, "signalLine": signal_line, "histogram": histogram}
