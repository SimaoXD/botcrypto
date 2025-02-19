import os


def calculateIchimoku(candles, params):
    conversionPeriod = params["conversionPeriod"]
    basePeriod = params["basePeriod"]
    spanBPeriod = params["spanBPeriod"]
    displacement = params["displacement"]
    if not isinstance(candles, list) or len(candles) < spanBPeriod:
        return None
    tenkan_sen = []
    kijun_sen = []
    for i in range(len(candles)):
        if i >= conversionPeriod - 1:
            recent = candles[i - conversionPeriod + 1 : i + 1]
            high_vals = [c["high"] for c in recent]
            low_vals = [c["low"] for c in recent]
            tenkan_sen.append((max(high_vals) + min(low_vals)) / 2)
        else:
            tenkan_sen.append(None)
        if i >= basePeriod - 1:
            recent = candles[i - basePeriod + 1 : i + 1]
            high_vals = [c["high"] for c in recent]
            low_vals = [c["low"] for c in recent]
            kijun_sen.append((max(high_vals) + min(low_vals)) / 2)
        else:
            kijun_sen.append(None)
    senkou_span_a = []
    for i in range(len(candles)):
        if tenkan_sen[i] is not None and kijun_sen[i] is not None:
            senkou_span_a.append((tenkan_sen[i] + kijun_sen[i]) / 2)
        else:
            senkou_span_a.append(None)
    senkou_span_b = []
    for i in range(len(candles)):
        if i >= spanBPeriod - 1:
            recent = candles[i - spanBPeriod + 1 : i + 1]
            high_vals = [c["high"] for c in recent]
            low_vals = [c["low"] for c in recent]
            senkou_span_b.append((max(high_vals) + min(low_vals)) / 2)
        else:
            senkou_span_b.append(None)
    senkou_span_a_shifted = [None] * displacement + senkou_span_a[
        : len(candles) - displacement
    ]
    senkou_span_b_shifted = [None] * displacement + senkou_span_b[
        : len(candles) - displacement
    ]
    return {
        "tenkanSen": tenkan_sen,
        "kijunSen": kijun_sen,
        "senkouSpanA": senkou_span_a_shifted,
        "senkouSpanB": senkou_span_b_shifted,
    }
