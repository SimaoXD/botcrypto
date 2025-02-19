from global_configuration import CONFIG

def evaluateSignalScore(
    rsi,
    dynamicRSIOversold,
    latestHistogram,
    stochastic,
    dynamicStochOversold,
    lastPrice,
    sma,
    bollinger,
    ichimoku,
):

    """
    Avalia a pontuação do sinal de compra com base em vários indicadores técnicos.

    Parâmetros:
    - rsi: Índice de Força Relativa (RSI)
    - dynamicRSIOversold: Limite dinâmico de sobrevenda para RSI
    - latestHistogram: Último valor do histograma MACD
    - stochastic: Oscilador Estocástico
    - dynamicStochOversold: Limite dinâmico de sobrevenda para Estocástico
    - lastPrice: Último preço do ativo
    - sma: Média Móvel Simples (SMA)
    - bollinger: Bandas de Bollinger
    - ichimoku: Indicadores Ichimoku

    Retorna:
    - score: Pontuação total do sinal de compra
    """
    score = 0

    if rsi is not None and rsi < dynamicRSIOversold:
        score += 1

    if latestHistogram is not None and latestHistogram < 0:
        score += 1

    if stochastic is not None and stochastic < dynamicStochOversold:
        score += 1

    if bollinger and lastPrice < bollinger["lower"]:
        score += 1

    if (
        ichimoku
        and ichimoku.get("kijunSen")
        and ichimoku["kijunSen"][-1] is not None
        and lastPrice < ichimoku["kijunSen"][-1]
    ):
        score += 1

    if sma is not None and lastPrice < sma * (1 - CONFIG["buyPriceBuffer"]):
        score += 1

    return score
