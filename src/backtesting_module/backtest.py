import asyncio
from global_configuration import CONFIG
from technite_indicators import (
    calculateEMA,
    calculateMACD,
    calculateRSI,
    calculateATR,
    calculateADX,
    calculateBollingerBands,
    calculateIchimoku,
    calculateSMA,
    calculateStochastic,
)
from signal_score import evaluateSignalScore


def parseCandle(candle):
    """Converte um candle histórico em um dicionário estruturado."""
    return {
        "open": float(candle[1]),
        "high": float(candle[2]),
        "low": float(candle[3]),
        "close": float(candle[4]),
        "volume": float(candle[5]),
    }


async def run_backtest(historical_candles, symbol_config):
    """
    Executa um backtest baseado nos candles históricos e configurações do símbolo.

    Parâmetros:
    - historical_candles: Lista de candles históricos
    - symbol_config: Configurações do símbolo a ser testado

    Retorna:
    - Dicionário com métricas do backtest (winRate, maxDrawdown, cumulativeReturn, simulatedPositions)
    """
    cash = 10000  # Capital inicial
    position = None
    peak = cash
    max_drawdown = 0
    wins = 0
    total_trades = 0
    simulated_positions = []
    candles = [parseCandle(c) for c in historical_candles]

    for i in range(CONFIG["smaPeriod"], len(candles)):
        current_slice = candles[: i + 1]
        prices = [c["close"] for c in current_slice]
        highs = [c["high"] for c in current_slice]
        lows = [c["low"] for c in current_slice]
        last_candle = current_slice[-1]
        last_price = last_candle["close"]

        # Cálculo dos indicadores técnicos
        rsi = calculateRSI(prices, CONFIG["rsiPeriod"])
        sma = calculateSMA(prices, CONFIG["smaPeriod"])
        macd_data = calculateMACD(prices, 12, 26, 9)
        bollinger = calculateBollingerBands(prices, CONFIG["smaPeriod"], 2)
        stochastic = calculateStochastic(highs, lows, prices, CONFIG["stochPeriod"])
        atr = calculateATR(current_slice, CONFIG["atrPeriod"])
        adx = calculateADX(current_slice, CONFIG["adxPeriod"])
        ichimoku = calculateIchimoku(current_slice, CONFIG["ichimoku"])

        # Ajuste dinâmico dos níveis de sobrevenda
        dynamicRSIOversold = symbol_config["rsiOversold"] - (
            atr / last_price * 10 if atr else 0
        )
        dynamicStochOversold = CONFIG["stochOversold"] - (
            atr / last_price * 10 if atr else 0
        )

        # Filtragem de candles de baixa volatilidade
        range_val = last_candle["high"] - last_candle["low"]
        if range_val / last_price < CONFIG["narrowRangeThreshold"]:
            continue
        if adx is not None and adx < CONFIG["adxThreshold"]:
            continue

        # Extração do histograma MACD
        latest_hist = (
            macd_data["histogram"][len(prices) - 1]
            if macd_data
            and macd_data.get("histogram")
            and (len(prices) - 1) < len(macd_data["histogram"])
            else None
        )

        # Avaliação do sinal
        score = evaluateSignalScore(
            rsi,
            dynamicRSIOversold,
            latest_hist,
            stochastic,
            dynamicStochOversold,
            last_price,
            sma,
            bollinger,
            ichimoku,
        )

        # Estratégia de compra/venda
        if position is None and score >= CONFIG["signalScoreThreshold"]:
            position = {"entry": last_price, "entryIndex": i}
            total_trades += 1
            simulated_positions.append({"type": "BUY", "price": last_price, "index": i})
            print(
                f"[Backtest {symbol_config['symbol']}] Compra em {last_price:.2f} (candle {i})."
            )
        elif position is not None and score < CONFIG["signalScoreThreshold"] - 1:
            profit = last_price - position["entry"]
            cash += profit
            total_trades += 1
            if profit > 0:
                wins += 1
            simulated_positions.append(
                {"type": "SELL", "price": last_price, "index": i, "profit": profit}
            )
            if cash > peak:
                peak = cash
            drawdown = (peak - cash) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            print(
                f"[Backtest {symbol_config['symbol']}] Venda em {last_price:.2f} (candle {i}) | Lucro: {profit:.2f}"
            )
            position = None

    # Cálculo de métricas
    win_rate = (wins / total_trades * 100) if total_trades else 0
    cumulative_return = ((cash - 10000) / 10000) * 100

    return {
        "winRate": win_rate,
        "maxDrawdown": max_drawdown * 100,
        "cumulativeReturn": cumulative_return,
        "simulatedPositions": simulated_positions,
    }
