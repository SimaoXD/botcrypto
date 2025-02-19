import asyncio
import math
import requests
from global_configuration import CONFIG, parseCandle, predictiveModels, positions
from predictive_model import PredictiveAnalysis
from technite_indicators import (
    calculateRSI,
    calculateSMA,
    calculateMACD,
    calculateBollingerBands,
    calculateStochastic,
    calculateATR,
    calculateADX,
    calculateIchimoku,
)
from signal_score import evaluateSignalScore
from send_order import send_order_with_retry


async def load_historical_data(symbol, interval, limit=500):
    endpoint = "/api/v3/klines"
    params = f"?symbol={symbol}&interval={interval}&limit={limit}"
    url = CONFIG["apiUrl"] + endpoint + params
    response = await asyncio.to_thread(requests.get, url)
    data = response.json()
    print(f"[DEBUG] Dados brutos recebidos para {symbol}: {data[:5]}")  # Imprime os 5 primeiros candles
    return [parseCandle(c) for c in data]

async def run_strategy_for_symbol(symbol_config):
    symbol = symbol_config["symbol"]
    quantity = symbol_config["quantity"]
    rsiOversold = symbol_config["rsiOversold"]
    rsiOverbought = symbol_config["rsiOverbought"]
    minVolume = symbol_config.get("minVolume", CONFIG["minVolume"])

    if symbol not in positions:
        positions[symbol] = {"positionOpen": False, "lastBuyPrice": None}
    trade_executed = False

    try:
        # Carrega 500 candles históricos
        candles = await load_historical_data(symbol, CONFIG["interval"], 500)
        prices = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        last_candle = candles[-1] if candles else None

        if (
            not last_candle
            or not last_candle["close"]
            or math.isnan(last_candle["close"])
        ):
            print(f"❌ [ERRO] Preço inválido no candle para {symbol}.")
            return
        last_price = last_candle["close"]

        print(f"\n=== [{symbol}] Ao Vivo ===")
        print(f"Preço atual: {last_price}")

        if last_candle["volume"] < minVolume:
            print(f"[{symbol}] Volume baixo ({last_candle['volume']}). Ignorando.")
            return

        rsi = calculateRSI(prices, CONFIG["rsiPeriod"])
        sma = calculateSMA(prices, CONFIG["smaPeriod"])
        macd_data = calculateMACD(prices, 12, 26, 9)
        bollinger = calculateBollingerBands(prices, CONFIG["smaPeriod"], 2)
        stochastic = calculateStochastic(highs, lows, prices, CONFIG["stochPeriod"])
        atr = calculateATR(candles, CONFIG["atrPeriod"])
        adx = calculateADX(candles, CONFIG["adxPeriod"])
        ichimoku = calculateIchimoku(candles, CONFIG["ichimoku"])

        if rsi is not None:
            print(f"RSI: {rsi:.2f}")
        if sma is not None:
            print(f"SMA: {sma:.2f}")
        if macd_data and macd_data.get("histogram"):
            latest_hist = macd_data["histogram"][-1]
            print(
                f"MACD Histogram: {latest_hist:.2f}"
                if latest_hist is not None
                else "N/A"
            )
        if bollinger:
            print(
                f"Bollinger -> Upper: {bollinger['upper']:.2f}, Middle: {bollinger['middle']:.2f}, Lower: {bollinger['lower']:.2f}"
            )
        if stochastic is not None:
            print(f"Estocástico: {stochastic:.2f}")
        if atr is not None:
            print(f"ATR: {atr:.2f}")
        if adx is not None:
            print(f"ADX: {adx:.2f}")
        if (
            ichimoku
            and ichimoku.get("kijunSen")
            and ichimoku["kijunSen"][-1] is not None
        ):
            print(f"Ichimoku -> Kijun-sen: {ichimoku['kijunSen'][-1]}")
        else:
            print("Ichimoku: Dados insuficientes para cálculo.")

        dynamicRSIOversold = rsiOversold - (atr / last_price * 10 if atr else 0)
        dynamicStochOversold = CONFIG["stochOversold"] - (
            atr / last_price * 10 if atr else 0
        )
        range_val = last_candle["high"] - last_candle["low"]
        dynamicNarrowRangeThreshold = CONFIG["narrowRangeThreshold"] * (
            atr / last_price * 2 if atr else 1
        )
        if range_val / last_price < dynamicNarrowRangeThreshold:
            print(f"[{symbol}] Mercado em consolidação (faixa estreita).")
            return
        if adx is not None and adx < CONFIG["adxThreshold"]:
            print(
                f"[{symbol}] ADX ({adx:.2f}) abaixo do threshold ({CONFIG['adxThreshold']}). Ignorando sinais."
            )
            return

        score = evaluateSignalScore(
            rsi,
            dynamicRSIOversold,
            (
                macd_data["histogram"][-1]
                if macd_data and macd_data.get("histogram")
                else None
            ),
            stochastic,
            dynamicStochOversold,
            last_price,
            sma,
            bollinger,
            ichimoku,
        )
        print(f"[{symbol}] Pontuação dos sinais: {score}")

        # Quantidade ajustada (posicionamento fracionado)
        adjusted_quantity = f"{float(quantity) * CONFIG['positionFraction']:.3f}"

        if (
            not positions[symbol]["positionOpen"]
            and score >= CONFIG["signalScoreThreshold"]
        ):
            print(f"[{symbol}] Sinal de COMPRA detectado (score: {score}).")
            await send_order_with_retry(symbol, adjusted_quantity, "BUY", last_price)
            positions[symbol]["positionOpen"] = True
            positions[symbol]["lastBuyPrice"] = last_price
            trade_executed = True
        elif (
            positions[symbol]["positionOpen"]
            and score < CONFIG["signalScoreThreshold"] - 1
        ):
            print(f"[{symbol}] Sinal de VENDA detectado (score: {score}).")
            await send_order_with_retry(symbol, adjusted_quantity, "SELL", last_price)
            positions[symbol]["positionOpen"] = False
            positions[symbol]["lastBuyPrice"] = None
            trade_executed = True

        if (
            positions[symbol]["positionOpen"]
            and positions[symbol]["lastBuyPrice"]
            and atr is not None
        ):
            dynamic_stop_loss = (
                positions[symbol]["lastBuyPrice"]
                * (1 - CONFIG["stopLossPercent"] / 100)
                - atr
            )
            dynamic_take_profit = (
                positions[symbol]["lastBuyPrice"]
                * (1 + CONFIG["takeProfitPercent"] / 100)
                + atr
            )
            if last_price <= dynamic_stop_loss:
                print(
                    f"[{symbol}] Stop-loss atingido ({last_price:.2f} <= {dynamic_stop_loss:.2f}). Vendendo."
                )
                await send_order_with_retry(
                    symbol, adjusted_quantity, "SELL", last_price
                )
                positions[symbol]["positionOpen"] = False
                positions[symbol]["lastBuyPrice"] = None
                trade_executed = True
            elif last_price >= dynamic_take_profit:
                print(
                    f"[{symbol}] Take-profit atingido ({last_price:.2f} >= {dynamic_take_profit:.2f}). Vendendo."
                )
                await send_order_with_retry(
                    symbol, adjusted_quantity, "SELL", last_price
                )
                positions[symbol]["positionOpen"] = False
                positions[symbol]["lastBuyPrice"] = None
                trade_executed = True

        if not trade_executed:
            if symbol not in predictiveModels:
                predictiveModels[symbol] = PredictiveAnalysis()
            await predictiveModels[symbol].train_model(prices)
            await predictiveModels[symbol].make_trading_decision(
                prices,
                last_price,
                positions,
                symbol,
                adjusted_quantity,
                send_order_with_retry,
            )

    except Exception as error:
        print(f"Erro na execução da estratégia ao vivo: {error}")