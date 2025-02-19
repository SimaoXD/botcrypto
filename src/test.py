import hashlib
import hmac
import math
import os
import asyncio
import json
import time
import requests
import websockets
import numpy as np
import tensorflow as tf
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
CONFIG = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "quantity": "0.00002000",
            "rsiOversold": 30,
            "rsiOverbought": 70,
        },
        {
            "symbol": "TRUMPUSDT",
            "quantity": "1.00000000",
            "rsiOversold": 30,
            "rsiOverbought": 70,
        },
        {
            "symbol": "BNBBRL",
            "quantity": "0.00300000",
            "rsiOversold": 36,
            "rsiOverbought": 38,
        },
        {"symbol": "USDTBRL", "quantity": "2", "rsiOversold": 30, "rsiOverbought": 70},
        {
            "symbol": "DOGEUSDT",
            "quantity": "4.00000000",
            "rsiOversold": 40,
            "rsiOverbought": 50,
        },
        {
            "symbol": "BNBETH",
            "quantity": "1.00000000",
            "rsiOversold": 40,
            "rsiOverbought": 60,
        },
        {
            "symbol": "SOLBNB",
            "quantity": "0.01000000",
            "rsiOversold": 30,
            "rsiOverbought": 70,
        },
        {
            "symbol": "ETCBNB",
            "quantity": "0.01000000",
            "rsiOversold": 35,
            "rsiOverbought": 65,
        },
        {
            "symbol": "BNBUSDT",
            "quantity": "0.00800000",
            "rsiOversold": 35,
            "rsiOverbought": 65,
        },
        {
            "symbol": "SOLUSDT",
            "quantity": "0.03000000",
            "rsiOversold": 35,
            "rsiOverbought": 65,
        },
        {
            "symbol": "SHIBUSDT",
            "quantity": "61170.00",
            "rsiOversold": 35,
            "rsiOverbought": 65,
        },
    ],
    "rsiPeriod": 14,
    "smaPeriod": 20,  # Para SMA e Bollinger Bands
    "stochPeriod": 14,
    "stochOversold": 20,
    "stochOverbought": 80,
    "adxPeriod": 14,
    "ichimoku": {
        "conversionPeriod": 9,
        "basePeriod": 26,
        "spanBPeriod": 52,
        "displacement": 26,
    },
    "atrPeriod": 14,
    "pollingInterval": 3000,  # em milissegundos
    "buyPriceBuffer": 0.01,  # sinal de compra: preço 1% abaixo da SMA
    "minProfitPercent": 2,
    "stopLossPercent": 2,
    "takeProfitPercent": 3,
    "minVolume": 10,
    "narrowRangeThreshold": 0.005,
    "adxThreshold": 15,
    "signalScoreThreshold": 1,
    "positionFraction": 1.0,
    "interval": "15m",
    "apiUrl": os.getenv("API_URL"),
    "apiKey": os.getenv("API_KEY"),
    "secretKey": os.getenv("SECRET_KEY"),
}

# Dicionários globais para armazenar estado
live_prices = {}  # Armazena os preços em tempo real
positions = {}  # Exemplo: {"DOGEUSDT": {"positionOpen": False, "lastBuyPrice": None}}
predictiveModels = {}  # Modelos preditivos para cada ativo


# ============================================================
# WEBSOCKET PARA MONITORAR PREÇOS EM TEMPO REAL
# ============================================================
async def start_websocket():
    """Conecta-se à Binance via WebSocket e atualiza preços em tempo real."""
    client = await AsyncClient.create(CONFIG["apiKey"], CONFIG["secretKey"])
    bm = BinanceSocketManager(client)

    # Criar conexões de WebSocket para cada ativo
    streams = [f"{symbol['symbol'].lower()}@trade" for symbol in CONFIG["symbols"]]
    ws = bm.multiplex_socket(streams)

    async with ws as stream:
        while True:
            data = await stream.recv()
            if "data" in data:
                event = data["data"]
                symbol = event["s"]
                price = float(event["p"])
                live_prices[symbol] = price  # Atualiza preço globalmente
                print(f"Preço atualizado: {symbol} = {price}")

    await client.close_connection()


# ============================================================
# EXECUTAR O WEBSOCKET (RODAR NO FUNDO)
# ============================================================
async def main():
    """Inicia o WebSocket e mantém o código rodando."""
    websocket_task = asyncio.create_task(start_websocket())

    # O código abaixo pode ser usado para lógica de trading futura
    while True:
        await asyncio.sleep(5)  # Aguarda 5 segundos antes de continuar
        print("Monitorando preços...")


# Inicia o loop assíncrono
# if __name__ == "__main__":
#     asyncio.run(main())


# ============================================================
# FUNÇÃO AUXILIAR: CONVERTE CANDLE EM OBJETO DESCRITIVO
# ============================================================
def parseCandle(candle):
    return {
        "open": float(candle[1]),
        "high": float(candle[2]),
        "low": float(candle[3]),
        "close": float(candle[4]),
        "volume": float(candle[5]),
    }


# ============================================================
# INDICADORES TÉCNICOS
# ============================================================


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


def calculateSMA(prices, period):
    if len(prices) < period:
        return None
    recent_prices = prices[-period:]
    return sum(recent_prices) / period


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


def calculateATR(candles, period):
    if len(candles) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i - 1]
        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"]),
        )
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period


def calculateADX(candles, period):
    if len(candles) < period + 1:
        return None
    trs = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(candles)):
        curr = candles[i]
        prev = candles[i - 1]
        tr = max(
            curr["high"] - curr["low"],
            abs(curr["high"] - prev["close"]),
            abs(curr["low"] - prev["close"]),
        )
        trs.append(tr)
        up_move = curr["high"] - prev["high"]
        down_move = prev["low"] - curr["low"]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    atr = sum(trs[:period])
    smooth_plus = sum(plus_dm[:period])
    smooth_minus = sum(minus_dm[:period])
    dxs = []
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
        smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
        smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period
        plus_di = 100 * (smooth_plus / (atr + 1e-10))
        minus_di = 100 * (smooth_minus / (atr + 1e-10))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        dxs.append(dx)
    if len(dxs) < period:
        return None
    return sum(dxs[:period]) / period


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


# ============================================================
# MODELO PREDITIVO – LSTM COM NORMALIZAÇÃO
# ============================================================
class PredictiveAnalysis:
    def __init__(self):
        self.model = self.create_model()
        self.train_counter = 0
        self.is_training = False

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, input_shape=(5, 1), return_sequences=False))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def normalize_prices(self, prices):
        base = prices[-1]
        return [p / base for p in prices]

    async def train_model(self, prices):
        if len(prices) < 6:
            return
        self.train_counter += 1
        if self.train_counter % 10 != 0:
            return  # Treina a cada 10 iterações
        if self.is_training:
            print("Treinamento já em andamento. Pulando nova chamada.")
            return
        self.is_training = True
        try:
            normalized = self.normalize_prices(prices)
            inputs = []
            outputs = []
            for i in range(len(normalized) - 5):
                inputs.append([[val] for val in normalized[i : i + 5]])
                outputs.append([normalized[i + 5]])
            xs = np.array(inputs, dtype=np.float32)
            ys = np.array(outputs, dtype=np.float32)
            # O treinamento é executado em thread separada para não bloquear o loop assíncrono
            await asyncio.to_thread(self.model.fit, xs, ys, epochs=50, verbose=0)
            print("📈 [LSTM] Modelo treinado incrementalmente.")
        except Exception as error:
            print("Erro no treinamento do modelo:", error)
        finally:
            self.is_training = False

    async def predict_next_price(self, prices):
        if len(prices) < 5:
            return None
        normalized = self.normalize_prices(prices)
        input_seq = [[val] for val in normalized[-5:]]
        input_tensor = np.array([input_seq], dtype=np.float32)
        prediction = self.model.predict(input_tensor)
        predicted_normalized = prediction[0][0]
        return predicted_normalized * prices[-1]

    async def make_trading_decision(
        self, prices, last_price, positions, symbol, quantity, order_func
    ):
        predicted = await self.predict_next_price(prices)
        if predicted is None:
            return
        print(f"🔮 [{symbol}] Previsão LSTM: {predicted:.2f}")
        dynamic_threshold = 1.01  # Ajustável conforme volatilidade
        if (
            not positions[symbol]["positionOpen"]
            and predicted > last_price * dynamic_threshold
        ):
            print(f"📈 [{symbol}] Previsão indica alta! Comprando.")
            await order_func(symbol, quantity, "BUY", last_price)
            positions[symbol]["positionOpen"] = True
            positions[symbol]["lastBuyPrice"] = last_price
        elif positions[symbol]["positionOpen"] and predicted < last_price * 0.99:
            print(f"📉 [{symbol}] Previsão indica queda! Vendendo.")
            await order_func(symbol, quantity, "SELL", last_price)
            positions[symbol]["positionOpen"] = False
            positions[symbol]["lastBuyPrice"] = None


# ============================================================
# ENVIO DE ORDENS COM RETRY
# ============================================================
async def getSymbolFilters(symbol):
    url = f"{CONFIG['apiUrl']}/api/v3/exchangeInfo"
    response = await asyncio.to_thread(requests.get, url)
    data = response.json()
    symbol_info = None
    for s in data["symbols"]:
        if s["symbol"] == symbol:
            symbol_info = s
            break
    if not symbol_info:
        raise Exception(f"[{symbol}] Símbolo não encontrado.")
    filters = {}
    for f in symbol_info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            filters["stepSize"] = float(f["stepSize"])
        elif f["filterType"] == "PRICE_FILTER":
            filters["tickSize"] = float(f["tickSize"])
    return filters


def adjustToStep(value, step_size):
    return math.floor(value / step_size) * step_size


async def sendOrder(symbol, quantity, side, price):
    # Obtém filtros do símbolo
    filters = await getSymbolFilters(symbol)
    quantity = adjustToStep(float(quantity), filters["stepSize"])
    price = adjustToStep(float(price), filters["tickSize"])
    if quantity <= 0 or not price or math.isnan(price):
        print(
            f"❌ Erro: quantidade ({quantity}) ou preço ({price}) inválidos para {symbol}"
        )
        return None

    order = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "quantity": f"{quantity:.8f}",
        "price": f"{price:.8f}",
        "timeInForce": "GTC",
        "timestamp": int(time.time() * 1000),
    }

    qs = "&".join([f"{k}={v}" for k, v in order.items()])
    signature = hmac.new(
        CONFIG["secretKey"].encode(), qs.encode(), hashlib.sha256
    ).hexdigest()
    order["signature"] = signature

    print("[DEBUG] Enviando ordem:", order)
    headers = {
        "X-MBX-APIKEY": CONFIG["apiKey"],
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = "&".join([f"{k}={v}" for k, v in order.items()])
    try:
        response = await asyncio.to_thread(
            requests.post,
            CONFIG["apiUrl"] + "/api/v3/order",
            data=data,
            headers=headers,
        )
        result = response.json()
        print("✅ Ordem enviada com sucesso:", result)
        return result
    except Exception as error:
        print("❌ Erro ao enviar ordem:", error)
        return None


async def send_order_with_retry(symbol, quantity, side, price, retries=3):
    if price is None or math.isnan(price):
        print(f"[{symbol}] Erro: Preço inválido: {price}")
        return None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[DEBUG] Tentativa {attempt}: Enviando ordem {side} para {symbol} a {price}"
            )
            result = await sendOrder(symbol, quantity, side, price)
            if result:
                print(f"[DEBUG] Ordem enviada com sucesso na tentativa {attempt}")
                return result
        except Exception as error:
            print(f"[{symbol}] Tentativa {attempt} falhou:", error)
            if attempt == retries:
                print(f"[{symbol}] Todas as tentativas falharam.")
                raise error
        await asyncio.sleep((2**attempt))
    return None


# ============================================================
# SISTEMA DE PONTUAÇÃO DOS SINAIS (SCORING)
# ============================================================
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


# ============================================================
# MÓDULO DE BACKTESTING
# ============================================================
async def run_backtest(historical_candles, symbol_config):
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

        rsi = calculateRSI(prices, CONFIG["rsiPeriod"])
        sma = calculateSMA(prices, CONFIG["smaPeriod"])
        macd_data = calculateMACD(prices, 12, 26, 9)
        bollinger = calculateBollingerBands(prices, CONFIG["smaPeriod"], 2)
        stochastic = calculateStochastic(highs, lows, prices, CONFIG["stochPeriod"])
        atr = calculateATR(current_slice, CONFIG["atrPeriod"])
        adx = calculateADX(current_slice, CONFIG["adxPeriod"])
        ichimoku = calculateIchimoku(current_slice, CONFIG["ichimoku"])

        dynamicRSIOversold = symbol_config["rsiOversold"] - (
            atr / last_price * 10 if atr else 0
        )
        dynamicStochOversold = CONFIG["stochOversold"] - (
            atr / last_price * 10 if atr else 0
        )

        range_val = last_candle["high"] - last_candle["low"]
        if range_val / last_price < CONFIG["narrowRangeThreshold"]:
            continue
        if adx is not None and adx < CONFIG["adxThreshold"]:
            continue

        latest_hist = (
            macd_data["histogram"][len(prices) - 1]
            if macd_data
            and macd_data.get("histogram")
            and (len(prices) - 1) < len(macd_data["histogram"])
            else None
        )

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

    win_rate = (wins / total_trades * 100) if total_trades else 0
    cumulative_return = ((cash - 10000) / 10000) * 100
    return {
        "winRate": win_rate,
        "maxDrawdown": max_drawdown * 100,
        "cumulativeReturn": cumulative_return,
        "simulatedPositions": simulated_positions,
    }


# ============================================================
# ESTRATÉGIA AO VIVO
# ============================================================
async def load_historical_data(symbol, interval, limit=500):
    endpoint = "/api/v3/klines"
    params = f"?symbol={symbol}&interval={interval}&limit={limit}"
    url = CONFIG["apiUrl"] + endpoint + params
    response = await asyncio.to_thread(requests.get, url)
    data = response.json()
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
        print(f"[{symbol}] Erro na estratégia:", error)


# ============================================================
# EXECUÇÃO AO VIVO: ITERA SOBRE OS ATIVOS
# ============================================================
async def run_live_strategy():
    for symbol_config in CONFIG["symbols"]:
        await run_strategy_for_symbol(symbol_config)


# ============================================================
# LOOP PRINCIPAL: EXECUÇÃO PERIÓDICA AO VIVO
# ============================================================
async def main():
    while True:
        await run_live_strategy()
        # Intervalo de execução definido em pollingInterval (convertido para segundos)
        await asyncio.sleep(CONFIG["pollingInterval"] / 1000.0)


if __name__ == "__main__":
    asyncio.run(main())
