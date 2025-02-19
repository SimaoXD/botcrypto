import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
CONFIG = {
    "symbols": [
        {
            "symbol": "BTCFDUSD",
            "quantity": "0.0020000",
            "rsiOversold": 40,  # Antes: 30
            "rsiOverbought": 60,  # Antes: 70
        },
        {
            "symbol": "BNBETH",
            "quantity": "1.00000000",
            "rsiOversold": 42,  # Antes: 40
            "rsiOverbought": 48,  # Antes: 60
        },
        {
            "symbol": "SOLBNB",
            "quantity": "0.01000000",
            "rsiOversold": 40,  # Antes: 30
            "rsiOverbought": 48,  # Antes: 70
        },
        {
            "symbol": "ETCBNB",
            "quantity": "0.01000000",
            "rsiOversold": 38,  # Antes: 35
            "rsiOverbought": 48,  # Antes: 65
        },
        {
            "symbol": "BNBFDUSD",
            "quantity": "0.00800000",
            "rsiOversold": 38,  # Antes: 35
            "rsiOverbought": 48,  # Antes: 65
        },
        {
            "symbol": "SOLFDUSD",
            "quantity": "0.03000000",
            "rsiOversold": 38,  # Antes: 35
            "rsiOverbought": 58,  # Antes: 65
        },
        {
            "symbol": "SHIBFDUSD",
            "quantity": "1.00",
            "rsiOversold": 38,  # Antes: 35
            "rsiOverbought": 58,  # Antes: 65
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


def parseCandle(candle):
    try:
        if len(candle) < 6:
            print(f"Candle inválido ou incompleto: {candle}")
            return None
        return {
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
        }
    except ValueError as e:
        print(f"Erro ao converter candle: {candle}\nErro: {e}")
        return None
    except IndexError as e:
        print(f"Indice inexistente no candle: {candle}\nErro: {e}")
        return None


# def parseCandle(candle):
#     return {
#         "timestamp": candle[0],
#         "open": float(candle[1]),
#         "high": float(candle[2]),
#         "low": float(candle[3]),
#         "close": float(candle[4]),
#         "volume": float(candle[5]),
#         "number_of_trades": int(candle[8]),
#         "buy_volume": float(candle[9]),  # Volume de compras do ativo
#         "sell_volume": float(candle[10]),  # Volume de vendas do ativo
#         "vwap": (
#             (float(candle[5]) * float(candle[4])) / float(candle[5])
#             if float(candle[5]) > 0
#             else 0
#         ),  # VWAP
#         "range": float(candle[2]) - float(candle[3]),  # Range
#         "is_bullish": float(candle[4])
#         > float(candle[1]),  # True se o fechamento for maior que a abertura
#     }
