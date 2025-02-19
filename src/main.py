from http import client
import os
import math
import time
import json
import asyncio
import requests
import hmac
import hashlib
import numpy as np
import tensorflow as tf
import websockets
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
from dotenv import load_dotenv

# Importando módulos internos
from global_configuration import (
    CONFIG,
    positions,
    predictiveModels,
    parseCandle,
    start_websocket,
)
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
from predictive_model import PredictiveAnalysis
from send_order import sendOrder, send_order_with_retry
from signal_score import evaluateSignalScore
from backtesting_module import run_backtest
from live_strategy import run_strategy_for_symbol

# Carrega variáveis de ambiente
load_dotenv()


# Função para carregar filtros de um arquivo JSON
def load_filters_from_file(symbol):
    try:
        with open(f"{symbol}_filters.json", "r") as file:
            filters = json.load(file)
        return filters
    except FileNotFoundError:
        print(
            f"Arquivo de filtros não encontrado para {symbol}. Execute o script exchangeinfo.py primeiro."
        )
        return None


# Ajustar quantidade mínima com base no valor notional mínimo
def adjust_quantity_for_min_notional(symbol_config):
    symbol = symbol_config["symbol"]
    filters = load_filters_from_file(symbol)

    if not filters:
        return symbol_config  # Se não conseguir carregar os filtros, retorna a configuração original

    try:
        min_notional = next(
            f["minNotional"] for f in filters if f["filterType"] == "MIN_NOTIONAL"
        )
    except StopIteration:
        print(f"Filtro de 'MIN_NOTIONAL' não encontrado para {symbol}.")
        return symbol_config

    price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    quantity = float(symbol_config["quantity"])
    notional_value = price * quantity

    if notional_value < float(min_notional):
        print(
            f"Ordem para {symbol} não atende ao valor notional mínimo. Ajustando quantidade."
        )
        adjusted_quantity = float(min_notional) / price
        symbol_config["quantity"] = f"{adjusted_quantity:.8f}"

    return symbol_config


# Ajusta todas as quantidades nos símbolos configurados
CONFIG["symbols"] = [
    adjust_quantity_for_min_notional(symbol_config)
    for symbol_config in CONFIG["symbols"]
]

# ============================================================
# INSTÂNCIA DO MODELO PREDITIVO – LSTM COM NORMALIZAÇÃO
# ============================================================
predictive_model = PredictiveAnalysis()  # Criando instância
model = predictive_model.create_model()


# ============================================================
# LOOP PRINCIPAL: EXECUÇÃO PERIÓDICA AO VIVO
# ============================================================
async def main():
    """Executa a estratégia de trading ao vivo em um loop infinito com intervalos configuráveis."""
    while True:
        print("Iniciando execução da estratégia.")
        for symbol_config in CONFIG["symbols"]:
            print(f"Iniciando estratégia para {symbol_config['symbol']}.")
            await run_strategy_for_symbol(
                symbol_config
            )  # Passa a configuração de cada símbolo
        print("Estratégias concluídas, aguardando próximo ciclo.")
        await asyncio.sleep(
            CONFIG["pollingInterval"] / 1000.0
        )  # Converte ms para segundos


if __name__ == "__main__":
    print("Iniciando o programa principal.")
    asyncio.run(main())  # Executa a função principal
