import os
import asyncio
from dotenv import load_dotenv
import tensorflow as tf
from .set_globals import CONFIG, positions, predictiveModels, live_prices
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager

load_dotenv()


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
