

import requests
import json
import os
from binance.client import Client

def get_symbol_filters(symbol):
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("SECRET_KEY")

    client = Client(api_key, api_secret)
    exchange_info = client.get_exchange_info()

    symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)
    if symbol_info:
        filters = {}
        for f in symbol_info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                filters["minPrice"] = float(f["minPrice"])
                filters["tickSize"] = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                filters["minQty"] = float(f["minQty"])
                filters["stepSize"] = float(f["stepSize"])
            elif f["filterType"] == "MIN_NOTIONAL":
                filters["minNotional"] = float(f["minNotional"])
        return filters
    else:
        return None

# Exemplo de uso
filters = get_symbol_filters("BTCUSDT")
print(f"Filtros salvos para BTCUSDT: {filters}")


# Exemplo de uso
filters = get_symbol_filters("BTCUSDT")
print(f"Filtros salvos para BTCUSDT: {filters}")


def save_filters_to_file(symbol, filters):
    with open(f"{symbol}_filters.json", "w") as file:
        json.dump(filters, file, indent=4)
    print(f"Filtros salvos para {symbol}: {filters}")


if __name__ == "__main__":
    symbols = ["BTCUSDT", "FDUSDBRL"]
    for symbol in symbols:
        filters = get_symbol_filters(symbol)
        if filters:
            save_filters_to_file(symbol, filters)

