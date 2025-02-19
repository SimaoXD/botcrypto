
import requests
import json
import os
from binance.client import Client

api_key = os.getenv("API_KEY")
api_secret = os.getenv("SECRET_KEY")

client = Client(api_key, api_secret)

account_info = client.get_account()
balances = account_info["balances"]

# Filtrar e exibir apenas os ativos que têm saldo livre maior que zero
for balance in balances:
    free = float(balance["free"])
    if free > 0:
        print(f"Asset: {balance['asset']}, Free: {balance['free']}")



def get_symbol_filters(symbol):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url).json()

    for s in response["symbols"]:
        if s["symbol"] == symbol:
            return s["filters"]

    return None

def save_filters_to_file(symbol, filters):
    with open(f"{symbol}_filters.json", "w") as file:
        json.dump(filters, file, indent=4)
    print(f"Filtros salvos para {symbol}: {filters}")

if __name__ == "__main__":
    symbols = ["BTCFDUSD", "FDUSDBNB"]
    for symbol in symbols:
        filters = get_symbol_filters(symbol)
        if filters:
            save_filters_to_file(symbol, filters)