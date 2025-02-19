# saldo.py
import os
from binance.client import Client


def get_balance(asset):
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("SECRET_KEY")

    client = Client(api_key, api_secret)
    account_info = client.get_account()
    balances = account_info["balances"]

    for balance in balances:
        free = float(balance["free"])
        if free > 0 and balance["asset"] == asset:
            return float(balance["free"])
    return 0.0
