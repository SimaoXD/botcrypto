import asyncio
import hashlib
import hmac
import math
import time
import requests
from global_configuration import CONFIG  # Importa configurações globais

async def getSymbolFilters(symbol):
    url = f"{CONFIG['apiUrl']}/api/v3/exchangeInfo"
    response = await asyncio.to_thread(requests.get, url)
    data = response.json()
    symbol_info = next((s for s in data["symbols"] if s["symbol"] == symbol), None)

    if not symbol_info:
        raise Exception(f"[{symbol}] Símbolo não encontrado.")

    filters = {
        "stepSize": float(next(f["stepSize"] for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE")),
        "tickSize": float(next(f["tickSize"] for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER")),
    }

    return filters

def adjustToStep(value, step_size):
    return math.floor(value / step_size) * step_size

async def sendOrder(symbol, quantity, side, price):
    filters = await getSymbolFilters(symbol)
    quantity = adjustToStep(float(quantity), filters["stepSize"])
    price = adjustToStep(float(price), filters["tickSize"])

    if quantity <= 0 or not price or math.isnan(price):
        print(f"❌ Erro: Quantidade ({quantity}) ou preço ({price}) inválidos para {symbol}")
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
    signature = hmac.new(CONFIG["secretKey"].encode(), qs.encode(), hashlib.sha256).hexdigest()
    order["signature"] = signature

    print("[DEBUG] Enviando ordem:", order)
    headers = {
        "X-MBX-APIKEY": CONFIG["apiKey"],
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = "&".join([f"{k}={v}" for k, v in order.items()])

    try:
        response = await asyncio.to_thread(requests.post, CONFIG["apiUrl"] + "/api/v3/order", data=data, headers=headers)
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
            print(f"[DEBUG] Tentativa {attempt}: Enviando ordem {side} para {symbol} a {price}")
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
