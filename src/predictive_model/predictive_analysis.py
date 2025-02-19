import asyncio
import numpy as np
import tensorflow as tf


class PredictiveAnalysis:
    def __init__(self):
        """
        Inicializa a classe com um modelo LSTM e variáveis de controle para treinamento.
        """
        self.model = self.create_model()
        self.train_counter = 0
        self.is_training = False

    def create_model(self):
        """
        Cria e compila um modelo LSTM para análise preditiva.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(50, input_shape=(5, 1), return_sequences=False),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def normalize_prices(self, prices):
        """
        Normaliza os preços em relação ao último valor da lista.
        """
        base = prices[-1] if prices[-1] != 0 else 1  # Evita divisão por zero
        return [p / base for p in prices]

    async def train_model(self, prices):
        """
        Treina o modelo LSTM de forma assíncrona a cada 10 iterações para evitar sobrecarga.
        """
        if len(prices) < 6:
            return

        self.train_counter += 1
        if self.train_counter % 10 != 0:
            return  # Treina apenas a cada 10 chamadas

        if self.is_training:
            print("Treinamento já em andamento. Pulando nova chamada.")
            return

        self.is_training = True
        try:
            normalized = self.normalize_prices(prices)
            inputs, outputs = [], []

            for i in range(len(normalized) - 5):
                inputs.append([[val] for val in normalized[i : i + 5]])
                outputs.append([normalized[i + 5]])

            xs = np.array(inputs, dtype=np.float32)
            ys = np.array(outputs, dtype=np.float32)

            await asyncio.to_thread(self.model.fit, xs, ys, epochs=50, verbose=0)
            print("📈 [LSTM] Modelo treinado incrementalmente.")
        except Exception as error:
            print("Erro no treinamento do modelo:", error)
        finally:
            self.is_training = False

    async def predict_next_price(self, prices):
        """
        Realiza uma previsão do próximo preço com base nos últimos 5 valores.
        """
        if len(prices) < 5:
            return None

        normalized = self.normalize_prices(prices)
        input_seq = [[val] for val in normalized[-5:]]
        input_tensor = np.array([input_seq], dtype=np.float32)

        prediction = self.model.predict(input_tensor, verbose=0)
        predicted_normalized = prediction[0][0]

        return predicted_normalized * prices[-1]

    async def make_trading_decision(
        self, prices, last_price, positions, symbol, quantity, order_func
    ):
        """
        Toma decisões de compra ou venda com base nas previsões do modelo.
        """
        predicted = await self.predict_next_price(prices)
        if predicted is None:
            return

        print(f"🔮 [{symbol}] Previsão LSTM: {predicted:.2f}")

        dynamic_threshold = 1.01  # Ajustável conforme volatilidade

        if (
            not positions[symbol].get("positionOpen", False)
            and predicted > last_price * dynamic_threshold
        ):
            print(f"📈 [{symbol}] Previsão indica alta! Comprando.")
            await order_func(symbol, quantity, "BUY", last_price)
            positions[symbol]["positionOpen"] = True
            positions[symbol]["lastBuyPrice"] = last_price
        elif (
            positions[symbol].get("positionOpen", False)
            and predicted < last_price * 0.99
        ):
            print(f"📉 [{symbol}] Previsão indica queda! Vendendo.")
            await order_func(symbol, quantity, "SELL", last_price)
            positions[symbol]["positionOpen"] = False
            positions[symbol]["lastBuyPrice"] = None
