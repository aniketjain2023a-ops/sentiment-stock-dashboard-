# src/backtest/backtest_engine.py

import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_cash=10000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.history = []

    def buy(self, ticker, price, quantity):
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self.history.append(f"Bought {quantity} of {ticker} at {price}")
        else:
            print("Not enough cash to buy")

    def sell(self, ticker, price, quantity):
        if self.positions.get(ticker, 0) >= quantity:
            self.positions[ticker] -= quantity
            self.cash += price * quantity
            self.history.append(f"Sold {quantity} of {ticker} at {price}")
        else:
            print(f"Not enough {ticker} to sell")

    def portfolio_value(self, market_prices):
        value = self.cash
        for ticker, qty in self.positions.items():
            value += market_prices.get(ticker, 0) * qty
        return value

if __name__ == "__main__":
    # Sample test
    engine = BacktestEngine()
    engine.buy("AAPL", 150, 10)
    engine.buy("TSLA", 700, 5)
    engine.sell("AAPL", 155, 5)
    market_prices = {"AAPL": 160, "TSLA": 720}
    print("Portfolio value:", engine.portfolio_value(market_prices))
    print("Trade history:", engine.history)