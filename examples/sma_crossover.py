from AlgorithmImports import *


class SmaCrossover(QCAlgorithm):
    """Classic SMA crossover"""
    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2020, 1, 1)
        self.SetCash(100_000)

        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.fast = self.SMA(self.symbol, 20, Resolution.Daily)
        self.slow = self.SMA(self.symbol, 50, Resolution.Daily)

    def OnData(self, data):
        if not (self.fast.IsReady and self.slow.IsReady):
            return
        if not data.ContainsKey(self.symbol) or data[self.symbol] is None:
            return

        invested = self.Portfolio[self.symbol].Invested
        if self.fast.Value > self.slow.Value and not invested:
            self.SetHoldings(self.symbol, 1.0)
        elif self.fast.Value < self.slow.Value and invested:
            self.Liquidate(self.symbol)
