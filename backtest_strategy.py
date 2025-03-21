import numpy as np

from backtesting import Strategy


class StopLossStrategy(Strategy):
    trade_size = 0.1

    def init(self):
        super().init()
        self.signal1 = self.I(lambda: self.data.Signal)

    def calculate_stop_loss(
        self, entry_price, pips=200, pip_value=0.0001, direction="long"
    ):
        """
        Calculate the stop loss distance given the entry price, number of pips, and pip value.

        Parameters:
        entry_price (float): The price at which the trade is entered.
        pips (int): The number of pips for the stop loss. Default is 200.
        pip_value (float): The value of one pip. Default is 0.0001 for most currency pairs.
        direction (str): 'long' or 'short' to indicate the trade direction.

        Returns:
        float: The stop loss price.
        """
        sl_distance = pips * pip_value
        if direction == "long":
            stop_loss_price = entry_price - sl_distance
        elif direction == "short":
            stop_loss_price = entry_price + sl_distance
        else:
            raise ValueError("direction must be 'long' or 'short'")

        return stop_loss_price

    def next(self):
        super().next()

        # Check if any trades are winning and close them
        for trade in self.trades:
            if trade.pl > 0:
                trade.close()

        # Handle new signals
        if self.signal1[-1] == 2 and not self.position:
            current_close = self.data.Close[-1]
            sl = self.calculate_stop_loss(
                entry_price=current_close, pips=250, pip_value=0.0001, direction="long"
            )
            self.buy(size=self.trade_size, sl=sl)

        elif self.signal1[-1] == 1 and not self.position:
            current_close = self.data.Close[-1]
            sl = self.calculate_stop_loss(
                entry_price=current_close, pips=250, pip_value=0.0001, direction="short"
            )
            self.sell(size=self.trade_size, sl=sl)


class BBRSIStrategy(Strategy):
    trade_size = 0.1
    slcoef = 3
    TPcoef = 2

    def init(self):
        super().init()
        self.signal1 = self.I(lambda: self.data.Signal)

    def next(self):
        super().next()
        slatr = self.slcoef * self.data.atr[-1]
        tpatr = self.TPcoef * self.data.atr[-1]

        if self.signal1 == 2 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] - slatr
            tp1 = self.data.Close[-1] + tpatr
            self.buy(sl=sl1, tp=tp1, size=self.trade_size)

        if self.signal1 == 1 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] + slatr
            tp1 = self.data.Close[-1] - tpatr
            self.sell(sl=sl1, tp=tp1, size=self.trade_size)


class MacdStrategy(Strategy):
    trade_size = 3000
    slcoef = 1.1
    TPSLRatio = 1.5
    TPcoef = 1.5
    # rsi_length = 16

    def init(self):
        super().init()
        self.signal1 = self.I(lambda: self.data.Signal)
        # df['RSI']=ta.rsi(df.Close, length=self.rsi_length)

    def next(self):
        super().next()
        slatr = self.slcoef * self.data.ATR[-1]
        TPSLRatio = self.TPSLRatio

        # if len(self.trades)>0:
        #     if self.trades[-1].is_long and self.data.RSI[-1]>=90:
        #         self.trades[-1].close()
        #     elif self.trades[-1].is_short and self.data.RSI[-1]<=10:
        #         self.trades[-1].close()

        if self.signal1 == 2 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] - slatr
            tp1 = self.data.Close[-1] + slatr * TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.trade_size)

        elif self.signal1 == 1 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] + slatr
            tp1 = self.data.Close[-1] - slatr * TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.trade_size)


class StopLossStrategy2(Strategy):
    trade_size = 0.99
    slcoef = 1.2  # 1.3
    TPcoef = 2  # 1.8

    def init(self):
        super().init()
        self.signal1 = self.I(lambda: self.data.Signal)

    def next(self):
        super().next()
        slatr = self.slcoef * self.data.atr[-1]
        TPcoef = self.TPcoef

        if len(self.trades) > 0:
            if self.trades[-1].is_long and self.data.rsi[-1] >= 90:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.rsi[-1] <= 10:
                self.trades[-1].close()

        if self.signal1 == 2 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] - slatr
            tp1 = self.data.Close[-1] + slatr * TPcoef
            self.buy(sl=sl1, tp=tp1, size=self.trade_size)

        elif self.signal1 == 1 and len(self.trades) == 0:
            sl1 = self.data.Close[-1] + slatr
            tp1 = self.data.Close[-1] - slatr * TPcoef
            self.sell(sl=sl1, tp=tp1, size=self.trade_size)


class TSMixerStrategy(Strategy):
    dp = None
    model = None
    window_size = 8

    def init(self):
        super().init()
        self.Close = self.I(lambda: self.data.Close).reshape(-1, 1)

        self.forecasts = self.I(
            lambda: np.repeat(np.nan, len(self.data)), name="forecast"
        )

        print(f'windows_size: {self.window_size}')

    def next(self):
        super().next()
   
        if len(self.data) <= self.window_size + 1:
            return

        scaled_prices = self.dp._make_dataset(
            self.dp.scaler.transform(
                self.Close[-(self.window_size + 1) :]
            ),
            shuffle=False,
        )

        predicted_close_price = self.dp.inverse_transform(
            self.model.predict(scaled_prices, verbose=0)[-1, :, :]
        )[0][0]

        self.forecasts[-1] = predicted_close_price

        if self.Close / predicted_close_price < 0.99:
            self.position.close()
            self.buy()

        elif self.Close / predicted_close_price > 1.01:
            self.position.close()
            self.sell()


# class TSMixerEnsembleStrategy(Strategy):
#     n1 = 5
#     n2 = 10

#     def init(self) -> None:
#         self.data_loader = data_loader
#         self.model = model
#         self.sma1 = self.I(SMA, self.data.Close, self.n1)
#         self.sma2 = self.I(SMA, self.data.Close, self.n2)

#         self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

#     def next(self) -> None:
#         if len(self.data) <= TSMIXER_CFG.WINDOW+1:
#             return

#         scaled_prices = self.data_loader._make_dataset(
#             self.data_loader.scaler.transform(self.data.df[["Close"]][-(TSMIXER_CFG.WINDOW+1):]),
#             shuffle=False
#             )

#         predicted_close_price = self.data_loader.inverse_transform(
#             self.model.predict(scaled_prices, verbose=0)[-1,:,:]
#             )[0][0]

#         self.forecasts[-1] = predicted_close_price

#         if crossover(self.sma1, self.sma2) and (self.data.Close / predicted_close_price < 0.99):
#             self.position.close()
#             self.buy()

#         elif crossover(self.sma2, self.sma1) and (self.data.Close / predicted_close_price > 1.01):
#             self.position.close()
#             self.sell()

if __name__ == "__main__":
    pass
