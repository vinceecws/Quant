class TradeEngine():

    def __init__(self, trade_interface):

        self._trade_interface = trade_interface

    def run(self):

        while not self._trade_interface.trading_environment.isEnd():
            print('Account Value: {} | Equity Value: {}'.format(self._trade_interface.trading_environment.getCurrentAccountValue(), self._trade_interface._trading_environment.getCurrentEquityCurve()))
            close = self._trade_interface.trading_environment.iterate()
            self._trade_interface.forward(close)
