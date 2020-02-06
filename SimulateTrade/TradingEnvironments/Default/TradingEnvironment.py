import numpy as np

#initialization
#initializes indicators (dict of list of Indicators)
#initializes environment variables (index, half-spread/commission etc.)

#execution
#takes in trades
#every iteration, returns closing price & indicator values
#update trade cost/earnings when enter/exit is called (enter & exit at next opening price)

#termination
#results = no. of iterations, no. of trades, stop/loss hits, realized & unrealized p/l, account value, actual account value, return %, trading service costs
#return results
#print results

#METHODS
#iterate: returns closing price for new bar, indicator values
#enterLong: calls enter with longEntry object
#enterShort: calls enter with shortEntry object
#enter: update the total trading costs to be subtracted from the account (half-spread/commission + unit prices)
#exit: update the total trading earnings to be added to the account
#getIndicatorValues: returns indicator values for all indicators for the bar

class LongEntry():

    def __init__(self, rate, units, stop_loss=None, take_profit=None):
        if rate <= 0.0:
            raise ValueError('rate must be > 0.0')
        self._rate = rate

        if units <= 0:
            raise ValueError('units must be > 0')
        self._units = units

        if stop_loss is not None and stop_loss >= rate:
            raise ValueError('stop loss must be < entry price')
        self._stop_loss = stop_loss

        if take_profit is not None and take_profit <= rate:
            raise ValueError('take profit must be > entry price')
        self._take_profit = take_profit
        self._in_trade = False

    @property
    def rate(self):
        return self._rate
    
    @property
    def units(self):
        return self._units
    
    @property
    def stop_loss(self):
        return self._stop_loss

    @property
    def take_profit(self):
        return self._take_profit

    @property
    def in_trade(self):
        return self._in_trade

    @property
    def trade_cost(self):
        return self._trade_cost

    @property
    def trade_earnings(self):
        return self._trade_earnings

    @stop_loss.setter
    def stop_loss(self, stop_loss):
        self._stop_loss = stop_loss

    @take_profit.setter
    def take_profit(self, take_profit):
        self._take_profit = take_profit

    def enter(self, entry_service_cost=1.0):
        if entry_service_cost < 0.0:
            raise ValueError('Service cost must be >= 0.0')
        self._in_trade = True

        self._trade_cost = self._rate * self._units + entry_service_cost

        return self._trade_cost, entry_service_cost

    def feedNewRates(self, bar_ohlcv, exit_service_cost=1.0):
        if exit_service_cost < 0.0:
            raise ValueError('Service cost must be >= 0.0')
        if not self._in_trade:
            raise AssertionError('Trade must be active first')
        if self._stop_loss and bar_ohlcv[2] <= self._stop_loss:
            return self.exit(self._stop_loss, exit_service_cost)
        elif self._take_profit and bar_ohlcv[1] >= self._take_profit:
            return self.exit(self._take_profit, exit_service_cost)

    def exit(self, new_rate, exit_service_cost=1.0):
        if exit_service_cost < 0.0:
            raise ValueError('Service cost must be >= 0.0')

        if not self._in_trade:
            raise AssertionError('Trade must be active first')

        self._in_trade = False

        self._trade_earnings = self._rate * self._units - exit_service_cost
        return self._trade_earnings, exit_service_cost, (self._trade_earnings - self._trade_cost)

    def getCurrentTradeValue(self, new_rate):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')

        return new_rate * self._units

class ShortEntry():

    def __init__(self, rate, units, stop_loss=None, take_profit=None):
        if rate <= 0.0:
            raise ValueError('rate must be > 0.0')
        self._rate = rate

        if units <= 0:
            raise ValueError('units must be > 0')
        self._units = units

        if stop_loss is not None and stop_loss >= rate:
            raise ValueError('stop loss must be < entry price')
        self._stop_loss = stop_loss

        if take_profit is not None and take_profit <= rate:
            raise ValueError('take profit must be > entry price')
        self._take_profit = take_profit
        self._in_trade = False

    @property
    def rate(self):
        return self._rate
    
    @property
    def units(self):
        return self._units
    
    @property
    def stop_loss(self):
        return self._stop_loss

    @property
    def take_profit(self):
        return self._take_profit

    @property
    def in_trade(self):
        return self._in_trade

    @property
    def trade_cost(self):
        return self._trade_cost

    @property
    def trade_earnings(self):
        return self._trade_earnings

    @stop_loss.setter
    def stop_loss(self, stop_loss):
        self._stop_loss = stop_loss

    @take_profit.setter
    def take_profit(self, take_profit):
        self._take_profit = take_profit

    def enter(self, entry_service_cost=1.0):
        if entry_service_cost < 0.0:
            raise ValueError('Service cost must be >= 0.0')
        self._in_trade = True

        self._trade_cost = self._rate * self._units + entry_service_cost

        return self._trade_cost, entry_service_cost

    def feedNewRates(self, bar_ohlcv, exit_service_cost=1.0):
        if exit_service_cost < 0.0:
            raise ValueError('Service cost must be >= 0.0')
        if not self._in_trade:
            raise AssertionError('Trade must be active first')
        if self._stop_loss and bar_ohlcv[2] >= self._stop_loss:
            return self.exit(self._stop_loss, exit_service_cost)
        elif self._take_profit and bar_ohlcv[1] <= self._take_profit:
            return self.exit(self._take_profit, exit_service_cost)

    def exit(self, new_rate, exit_service_cost=1.0):
        if exit_service_cost < 0.0:
            raise ValueError('Service cost must be >= 0.0')

        if not self._in_trade:
            raise AssertionError('Trade must be active first')

        self._in_trade = False

        self._trade_earnings = (self._rate - new_rate) * self._units - exit_service_cost
        return self._trade_earnings, exit_service_cost, (self._trade_earnings - self._trade_cost)

    def getCurrentTradeValue(self, new_rate):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')

        return (self._rate + (self._rate - new_rate)) * self._units

class TradingEnvironment():

    def __init__(self, ohlcv, start, initial_account_value=10000):

        self._ohlcv = ohlcv

        if initial_account_value < 500.0:
            raise ValueError('Initial account value must be >= 500.00')

        #ENVIRONMENT VARIABLES
        self._start = start
        self._initial_account_value = initial_account_value
        self._bar = self._start - 1
        self._iteration = 0
        self._no_of_trades = 0
        self._take_profit_hits = 0
        self._stop_loss_hits = 0
        self._service_cost = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype) #costs of trades
        self._net_earnings = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype) #net earnings of trades
        self._account_value = np.full((self._ohlcv.shape[0], 1), initial_account_value, dtype=self._ohlcv.dtype) #actual value of account
        self._equity_curve = np.full((self._ohlcv.shape[0], 1), initial_account_value, dtype=self._ohlcv.dtype) #net increasing/decreasing value of account
        self._in_trade = False
        self._trade = None

    @property
    def start(self):
        return self._start
    
    @property
    def initial_account_value(self):
        return self._initial_account_value
    
    @property
    def bar(self):
        return self._bar
    
    @property
    def iteration(self):
        return self._iteration
    
    @property
    def no_of_trades(self):
        return self._no_of_trades

    @property
    def take_profit_hits(self):
        return self._take_profit_hits
    
    @property
    def stop_loss_hits(self):
        return self._stop_loss_hits
    
    @property
    def service_cost(self):
        return self._service_cost

    @property
    def net_earnings(self):
        return self._net_earnings
     
    @property
    def account_value(self):
        return self._account_value

    @property
    def equity_curve(self):
        return self._equity_curve
    
    @property
    def in_trade(self):
        return self._in_trade

    @property
    def trade(self):
        return self._trade
    
    def iterate(self):
        if self.isEnd():
            return

        self._bar += 1

        self._iteration += 1
        self._initializeAccountValue()
        self._initializeEquityCurve()

        if self._in_trade:
            values = self._trade.feedNewRates(self._ohlcv[self._bar])

            if values:
                trade_earnings, service_cost, net_earnings = values
                self._autoExit(trade_earnings, service_cost, net_earnings)

        return self._ohlcv[bar, 3]

    def isEnd(self):
        return self._bar >= self._ohlcv.shape[0] - 1

    def enterLong(self, units, stop_loss=None, take_profit=None):
        if self._in_trade:
            raise AssertionError('Account currently has an active trade')
        self._enter('long', units, stop_loss, take_profit)

    def enterShort(self, units, stop_loss=None, take_profit=None):
        if self._in_trade:
            raise AssertionError('Account currently has an active trade')
        self._enter('short', units, stop_loss, take_profit)

    def _enter(self, order_type, units, stop_loss=None, take_profit=None):
        if order_type != 'long' and order_type != 'short':
            raise ValueError('Order must be either long or short')

        if self._bar >= self._ohlcv.shape[0] - 1: #ignore entry at last bar of data
            return

        self._in_trade = True

        if order_type is 'long':
            self._trade = LongEntry(self.getCurrentPrice(), units, stop_loss, take_profit) #enter at next opening price

        else:
            self._trade = ShortEntry(self.getCurrentPrice(), units, stop_loss, take_profit) #enter at next opening price

        trade_cost, service_cost = self._trade.enter()

        self._updateAccountValue(-trade_cost)
        self._service_cost[self._bar, 0] = service_cost

    def exit(self):
        if self._bar >= self._ohlcv.shape[0] - 1: #ignore exit at last bar of data
            return
            
        self._in_trade = False

        trade_earnings, service_cost, net_earnings = self._trade.exit(self._ohlcv[self._bar + 1, 0]) #exit at next opening price

        self._updateAccountValue(trade_earnings)
        self._updateEquityCurve(net_earnings)
        self._service_cost[self._bar, 0] = service_cost
        self._net_earnings[self._bar, 0] = net_earnings

        self._trade = None
        self._no_of_trades += 1

    def _autoExit(self, trade_earnings, service_cost, net_earnings):

        self._in_trade = False
        self._updateAccountValue(trade_earnings)
        self._updateEquityCurve(net_earnings)
        self._service_cost[self._bar, 0] = service_cost
        self._net_earnings[self._bar, 0] = net_earnings
        self._trade = None
        self._no_of_trades += 1

        if net_earnings > 0:
            self._takeProfitExit()
        else:
            self._stopLossExit()

    def _stopLossExit(self):
        self._stop_loss_hits += 1

    def _takeProfitExit(self):
        self._take_profit_hits += 1

    def _initializeAccountValue(self):
        self._account_value[self._bar, 0] = self._account_value[self._bar - 1, 0]
    def _initializeEquityCurve(self):
        self._equity_curve[self._bar, 0] = self._equity_curve[self._bar - 1, 0]

    def _updateAccountValue(self, value):
        self._account_value[self._bar, 0] = self._account_value[self._bar, 0] + value

    def _updateEquityCurve(self, value):
        self._equity_curve[self._bar, 0] = self._equity_curve[self._bar, 0] + value

    def getResults(self):
        return self._iteration + 1, self._no_of_trades, self._stop_loss_hits, \
                self._equity_curve, self._account_value, self.getReturn(), self.service_cost

    def getCurrentAccountValue(self):
        return self._account_value[self._bar, 0]

    def getCurrentEquityCurve(self):
        return self._equity_curve[self._bar, 0]

    def getCurrentPrice(self):
        return self._ohlcv[self._bar + 1, 0]

    def printResults(self):
        print('RESULTS:')
        print('No. of bars: {}'.format(self._iteration))
        print('No. of trades: {}'.format(self._no_of_trades))
        print('Take-profit hits: {}'.format(self._take_profit_hits))
        print('Stop-loss hits: {}'.format(self._stop_loss_hits))
        print('Realized P/L: {}'.format(self.getCurrentEquityCurve() - self._initial_account_value))
        if self._in_trade:
            print('Unrealized P/L: {}'.format(self._trade.getCurrentTradeValue(self._ohlcv[self._bar, 3])))
        print('End account value: {}'.format(self.getCurrentAccountValue()))
        print('Return: {:.2f}%'.format(self.getReturn()))
        print('Total cost: {}'.format(np.sum(self._service_cost)))

    def getReturn(self):

        return ((self.getCurrentAccountValue() - self._initial_account_value) / self._initial_account_value) * 100

    '''
        Trading System Metrics:
        The metrics are numerous ways to gauge the profitability, accuracy and performance consistency of a trading algorithm.

    '''

    def profitFactor(self):
        profit = np.sum(self._net_earnings[np.where(self._net_earnings > 0)[0]])
        loss = np.abs(np.sum(self._net_earnings[np.where(self._net_earnings < 0)[0]]))

        return profit / loss

    def CAR(self):
        n = (self._ohlcv.shape[0] - self._start - 1) / 365
        ending_balance = self._equity_curve[-1, 0]
        beginning_balance = self._equity_curve[self._start, 0]

        car = pow(ending_balance / beginning_balance, 1 / n) - 1

        return car

    def MDD(self):
        ending_drawdown = np.argmax(np.maximum.accumulate(self._equity_curve) - self._equity_curve)
        beginning_drawdown = np.argmax(self._equity_curve[:ending_drawdown])

        #(Trough value - Peak value) / Peak value
        mdd = (self._equity_curve[ending_drawdown, 0] - self._equity_curve[beginning_drawdown, 0]) / self._equity_curve[beginning_drawdown, 0]

        return mdd

    def MAR(self):
        return self.CAR() / np.abs(self.MDD())

    def printMetrics(self):

        print('METRICS:')
        print('Profit Factor: {}'.format(self.profitFactor()))
        print('Compounded Annual Return (CAR): {}'.format(self.CAR()))
        print('Maximum Drawdown (MDD): {}'.format(self.MDD()))
        print('CAR/MDD (MAR): {}'.format(self.MAR()))

