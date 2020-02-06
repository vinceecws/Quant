from ..Default.TradingEnvironment import *
import numpy as np

class ForexLongEntry(LongEntry):
    
    def __init__(self, rate, units, pip_denomination, stop_loss=None, take_profit=None, trade_margin=0.05):
        super(ForexLongEntry, self).__init__(rate, units, stop_loss, take_profit)
        self._pip_denomination = pip_denomination
        self._trade_margin = trade_margin
        self._value_per_pip = self._pip_denomination * self._units
        self._entry_half_spread = None
        self._exit_half_spread = None

    @property
    def value_per_pip(self):
        return self._value_per_pip

    @property
    def trade_margin(self):
        return self._trade_margin

    @property
    def entry_half_spread(self):
        return self._entry_half_spread

    @property
    def exit_half_spread(self):
        return self._exit_half_spread

    def enter(self, entry_half_spread):
        self._in_trade = True
        if entry_half_spread < 0.0:
            raise ValueError('half-spread must be >= 0.0')
        self._entry_half_spread = entry_half_spread

        self._margin_cost = self._rate * self._units * self._trade_margin
        half_spread_cost = self._entry_half_spread * self._units * self._trade_margin
        self._trade_cost = self._margin_cost + half_spread_cost

        return self._trade_cost, half_spread_cost

    def feedNewRates(self, bar_ohlcv, exit_half_spread):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')
        if self._stop_loss and bar_ohlcv[2] <= self._stop_loss:
            return self.exit(self._stop_loss, exit_half_spread)
        elif self._take_profit and bar_ohlcv[1] >= self._take_profit:
            return self.exit(self._take_profit, exit_half_spread)

    def exit(self, new_rate, exit_half_spread):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')
        if exit_half_spread < 0.0:
            raise ValueError('half-spread must be >= 0.0')
        self._in_trade = False
        self._exit_half_spread = exit_half_spread

        margin_earnings = self.pip(new_rate - self._rate) * self._value_per_pip
        half_spread_cost = self.exit_half_spread * self._units * self._trade_margin
        self._trade_earnings = self._margin_cost + margin_earnings - half_spread_cost
        return self._trade_earnings, half_spread_cost, (self._trade_earnings - self._trade_cost)

    def getCurrentTradeValue(self, new_rate):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')

        return new_rate * self._units * self._trade_margin

    def pip(self, value):
        return np.round(value / self._pip_denomination, 1)

class ForexShortEntry(ShortEntry):

    def __init__(self, rate, units, pip_denomination, stop_loss=None, take_profit=None, trade_margin=0.05):
        if rate <= 0.0:
            raise ValueError('rate must be > 0.0')
        self._rate = rate

        if units <= 0:
            raise ValueError('units must be > 0')
        self._units = units
        self._pip_denomination = pip_denomination
        if stop_loss is not None and stop_loss <= rate:
            raise ValueError('stop loss must be > entry price')
        self._stop_loss = stop_loss

        if take_profit is not None and take_profit >= rate:
            raise ValueError('take profit must be < entry price')
        self._take_profit = take_profit
        self._trade_margin = trade_margin
        self._value_per_pip = self._pip_denomination * self._units
        self._entry_half_spread = None
        self._exit_half_spread = None
        self._in_trade = False

    @property
    def rate(self):
        return self._rate
    
    @property
    def units(self):
        return self._units

    @property
    def pip_denomination(self):
        return self._pip_denomination
    
    @property
    def stop_loss(self):
        return self._stop_loss

    @property
    def take_profit(self):
        return self._take_profit

    @property
    def trade_margin(self):
        return self._trade_margin

    @property
    def value_per_pip(self):
        return self._value_per_pip

    @property
    def value_per_pip(self):
        return self._value_per_pip

    @property
    def entry_half_spread(self):
        return self._entry_half_spread

    @property
    def exit_half_spread(self):
        return self._exit_half_spread

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

    def enter(self, entry_half_spread):
        self._in_trade = True
        if entry_half_spread < 0.0:
            raise ValueError('half-spread must be >= 0.0')
        self._entry_half_spread = entry_half_spread

        self._margin_cost = self._rate * self._units * self._trade_margin
        half_spread_cost = self._entry_half_spread * self._units * self._trade_margin
        self._trade_cost = self._margin_cost + half_spread_cost

        return self._trade_cost, half_spread_cost

    def feedNewRates(self, bar_ohlcv, exit_half_spread):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')
        if self._stop_loss and bar_ohlcv[1] >= self._stop_loss:
            return self.exit(self._stop_loss, exit_half_spread)
        elif self._take_profit and bar_ohlcv[2] <= self._take_profit:
            return self.exit(self._take_profit, exit_half_spread)

    def exit(self, new_rate, exit_half_spread):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')
        if exit_half_spread < 0.0:
            raise ValueError('half-spread must be >= 0.0')
        self._in_trade = False
        self._exit_half_spread = exit_half_spread

        #Emulates long entry: Price increasing in a short entry is equal to price decreasing in a long entry, and vice versa
        margin_earnings = self.pip(self._rate - new_rate) * self._value_per_pip
        half_spread_cost = self.exit_half_spread * self._units * self._trade_margin
        self._trade_earnings = self._margin_cost + margin_earnings - half_spread_cost
        return self._trade_earnings, half_spread_cost, (self._trade_earnings - self._trade_cost)

    def getCurrentTradeValue(self, new_rate):
        if not self._in_trade:
            raise AssertionError('Trade must be active first')

        return (self._rate + (self._rate - new_rate)) * self._units * self._trade_margin

    def pip(self, value):
        return np.round(value / self._pip_denomination, 1)

class ForexTradingEnvironment(TradingEnvironment):

    def __init__(self, ohlcv, start, pip_denomination, min_spread=1, initial_account_value=10000, trade_margin=0.05):
        super(ForexTradingEnvironment, self).__init__(ohlcv, start, initial_account_value)
        self._pip_denomination = pip_denomination
        self._min_spread = min_spread
        self._trade_margin = trade_margin

    def iterate(self):
        if self.isEnd():
            return

        self._bar += 1

        self._iteration += 1
        self._initializeAccountValue()
        self._initializeEquityCurve()

        if self._in_trade:
            values = self._trade.feedNewRates(self._ohlcv[self._bar], self._getHalfSpread())

            if values:
                trade_earnings, service_cost, net_earnings = values
                self._autoExit(trade_earnings, service_cost, net_earnings)

        return self._ohlcv[self._bar, 3]

    def _enter(self, order_type, units, stop_loss=None, take_profit=None):
        if order_type != 'long' and order_type != 'short':
            raise ValueError('Order must be either long or short')

        if self._bar >= self._ohlcv.shape[0] - 1: #ignore entry at last bar of data
            return

        self._in_trade = True

        if order_type is 'long':
            self._trade = ForexLongEntry(self.getCurrentPrice(), units, self._pip_denomination, stop_loss, take_profit) #enter at next opening price

        else:
            self._trade = ForexShortEntry(self.getCurrentPrice(), units, self._pip_denomination, stop_loss, take_profit) #enter at next opening price

        trade_cost, service_cost = self._trade.enter(self._getHalfSpread())
        self._updateAccountValue(-trade_cost)
        self._service_cost[self._bar, 0] = service_cost

    def exit(self):
        if self._bar >= self._ohlcv.shape[0] - 1: #ignore exit at last bar of data
            return

        self._in_trade = False

        trade_earnings, service_cost, net_earnings = self._trade.exit(self._ohlcv[self._bar + 1, 0], self._getHalfSpread())

        self._updateAccountValue(trade_earnings)
        self._updateEquityCurve(net_earnings)
        self._service_cost[self._bar, 0] = service_cost
        self._net_earnings[self._bar, 0] = net_earnings

        self._trade = None
        self._no_of_trades += 1

    '''
        getHalfSpread: 
        Calculates the half-spread cost based on the mean and standard deviation of both volatility and volume within a given window.
        Typical window size is 30 days.
        Volume/volatility is categorized into 3 types:
            1) Volume/volatility is average (within 1 standard deviation of past data [inclusive])
            2) Volume/volatility is low (below 1 standard deviation of past data [exclusive])
            3) Volume/volatility is high (above 1 standard deviation of past data [exclusive])
        Based on the above criteria, a multiplier is assigned as follows:
            1) Average volume or Average volatility -> 0.65
            1) High volume or Low volatility -> 0.5
            2) Low volume or High volatility -> 1.0

        Parameter(s):
            bar: The bar of which the spread value is calculated
            min_spread: Minimum spread value allowed for the index (pips) (default: 1)
            window: Timeframe of data (default: 30)

        Return(s):
            spread: The spread cost based on minimum spread, volume and volatility multiplier
    '''

    def _getHalfSpread(self, window=30): #Calculates half spread for next bar
        if window < 1:
            raise ValueError('Window size must be >= 1')

        min_spread_val = self._min_spread * self._pip_denomination
        first = max(0, self._bar - (window - 1)) #First bar of window
        volume = self._ohlcv[self._bar, 4] 
        volatility = self._ohlcv[self._bar, 1] - self._ohlcv[self._bar, 2]
        volume_std = np.std(self._ohlcv[first:self._bar, 4])
        volume_mean = np.mean(self._ohlcv[first:self._bar, 4])
        volatility_std = np.std(self._ohlcv[first:self._bar, 1] - self._ohlcv[first:self._bar, 2])
        volatility_mean = np.mean(self._ohlcv[first:self._bar, 1] - self._ohlcv[first:self._bar, 2])

        if (volume_mean - volume_std) <= volume <= (volume_mean + volume_std):
            volume_multiplier = 0.65 #Average volume
        elif volume < (volume_mean - volume_std):
            volume_multiplier = 1 #Low volume
        else:
            volume_multiplier = 0.5 #High volume

        if (volatility_mean - volatility_std) <= volatility <= (volatility_mean + volatility_std):
            volatility_multiplier = 0.65 #Average volatility
        elif (volatility < -volatility_std):
            volatility_multiplier = 0.5 #Low volatility
        else:
            volatility_multiplier = 1 #High volatility

        return (volume_multiplier + volatility_multiplier) * min_spread_val


